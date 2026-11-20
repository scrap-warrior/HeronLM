"""
Training and evaluation for the arms.

Two things this file is not:
  - it is not the reference. See phase0/__init__.py for why the separation matters.
  - it is not a quality run. At TINY scale these numbers validate the harness: they show that a
    mechanism changes loss at all, which is the precondition for a gate being decidable at the real
    scale. A tiny arm cannot decide a gate whose threshold is stated in nats at 200M params.

The same code runs at either scale, which is why nothing here assumes CPU. `device()` picks CUDA when a
card is present, and the arms and thresholds are the same in both cases.

Two things here exist for the gate-scale run rather than for the tiny test:
  - a `SampleStream` supplies fresh samples on demand. A fixed 64-sample pool repeated for a long run is a
    memorization measurement, not a depth measurement.
  - an arm can checkpoint and resume. A gate arm runs for hours, and a crash that loses it is a data point
    that has to be bought again. The checkpoint stores both RNG states and the recorded orders, so the
    resumed run is the run that would have happened without the interruption, not an approximation of it.
"""

import dataclasses
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from heronlm.memory import slot_ids
from heronlm.model import HeronLM
from phase0.arms import Arm, RunSpec
from phase0.corpus import Sample, SampleStream, make_corpus
from phase0.tokens import TokenizedCorpus, TokenStream


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def param_groups(model: HeronLM, lr: float, wd: float) -> list[dict]:
    """
    Two groups, because the table rows are not decayed.

    The rule in config is that table rows go through Adam without weight decay and only their
    projections go through Muon. Decaying everything uniformly would shrink rows that are updated sparsely,
    which is the opposite of what a lookup table needs: a row that is hit twice a month should not be
    pulled toward zero by a decay term that runs every step.

    What this function actually does, stated plainly: it applies AdamW to everything, including the
    projections the rule says go through Muon. Muon is not implemented here. So the loss numbers these arms
    produce are AdamW numbers, and a gate result cannot be read as evidence about the optimizer rule - it is
    evidence about the mechanism the arm varies, measured under one optimizer that is not the one the design
    claims for the target model. Either Muon is implemented before the gate run, or the run record has to say
    that the rule was not exercised.
    """
    decay, nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nodecay if "memory.table" in name else decay).append(p)
    return [
        {"params": decay, "weight_decay": wd},
        {"params": nodecay, "weight_decay": 0.0},
    ]


def _batch(samples: list[Sample], dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.tensor([list(s.tokens) for s in samples], device=dev)
    targets = torch.tensor([list(s.targets) for s in samples], device=dev)
    return tokens, targets


def _to_cpu(value):
    """
    Copy every tensor in a checkpoint payload to CPU, leaving RNG states and plain data alone.

    Saving directly from a CUDA model would keep the tensors on the card; a checkpoint is storage, not
    resident state. The recursive form is needed because the optimizer state is nested dictionaries of
    tensors and parameter-group metadata.
    """
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(v) for v in value)
    return value


def _run_key(arm: Arm, corpus_id: str) -> dict:
    """
    The fields that make a checkpoint reusable for this arm.

    `steps` is deliberately absent: a checkpoint written at step 500 has to be resumable by a run that was
    later extended to 2000 steps. Everything that changes what the weights or the data mean is present, so a
    checkpoint from a different arm, seed, depth distribution, model config or corpus is rejected rather than
    resumed into a run whose result would not belong to this gate.
    """
    spec = arm.spec
    return {
        "gate": arm.gate,
        "name": arm.name,
        "cfg": dataclasses.asdict(arm.cfg),
        "seed": spec.seed,
        "batch": spec.batch,
        "seq_len": spec.seq_len,
        "lr": spec.lr,
        "wd": spec.wd,
        "train_k": tuple(spec.train_k),
        "families": tuple(spec.families),
        "n_train": spec.n_train,
        "n_test": spec.n_test,
        # The corpus id is what separates a synthetic stream from a token-shard corpus and one token corpus
        # from another. Without it, a checkpoint resumed against different data would produce a loss that is
        # not comparable to its own earlier history.
        "corpus": corpus_id,
    }


def training_key(arm: Arm, corpus_id: str) -> str:
    """
    The identity of a training run, independent of the gate that is using it.

    The five gate baselines are the same config and the same spec, seeded the same way, so training them five
    times would produce five bit-identical models at five times the cost. This key is what lets the driver
    train one and reuse it. It deliberately excludes `gate`, `name`, `variable` and `claim`: those label the
    comparison, they do not change what the optimizer does. It includes every config field and every spec
    field that feeds training, plus the corpus id, so an arm that differs in anything that can move a weight
    gets a different key and is trained separately.

    `steps` is included here, unlike in the checkpoint run key. A checkpoint is meant to be extended, so it
    must survive a larger `steps`; a shared training result is not, so two runs with different step counts
    must not be confused.
    """
    spec = arm.spec
    payload = {
        "cfg": dataclasses.asdict(arm.cfg),
        "seed": spec.seed,
        "steps": spec.steps,
        "batch": spec.batch,
        "seq_len": spec.seq_len,
        "lr": spec.lr,
        "wd": spec.wd,
        "train_k": tuple(spec.train_k),
        "eval_k": tuple(spec.eval_k),
        "families": tuple(spec.families),
        "n_train": spec.n_train,
        "n_test": spec.n_test,
        "corpus": corpus_id,
    }
    # default=str covers config values that are not JSON-native (tuples are already converted above); the key
    # is an identity string, not data that anything parses back.
    return json.dumps(payload, sort_keys=True, default=str)


# --- What the arithmetic actually is ------------------------------------------------
#
# A gate run on a card is not the same arithmetic as the reference tests on this machine. Three things decide
# that: the torch build, whether TF32 is allowed for fp32 matmul, and whether anything casts. The reference
# path is fp32 and nothing in the harness casts, so the only silent difference is TF32, and it is silent
# because its default is set by the device rather than by the code. A loss number that came from TF32 matmul
# is a different number from the same arm on CPU, and the parity claim in the reference package is a claim
# about one environment, not about every environment.
#
# So the flags are read from torch rather than assumed, and both are written into the run record. Turning
# TF32 off is the default because the gate numbers are meant to be comparable with the reference; a run that
# wants TF32 for throughput has to ask for it, and the record then says which arithmetic produced the result.

def precision_flags() -> dict:
    """The arithmetic settings in effect, read from torch rather than from what the code intends."""
    return {
        "torch": torch.__version__,
        "torch_cuda_build": str(torch.version.cuda),
        "device": str(device()),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "dtype": "float32",
        "autocast": "not used anywhere in the harness",
    }


def set_tf32(allow: bool) -> dict:
    """Set both TF32 flags and return what is in effect, so the record states the value that was used."""
    torch.backends.cuda.matmul.allow_tf32 = bool(allow)
    torch.backends.cudnn.allow_tf32 = bool(allow)
    return precision_flags()


# --- The recorded orders, in the form a long run can afford -------------------------
#
# `data_order` and `k_order` are what make a between-arm loss difference attributable to the arm's variable,
# so they have to survive the run. At the token budget ROADMAP line 95 states, an arm records around a
# million steps: as a list of lists of Python ints that is tens of millions of objects, and once the arms run
# for the gate budget the record costs more than the weights do.
#
# Fixed-width arrays carry the same facts at 4 bytes per token position and 1 byte per step. The row
# boundaries are reconstructed from the batch size, which is part of the run key, so nothing is lost.
#
# The hash is the part that makes the compact form checkable: `sha256(order bytes + k bytes)` is recorded, so
# a resume can verify that the order it is reading is the order that was written. A record that lost half its
# rows would otherwise resume quietly with a shorter order, and the run's data order would no longer be the
# one the arm was measured under.

def orders_sha256(data: np.ndarray, k: np.ndarray) -> str:
    """Hash the canonical bytes of both orders. Byte order is fixed by the dtypes, not by the platform."""
    digest = hashlib.sha256()
    digest.update(data.tobytes())
    digest.update(k.tobytes())
    return digest.hexdigest()


def order_arrays(data_order: list, k_order: list, batch: int) -> dict:
    """The orders as fixed-width arrays plus the hash of their bytes."""
    data = np.fromiter((int(i) for row in data_order for i in row), dtype="<u4",
                       count=sum(len(row) for row in data_order))
    k = np.fromiter((int(i) for i in k_order), dtype="u1", count=len(k_order))
    if data.size % batch != 0:
        raise ValueError(f"data order has {data.size} entries, not a multiple of batch {batch}")
    if k.size != data.size // batch:
        raise ValueError(f"k order has {k.size} entries, expected {data.size // batch} rows")
    return {"data": data, "k": k, "batch": batch, "sha256": orders_sha256(data, k)}


def orders_from_arrays(orders: dict, batch: int) -> tuple[list, list]:
    """Rebuild the list form the tests and the report use, and verify the record against its own hash."""
    data, k = orders["data"], orders["k"]
    if data.size % batch != 0:
        raise ValueError(f"checkpoint order has {data.size} entries, not a multiple of batch {batch}")
    if orders_sha256(data, k) != orders["sha256"]:
        raise ValueError(
            f"checkpoint order record does not match its own hash ({orders['sha256']}): the order is "
            f"truncated or was written by a different record. Refusing to resume on an order that cannot "
            f"be verified, because the data order is the reason a between-arm difference means anything."
        )
    if k.size != data.size // batch:
        raise ValueError(f"checkpoint k order has {k.size} entries, expected {data.size // batch} rows")
    return [list(row) for row in data.reshape(-1, batch).tolist()], [int(i) for i in k.tolist()]


def orders_from_payload(loaded: dict) -> tuple[list, list, str]:
    """
    Read the recorded orders from a checkpoint payload, in whichever form it stored them.

    Format 2 stores the compact arrays; format 1 stored Python lists. Both are accepted, because a gate run
    that was interrupted before the record form changed has to remain resumable, and the hash is recomputed
    either way so the record can state the same fact for both.
    """
    batch = int(loaded["run_key"]["batch"])
    if "orders" in loaded:
        data_order, k_order = orders_from_arrays(loaded["orders"], batch)
        return data_order, k_order, orders_sha256(loaded["orders"]["data"], loaded["orders"]["k"])
    data_order = [list(row) for row in loaded["data_order"]]
    k_order = [int(i) for i in loaded["k_order"]]
    return data_order, k_order, orders_sha256(
        order_arrays(data_order, k_order, batch)["data"], order_arrays(data_order, k_order, batch)["k"])


def _archive_current(path: Path, step: int) -> Path | None:
    """
    Move the checkpoint that is about to be replaced into a numbered archive.

    The current file stays the current file: `--resume` reads it, and a rotation policy that renamed it would
    change what a resumed run finds. Archiving the old one first means the newest N are all present, in one
    directory, with the newest at the name the run already expects.
    """
    if not path.exists():
        return None
    archive = path.with_name(f"{path.stem}.{step:07d}.pt")
    os.replace(path, archive)
    return archive


def _keep_newest(path: Path, keep: int) -> list[str]:
    """
    Keep the newest `keep - 1` archives next to the current checkpoint; delete the rest.

    `keep = 0` prunes nothing: every record the run wrote survives, which is what a run that wants to be able
    to go back to any of its positions needs. The distinction matters because a policy that keeps nothing is
    not the same as a policy that keeps everything, and the code path that writes archives is shared by both.
    """
    stem = path.stem
    if "." in stem:
        raise ValueError(f"checkpoint name {path.name} contains a dot; the archive name is built from it")
    archives = [p for p in path.parent.glob(f"{stem}.*.pt") if p.name != path.name]
    if not archives:
        return [path.name]
    def step_of(p: Path) -> int:
        return int(p.name[len(stem) + 1: p.name.rindex(".")])
    archives.sort(key=step_of)
    if keep:
        for old in archives[:max(0, len(archives) - (keep - 1))]:
            old.unlink()
    kept = sorted([p for p in path.parent.glob(f"{stem}.*.pt") if p.name != path.name], key=step_of)
    return [p.name for p in kept] + [path.name]


def _save_checkpoint(path: str | Path, arm: Arm, step: int, model: HeronLM,
                     opt: torch.optim.Optimizer, k_rng: random.Random, data_rng: random.Random,
                     history: list, data_order: list, k_order: list, train_loss: float,
                     corpus_id: str, keep: int = 0, archive_step: int | None = None) -> list[str]:
    """
    Write one checkpoint atomically.

    The temporary file matters: a run that is killed while writing must leave the previous good checkpoint in
    place, not a truncated file that the next resume rejects. os.replace is atomic on both NTFS and POSIX for
    a same-directory rename.

    `keep` is the rotation policy. Writing every checkpoint a long run produces is not a policy, it is an
    accident: at the gate token budget with a save every 500 steps, one arm writes around two thousand files
    of about 3.4 GB each, which is terabytes for one arm and beyond any storage quota a free instance has.
    With `keep = N` the newest N records exist and the older ones are deleted, so the run can be resumed from
    any of them and the storage cost is bounded by N. `keep = 0` keeps every record the run writes, which is
    what a run that wants to be able to go back to any of its positions needs, and what the short tests do
    because they read a record back. `keep = 1` keeps only the current record.

    Archiving is what a retained record requires: the current file has to stay the current file, because that
    is the name `--resume` reads, so every record other than the current one survives under a numbered name. A
    policy that retained records without archiving them would simply overwrite them, which is the difference
    between "keeps one" and "keeps everything".

    `archive_step` is the step of the record being replaced, so the archive is named by the run position it
    holds rather than by a counter.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = [path.name]
    if keep != 1 and archive_step is not None:
        _archive_current(path, archive_step)
        kept = _keep_newest(path, keep)
    orders = order_arrays(data_order, k_order, arm.spec.batch)
    payload = {
        # Format 2 is the compact order record. Format 1 stored the orders as Python lists; a checkpoint
        # written in format 1 is still read (see `orders_from_payload`), so an interrupted run does not become
        # unresumable because the record form changed.
        "format": 2,
        "run_key": _run_key(arm, corpus_id),
        "step": step,
        "model": _to_cpu(model.state_dict()),
        "optimizer": _to_cpu(opt.state_dict()),
        # Both RNGs are state, not decoration. Without the k stream, a resumed sampled-k arm would draw a
        # different depth sequence; without the data stream, the fixed-pool path would replay different
        # sequences. The streaming path derives its indices from the step number, so it needs no cursor.
        "k_rng": k_rng.getstate(),
        "data_rng": data_rng.getstate(),
        # `history` is only populated when a run logs, so at gate scale it is small: a run that logs every
        # step of a million-step arm writes a million rows per checkpoint, which is the reason the driver logs
        # at an interval rather than every step.
        "history": list(history),
        "orders": orders,
        "orders_sha256": orders["sha256"],
        "train_loss": train_loss,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return sorted(kept)


def train_arm(arm: Arm, corpus: dict[str, list[Sample]], dev: torch.device | None = None,
              log_every: int = 0, *, train_stream: SampleStream | TokenStream | None = None,
              checkpoint_path: str | Path | None = None, checkpoint_every: int = 0,
              checkpoint_keep: int = 0, resume: bool = False,
              capture_initial_state: bool = True) -> dict:
    """
    One arm.

    The seed is set before the model is built, so two arms of the same gate start from identical weights.
    The loop depth is drawn from its own generator, so an arm that changes the training distribution (G0.6)
    still sees the same sequences in the same order as its baseline. Without both, a loss difference between
    arms would not be attributable to the flag being tested.

    `k` is sampled per batch rather than fixed per run, even when `train_k` has one value, so the same code
    path serves gate G0.6, where the variable is the distribution itself.

    Data comes from one of two places:
      - `train_stream`, the gate-scale path: sample i is generated on demand from the sample index, so a long
        run is not a repeated pass over a fixed pool and two arms still see identical data;
      - `corpus["train"]`, the fixed pool used by the Phase 1 driver and by the short tests, where the point
        is a reproducible table build rather than a data budget.

    A checkpoint stores weights, optimizer state, both RNG states and the recorded orders. `resume` validates
    the run key before loading, so a checkpoint from a different arm or config is rejected instead of being
    resumed into a model whose result would not belong to this gate. `checkpoint_keep` is the rotation policy
    for that directory (see `_save_checkpoint`): 0 keeps every record the run writes, which is what the short
    tests need because they read a checkpoint back, and a bounded N is what a multi-day run needs.

    The run is explicitly put in training mode before the first step. No module in `heronlm` has train-mode
    behaviour today - there is no dropout and no batch norm anywhere in the reference - so the call changes no
    number, but the record states which mode produced its loss, and a future module that does have mode-
    dependent behaviour cannot then be described as if the numbers came from the inference path.

    `precision` is read from torch at the start of the run rather than asserted, so a gate number can be
    attributed to the arithmetic that produced it. On a CPU build the TF32 flags are reported as they are; on a
    card they are what the driver set, which is why the driver sets them before calling this.
    """
    dev = dev or device()
    spec = arm.spec
    cfg = arm.cfg
    if checkpoint_every and checkpoint_path is None:
        raise ValueError("checkpoint_every needs a checkpoint_path")
    if resume and checkpoint_path is None:
        raise ValueError("resume needs a checkpoint_path")
    if train_stream is None and not corpus.get("train"):
        raise ValueError("training needs either a train_stream or a non-empty corpus['train']")

    # The corpus identity is part of the checkpoint key, not decoration. A synthetic stream and a token-shard
    # corpus can share a seed, a length and an arm name; without this field a resume would load weights from
    # one data source and continue on the other, and the resulting loss would belong to neither run.
    if train_stream is not None:
        corpus_id = getattr(
            train_stream, "corpus_id",
            f"synthetic:{spec.seed}:{spec.seq_len}:{','.join(spec.families)}",
        )
    else:
        corpus_id = f"fixed:{spec.seed}:{spec.seq_len}:{','.join(spec.families)}:{spec.n_train}"

    torch.manual_seed(spec.seed)
    model = HeronLM(cfg).to(dev)
    # Stated as mode, not assumed: nothing in the reference changes behaviour between train and eval, but the
    # record has to say which mode the loss came from, and eval() is called again inside `evaluate`.
    model.train()
    # Read at the start of the run, so a record that says tf32 was off can be checked against what was in
    # effect when the first step ran rather than against what the caller intended.
    precision = precision_flags()
    # A full clone of every parameter is a test aid: it lets a test check that two arms started from the same
    # weights. At PHASE0 scale it is 1.8 GB held next to the model, so the gate driver turns it off. Left on,
    # it would double the resident weight memory and prove nothing a finished run still needs.
    initial_state = {n: p.detach().clone() for n, p in model.named_parameters()} if capture_initial_state else {}

    opt = torch.optim.AdamW(param_groups(model, spec.lr, spec.wd))

    # Two generators on purpose: the loop depth and the data order are separate variables. A gate that
    # changes train_k must not change which sequences the arm sees, or the comparison has two differences
    # instead of one. The streaming path does not use data_rng at all, but the fixed-pool path does.
    k_rng = random.Random(spec.seed + 2)
    data_rng = random.Random(spec.seed + 1)

    # The order is recorded rather than re-derived in the test, so the test checks what the run actually did.
    # Data order and sampled k are the two things that must be shared or explicitly controlled between arms:
    # shared data order means a loss difference is attributable to the arm's variable, and the k order is the
    # variable for gate G0.6, so it has to be observable rather than assumed.
    history: list[tuple[int, int, float]] = []
    data_order: list[list[int]] = []
    k_order: list[int] = []
    start_step = 0
    train_loss = float("nan")
    kept_checkpoints: list[str] = []

    if resume:
        path = Path(checkpoint_path)
        if path.exists():
            loaded = torch.load(path, map_location=dev, weights_only=False)
            # Both record forms are accepted. Format 1 stored the orders as Python lists, format 2 as fixed-
            # width arrays; a run interrupted before the record form changed has to stay resumable.
            if loaded.get("format") not in (1, 2):
                raise ValueError(f"unknown checkpoint format in {path}: {loaded.get('format')}")
            if loaded.get("run_key") != _run_key(arm, corpus_id):
                raise ValueError(
                    f"checkpoint {path} belongs to a different arm, configuration or corpus; "
                    f"refusing to resume"
                )
            model.load_state_dict(loaded["model"])
            opt.load_state_dict(loaded["optimizer"])
            k_rng.setstate(loaded["k_rng"])
            data_rng.setstate(loaded["data_rng"])
            start_step = int(loaded["step"])
            history = [tuple(entry) for entry in loaded["history"]]
            data_order, k_order, orders_hash = orders_from_payload(loaded)
            train_loss = float(loaded["train_loss"])
            # A resumed run verifies the order it reads against the hash stored with it, so a truncated or
            # edited record cannot quietly become the order the arm was measured under. The verification
            # happens inside `orders_from_payload`; the hash is kept here because it is what the record reports.
            if orders_hash != loaded.get("orders_sha256", orders_hash):
                raise ValueError(
                    f"checkpoint {path} stores an order hash ({loaded.get('orders_sha256')}) that does not "
                    f"match the order bytes it holds ({orders_hash})"
                )
            # The initial state is not stored: it is a test aid, and a copy of every parameter would roughly
            # double each checkpoint. A resumed run reports an empty dict and is not used to check that two
            # arms started from the same weights - the fresh run is.
            initial_state = {}
            # The record reports what is on disk at the moment of the resume, not what this run will write.
            kept_checkpoints = [path.name]
        # A missing checkpoint is not an error: --resume on a first run starts it, which is what makes the
        # flag safe to leave on in a long-running command.

    if start_step > spec.steps:
        raise ValueError(f"checkpoint is at step {start_step}, beyond the requested {spec.steps}")

    # `saved_at` doubles as the step of the record currently on disk, which is what the archive name is built
    # from. On a resumed run it starts at the step the loaded checkpoint holds, so the first write after a
    # resume archives the record it is replacing with the correct position rather than with 0.
    saved_at = start_step
    for step in range(start_step, spec.steps):
        k = int(k_rng.choice(list(spec.train_k)))
        if train_stream is not None:
            # Global step index, not a draw from a pool. The stream makes sample i a pure function of i, so
            # the data order is shared by construction and a resumed run knows exactly where it stopped.
            start = step * spec.batch
            idx = list(range(start, start + spec.batch))
            samples = train_stream.sample_batch(start, spec.batch)
        else:
            idx = [data_rng.randrange(len(corpus["train"])) for _ in range(spec.batch)]
            samples = [corpus["train"][i] for i in idx]
        tokens, targets = _batch(samples, dev)

        logits = model.forward(tokens, k, None, None, slot_ids(tokens, cfg))
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1), ignore_index=-1)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        data_order.append([int(i) for i in idx])
        k_order.append(k)
        train_loss = float(loss.detach())
        if log_every and (step + 1) % log_every == 0:
            # .detach() because history is a record, not part of the graph. Without it the history holds a
            # reference to every logged step's autograd graph, and the value itself is a tensor with
            # requires_grad rather than a number.
            history.append((step + 1, k, train_loss))
        if checkpoint_path is not None and checkpoint_every and (step + 1) % checkpoint_every == 0:
            kept_checkpoints = _save_checkpoint(checkpoint_path, arm, step + 1, model, opt, k_rng, data_rng,
                                                history, data_order, k_order, train_loss, corpus_id,
                                                keep=checkpoint_keep, archive_step=saved_at)
            saved_at = step + 1

    # Always leave a checkpoint at the end of the run, so a completed arm can be evaluated or extended without
    # retraining. Skipped when that exact step was already written by the periodic save.
    if checkpoint_path is not None and spec.steps > start_step and saved_at != spec.steps:
        kept_checkpoints = _save_checkpoint(checkpoint_path, arm, spec.steps, model, opt, k_rng, data_rng,
                                            history, data_order, k_order, train_loss, corpus_id,
                                            keep=checkpoint_keep, archive_step=saved_at)

    # The compact form of the orders, computed once at the end so the record can state the same hash that a
    # resumed run verifies against. `steps_recorded` counts the rows the record holds, which on a resumed run
    # includes the steps the checkpoint was written at - the run's own steps plus what it inherited.
    orders = order_arrays(data_order, k_order, spec.batch)
    return {
        "arm": arm,
        "model": model,
        "history": history,
        "initial_state": initial_state,
        "data_order": data_order,
        "k_order": k_order,
        "train_loss": train_loss,
        "device": str(dev),
        "resumed_from": start_step if resume and start_step > 0 else None,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "corpus": corpus_id,
        # What the record says about the run, as distinct from what the run did: the mode the loss came from,
        # the arithmetic that produced it, the hash of the recorded order bytes, the number of steps the record
        # holds, and the checkpoint files that exist now.
        "mode": "train",
        "precision": precision,
        "orders_sha256": orders["sha256"],
        "steps_recorded": len(k_order),
        "tokens_recorded": len(k_order) * spec.seq_len,
        "checkpoint_keep": checkpoint_keep,
        "checkpoints_kept": kept_checkpoints,
    }


@torch.no_grad()
def evaluate(result: dict, samples: list[Sample], k: int, dev: torch.device | None = None) -> dict:
    """
    Held-out loss at a given loop count.

    Every position that has a target is scored, and the last position of the sequence has none, so it is
    skipped rather than scored against -1. Scoring it against a sentinel value is the kind of thing that
    quietly produces a number and calls it a loss.
    """
    dev = dev or device()
    model = result["model"]
    cfg = model.cfg
    model.eval()

    total, scored = 0.0, 0
    by_family: dict[str, list[float]] = {}
    for s in samples:
        tokens = torch.tensor([list(s.tokens)], device=dev)
        targets = torch.tensor([list(s.targets)], device=dev)
        logits = model.forward(tokens, k, None, None, slot_ids(tokens, cfg))

        mask = targets[0] >= 0
        if int(mask.sum()) == 0:
            continue
        loss = float(F.cross_entropy(logits[0][mask], targets[0][mask]))
        total += loss
        scored += 1
        by_family.setdefault(s.family, []).append(loss)

    return {
        "k": k,
        "loss": total / scored if scored else float("nan"),
        "by_family": {f: sum(v) / len(v) for f, v in by_family.items()},
    }


def _checkpoint_name(gate: str, arm: Arm) -> str:
    """
    A Windows-safe file name, and one the rotation policy can read.

    Arm names contain = and @, so the name is sanitized rather than used as given. A dot is excluded for a
    second reason: the archive name is built from the stem, so a name that already contains a dot cannot be
    ordered by step. Gate ids are written with a dot in ROADMAP (G0.1), which is why the gate is sanitized as
    well as the arm name - G0.1 becomes G0_1, and the record name stays readable by the pruner.
    """
    def safe(part: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in part)
    return f"{safe(gate)}_{safe(arm.name)}.pt"


def run_gate(gate: str, arms: list[Arm], dev: torch.device | None = None, log_every: int = 0,
             checkpoint_dir: str | Path | None = None, checkpoint_every: int = 0,
             checkpoint_keep: int = 0, resume: bool = False,
             corpus_dir: str | Path | None = None,
             cache: dict[str, dict] | None = None) -> dict:
    """
    Train every arm of one gate and evaluate it at every eval_k.

    Two data sources, with the same interface:

      - `corpus_dir=None`: a `SampleStream` over the synthetic task families, with the held-out eval
        sequences excluded. It shows that each mechanism changes loss, but its task space is finite, so it
        cannot honestly spend a 1-2B-token budget;
      - `corpus_dir=<path>`: uint32 token shards over real text (see `phase0.tokens`). Train and val are
        separate shards, so the held-out split cannot leak into training, and the windowed stream is large
        enough to make the roadmap's token budget meaningful.

    In both cases, eval is built first and the stream is constructed from the same seed, so every arm of the
    gate sees the same data in the same order without sharing a mutable generator.

    `cache` is the cross-gate training reuse. Its key is `training_key`, which omits the gate labels, so the
    five identical baselines train once and later gates receive the same model and the same `by_k`. A cache
    hit skips training *and* evaluation, because both are pure functions of the key. The driver decides how
    long an entry lives; `run_gate` only reads and fills the dict.

    `checkpoint_every` and `checkpoint_keep` are two separate knobs because they answer two different
    questions: how often a record is written, and how many records survive. A run that writes every 500 steps
    and keeps three is a checkpointing policy; a run that writes every 500 steps and keeps everything is a
    storage problem. On a shared box the second one ends the run before the gate is decided.
    """
    dev = dev or device()
    token_corpus = TokenizedCorpus.load(corpus_dir) if corpus_dir is not None else None
    results = []
    try:
        for arm in arms:
            spec = arm.spec
            if token_corpus is not None:
                train_split = token_corpus.split("train", spec.seq_len)
                val_split = token_corpus.split("val", spec.seq_len)
                if spec.n_test > len(val_split):
                    raise ValueError(
                        f"eval set of {spec.n_test} windows exceeds the {len(val_split)} available in the "
                        f"val split at seq_len {spec.seq_len}; add validation text or lower n_test"
                    )
                eval_stream = TokenStream(val_split, spec.seed)
                corpus = {"train": [], "test": [eval_stream[i] for i in range(spec.n_test)], "build": []}
                stream = TokenStream(train_split, spec.seed)
            else:
                # n_train=0: the fixed train split is not used on this path, and building it would only
                # allocate a pool that the stream immediately replaces.
                corpus = make_corpus(spec.seed, length=spec.seq_len, n_train=0, n_test=spec.n_test,
                                     families=spec.families)
                stream = SampleStream(spec.seed, length=spec.seq_len, families=spec.families,
                                      split="train", exclude=[s.tokens for s in corpus["test"]])

            key = training_key(arm, stream.corpus_id)
            if cache is not None and key in cache:
                hit = cache[key]
                results.append({
                    "arm": arm,
                    # A shallow copy so the current gate's arm label does not overwrite the owner's in the
                    # cached result. The model and by_k are shared on purpose; both are read-only after this
                    # point (evaluation runs under no_grad and quantization deep-copies).
                    "result": {**hit["result"], "arm": arm},
                    "by_k": hit["by_k"],
                    "reused": True,
                    "trained_as": hit["result"]["arm"].name,
                })
                continue

            checkpoint_path = None
            if checkpoint_dir is not None:
                checkpoint_path = Path(checkpoint_dir) / _checkpoint_name(gate, arm)
            result = train_arm(arm, corpus, dev, log_every, train_stream=stream,
                               checkpoint_path=checkpoint_path, checkpoint_every=checkpoint_every,
                               checkpoint_keep=checkpoint_keep, resume=resume,
                               capture_initial_state=False)
            # The split is carried on the result so the measurement and the training run use the same data.
            result["corpus_test"] = corpus["test"]
            by_k = {k: evaluate(result, corpus["test"], k, dev) for k in spec.eval_k}
            results.append({"arm": arm, "result": result, "by_k": by_k, "reused": False})
            if cache is not None:
                cache[key] = {"result": result, "by_k": by_k}
    finally:
        if token_corpus is not None:
            token_corpus.close()
    return {"gate": gate, "arms": results}
