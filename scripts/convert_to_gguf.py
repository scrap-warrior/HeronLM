"""
Phase 4.1 CLI: a checkpoint in, the files the engine loads out.

`heronlm.converter` is the part that knows what a file must contain; this is the part that decides which
checkpoint and which vocabulary to feed it. Three rules the CLI keeps, and one asymmetry it states.

1. Output names are read out of the metadata, not invented here. The metadata records the set of artifacts a
   loader has to open - `core_file`, one name per routed expert, `table.storage_path` - because a hot-swap
   resolves a routed slot to a path and a host-side probe needs one. If the CLI named the core file itself,
   the name would exist in one place only, and no test could check that the files on disk are the files the
   metadata claims. tests/test_converter.py asserts the names written equal the names recorded.

2. `--quant` accepts only the scheme names the harness records, plus F32. A converter that invented its own
   spelling for "the single-block int4 version that is simulated" would make a measured quality delta
   unattributable: the number could be coming from the scheme or from the name. The choices are closed for the
   same reason. ROADMAP's phone tier is Q4_K_M, and this converter cannot produce it: ggml's mixed-type schemes
   need an imatrix and per-layer types, which is a different job from writing one file's tensors. A Q4_K_M file
   is not something `--quant` can be handed a name for.

3. The tokenizer file is read as JSON, not through the `tokenizers` package. The vocabulary and the merges are
   the facts the metadata needs, and both are plain JSON fields; depending on a tokenizer library to read them
   would make the reference's metadata depend on a library it does not use for encoding. Ids at or above the
   model's vocabulary are rejected rather than stored: the Qwen file carries added tokens - vision, tool-call
   and file separators - whose ids sit above the embedding range, and a vocabulary written to the file has to
   be a vocabulary the model can embed.

The byte-fallback key is the one tokenizer fact the CLI cannot decide from the vocabulary alone, so it reports
which of the two backed the value it wrote. `heronlm.metadata.byte_level_rule` is the rule: a vocabulary can show
that its base alphabet is one symbol per byte, but whether the tokenizer uses byte positions for bytes it cannot
find is the file's claim, and the pinned Qwen file claims false. A claim the vocabulary cannot back is refused,
so the CLI never writes a rule this vocabulary has no symbols for.

The asymmetry, stated rather than papered over: `router.prototypes` is not an `nn.Parameter`, so a checkpoint's
`state_dict()` cannot carry trained prototypes. Converting a checkpoint without saying where the directions
came from would write the placeholder random directions a freshly-built model has, in a file that looks like a
trained file. So the CLI requires an explicit `--prototypes` source, and refuses otherwise. The provenance is
printed for every run, including the case where the placeholder is allowed on purpose.

Run:
  python scripts/convert_to_gguf.py --scale full \
      --checkpoint runs/phase0/<arm>.pt \
      --tokenizer tokenizer/qwen2.5/tokenizer.json \
      --out-dir out/heron \
      --prototypes runs/phase0/prototypes.pt \
      --quant q4_0_single_block_sim
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from gguf import LlamaFileType

from heronlm import converter
from heronlm.config import FULL, PHASE0, TINY
from heronlm.metadata import base_alphabet, byte_level_rule, metadata
from heronlm.model import HeronLM

SCALES = {"tiny": TINY, "phase0": PHASE0, "full": FULL}

# The scheme names this CLI accepts. Taken from converter.SCHEMES - the table the harness's names are mirrored
# into - rather than from a list written here. `int8_per_row` is excluded because its subject is the table,
# which is not a tensor: it has no ggml type to write, so it is not a choice a file can be quantized with.
# "F32" is not a scheme, it is the absence of one: the reference's own bytes, which is what a parity test is
# written against.
TENSOR_SCHEMES = [name for name, quant in converter.SCHEMES.items() if quant is not None]
QUANT_CHOICES = ["F32"] + sorted(TENSOR_SCHEMES)


def read_tokenizer(path: Path, cfg) -> tuple[list[str], list[str], dict]:
    """
    The vocabulary and the merges, as the tokenizer file states them.

    Returns the facts the metadata cannot be written without, plus what the file itself said, so the run report
    can show that the claims the metadata makes about the tokenizer are the claims the tokenizer file makes.
    The ids are rejected at the vocabulary limit rather than stored: every id in the file has to be an id the
    model can embed, and every id the n-gram table was built with.

    A gap in the vocabulary is a failure rather than a blank entry, because the metadata writes the vocabulary
    as an ordered array: a missing entry shifts every id after it, and the table entries built with those ids
    would still look valid.

    The byte-fallback fact is returned as three fields, because they are three different facts: the file's own
    claim, what the vocabulary can show by itself, and the value the metadata can therefore write. The old
    version of this reader refused any file that did not claim byte fallback, which meant the reference was
    enforcing its own assumption instead of reporting the tokenizer's - and the pinned Qwen file contradicts
    that assumption, so the refusal was rejecting the file it was supposed to be checking.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    model = raw.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("vocab"), dict):
        raise ValueError(f"{path} has no model.vocab; it is not a tokenizer.json")
    if model.get("type") != "BPE":
        # `tokenizer.ggml.model` is written as "gpt2", which is the format's name for BPE. A file of another
        # type cannot be described by the metadata this CLI writes, so the disagreement is stated here instead
        # of being written into the file as a claim.
        raise ValueError(f"{path} is {model.get('type')!r}, not BPE; the metadata describes a BPE tokenizer")

    vocab = model["vocab"]
    merges = [str(m) for m in model.get("merges", [])]
    added = [a for a in raw.get("added_tokens", []) if a.get("id", -1) < cfg.vocab_size]

    # The vocabulary as an ordered array, which is what the metadata writes and what a reader indexes by id.
    tokens: list[str | None] = [None] * cfg.vocab_size
    for token, index in vocab.items():
        if index >= cfg.vocab_size:
            raise ValueError(
                f"{path} has {token!r} at id {index}, above the model's vocabulary of {cfg.vocab_size}. "
                "An id the model cannot embed cannot be part of the file's vocabulary."
            )
        tokens[index] = str(token)
    for entry in added:
        # An added token inside the embedding range is written: it is a token the model can embed, and the chat
        # tags are exactly those. Above the range, it is not stored at all.
        tokens[entry["id"]] = str(entry["content"])
    if any(t is None for t in tokens):
        raise ValueError(f"{path} leaves {sum(t is None for t in tokens)} ids empty; no id can be missing")

    # The claim the file states, and the fact the vocabulary can back on its own. They are different facts, and
    # the metadata can only write the one it has: `byte_level_rule` takes the file's claim when the file states
    # it, and falls back to the vocabulary when the vocabulary is the only authority. The refusal below is the
    # case where a file claims a rule its own symbol set does not have - which is a claim the reference would be
    # making on the file's behalf, so it is not written.
    claim = model.get("byte_fallback")
    value, backed_by = byte_level_rule(tokens, claim)
    entailed = base_alphabet(tokens) is not None

    facts = {
        "path": str(path),
        "tokens": len(tokens),
        "merges": len(merges),
        "claim": claim,
        # What the vocabulary can show by itself, kept separate from the value written so a divergence between
        # the two is visible in the report rather than resolved by whichever constant was here first.
        "byte_level_entailed": entailed,
        "byte_fallback": value,
        "backed_by": backed_by,
        # The decoder is where a byte-level BPE's byte-to-symbol table lives. It is recorded as the file's own
        # field, because that is the fact the vocabulary cannot show: a byte-shaped alphabet does not say the
        # tokenizer reads bytes from it.
        "decoder_type": (raw.get("decoder") or {}).get("type"),
        "added_tokens_written": [str(a["content"]) for a in added],
        "added_tokens_rejected": [str(a["content"]) for a in raw.get("added_tokens", [])
                                  if a.get("id", -1) >= cfg.vocab_size],
    }
    return tokens, merges, facts


def checkpoint_state(cfg, checkpoint: Path) -> dict:
    """
    The checkpoint's weights, checked against the config the file is being written for.

    Two shapes are accepted because both exist in practice: a training checkpoint (a payload with `model` and
    the optimizer and RNG states) and a plain state dict. A payload that records the run it came from is
    compared field by field, so a checkpoint trained at one scale cannot be converted under a metadata dict
    that describes another. The shapes would match at every field the loader allocates from, and the metadata
    would be describing an architecture the bytes were not trained in.

    A checkpoint produced by the reference cannot carry slot 0 at all: the reference's generalist module is
    outside `state_dict()`. So slot-0 keys here mean the bytes came from a model where slot 0 was an ordinary
    expert, and there is no name for them - `blk.{bid}.ffn_*` already names the same thing. That is refused
    rather than reported, because writing them would need a name the metadata does not claim, and the refusal
    happens before any file is opened.
    """
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{checkpoint} is not a checkpoint payload")
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise ValueError(f"{checkpoint} has no model state dict")

    recorded = payload.get("run_key", {}).get("cfg")
    if recorded is not None:
        def as_data(value):
            return {k: (tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in value.items()}

        differ = [k for k, v in as_data(dataclasses.asdict(cfg)).items() if as_data(recorded).get(k) != v]
        if differ:
            raise ValueError(
                f"{checkpoint} was trained at a config that disagrees with --scale on {sorted(differ)}; "
                "the metadata would describe an architecture these bytes were not trained in"
            )

    # The generalist is the block's base FFN, so its bytes are already named by the core file. A checkpoint that
    # carries slot 0 came from a model where slot 0 was an ordinary expert, which is a different architecture
    # from the one the metadata describes. "keys" rather than "parameters": the reference has no parameters at
    # slot 0, so these bytes are keys it cannot produce, and refusing them is the same statement as the naming
    # rule rather than a rule about tensors.
    at_generalist = [k for k in state if k.startswith(f"experts.{cfg.generalist_slot}.")]
    if at_generalist:
        raise ValueError(
            f"{checkpoint} carries {len(at_generalist)} keys at slot {cfg.generalist_slot}. "
            f"The reference excludes slot {cfg.generalist_slot} from state_dict(), so a checkpoint it produces "
            "cannot carry them: slot 0 is each expert block's base FFN, already named by blk.{bid}.ffn_*. "
            "These bytes have no name, so nothing is written for them."
        )
    return state


def resolve_prototypes(model: HeronLM, cfg, source: str | None, allow_placeholder: bool) -> str:
    """
    The router's directions, with where they came from stated.

    `router.prototypes` is not a Parameter, so a checkpoint cannot carry it. That is not a detail to work
    around: it means the bytes the file records for the router are either directions someone produced - the
    clustered teacher activations a prototype set is defined as - or the random directions a freshly-built
    model has. Only the first is a router in the sense the routing gate measures, so the second has to be asked
    for by name.

    Returns the provenance string the run report prints.
    """
    shape = (cfg.n_experts, cfg.d_model)
    if source:
        path = Path(source)
        loaded = np.load(str(path)) if path.suffix == ".npy" else torch.load(str(path), map_location="cpu",
                                                                              weights_only=True)
        array = np.asarray(loaded, dtype=np.float32)
        if array.shape != shape:
            raise ValueError(f"{path} holds {array.shape}, not the router's {shape}")
        model.router.prototypes = torch.from_numpy(array)
        return f"read from {path} as {shape}; provenance is whatever produced it, which the CLI cannot know"
    if not allow_placeholder:
        raise ValueError(
            f"router.prototypes is not in a checkpoint's state_dict, so this run would write the model's "
            f"placeholder directions ({shape}). Give an explicit source with --prototypes, or say the "
            "placeholder is intended with --allow-placeholder-prototypes."
        )
    return "placeholder: the directions this model was built with, not directions that were trained"


def quant_rule(name: str):
    """
    The scheme -> ggml type rule `tensor_entries` applies, as a function of module path and shape.

    F32 means no rule at all: the reference's bytes. A named scheme still leaves scalars and sub-block
    parameters as F32, because ggml's smallest block is 32 elements - so a file written with `--quant q8_0` is
    a mostly-Q8_0 file with F32 exceptions, and the report has to show that rather than print one type.
    """
    if name == "F32":
        return lambda path, shape: None
    return lambda path, shape: converter.quant_type_for(path, shape, dense=name)


def describe_type(value: int) -> str:
    """A file type as its name, so the report shows what the header records rather than a number."""
    for member in LlamaFileType:
        if int(member.value) == value:
            return member.name
    return str(value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="convert_to_gguf.py", description="Write the files the HeronLM engine loads.")
    ap.add_argument("--scale", required=True, choices=sorted(SCALES), help="which config the file describes")
    ap.add_argument("--checkpoint", default=None,
                    help="a training checkpoint or a plain state dict; without it the model is freshly built")
    ap.add_argument("--tokenizer", required=True, help="path to the Qwen tokenizer.json")
    ap.add_argument("--out-dir", required=True, help="where the files go; names come from the metadata")
    ap.add_argument("--quant", default="F32", choices=QUANT_CHOICES,
                    help="the scheme dense weights are written under; F32 is the reference's own bytes")
    ap.add_argument("--prototypes", default=None,
                    help="the router's directions, from the clustered teacher activations (.pt or .npy)")
    ap.add_argument("--allow-placeholder-prototypes", action="store_true",
                    help="write the model's own random directions, and say so")
    ap.add_argument("--no-table-blob", action="store_true",
                    help="skip the blob, when the table is produced by a different step")
    args = ap.parse_args(argv)

    cfg = SCALES[args.scale]()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens, merges, tokenizer_facts = read_tokenizer(Path(args.tokenizer), cfg)
    model = HeronLM(cfg).eval()
    if args.checkpoint:
        # Read the state dict once, so the report below describes the bytes that were loaded rather than a
        # second read of the same file. `checkpoint_state` refuses a checkpoint carrying slot 0, so every key
        # loaded here is a key a file can name.
        state = checkpoint_state(cfg, Path(args.checkpoint))
        model.load_state_dict(state)
    provenance = resolve_prototypes(model, cfg, args.prototypes, args.allow_placeholder_prototypes)

    # The metadata is written once and read from for every file, so the core file and the per-expert files state
    # the same facts. `expert_file_slot` is the one key that differs, and write_gguf adds it. The byte-fallback
    # value is passed as the tokenizer file's claim rather than decided here, so the metadata records the same
    # value the report above says it was backed by.
    md = metadata(cfg, tokens, merges, claim=tokenizer_facts["claim"])
    rule = quant_rule(args.quant)

    # Every name used below is read out of the metadata, which is the only place the naming lives.
    core_name = md["general.arch.heron.core_file"]
    expert_names = md["general.arch.heron.expert_files"]
    blob_name = md["general.arch.heron.table.storage_path"]

    print(f"scale      : {args.scale}")
    print(f"tokenizer  : {tokenizer_facts['path']} "
          f"({tokenizer_facts['tokens']} tokens, {tokenizer_facts['merges']} merges)")
    print(f"byte_fallback: {tokenizer_facts['byte_fallback']} <- {tokenizer_facts['backed_by']}")
    if tokenizer_facts["claim"] is not None and tokenizer_facts["byte_level_entailed"] != tokenizer_facts["claim"]:
        # The two authorities disagree, so the value written is the file's and the disagreement is stated rather
        # than resolved by whichever one the code happened to read first. The pinned Qwen file is exactly this
        # case: it claims no byte fallback while its base alphabet at ids 0..255 is 256 single-character
        # symbols, so its byte-level behaviour is a property of its ByteLevel decoder, not of a fallback rule.
        print(f"             the vocabulary's base alphabet is byte-shaped ({tokenizer_facts['byte_level_entailed']}), "
              f"the file claims {tokenizer_facts['claim']}; decoder.type={tokenizer_facts['decoder_type']}")
    print(f"prototypes : {provenance}")
    print(f"quant      : {args.quant} -> "
          f"{converter.SCHEMES.get(args.quant, 'no ggml type')}; "
          f"excluded params stay F32 (ggml's smallest block is {converter.MIN_QUANT_BLOCK} elements)")
    print(f"out dir    : {out_dir}")

    print()
    core = converter.tensor_entries(model, cfg, quant=rule)
    report = converter.write_gguf(out_dir / core_name, cfg, md, core)
    print(f"{core_name}  {len(core)} tensors  {describe_type(report['file_type'])}")

    for slot in range(cfg.n_experts):
        if slot == cfg.generalist_slot:
            # The name stays in the metadata so slot indices and names line up, but no file is written for it:
            # the generalist is each expert block's base FFN, which the core file already carries. The module
            # the reference builds at this index is outside state_dict(), so a checkpoint it produces has no
            # bytes here - which is why this line reports the slot as having no content rather than as a file
            # that was skipped.
            print(f"{expert_names[slot]}  not written: slot {slot} is the generalist, and its tensors ARE "
                  "the core file's base FFN. The reference's module at this index is excluded from "
                  "state_dict(), so no checkpoint carries bytes a file for this slot would need.")
            continue
        entries = converter.tensor_entries(model, cfg, expert_file_slot=slot, quant=rule)
        report = converter.write_gguf(out_dir / expert_names[slot], cfg, md, entries, expert_file_slot=slot)
        print(f"{expert_names[slot]}  {len(entries)} tensors  {describe_type(report['file_type'])}")

    if args.no_table_blob:
        print(f"{blob_name}  skipped (--no-table-blob); the metadata still records it as the table's path")
    else:
        blob = converter.write_table_blob(model.memory.table.weight, cfg)
        (out_dir / blob_name).write_bytes(blob)
        rows, width = cfg.ngram_memory_entries, cfg.ngram_memory_dim
        print(f"{blob_name}  {rows} rows x {width} codes + {rows} scales  {len(blob)} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
