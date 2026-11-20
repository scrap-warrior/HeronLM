"""
Reader tests, at TINY.

`tests/test_converter.py` checks what a writer produces. These check what a reader can honour, which is a
different set of failures. A writer that writes a name the metadata does not claim is a bug the writer can catch
before anything is written; a reader that guesses a name, a slot, or a shape is a bug that shows up only as
different numbers, in a model that otherwise looks fine. So most of what follows points a hand-built file at the
reader: a writer that produces a bad file was already caught by the converter tests, and a reader being handed one
is what this phase has to prove.

The single most important test here is the one that refuses a file rather than loading it. A loader that loads a
file it cannot explain fills the missing pieces with initialization and keeps going, and a randomly initialized
router produces perfectly plausible logits. So "refused" is the honest result for every case where the reader would
otherwise have to guess, and the tests assert the refusal message as well as the refusal.

Run: python tests/test_loader.py
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Appended, not inserted, so `phase0` still resolves to the package at the project root rather than to
# scripts/phase0.py. The CLI's quant rule is used here because it is the rule the CLI applies, and a test that
# invented its own rule would be testing a rule nobody uses.
sys.path.append(str(ROOT / "scripts"))

import numpy as np
import torch
import convert_to_gguf
from gguf import LlamaFileType, quants
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as Quant

from heronlm import converter, loader
from heronlm.config import HeronConfig, TINY
from heronlm.memory import hash_ngram
from heronlm.metadata import metadata, placeholder_vocabulary
from heronlm.model import HeronLM
from heronlm.norms import EPS, PARITY_RULE, normalized_weight
from phase0 import quant as harness_quant


def tiny_model():
    """Deterministic: the same parameter bytes on every run, so a byte comparison is meaningful."""
    torch.manual_seed(0)
    return HeronLM(TINY())


def md_for(cfg):
    return metadata(cfg, placeholder_vocabulary(cfg.vocab_size), ["merge a b"])


def tokens_for(cfg, n=40):
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, cfg.vocab_size, (1, n), generator=g)


def write_file(model, cfg, md, tmp, *, quant="F32", slot=None):
    """One file, written the way the CLI writes it, and read back with the reader under test."""
    entries = converter.tensor_entries(model, cfg, expert_file_slot=slot, quant=convert_to_gguf.quant_rule(quant))
    path = tmp / ("heron_core.gguf" if slot is None else f"expert_{slot}.gguf")
    converter.write_gguf(path, cfg, md, entries, expert_file_slot=slot)
    return path


def write_set(model, cfg, tmp, quant="F32"):
    """The whole set: core file, one file per routed expert, and the blob."""
    md = md_for(cfg)
    files = [write_file(model, cfg, md, tmp, quant=quant)]
    files += [write_file(model, cfg, md, tmp, quant=quant, slot=s) for s in range(1, cfg.n_experts)]
    blob = tmp / "heron_table.bin"
    blob.write_bytes(converter.write_table_blob(model.memory.table.weight.detach(), cfg))
    return files, blob


def hand(md, tensors, path="hand-built.gguf"):
    """
    A file built by hand rather than by the writer.

    This is what makes the refusal paths testable: a hand-built file is a file the writer would never produce, so
    testing it tests the reader rather than testing the writer twice.
    """
    return loader.GGUFFile(path, dict(md), dict(tensors))


def held(file, name):
    """One tensor entry as the reader reports it, so a hand-built entry can be compared against a read one."""
    return file.tensors[name]


def loaded_value(model, module_path):
    """The value the loader put at a module path, as numpy, for a byte comparison."""
    owner, attr = loader._target(model, module_path)
    current = getattr(owner, attr)
    return current.detach().numpy() if isinstance(current, torch.nn.Parameter) else current.detach().numpy()


def test_the_round_trip_at_f32_loads_the_same_logits_as_the_reference():
    """
    The proof the claims are consumable.

    A converter can state a name, a shape and a block in metadata and still produce a file a loader cannot use,
    because nothing in the format forces a loader to agree with the metadata: it forces the loader to agree with
    the bytes. So the test is not "the metadata matches the dict it was given", it is "a module rebuilt from the
    bytes computes the same numbers as the module the bytes came from".

    F32 is the only scheme this can assert as bit-exact, for two reasons that are the same reason: a quantized
    file is by definition not the reference's values, and the reference normalizes at forward time, so a loaded
    file's values are only comparable with the reference's values if nothing was changed on the way in.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = TINY()
        ref = tiny_model()
        files, blob = write_set(ref, cfg, tmp, quant="F32")
        opened = [loader.open_file(p) for p in files]
        loaded = loader.load_model(opened, blob=loader.open_blob(blob, opened[0]))

        # The config as the file states it, field by field, at the precision the format has. The one field that
        # cannot be compared as equal is `full_attention_every`: the file states the per-block result, not the
        # interval that produced it, so what has to match is the per-block rule - the half the file does state.
        for key, field_name in loader.METADATA_FIELDS.items():
            if key not in opened[0].metadata:
                continue
            stated = opened[0].metadata[key]
            value = getattr(loaded.cfg, field_name)
            if isinstance(stated, bool):
                assert bool(stated) == value, (key, stated, value)
            elif isinstance(stated, (list, tuple)):
                assert [int(x) for x in stated] == value, (key, stated, value)
            elif isinstance(stated, str):
                assert str(stated) == str(value), (key, stated, value)
            elif isinstance(stated, (float, np.floating)):
                assert loader.stored_float(stated) == loader.stored_float(value), (key, stated, value)
            else:
                assert int(stated) == int(value), (key, stated, value)
        assert loaded.cfg.full_attention_every != cfg.full_attention_every, (
            loaded.cfg.full_attention_every, cfg.full_attention_every)
        assert all(loaded.cfg.mixer_mode(b) == cfg.mixer_mode(b) for b in range(cfg.n_blocks())), (
            [loaded.cfg.mixer_mode(b) for b in range(cfg.n_blocks())],
            [cfg.mixer_mode(b) for b in range(cfg.n_blocks())])

        # Every parameter the claim can name, checked against the reference's own bytes.
        for path, source in sorted(loaded.loaded.items()):
            if "[" in path:
                continue  # a row of the shared read weights; checked as a whole below
            expected = ref.state_dict()[path] if path in ref.state_dict() else ref.router.prototypes
            got = loaded_value(loaded.model, path)
            assert np.array_equal(got, expected.detach().numpy()), (source, path, got.shape, expected.shape)

        logits_a = ref.forward(tokens_for(cfg), loop=2, active_slot=1)
        logits_b = loaded.forward(tokens_for(cfg), loop=2, active_slot=1)
        assert torch.equal(logits_a, logits_b), float((logits_a - logits_b).abs().max().detach())
        assert float((logits_a - logits_b).abs().max().detach()) == 0.0

        # The two structures a file cannot express as one tensor: rows, and a tensor that is not a Parameter.
        assert torch.equal(loaded.model.stream.read_weights, ref.stream.read_weights)
        assert torch.equal(loaded.model.router.prototypes, ref.router.prototypes)
        # The table is not a tensor in a file, so its values come from the blob - and the blob is always the
        # harness's int8 rows, whatever scheme the file was written with. So the comparison here is against the
        # harness's stored row read back, not against the reference's raw Parameter: at TINY the reference's table
        # is a freshly initialized float tensor, and "bit-exact" is not a claim an int8 blob can support.
        harness_values = harness_quant.dequantized(ref.memory.table.weight, harness_quant.TABLE_INT8).detach().numpy()
        assert np.allclose(loader.open_blob(blob, opened[0]).values(), harness_values), harness_values[0][:8]

        # Slot 0's tensors are the only ones a file has no destination for, and the report says so. They exist as
        # non-persistent buffers: the harness sees the generalist as a module, but no checkpoint carries its bytes
        # and the load path does not expect them, so nothing here can be filled - forward() reads block.ffn for
        # slot 0. This is the reference's shape recorded as a gap rather than quietly left at initialization, and
        # the number is the module's shape, not a count the test chose: four tensors per slot, times the slots the
        # reference instantiates. They are buffers and not parameters, which is what the exclusion is built on, so
        # the assertion is made against both lists rather than against the gap count alone.
        slot0 = f"experts.{cfg.generalist_slot}."
        generalist = [p for p in loaded.gaps if p.startswith(slot0)]
        assert len(generalist) == 4 * cfg.expert_slots, (len(generalist), sorted(loaded.gaps))
        assert all("has no file" in loaded.gaps[p] for p in generalist), sorted(loaded.gaps)[:3]
        assert all(p in dict(loaded.model.named_buffers()) for p in generalist), sorted(generalist)
        assert not any(p.startswith(slot0) for p in loaded.loaded), sorted(loaded.loaded)


def test_the_loader_dequantizes_and_refuses_a_file_that_states_another_order():
    """
    The order rule asserted through the reader.

    `PARITY_RULE` says normalization happens after dequantization. The only way that rule survives a reader is if
    the reader hands the module the stored values: the moment a loader normalizes on the way in, the order is
    decided by the loader rather than by the rule, and a parity test can no longer tell which of the two produced
    the numbers. So there is no normalize step in the loader at all, and what it can do is refuse a file that
    states a different order.

    The limit is worth stating, because it is the limit of what a reader can prove: for the unit-norm arm
    normalization is idempotent, so a file written from pre-normalized weights is indistinguishable by looking at
    loaded values. What catches it is a byte comparison against the reference, which is what the converter test
    does. What this test can show is the two things the reader is actually responsible for: the values it hands
    over are the file's, and a file claiming a different order is refused.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = TINY()
        ref = tiny_model()
        files, blob = write_set(ref, cfg, tmp, quant="F32")
        opened = [loader.open_file(p) for p in files]
        loaded = loader.load_model(opened, blob=loader.open_blob(blob, opened[0]))

        # The loaded weight is the reference's raw weight, not the normalized one. If the loader had normalized,
        # this assertion would fail for any weight that is not already unit-norm, so the test picks one that is
        # not - otherwise the assertion could not fail and would not be a test.
        path = "blocks.0.ffn.gate.weight"
        raw = ref.state_dict()[path].detach().numpy()
        normalized = normalized_weight(ref.state_dict()[path], cfg).detach().numpy()
        assert not np.array_equal(raw, normalized), "the chosen weight is already unit-norm; nothing is asserted"
        assert np.array_equal(loaded_value(loaded.model, path), raw), "the loader normalized on the way in"

        md = md_for(cfg)
        md["general.arch.heron.weight_norm_order"] = "before_quant"
        problems = loader.file_problems(hand(md, converter.read_tensors(files[0])))
        assert any("cannot honour" in p and "weight_norm_order" in p for p in problems), problems

        # And the reference's own order is accepted, which is what makes the check a check rather than a refusal
        # of everything.
        assert not any("weight_norm_order" in p for p in loader.file_problems(opened[0])), loader.file_problems(opened[0])


def test_a_quantized_file_is_reported_as_measured_error_and_f32_stays_bit_exact():
    """
    Quantization is reported as measured error, never as bit-exactness.

    The point of separating the two is that a parity test that tolerates a quantization error cannot tell whether
    the error came from the scheme or from the loader. So the assertion here is the shape of the result, not the
    size: F32 is exactly zero, and every quantized file is nonzero and ordered by what the schemes are known to
    cost. The numbers themselves are printed as measurements, not asserted as thresholds, because a threshold on a
    rounding error at TINY would be a number nobody could reproduce at the gate scale.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = TINY()
        ref = tiny_model()
        tokens = tokens_for(cfg)

        errors = {}
        # The scheme names are the harness's, not ggml's, because the CLI accepts exactly these: `q4_0` is what
        # the file records as a type, `q4_0_single_block_sim` is what the harness recorded as the single-block
        # int4 arm being simulated. A test that used the type name as the scheme name would be testing a name the
        # converter's table does not have.
        q4 = "q4_0_single_block_sim"
        for scheme in ("F32", "q8_0", q4):
            files, blob = write_set(ref, cfg, tmp, quant=scheme)
            opened = [loader.open_file(p) for p in files]
            loaded = loader.load_model(opened, blob=loader.open_blob(blob, opened[0]))
            weight_error = max(
                float(np.abs(loaded_value(loaded.model, path) - ref.state_dict()[path].detach().numpy()).max())
                for path, source in sorted(loaded.loaded.items())
                if "[" not in path and path in ref.state_dict()
            )
            logits_error = float((ref.forward(tokens, loop=2, active_slot=1)
                                  - loaded.forward(tokens, loop=2, active_slot=1)).abs().max().detach())
            errors[scheme] = (weight_error, logits_error)

        assert errors["F32"] == (0.0, 0.0), errors["F32"]
        for scheme in ("q8_0", q4):
            assert errors[scheme][0] > 0.0, (scheme, errors[scheme])
            assert errors[scheme][1] > 0.0, (scheme, errors[scheme])
        assert errors["q8_0"][0] < errors[q4][0], errors
        assert errors["q8_0"][1] < errors[q4][1], errors

        # The relative estimates and the measured error are not the same fact, and the test says which one it is
        # looking at. 8.5 effective bits for q8_0 and about 4.3 for q4_0 describe the storage; the numbers above
        # are what a forward pass actually produced, and a report that printed the first as the second would be
        # reporting an estimate as a measurement.
        print("  measured error per scheme:", {k: round(v[0], 4) for k, v in errors.items()})


def test_names_are_resolved_only_from_the_claim_and_a_hand_built_name_is_refused_both_ways():
    """
    The claim read in both directions.

    If the loader carried its own copy of the name mapping, a file holding a tensor the metadata does not describe
    would still load - and loading it would mean placing bytes somewhere the file never said they belonged. So a
    name outside the claim is refused, and a name the claim covers that the file does not hold is refused too,
    because a missing tensor is not an error: it is a parameter that keeps whatever the freshly built module
    happened to initialize it to.

    The claim is written as patterns with placeholders, so matching a name is a substitution against the block
    index the name carries, not a dictionary lookup. That is what `name_pattern` does, and the test asserts it on
    its own because a family pattern cannot be matched by equality.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = TINY()
        ref = tiny_model()
        path = write_file(ref, cfg, md_for(cfg), tmp)
        opened = loader.open_file(path)
        tensors = converter.read_tensors(path)

        # A name the claim cannot explain. `blk.6.ffn_gate_exps` is a real name for slot 1, but the core file has
        # no slot claim, so its scope is the non-expert patterns and this name is outside it.
        extra = dict(tensors)
        extra["blk.6.ffn_gate_exps"] = dict(tensors["blk.0.ffn_gate"])
        problems = loader.name_problems(hand(opened.metadata, extra))
        assert any("the claim cannot explain it" in p for p in problems), problems

        # A name the claim covers that the file does not hold.
        missing = {k: v for k, v in tensors.items() if k != "blk.0.attn_q"}
        problems = loader.name_problems(hand(opened.metadata, missing))
        assert any("the claim covers it but the file does not hold it" in p and p.startswith("blk.0.attn_q")
                   for p in problems), problems

        # `pattern_for` is the reader's own refusal, and it refuses rather than guessing. A guessed name is the
        # same failure as a dictionary lookup that returns None: the tensor lands somewhere plausible.
        try:
            loader.pattern_for("blk.0.not_a_field", opened)
        except ValueError as e:
            assert "cannot explain this name" in str(e), str(e)
        else:
            raise AssertionError("a name outside the claim must be refused, not guessed")

        # The block index in a name is what carries the substitution. `name_pattern` has to recover the family,
        # not the individual tensor, or a file at block 6 would not match the claim that covers every block.
        assert loader.name_pattern("blk.6.attn_q") == "blk.{bid}.attn_q", loader.name_pattern("blk.6.attn_q")
        # The claim's key side is the module pattern, which is the TENSOR_NAMES spelling - `blocks.{bid}.attn.q
        # .weight`, not the name spelled with an underscore. Written as the dict the claim actually is, so the
        # test reads the claim rather than restating it.
        assert opened.claims()["blocks.{bid}.attn.q.weight"] == "blk.{bid}.attn_q"

        # And the loaded model has to agree with the reference about which block is which, because that is the
        # half a reader that reconstructed names from config arithmetic would get wrong without noticing. Read in
        # the reader's direction: every name the file holds, resolved back through the claim.
        assert sorted(loader.pattern_for(n, opened) for n in opened.tensors) == sorted(opened.names().values()), (
            sorted(opened.names())[:3])


def test_the_slot_comes_from_the_claim_and_a_file_stating_slot_0_or_no_slot_is_refused():
    """
    Which expert a file is cannot be read from its bytes.

    Five per-expert files have the same tensor names and the same shapes; only the slot claim says which domain
    the bytes are for. So a file with no slot claim is refused, and slot 0 is refused for a different reason: the
    generalist is each expert block's base FFN, so its bytes are already named by `blk.{bid}.ffn_*` and a file for
    slot 0 has no content to hold.

    The `{eid}` and `{slot}` placeholders stay unresolved in a name for the same reason. A name can say which
    block it is; it cannot say which expert. That is why `module_path` fills them from the claim and refuses when
    there is no claim to fill them from.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = TINY()
        ref = tiny_model()
        path = write_file(ref, cfg, md_for(cfg), tmp, slot=1)
        opened = loader.open_file(path)
        assert opened.slot() == 1, opened.slot()
        layers = opened.expert_blocks()
        assert layers == cfg.expert_blocks(), (layers, cfg.expert_blocks())

        # Slot 0.
        md = dict(opened.metadata)
        md["general.arch.heron.expert_file_slot"] = 0
        problems = loader.file_problems(hand(md, opened.tensors))
        assert any("Slot 0 is the generalist" in p for p in problems), problems

        # No slot claim at all. `open_file` refuses on the names, because without a slot it cannot tell whether
        # this is a core file or a per-expert file; `module_path` is where the reason is stated, and both have to
        # be checked because they are different checks.
        md = dict(opened.metadata)
        del md["general.arch.heron.expert_file_slot"]
        file = hand(md, opened.tensors)
        problems = loader.file_problems(file)
        assert any("states no slot" in p for p in problems), problems
        try:
            loader.module_path("blk.6.ffn_gate_exps", file, cfg)
        except ValueError as e:
            assert "cannot be loaded without a slot claim" in str(e), str(e)
        else:
            raise AssertionError("an expert name without a slot claim must be refused")

        # A name that lands in a block the claim does not put slots in. The name is still explainable by the
        # claim's pattern, so this is a different failure from an unexplainable name and has to be refused
        # separately: the block index is the fact, not the pattern. At TINY the expert blocks are [6, 7], so blk.5
        # has a base FFN named blk.5.ffn_* and no slot the claim could name.
        file = hand(opened.metadata, {name: v for name, v in opened.tensors.items() if name.startswith("blk.5")})
        try:
            loader.module_path("blk.5.ffn_gate_exps", file, cfg)
        except ValueError as e:
            assert "not one of this file's slot layers [6, 7]" in str(e), str(e)
            assert "block 5" in str(e) and "base FFN" in str(e), str(e)
        else:
            raise AssertionError("a slot name in a block the claim does not use for slots must be refused")

        # The same pattern, for a block the claim does cover, resolves - which is the half that shows the refusal
        # above is about the block rather than about the name.
        assert loader.module_path("blk.6.ffn_gate_exps", file, cfg) == "experts.1.0.ffn.gate.weight", (
            loader.module_path("blk.6.ffn_gate_exps", file, cfg))

        # The slot is resolved to the module's index within the set, not to a module of its own. Slot 1 at TINY
        # has one slot per file, so the path is `experts.1.0.ffn.gate.weight` and the round trip puts the file's
        # bytes at exactly that path.
        #
        # `load_model` needs a core file, and not as a formality: the config it builds comes from the core file's
        # claims, so a per-expert file alone cannot be loaded as a model even though it states the same shapes.
        # The core file here is what the expert file's bytes are placed into, and it is also what the report
        # describes the expert file against.
        core = loader.open_file(write_file(ref, cfg, md_for(cfg), tmp))
        loaded = loader.load_model([core, loader.open_file(path)], blob=None)
        assert np.array_equal(loaded_value(loaded.model, "experts.1.0.ffn.gate.weight"),
                              ref.state_dict()["experts.1.0.ffn.gate.weight"].detach().numpy())


def test_read_weight_rows_are_assembled_in_block_order_and_a_dropped_row_is_refused():
    """
    One tensor in the module, one tensor per block in the file.

    A file that wrote the aliases as n separate tensors would look identical byte for byte and load identically,
    so what is asserted here is the structure the claim states, not only the values. The row index is the block
    index because `ResidualStream.read` is called with the block index - it is not a sublayer position, and the
    three parts of the block map do not reorder the rows.

    The check that rules out a file that dropped one row is the shape fact, not the count of names: nine names
    that all resolve is what a dropped row looks like until the number of rows is compared with the number the
    file states.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = TINY()
        ref = tiny_model()
        path = write_file(ref, cfg, md_for(cfg), tmp)
        opened = loader.open_file(path)
        loaded = loader.load_model([opened], blob=None)

        shape = [int(x) for x in opened.metadata["general.arch.heron.gr.read_weights_shape"]]
        assert shape == [cfg.n_blocks(), cfg.max_loop, cfg.gr_branches], shape
        rows = sorted(int(name.split(".")[1]) for name in opened.tensors if name.endswith("residual_alpha"))
        assert rows == list(range(cfg.n_blocks())), rows
        assert np.array_equal(loaded.model.stream.read_weights.detach().numpy(),
                              ref.stream.read_weights.detach().numpy())
        for bid in rows:
            assert np.array_equal(loaded.model.stream.read_weights[bid].detach().numpy(),
                                  ref.stream.read_weights[bid].detach().numpy()), bid

        # A dropped row, refused at the point a reader would otherwise assemble n-1 rows and call them the tensor.
        missing = {k: v for k, v in opened.tensors.items() if k != "blk.3.residual_alpha"}
        problems = loader.name_problems(hand(opened.metadata, missing))
        assert any(p.startswith("blk.3.residual_alpha") for p in problems), problems

        # A row that is not the shape the file states: the claim is still satisfiable by name, so this is the
        # check that has to be made after the names, and it is the one that catches a header describing a row as
        # a different tensor than the module has.
        wrong = dict(opened.tensors)
        wrong["blk.3.residual_alpha"] = dict(wrong["blk.3.residual_alpha"], dims=[3, 3, 2],
                                             n_bytes=18 * 4)
        try:
            loader.value_for("blk.3.residual_alpha", hand(opened.metadata, wrong), cfg)
        except ValueError as e:
            assert "are not the reverse of the module's" in str(e), str(e)
        else:
            raise AssertionError("a row whose dims are not the reverse of the module's shape must be refused")

        # Rows are the only form this parameter can take in a file, so coverage has to know that before it asks
        # the mapping for a name - and the mapping raises for it by design.
        try:
            converter.tensor_name("stream.read_weights", cfg)
        except ValueError as e:
            assert "residual_alpha" in str(e), str(e)
        else:
            raise AssertionError("the shared read weights have no single name")
        assert "stream.read_weights" not in loaded.gaps, loaded.gaps


def test_router_prototypes_reach_the_module_although_a_state_dict_cannot_carry_them():
    """
    The one thing neither a state dict nor a parameter loop carries on its own.

    The router stores its prototypes as a plain tensor, not as a Parameter, so `state_dict()` does not contain
    them and a state-dict-driven loader would produce a model with no router and no error. This is the reason
    `assign` branches on the type at all, and the test asserts the asymmetry rather than assuming parameters and
    tensors are the same thing.

    It is also why the loader reports them as a gap when a file omits them: a loaded model with a placeholder
    router computes plausible logits through the wrong directions, and routing is the thing the gate measures.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = TINY()
        ref = tiny_model()
        path = write_file(ref, cfg, md_for(cfg), tmp)
        opened = loader.open_file(path)

        assert "router.prototypes" not in ref.state_dict(), "the asymmetry is what the test is about"
        assert "router.prototypes" not in dict(ref.named_parameters()), "the router has no Parameter"

        loaded = loader.load_model([opened], blob=None)
        name = converter.tensor_name("router.prototypes", cfg)
        assert name in opened.tensors, sorted(opened.tensors)
        assert np.array_equal(loaded.model.router.prototypes.detach().numpy(),
                              ref.router.prototypes.detach().numpy()), name

        # A file that holds every name it claims except this one is refused, so the gap cannot be reached by a
        # reader that was handed a file. Reported as a gap is the honest word for a parameter no file can supply;
        # here the file could have supplied it and did not, which is a different reason and has to read as one.
        missing = {k: v for k, v in opened.tensors.items() if k != name}
        problems = loader.name_problems(hand(opened.metadata, missing))
        assert any(p == f"{name}: the claim covers it but the file does not hold it" for p in problems), problems
        gaps = loader.coverage(ref, [hand(opened.metadata, missing)], {}, None)
        assert gaps.get("router.prototypes", "").startswith(f"named {name!r}"), gaps
        assert "not a Parameter" in gaps["router.prototypes"], gaps["router.prototypes"]


def test_scalars_and_sub_block_tensors_stay_exact_through_a_quantized_file():
    """
    A quantized file is not exception-free, and the exceptions have to stay exact.

    ggml's smallest block is 32 elements, so a scalar and a tensor narrower than one block cannot be quantized
    at all. They stay F32 inside a mostly-quantized file, which means the header records a dominant type with an
    exception, and the loader learns each tensor's type from the tensor itself rather than from the header. So the
    values that cannot be quantized have to be the reference's bytes exactly while the values around them are
    approximate - and if a loader read the header as a rule instead of as a record, the exact ones would be read
    as approximate too.

    The per-expert file at TINY is the case that shows the exception: its only scalar is the slot's gate scale.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = TINY()
        ref = tiny_model()
        md = md_for(cfg)
        rule = convert_to_gguf.quant_rule("q8_0")

        # A core file, so the per-expert file has something to be loaded against. The expert file cannot be a
        # model on its own - the config comes from the core file's claims - so the loaded model is the pair.
        core = loader.open_file(write_file(ref, cfg, md, tmp, quant="q8_0"))

        for slot in range(1, cfg.n_experts):
            entries = converter.tensor_entries(ref, cfg, expert_file_slot=slot, quant=rule)
            exceptions = [e for e in entries if e.quant_type is None]
            assert exceptions and all(e.shape[-1] < converter.MIN_QUANT_BLOCK for e in exceptions), (
                slot, [e.shape for e in exceptions])

            path = write_file(ref, cfg, md, tmp, quant="q8_0", slot=slot)
            opened = loader.open_file(path)
            loaded = loader.load_model([core, opened], blob=None)

            for name, entry in sorted(opened.tensors.items()):
                module_path = loader.module_path(name, opened, cfg)
                got = loaded_value(loaded.model, module_path)
                expected = ref.state_dict()[module_path].detach().numpy()
                exact = np.array_equal(got, expected)
                if entry["type"] == "F32":
                    assert exact, (slot, name, entry["type"])
                else:
                    assert not exact, (slot, name, "a quantized tensor is reported as exact")

            # The header records the dominant type, and the exceptions are what make it a record rather than a
            # statement about every tensor.
            assert int(converter.read_metadata(path)["general.file_type"]) == int(LlamaFileType.MOSTLY_Q8_0), slot

        # The core file's own exceptions: the router's prototypes, the read-weight rows, and every scalar. The
        # rows are excluded from quantization by the harness's rule, so the file records F32 for them; the test
        # asserts the reason is recorded, not only the type.
        #
        # The reasons are read from the entries the file holds rather than by looking every state-dict path up in
        # the mapping, because one set of paths in the mapping cannot be looked up: slot 0's. The reference's
        # module at that index holds non-persistent buffers, so `state_dict()` carries nothing for it, and
        # `tensor_name` refuses those paths because a file has no destination for them. A core file's entries are
        # exactly the paths the mapping can name, so the entries are the honest set to iterate.
        core_entries = converter.tensor_entries(ref, cfg, quant=rule)
        reasons = {e.module_path: converter.scheme_for_path(e.module_path) for e in core_entries
                   if e.module_path in converter.EXCLUSION_REASONS or "memory.table" in e.module_path}
        excluded = [e for e in core_entries if e.quant_type is None]
        assert excluded, "the core file has no exceptions, which contradicts the format's block minimum"
        for e in excluded:
            assert e.shape[-1] < converter.MIN_QUANT_BLOCK or e.quant_type is None, (e.name, e.shape)
        assert any("row" in str(reasons.get(e.module_path, "")) or e.module_path in converter.EXCLUSION_REASONS
                   or e.shape[-1] < converter.MIN_QUANT_BLOCK for e in excluded), sorted(reasons.items())[:3]


def test_the_table_blob_rounds_the_way_the_harness_quantizer_rounds():
    """
    The blob is the one place the writer and the reader meet on arithmetic rather than on names.

    The table is not a ggml tensor, so there is no shared dequantize function to agree with: the writer rounds
    with the harness's scheme and the reader multiplies codes by the stored scale. If the reader recomputed the
    scale from the dequantized value, re-rounding would produce different codes, and the parity invariant the
    harness records - that the stored scale is what reproduces the codes - would be broken in the reader while
    looking correct in the writer. So the reader's arithmetic is asserted against the harness's, element for
    element, at TINY.

    `qmax` is 128 rather than 127 because the int8 range is not symmetric. That difference is the effective-bits
    question, so the test asserts the range as data and the divisor as what falls out of it.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = TINY()
        ref = tiny_model()
        md = md_for(cfg)

        blob_bytes = converter.write_table_blob(ref.memory.table.weight.detach(), cfg)
        path = tmp / "heron_table.bin"
        path.write_bytes(blob_bytes)

        # The blob has no header, so every fact about its shape comes from the claims; a hand-built file that
        # carries only those claims is enough, because there is no tensor-info block to read.
        blob = loader.open_blob(path, hand(md, {}))

        codes, scales = harness_quant.quantize(ref.memory.table.weight, harness_quant.TABLE_INT8)
        stored = harness_row(codes, scales, cfg.ngram_memory_entries, cfg.ngram_memory_dim)
        assert np.array_equal(blob.codes, stored[0]), (blob.codes[0][:8], stored[0][0][:8])
        assert np.array_equal(blob.scales, stored[1]), (blob.scales[:8], stored[1][:8])

        # The probe arithmetic: code x the stored scale, never a recomputed max-abs. This is the operation the
        # engine performs host-side, so it has to be the operation the reader performs.
        slot = hash_ngram((7, 11, 13, 17), cfg.ngram_memory_entries)
        row = blob.row(slot)
        expected = (blob.codes[slot].astype(np.float32) * np.float32(blob.scales[slot])).reshape(cfg.ngram_memory_dim)
        assert np.array_equal(row, expected), row[:4]
        # The harness's own dequantized row is the same values, which is the check that the reader's arithmetic is
        # the writer's rounding read back rather than a second rounding.
        harness_values = harness_quant.dequantized(ref.memory.table.weight, harness_quant.TABLE_INT8)
        assert np.allclose(row, harness_values[slot].detach().numpy()), (row[:4], harness_values[slot][:4])

        # qmax as the range states it, and the divisor that falls out of it. Asserted as data because the
        # effective-bits number in the ROADMAP depends on which of the two is quoted. The harness's scheme states
        # the range as two fields rather than as a divisor, so the divisor is computed here from the range the
        # scheme holds - which is the only way the two can be shown to agree.
        lo, hi = converter.TABLE_CODE_RANGE
        assert (lo, hi) == (harness_quant.TABLE_INT8.code_min, harness_quant.TABLE_INT8.code_max), (lo, hi)
        assert (lo, hi) == (-128, 127), (lo, hi)
        assert max(abs(lo), abs(hi)) == max(abs(harness_quant.TABLE_INT8.code_min),
                                             abs(harness_quant.TABLE_INT8.code_max)) == 128

        # A blob whose length does not match the claimed shape cannot be read as rows, and counting is the only way
        # to find that out.
        truncated = tmp / "short.bin"
        truncated.write_bytes(blob_bytes[:-1])
        try:
            loader.open_blob(truncated, hand(md, {}))
        except ValueError as e:
            assert "holds" in str(e), str(e)
        else:
            raise AssertionError("a blob with the wrong number of bytes must be refused")

        order = dict(md)
        order["general.arch.heron.table.blob_order"] = ["scales", "codes"]
        try:
            loader.open_blob(path, hand(order, {}))
        except ValueError as e:
            assert "codes then scales" in str(e), str(e)
        else:
            raise AssertionError("a blob that states a different order must be refused")


def harness_row(codes, scales, n_rows, width):
    """
    The harness's stored row, as numpy, for a byte comparison.

    Mirrors the helper in tests/test_converter.py for the same reason: `quantize` returns tensors built from a
    Parameter, so they carry grad and cannot be turned into numpy directly, and a per-row scheme returns codes
    shaped (rows, 1, width), so a row comparison has to reshape rather than compare the block view.
    """
    return (
        codes.detach().to(torch.int8).reshape(n_rows, width),
        scales.detach().reshape(-1).to(torch.float16),
    )


def test_a_header_that_states_a_type_the_bytes_cannot_be_is_refused():
    """
    The byte-count check, on its own.

    The shape check and the byte-count check are two halves of the same assertion, and each can pass while the
    tensor is wrong: a transposed tensor has the right element count, and a header stating a type the bytes cannot
    be has the right shape. So the test hands the reader a file whose header says Q8_0 over F32 bytes, which is
    the failure a writer that trusted a stored type rather than the bytes would produce.

    The reader learns each tensor's type from the tensor itself, so `general.file_type` is not checked here at all
    - a mislabelled header is a record that disagrees with the file, not a tensor that is wrong.

    The tensor this test uses is not square, which is what makes the second half of the check possible at all: for
    a square tensor the transpose is the same shape and the same element count, so a transposed read would be
    indistinguishable. `blk.0.ffn_gate` is (ffn_width, d_model), and at TINY those are different numbers.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = TINY()
        ref = tiny_model()
        path = write_file(ref, cfg, md_for(cfg), tmp)
        opened = loader.open_file(path)
        name = "blk.0.ffn_gate"
        shape = converter.expected_shapes(cfg)["blocks.{bid}.ffn.gate.weight"]
        assert shape[0] != shape[1], "a square tensor cannot show that a transposed read gives different values"

        wrong_type = dict(opened.tensors[name], type="Q8_0")
        try:
            loader.value_for(name, hand(opened.metadata, {name: wrong_type}), cfg)
        except ValueError as e:
            assert "stores" in str(e) and "the file says" in str(e), str(e)
        else:
            raise AssertionError("a header stating a type the bytes cannot be must be refused")

        # The reverse, with bytes that would be right for the claimed type but a shape that is the transpose.
        # The file's own dims are already the reverse of the module's shape, so a wrong one is the module's shape
        # written as the file's dims: the same numbers, read as a different matrix.
        flipped = dict(opened.tensors[name], dims=list(shape))
        try:
            loader.value_for(name, hand(opened.metadata, {name: flipped}), cfg)
        except ValueError as e:
            assert "are not the reverse of the module's" in str(e), str(e)
        else:
            raise AssertionError("a tensor whose dims are the transpose of the module's shape must be refused")

        # Within a quantized block, ggml stores the first block_size elements of the flattened tensor, so the
        # dequantized values come back in the order the module wrote them: reshape is the right operation here and
        # transpose is not. Asserted by writing the values back through the reference's own element order.
        q = write_file(ref, cfg, md_for(cfg), tmp, quant="q8_0")
        opened_q = loader.open_file(q)
        got = loader.value_for(name, opened_q, cfg)
        entry = opened_q.tensors[name]
        assert got.shape == shape, got.shape
        flat = np.asarray(quants.dequantize(np.asarray(entry["data"]).reshape(-1), Quant.Q8_0))
        assert flat.shape == (int(np.prod(shape)),), flat.shape
        assert np.array_equal(got, flat.reshape(shape)), "reshape is the operation, not a view of the block"

        # The same bytes read through the reverse shape are the same numbers in a different arrangement, so they
        # are not the same tensor. That is the reason the check is against the reverse of the module's shape rather
        # than against its element count: element count cannot tell the two apart, and only the shape says which
        # arrangement the writer meant. A comparison against the reference's raw bytes cannot show this, because a
        # quantized file is not the reference's bytes.
        transposed = flat.reshape(tuple(reversed(shape))).T
        assert not np.array_equal(got, transposed), "the reshape check is vacuous for this tensor's shape"

        # And the reader has to refuse a type it cannot read, because `GGML_QUANT_SIZES` is the only place the
        # storage size comes from.
        unknown = dict(opened.tensors[name], type="NOT_A_TYPE")
        try:
            loader.value_for(name, hand(opened.metadata, {name: unknown}), cfg)
        except ValueError as e:
            assert "cannot be read" in str(e), str(e)
        else:
            raise AssertionError("a type the format does not have must be refused")


def test_numeric_claims_are_compared_at_the_precision_the_format_has():
    """
    A float metadata field is float32, so `1e-06` does not survive the round trip.

    That is not a bug to work around; it is a fact about what a file can state. A reader that compares with
    Python's precision refuses a faithful writer, which is the worst kind of check: it reports a correct file as a
    wrong one. So every numeric claim a loader acts on is compared as float32, and the test asserts both directions
    - a faithful writer's value is accepted, and a value that differs at float32 precision is refused.

    The refusal matters because these numbers are module constants rather than config fields: a loader given a
    different eps has no way to honour it without changing a constant, so it has to refuse.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = TINY()
        ref = tiny_model()
        path = write_file(ref, cfg, md_for(cfg), tmp)
        opened = loader.open_file(path)

        stated = opened.metadata["general.arch.heron.norm_eps"]
        assert loader.stored_float(stated) == loader.stored_float(EPS), (stated, EPS)
        assert float(stated) != EPS, "the format's precision is what the test is asserting on"
        assert not any("norm_eps" in p for p in loader.claim_problems(opened)), loader.claim_problems(opened)

        md = dict(opened.metadata)
        md["general.arch.heron.norm_eps"] = float(np.float32(2e-6))
        problems = loader.claim_problems(hand(md, opened.tensors))
        assert any("norm_eps" in p for p in problems), problems
        assert any("cannot honour" in p for p in problems), problems

        # The same argument for the rope base, which is a constant the reference builds angles from.
        md = dict(opened.metadata)
        md["general.arch.heron.rope.freq_base"] = float(np.float32(500000.0))
        problems = loader.claim_problems(hand(md, opened.tensors))
        assert any("rope.freq_base" in p for p in problems), problems

        # And the hash, which a loader cannot honour at all: a different hash lands on a different slot, and a
        # loaded model that disagrees about slots is a different model.
        md = dict(opened.metadata)
        md["general.arch.heron.table.hash"] = "sha256_modulo_entries"
        problems = loader.claim_problems(hand(md, opened.tensors))
        assert any("blake2b" in p for p in problems), problems

        # A claim the file does not state is not refused, because nothing in a file can state a constant: the
        # loader uses the reference's value and says so, which is the difference between a claim and a fact.
        md = dict(opened.metadata)
        del md["general.arch.heron.norm_eps"]
        assert not any("norm_eps" in p for p in loader.claim_problems(hand(md, opened.tensors))), (
            loader.claim_problems(hand(md, opened.tensors)))


def test_the_mixer_rule_is_checked_against_a_config_from_outside_the_file():
    """
    The one behaviour fact that only a claim states, checked in the only direction it can be checked.

    A full-attention block and a sliding-window block have the same tensors, so this cannot be derived from a name
    or a shape. What a file states is the per-block result; the interval that produced the result is not in the
    file. So a file checked against the config built from its own claims cannot fail when the pattern has a single
    full block: any interval at least as large as the output block produces the same pattern, and the pattern is
    the behaviour for the blocks the file has.

    That is why the check has to be able to say three different things, and the test asserts all three:
      - the pattern implies an interval, and a config claiming a different one is refused;
      - the pattern implies nothing, and the loaded model's per-block rule still matches the reference's;
      - the pattern cannot be produced by any interval, which is a refusal rather than a report.
    """
    cfg = TINY()

    # Case 1: a pattern that implies an interval, checked against a config that claims a different one. Output
    # start 2, interval 3 implies full at 2 and 5; a config with 4 cannot produce that.
    md = {"general.arch.heron.loop.input_layers": [0],
          "general.arch.heron.loop.layers": [1],
          "general.arch.heron.loop.output_layers": [2, 3, 4, 5],
          "general.arch.heron.layer_pattern": [0, 0, 1, 0, 0, 1]}
    file = hand(md, {})
    other = HeronConfig(input_blocks=1, loop_blocks=1, output_blocks=4, full_attention_every=4)
    problems = loader.check_mixer_rule(other, file)
    assert any("the pattern implies 3" in p for p in problems), problems

    # Case 2: the same pattern against a config that agrees with it.
    same = HeronConfig(input_blocks=1, loop_blocks=1, output_blocks=4, full_attention_every=3)
    assert not loader.check_mixer_rule(same, file), loader.check_mixer_rule(same, file)

    # Case 3: a pattern no interval produces. Written with the key the claim reads, because `layer_pattern` on its
    # own is not a claim - the claim is the metadata key, and a test that wrote the short form would be checking a
    # key the reader never looks at.
    md = dict(md, **{"general.arch.heron.layer_pattern": [0, 0, 1, 1, 0, 1]})
    problems = loader.check_mixer_rule(same, hand(md, {}))
    assert any("no interval produces this pattern" in p for p in problems), problems

    # Case 4: the pattern that TINY's file states, which has one full block and therefore implies nothing. The
    # loaded config keeps the dataclass interval, and the per-block rule still matches the reference's for every
    # block - which is the honest way to state what a reader can reproduce from a file.
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ref = tiny_model()
        opened = loader.open_file(write_file(ref, cfg, md_for(cfg), tmp))
        loaded = loader.load_model([opened], blob=None)
        assert loaded.cfg.full_attention_every != cfg.full_attention_every, (
            loaded.cfg.full_attention_every, cfg.full_attention_every)
        assert all(loaded.cfg.mixer_mode(b) == cfg.mixer_mode(b) for b in range(cfg.n_blocks())), (
            [loaded.cfg.mixer_mode(b) for b in range(cfg.n_blocks())],
            [cfg.mixer_mode(b) for b in range(cfg.n_blocks())])
        assert not loader.check_mixer_rule(loaded.cfg, opened), loader.check_mixer_rule(loaded.cfg, opened)
        kind, implied = loader.attention_interval([int(x) for x in opened.metadata["general.arch.heron.layer_pattern"]],
                                                  cfg.output_start_block())
        assert (kind, implied) == ("underdetermined", None), (kind, implied)
        assert "not implied" in loaded.derived()["full_attention_every"], loaded.derived()

        # A pattern that implies nothing is honoured by every interval at least as large as the output block, so a
        # config that differs from the one the file was written from is not necessarily wrong. Interval 2 is the
        # reference's own; interval 3 would put the next full block at 9, outside an 8-block model, so it produces
        # the same rule as the file; interval 5 the same. What can still be refused is an interval that produces a
        # different rule - one smaller than the output block, which puts full attention where the file says
        # sliding window.
        for interval in (2, 3, 5):
            honouring = HeronConfig(input_blocks=cfg.input_blocks, loop_blocks=cfg.loop_blocks,
                                    output_blocks=cfg.output_blocks, full_attention_every=interval)
            assert not loader.check_mixer_rule(honouring, opened), (interval,
                                                                    loader.check_mixer_rule(honouring, opened))
        contradicted = HeronConfig(input_blocks=cfg.input_blocks, loop_blocks=cfg.loop_blocks,
                                   output_blocks=cfg.output_blocks, full_attention_every=1)
        problems = loader.check_mixer_rule(contradicted, opened)
        assert problems and any("block 7" in p for p in problems), problems

        # The same file against the config it was written from, in the direction a reader cannot check from the
        # file alone: the interval is outside the file, so the check only means something when a config arrives.
        assert not loader.check_mixer_rule(cfg, opened), loader.check_mixer_rule(cfg, opened)
        wrong = HeronConfig(input_blocks=cfg.input_blocks, loop_blocks=cfg.loop_blocks,
                            output_blocks=cfg.output_blocks, full_attention_every=3)
        assert not loader.check_mixer_rule(wrong, opened), loader.check_mixer_rule(wrong, opened)

        # A pattern of the wrong length is refused as a length, not as a cache rule.
        md = dict(opened.metadata, **{"general.arch.heron.layer_pattern": [0] * (cfg.n_blocks() - 1)})
        problems = loader.check_mixer_rule(cfg, hand(md, opened.tensors))
        assert any("config expects" in p for p in problems), problems


def test_what_a_file_cannot_state_is_reported_as_unrecorded_and_not_guessed():
    """
    The fields a file cannot state, with the reason for each.

    A loader that guessed a training default would produce a model that looks like the reference and was decided
    by the reader. So the config is built from the claims only, and what the file never said is returned as
    `unrecorded`. The distinction between `unrecorded` and `derived` is the point of this test: a pattern with
    two or more full blocks implies its interval, a pattern with one block implies nothing at all, and both are
    honoureable by a config that holds an interval.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = TINY()
        ref = tiny_model()
        opened = loader.open_file(write_file(ref, cfg, md_for(cfg), tmp))
        loaded = loader.load_model([opened], blob=None)

        un = loaded.unrecorded
        assert set(un) == set(loader.unrecorded()), sorted(un)
        for field in ("loop_curriculum_expand", "ngram_optimizer", "full_attention_every", "ngpt", "expert_slots"):
            assert field in un, sorted(un)
            assert un[field], f"{field} has no reason stated"

        # The fields the file does state have to come back as the config, not as the reader's types. This is where
        # the two are reconciled once, so the test asserts the reconciliation rather than trusting it.
        for key, field_name in loader.METADATA_FIELDS.items():
            if key in opened.metadata:
                value = getattr(loaded.cfg, field_name)
                stated = opened.metadata[key]
                if isinstance(stated, (list, tuple)):
                    assert [int(x) for x in stated] == value, (key, stated, value)
                elif isinstance(stated, bool):
                    assert bool(stated) == value, (key, stated, value)
                elif isinstance(stated, (int, np.integer)):
                    assert int(stated) == value, (key, stated, value)
                    assert isinstance(value, int), (key, type(value))

        # `expert_slots` is not a key a file states; `expert_slot_layers` is. The count follows only because the
        # design puts one slot in each block, which is what makes a name per slot possible at all. A file that
        # repeated a block is refused for that reason, not for a shape reason.
        md = dict(opened.metadata)
        md["general.arch.heron.expert_slot_layers"] = [6, 6]
        file = hand(md, {k: v for k, v in opened.tensors.items() if not k.startswith("blk.6")})
        try:
            loader.config_from_metadata(file)
        except ValueError as e:
            assert "cannot state" in str(e), str(e)
        else:
            raise AssertionError("a slot claim that repeats a block must be refused")
        assert "one slot in each block" in loaded.unrecorded["expert_slots"], loaded.unrecorded["expert_slots"]

        # `ngram_orders` is a stated claim, so it is not in `unrecorded` - and the loaded model has to be able to
        # say it came from the file.
        assert "ngram_orders" not in loaded.unrecorded, sorted(loaded.unrecorded)
        assert tuple(loaded.cfg.ngram_orders) == tuple(cfg.ngram_orders), loaded.cfg.ngram_orders
        assert [int(x) for x in opened.metadata["general.arch.heron.ngram.orders"]] == list(cfg.ngram_orders)

        # The report prints what it cannot state, so a person sees it. `derived` is the line that says whether the
        # pattern implied the interval, and it has to be printed separately.
        report = loaded.report()
        assert "unrecorded  full_attention_every" in report, report
        assert "derived  full_attention_every" in report, report
        print("  report for a core file alone:", report.splitlines()[-1])


def test_coverage_accounts_for_every_tensor_with_its_reason():
    """
    Every tensor the reference has - parameters and buffers together - and whether any of these files can supply
    its bytes.

    Reported rather than assumed, because the gaps are the interesting part and each has a different reason. The
    reasons come from `converter.tensor_name`, the same mapping the writer used, so the two directions cannot
    disagree about what a name is. What is not a gap is decided by what was loaded, because a parameter can arrive
    in a form a name lookup cannot express: `stream.read_weights` has no single name, and the hash table is not a
    tensor at all.

    The accounting has to cover buffers as well as parameters, because the generalist's tensors are buffers and the
    parameter loop cannot see them. A report built only from `named_parameters()` would leave slot 0 out of the
    accounting entirely - neither loaded nor gap - which is the same kind of hole as the table being called a gap
    while the loader held its values.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = TINY()
        ref = tiny_model()
        core = loader.open_file(write_file(ref, cfg, md_for(cfg), tmp))

        # A core file alone: the table is not a tensor, so only the blob can supply it, and the generalist's
        # tensors have no destination in any file. Slot 0 is reported even though it is not a parameter, which is
        # what this test is checking as much as the reasons: the gap count cannot come from a parameter loop.
        slot0 = f"experts.{cfg.generalist_slot}."
        assert not any(p.startswith(slot0) for p, _ in ref.named_parameters()), sorted(ref.named_parameters())[:3]
        gaps = loader.coverage(ref, [core], {}, None)
        assert gaps.get("memory.table.weight") == "not a tensor in any file; only the blob can supply it", gaps
        assert any(p.startswith(slot0) for p in gaps), sorted(gaps)
        assert all(p in dict(ref.named_buffers()) for p in gaps if p.startswith(slot0)), sorted(gaps)
        assert "stream.read_weights" in gaps, sorted(gaps)
        assert gaps["stream.read_weights"] == "the claim names one row per block; no file holds a row", gaps

        # The same files, with the rows supplied as n names and the table supplied by a blob: both are supplied,
        # so neither is a gap, and the report says what remains.
        (tmp / "heron_table.bin").write_bytes(converter.write_table_blob(ref.memory.table.weight.detach(), cfg))
        loaded = loader.load_model([core], blob=None)
        gaps = loader.coverage(ref, [core], loaded.loaded, None)
        assert "stream.read_weights" not in gaps, gaps
        assert "memory.table.weight" in gaps, gaps
        gaps_with_blob = loader.coverage(ref, [core], loaded.loaded, loader.open_blob(tmp / "heron_table.bin", core))
        assert "memory.table.weight" not in gaps_with_blob, gaps_with_blob
        # A core file alone cannot supply any expert, so the gaps are every expert's tensors - slot 0's because
        # no file can name them, slots 1 and 2's because the claim names them and this file does not hold them. Two
        # different reasons, and the report has to keep them apart.
        assert all(p.startswith("experts.") for p in gaps_with_blob), sorted(gaps_with_blob)
        slot0 = [p for p in gaps_with_blob if p.startswith("experts.0.")]
        assert slot0 and all("has no file" in gaps_with_blob[p] for p in slot0), sorted(gaps_with_blob)[:4]
        others = [p for p in gaps_with_blob if not p.startswith("experts.0.")]
        assert others and all("no file holds it" in gaps_with_blob[p] for p in others), sorted(gaps_with_blob)[:4]

        # The router: supplied when the file holds it, reported with its reason when the file does not. It is not a
        # Parameter, so a state dict would not carry it either, and the reason has to read as that rather than as
        # a missing name.
        name = converter.tensor_name("router.prototypes", cfg)
        missing = {k: v for k, v in core.tensors.items() if k != name}
        gaps = loader.coverage(ref, [hand(core.metadata, missing)], {}, None)
        assert "router.prototypes" in gaps and "not a Parameter" in gaps["router.prototypes"], gaps

        # And the whole model, loaded: every tensor the reference has accounted for, either by a file that supplied
        # its bytes or by a gap that says why no file can. `loaded` plus `gaps` is the whole set only once the blob
        # is counted as a third way a parameter can be supplied, which is the honest form: the table is neither a
        # name nor a gap, it is bytes in a file that is not a GGUF file. A report that counted only names and gaps
        # would call the table a gap while the loader was holding its values. The accounting is over parameters and
        # buffers together, because the generalist's tensors are buffers: a loop over `named_parameters()` alone
        # would report a complete accounting while leaving slot 0 out of it.
        files, blob = write_set(ref, cfg, tmp)
        opened = [loader.open_file(p) for p in files]
        loaded = loader.load_model(opened, blob=loader.open_blob(blob, opened[0]))
        blob_supplied = {"memory.table.weight"}
        # `loaded` is keyed by module path, which is the form the accounting needs: a parameter is accounted for
        # when some file supplied its bytes, and the name that supplied them is a fact about the file rather than
        # about the module. A row is an element selection on one Parameter, so the key is `stream.read_weights[3]`
        # while the parameter is `stream.read_weights`; the brackets are stripped because a row is the same
        # parameter in the only form a file can hold it in. Without this the accounting would report one parameter
        # as unaccounted while the file that supplied it was in the report.
        accounted = {p.split("[")[0] for p in loaded.loaded} | set(loaded.gaps) | blob_supplied
        for path, _ in list(ref.named_parameters()) + list(ref.named_buffers()):
            assert path in accounted, (path, sorted(loaded.gaps))
        assert any(p.startswith("experts.0.") for p in loaded.gaps), sorted(loaded.gaps)
        assert "router.prototypes" in accounted, sorted(loaded.gaps)
        assert "memory.table.weight" not in loaded.gaps, loaded.gaps


def test_the_loader_refuses_a_file_and_reports_what_it_did_not_fill():
    """
    The refusal and the report, together, on a file that is complete except for one tensor.

    This is the failure mode the phase is about: a loader that loads a file it cannot explain produces a model
    with one parameter at its initialization, and a model with a randomly initialized router or a randomly
    initialized expert produces logits that look exactly like logits. So the test asserts that a reader which was
    handed a file refuses it, and that a reader that was handed a set reports the pieces it could not fill - the
    two behaviours have to be different, because a reader that reported everything as missing would never load
    anything and a reader that loaded everything would never refuse anything.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = TINY()
        ref = tiny_model()
        files, blob = write_set(ref, cfg, tmp)
        opened = [loader.open_file(p) for p in files]
        loaded = loader.load_model(opened, blob=loader.open_blob(blob, opened[0]))
        assert "gap" in loaded.report(), loaded.report()

        # A per-expert file that also carried a core tensor is refused: the file is one slot set, and a file that
        # mixes the two cannot be loaded as a unit.
        core = opened[0]
        expert = opened[1]
        mixed = dict(expert.tensors)
        mixed["blk.0.attn_q"] = dict(core.tensors["blk.0.attn_q"])
        problems = loader.file_problems(hand(expert.metadata, mixed))
        assert any("the claim cannot explain it" in p for p in problems), problems

        # The same tensor, in a core file that claims a name for it, is accepted - which is the other half of the
        # scope check: the claim is model-wide, so a per-expert file is checked against the patterns it is allowed
        # to hold, not against the whole claim. `scope()` answers with module patterns, which is the half of the
        # claim that says what a file may hold; the name is what the claim maps that pattern to.
        assert not any("blk.0.attn_q" in p for p in loader.file_problems(core)), loader.file_problems(core)
        assert "blocks.{bid}.attn.q.weight" not in expert.scope(), expert.scope()
        assert "blocks.{bid}.attn.q.weight" in core.scope(), core.scope()

        # A file that states a slot it cannot hold: the writer refuses to write one, and the reader refuses to
        # load one, for the same reason stated in the same words.
        md = dict(expert.metadata)
        md["general.arch.heron.expert_file_slot"] = 0
        try:
            converter.tensor_entries(ref, cfg, expert_file_slot=0)
        except ValueError as e:
            assert "has no file" in str(e), str(e)
        else:
            raise AssertionError("the writer must refuse slot 0")
        assert any("has no content" in p for p in loader.file_problems(hand(md, expert.tensors))), (
            loader.file_problems(hand(md, expert.tensors)))


def test_the_loaded_model_and_the_reference_disagree_only_where_the_file_says_they_may():
    """
    The forward pass through a loaded model, in the form a reader can actually run.

    The memory path is not run: the table is read host-side through the blob, so a probe has to be fed in as
    `slot_ids`. Calling forward without them is the honest form of "the file describes the model except for the
    blob", and the blob is asserted separately rather than pretending a GGUF file contains a probe op.

    What is asserted here is that the loaded model can be run the way the reference is run, on the same input, and
    that its routing decision comes from the file's prototypes rather than from the module it was built with. A
    reader that left the prototypes at their placeholder values would produce routing that looked fine and was the
    reader's choice, not the file's.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = TINY()
        ref = tiny_model()
        files, blob = write_set(ref, cfg, tmp)
        opened = [loader.open_file(p) for p in files]
        loaded = loader.load_model(opened, blob=loader.open_blob(blob, opened[0]))

        tokens = tokens_for(cfg, 32)
        for active_slot in range(1, cfg.n_experts):
            for loop in range(1, cfg.max_loop + 1):
                a = ref.forward(tokens, loop=loop, active_slot=active_slot)
                b = loaded.forward(tokens, loop=loop, active_slot=active_slot)
                assert torch.equal(a, b), (loop, active_slot, float((a - b).abs().max()))

        # The probe arithmetic, on the same slot, through both sides. The reference reads the table as a Parameter;
        # the engine reads codes plus scales. At F32 the two are the same values, which is the check that the
        # loader's blob path and the reference's parameter path describe one table rather than two.
        from heronlm.memory import hash_ngram, slot_ids
        prefix = [[7, 11, 13, 17]]
        order = max(cfg.ngram_orders)
        slots = slot_ids(tokens, cfg, prefix)
        values = loader.open_blob(tmp / "heron_table.bin", opened[0]).values()
        for pos in range(slots.shape[1]):
            slot = int(slots[0][pos])
            row = values[slot]
            expected = ref.memory.table.weight[slot].detach().numpy()
            assert np.array_equal(row, expected), (pos, slot, row[:4], expected[:4])
            # The key is the rolling window that ends at this token, not the whole prefix plus the token. At
            # position 0 the prefix is exactly the order long, so the window is the last `order` tokens of the
            # history and the first prefix token has already left it. Asserted as the rule rather than as a fixed
            # tuple because the rule is what the engine computes incrementally per token, and keying on a shorter
            # n-gram is the silent version of this mistake.
            key = (prefix[0] + tokens[0, : pos + 1].tolist())[-order:]
            assert len(key) == order, (pos, key)
            assert slot == hash_ngram(tuple(key), cfg.ngram_memory_entries), (pos, key, slot)

        # The routing decision, from the file's directions. Same input, same prototypes, same decision - and the
        # decision is asserted as data rather than as a score, because the score is what the gate measures later
        # and not something a reader can check at TINY.
        for path in ("router.prototypes",):
            assert np.array_equal(loaded.model.router.prototypes.detach().numpy(),
                                  ref.router.prototypes.detach().numpy()), path
        a = ref.forward(tokens, loop=1, active_slot=None)
        b = loaded.forward(tokens, loop=1, active_slot=None)
        assert torch.equal(a, b), float((a - b).abs().max())


def test_a_file_that_is_a_record_of_a_run_can_state_only_what_a_run_knew():
    """
    Provenance, as the reader sees it.

    The router's prototypes cannot come from a checkpoint, because they are not a Parameter. Whatever produced
    them - clustered teacher activations, a placeholder, a hand-written file - is a fact a reader cannot recover
    from bytes. So the loader prints the claim it can check (the shape, the name, that the values are the file's)
    and does not assert what it cannot know (where the directions came from). The test asserts that distinction,
    because a metadata key that states an unbackable provenance is a claim a reader would have to accept on trust.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = TINY()
        ref = tiny_model()
        opened = loader.open_file(write_file(ref, cfg, md_for(cfg), tmp))
        name = converter.tensor_name("router.prototypes", cfg)

        # What a file can state about the prototypes: the shape and the name. Both are checked, in the direction
        # the reader has to check them - the shape from the config, the name from the claim.
        entry = opened.tensors[name]
        pattern = loader.pattern_for(name, opened)
        assert pattern == "router.prototypes", pattern
        assert tuple(reversed(entry["dims"])) == converter.expected_shapes(cfg)[pattern], entry["dims"]
        assert np.array_equal(loader.value_for(name, opened, cfg).reshape(-1),
                              ref.router.prototypes.detach().numpy().reshape(-1))

        # A file that states a shape the values cannot be is refused, which is the same check as any tensor.
        wrong = dict(entry, dims=[cfg.n_experts, cfg.d_model - 1])
        try:
            loader.value_for(name, hand(opened.metadata, {name: wrong}), cfg)
        except ValueError as e:
            assert "are not the reverse of the module's" in str(e), str(e)
        else:
            raise AssertionError("a claimed shape the bytes cannot be must be refused")

        # And the reader's own limit, stated rather than assumed: a loaded model's routing comes from whatever the
        # file holds, and the reader cannot say what produced it. The report says "unrecorded" for every field no
        # file can state, and the only provenance a file can state is the tokenizer's, which is where the pinned
        # vocabulary's SHA-256 belongs - not here.
        un = loader.unrecorded()
        assert "ngram_optimizer" in un and "training fact" in un["ngram_optimizer"], un
        assert "loop_curriculum_expand" in un and "how a model was trained" in un["loop_curriculum_expand"], un
        assert not any("provenance" in k for k in un), sorted(un)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:
                failures += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
