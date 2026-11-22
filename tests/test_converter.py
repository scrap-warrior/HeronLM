"""
Converter tests, at TINY.

These are not quality tests and they are not speed tests. They check the two things a converter can get
wrong without anything in the output saying so:

  - a tensor written under a name the metadata does not claim, or into a block the config does not put it in.
    The shapes stay right and the file opens, so the only symptom is a model that computes different numbers.
  - bytes that are not the reference's bytes - a pre-normalized weight, a transposed element order, a scale
    recomputed from the dequantized value instead of the stored one.

Everything is written to a temporary file and read back with the same reader the engine would use, because a
test that compares a dict against the dict it was given proves nothing about the file.

Run: python tests/test_converter.py
"""

import contextlib
import dataclasses
import hashlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The CLI is appended, not inserted, so the `phase0` package below still resolves to the package at the project
# root rather than to scripts/phase0.py. Importing the CLI as a module rather than shelling out is what lets the
# test assert on the report it prints.
sys.path.append(str(ROOT / "scripts"))

import numpy as np
import torch
import convert_to_gguf
from gguf import LlamaFileType, quants
from gguf import GGUFWriter
from gguf.constants import GGML_QUANT_SIZES, Keys
from gguf.constants import GGMLQuantizationType as Quant
from gguf.constants import TokenType

from heronlm import converter
from heronlm.config import FULL, TINY
from heronlm.metadata import (
    BASE_ALPHABET_SIZE,
    SPECIAL_TAGS,
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_NORMAL,
    TENSOR_NAMES,
    base_alphabet,
    byte_level_rule,
    metadata,
    placeholder_vocabulary,
    special_token_ids,
    token_types,
)
from heronlm.model import HeronLM
from phase0 import quant as harness_quant


def tiny_model():
    """Deterministic: the same parameter bytes on every run, so a byte comparison is meaningful."""
    torch.manual_seed(0)
    return HeronLM(TINY())


def params_of(model):
    """
    The reference's own bytes, by module path, for byte comparisons.

    `named_parameters()` is the whole set except one: the router stores its prototypes as a plain attribute, so
    they are not in it. They are still bytes the file must contain, so a byte comparison needs them - and the
    test that asserts the asymmetry reads `named_parameters()` directly rather than this helper.
    """
    out = {name: p.detach().numpy() for name, p in model.named_parameters()}
    out["router.prototypes"] = model.router.prototypes.detach().numpy()
    return out


def reference_bytes(params, module_path):
    """
    The reference's bytes for one entry.

    A read-weight row names an element selection on the shared tensor, so it is resolved back to the row;
    every other path is a parameter as it is. This is the only place the two kinds of path are reconciled, so
    the byte comparisons below do not each carry their own parsing.
    """
    if "[" not in module_path:
        return params[module_path]
    row = int(module_path.split("[")[1].rstrip("]"))
    return params["stream.read_weights"][row]


def metadata_for(cfg, tokens=None, merges=("merge a b",)):
    return metadata(cfg, tokens or placeholder_vocabulary(cfg.vocab_size), list(merges))


def element(value):
    """A metadata element as the read side reports it: the reader gives numpy integers for numbers."""
    return int(value) if isinstance(value, (int, np.integer)) else value


def quant_bytes(shape, quant_type):
    """The storage a quantized tensor occupies: one block header plus codes, per block."""
    block, type_size = GGML_QUANT_SIZES[quant_type]
    return int(np.prod(shape)) // block * type_size


def harness_row(codes, scales, n_rows, width):
    """
    The harness's stored row, as numpy, for a byte comparison.

    Two details that are easy to get wrong and both show up as a comparison returning False rather than as an
    error: `quantize` returns tensors built from a Parameter, so they carry grad and cannot be turned into
    numpy directly; and the codes come back shaped as the scheme blocked them, which for a per-row scheme is
    (rows, 1, width), so a row comparison has to reshape rather than compare the block view.
    """
    return (
        codes.detach().to(torch.int8).reshape(n_rows, width),
        scales.detach().reshape(-1).to(torch.float16),
    )


def dequantized(entry):
    """The file's packed bytes read back as values, shaped as the module named them."""
    values = np.asarray(quants.dequantize(entry.data.reshape(-1), entry.quant_type))
    return values.reshape(entry.shape)


def test_every_tensor_has_a_name_and_the_names_the_file_uses_are_the_names_it_claims():
    """
    Two directions, both checked.

    Forward: every tensor the reference has is either named by TENSOR_NAMES or is deliberately not a
    tensor. Backward: every name the file uses is one the metadata claims, resolved from the same dict. A
    converter with its own private copy of the names would pass the forward check and fail this one.

    The forward check is over parameters and buffers together, because the two lists are not the same set:
    slot 0's tensors are non-persistent buffers, so a loop over `named_parameters()` alone would never reach the
    paths the mapping refuses.

    The asymmetry is worth asserting on its own. `router.prototypes` is not a Parameter - the router stores it
    as a plain tensor - so `state_dict()` does not carry it, and a state-dict-driven converter would write a
    file with no router tensors and no error. The file has to have it, so the test asserts the file has
    something the state dict does not.
    """
    cfg = TINY()
    model = tiny_model()

    for path, _ in list(model.named_parameters()) + list(model.named_buffers()):
        pattern = converter.module_pattern(path)
        assert pattern in TENSOR_NAMES, f"{path} is not covered by TENSOR_NAMES"
        name = TENSOR_NAMES[pattern]
        if pattern == "memory.table.weight":
            assert name is None, "the hash table is recorded as not a tensor, in the same place as the names"
            continue
        if pattern.startswith("experts."):
            expert = int(path.split(".")[1])
            if expert == cfg.generalist_slot:
                # The generalist is the block's base FFN, so these tensors have no destination. They are buffers
                # rather than parameters, which is what makes them absent from a checkpoint; the mapping refuses
                # the paths either way, so the refusal says the same thing from either list and has to be reached
                # from this one.
                try:
                    converter.tensor_name(path, cfg)
                except ValueError as e:
                    assert "has no file" in str(e), (path, str(e))
                else:
                    raise AssertionError(f"slot {cfg.generalist_slot} must not have a tensor name: {path}")
                continue
            assert "{bid}" in name, f"{path}: an expert slot's block comes from the config, not the path"
            continue
        assert name is not None, f"{path}: no name"

    # A bare lookup on the shared tensor has no block index, so it cannot name anything. That is why
    # tensor_name_for_row exists, and why the row path carries the index in brackets.
    try:
        converter.tensor_name("stream.read_weights", cfg)
    except ValueError as e:
        assert "one row per block" in str(e), str(e)
    else:
        raise AssertionError("the shared read weights have no single name")

    # The shape table must cover every key, including the one that is not a tensor.
    assert set(converter.expected_shapes(cfg)) == set(TENSOR_NAMES)

    entries = converter.tensor_entries(model, cfg)
    claimed = {line.split(" -> ")[1] for line in metadata_for(cfg)["general.arch.heron.tensor_names"]}
    assert claimed == {name for name in TENSOR_NAMES.values() if name is not None}, (
        "the metadata records the mapping as data, so the two lists cannot be two different copies"
    )
    assert "memory.table.weight" not in " ".join(claimed), "the blob must not appear as a tensor name"

    for entry in entries:
        pattern = converter.module_pattern(entry.module_path)
        resolved = claimed_patterns(cfg)[pattern]
        assert entry.name in resolved, (entry.name, pattern, resolved)

    # The router: in the file, not in the state dict.
    assert "router.prototypes" in {e.module_path for e in entries}
    assert "router.prototypes" not in dict(model.named_parameters()), "prototypes are not a Parameter"
    assert "router_prototypes" in {e.name for e in entries}


def claimed_patterns(cfg):
    """
    The metadata's claimed names, as patterns resolved by the config.

    `{bid}` is resolved two different ways in the mapping - by the path for a block's own tensors, by the
    config for an expert slot - so resolving it here through `converter.tensor_name` keeps this test reading
    from the same source the converter does rather than recomputing the pattern arithmetic.

    The shared read weights are the one pattern whose name depends on a block index that is not in the path,
    so they are resolved row by row: the pattern names a tensor family, the file holds one tensor per block.
    """
    out = {}
    for pattern, name in TENSOR_NAMES.items():
        if name is None:
            continue
        if pattern == "stream.read_weights":
            out[pattern] = {converter.tensor_name_for_row(bid, cfg) for bid in range(cfg.n_blocks())}
        elif pattern.startswith("experts."):
            out[pattern] = {
                name.replace("{bid}", str(cfg.expert_blocks()[slot])).replace("{eid}", str(eid)).replace(
                    "{slot}", str(slot)
                )
                for eid in range(cfg.n_experts)
                if eid != cfg.generalist_slot
                for slot in range(cfg.expert_slots)
            }
        elif pattern.startswith("blocks."):
            out[pattern] = {name.replace("{bid}", str(bid)) for bid in range(cfg.n_blocks())}
        else:
            out[pattern] = {name}
    return out


def test_shared_read_weights_are_written_as_one_row_per_block():
    """
    One Parameter object, many names.

    `state_dict` carries the shared read weights once, and again as an alias in every block. A converter that
    wrote one tensor would produce a file the loader cannot index per block; a converter that wrote the aliases
    would write the same bytes n_blocks times under different names, which is the same file with a different
    tensor count. The entry list is one row per block, and the row's bytes are the block's row.
    """
    cfg = TINY()
    model = tiny_model()
    entries = converter.tensor_entries(model, cfg)
    rows = [e for e in entries if converter.module_pattern(e.module_path) == "stream.read_weights"]

    assert len(rows) == cfg.n_blocks(), (len(rows), cfg.n_blocks())
    assert len({e.name for e in rows}) == len(rows), "row names must be distinct"
    shared = params_of(model)["stream.read_weights"]
    # The whole tensor is one Parameter: one row per block, each row a branch vector. The file has one tensor
    # per block because a loader indexes it per block, so the entry's shape is a row, not the tensor.
    assert shared.shape == (cfg.n_blocks(), cfg.max_loop, cfg.gr_branches), shared.shape

    for e in rows:
        bid = int(e.module_path.split("[")[1].rstrip("]"))
        assert e.name == converter.tensor_name_for_row(bid, cfg), e.name
        assert e.shape == (cfg.max_loop, cfg.gr_branches), (e.name, e.shape)
        assert np.array_equal(e.data, shared[bid]), f"{e.name} is not the config's row {bid}"
        # The row is written as F32 for the reason the harness records: quantizing it would change which
        # branch a block reads, which is the mechanism under test rather than a rounding detail.
        assert e.quant_type is None, f"{e.name}: read weights must not be quantized"
        assert "read mixture" in harness_quant.EXCLUSION_REASONS["stream.read_weights"], e.name
    # Rows past the first exist for blocks that never loop, and they are written: the reference has those
    # values, so a file that wrote only row 0 would be a file the loader cannot index at pass k.
    assert any(int(e.module_path.split("[")[1].rstrip("]")) >= cfg.input_blocks for e in rows), (
        "no row belongs to a block that never loops",
        [e.module_path for e in rows],
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "core.gguf"
        converter.write_gguf(path, cfg, metadata_for(cfg), entries)
        tensors = converter.read_tensors(path)
        for e in rows:
            bid = int(e.module_path.split("[")[1].rstrip("]"))
            # A row is square at TINY, so the reversed dims are the same dims and the shape fact cannot
            # distinguish the two readings. The byte order can, so that is what is asserted: the file's bytes
            # are the module's row, in the module's order.
            assert np.array_equal(tensors[e.name]["data"].reshape(-1), shared[bid].reshape(-1)), e.name

    # The metadata names the shared tensor as one pattern with a block placeholder, not as n_blocks names:
    # the file's rows are one tensor family, and the metadata says so.
    assert metadata_for(cfg)["general.arch.heron.gr.read_weights_name"] == TENSOR_NAMES["stream.read_weights"], (
        metadata_for(cfg)["general.arch.heron.gr.read_weights_name"],
        TENSOR_NAMES["stream.read_weights"],
    )


def test_file_dims_are_the_reverse_of_the_module_shape_and_the_bytes_stay_in_element_order():
    """
    The convention, asserted as a pair.

    GGUF records dims fastest-first, so a [out, in] weight is recorded as [in, out]. Handing the writer
    `array.T` produces the same element count with a transposed element order, and no shape check can see it:
    the dims are right, the meaning is wrong. So the test asserts the dims are reversed AND the bytes are the
    module's own order, which is what a ggml loader reads as the reference's matrix.
    """
    cfg = TINY()
    model = tiny_model()
    entries = converter.tensor_entries(model, cfg)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "core.gguf"
        converter.write_gguf(path, cfg, metadata_for(cfg), entries)
        tensors = converter.read_tensors(path)

        assert len(tensors) == len(entries)
        for entry in entries:
            t = tensors[entry.name]
            assert t["dims"] == tuple(reversed(entry.shape)), (entry.name, t["dims"], entry.shape)
            assert t["type"] == "F32", entry.name
            assert np.array_equal(t["data"].reshape(-1), entry.data.reshape(-1)), (
                f"{entry.name}: bytes are not the module's element order"
            )
            # ggml's indexing: the first recorded dim varies fastest. Reading the same bytes that way - a
            # Fortran-order reshape of the flat buffer into the file's dims - gives the module's matrix
            # transposed, which is what a ggml matmul expects from a weight. Asserting it alongside the flat
            # byte order is what rules out the transposed-array case: handing the writer `array.T` keeps the
            # element count and changes the order of the elements, so the dims alone cannot tell the two apart.
            if len(entry.shape) == 2:
                ggml_view = t["data"].reshape(-1).reshape(t["dims"], order="F")
                assert np.array_equal(ggml_view, entry.data.T), (entry.name, "ggml indexing")
            assert t["n_bytes"] == entry.data.size * 4, (entry.name, t["n_bytes"])

        # A scalar is written as a one-element tensor: the value is unchanged, only the shape fact.
        scalars = [e for e in entries if e.shape == (1,)]
        assert scalars, "the reference has scalar parameters"
        for entry in scalars:
            assert converter.read_tensors(path)[entry.name]["dims"] == (1,), entry.name
            assert float(converter.read_tensors(path)[entry.name]["data"][0]) == float(entry.data[0]), entry.name


def test_metadata_round_trips_through_the_file_and_nothing_is_silently_dropped():
    """
    The metadata as the file states it, against the metadata as it was written.

    Two format behaviours make this worth testing rather than trusting: the writer cannot store an empty
    array, so an empty merges list is not in the file at all; and floats round-trip at float32, so 0.1 comes
    back as 0.10000000149011612. Both are reported by `write_gguf`, and the test asserts the report matches
    the file, which is the only place a dropped fact becomes visible.
    """
    cfg = TINY()
    entries = converter.tensor_entries(tiny_model(), cfg)
    md = metadata_for(cfg, merges=())

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "core.gguf"
        report = converter.write_gguf(path, cfg, md, entries)
        read_back = converter.read_metadata(path)

        assert report["skipped"] == ["tokenizer.ggml.merges"], report["skipped"]
        assert "tokenizer.ggml.merges" not in read_back, "the file says it wrote something it could not"
        assert set(report["metadata_keys"]) == set(read_back), (
            f"the file holds {sorted(set(read_back) - set(report['metadata_keys']))} more keys than it claims"
        )

        for key, value in read_back.items():
            given = md.get(key)
            if given is None:
                # The file type is the one fact the metadata dict does not state, because it is a fact about
                # what the writer wrote. It is derived from the entries here.
                assert key == "general.file_type", key
                assert value == converter.file_type_for(entries), (value, converter.file_type_for(entries))
                continue
            if isinstance(given, list):
                assert value == [element(x) for x in given], key
            elif isinstance(given, float):
                assert abs(value - given) < 1e-6, (key, value, given)
            else:
                assert value == given, (key, value, given)

        # A fact that can be written is written. The test above compares every key, so this asserts the
        # specific case where an array could have been dropped and was not.
        assert len(read_back["tokenizer.ggml.tokens"]) == cfg.vocab_size
        assert read_back["general.arch.heron.tensor_names"] == metadata_for(cfg)["general.arch.heron.tensor_names"]


def test_a_block_name_mismatch_is_caught():
    """
    The failure the converter exists to prevent, made concrete.

    A tensor in the right shape at the wrong block is the quiet one: it loads, it passes every shape check,
    and it produces different numbers. Here a q projection that belongs to block 4 is given block 3's name, so
    the entry is internally consistent - the shape is what the config says - and only the mapping knows it is
    wrong.
    """
    cfg = TINY()
    model = tiny_model()
    entries = converter.tensor_entries(model, cfg)

    good = [e for e in entries if e.module_path == "blocks.4.attn.q.weight"][0]
    bad = converter.Entry(
        module_path="blocks.4.attn.q.weight",
        name="blk.3.attn_q",
        shape=good.shape,
        quant_type=None,
        data=good.data,
    )
    problems = converter.check_entries([bad], cfg)
    assert problems, "a name that is not the mapping's name must be reported"
    assert "blk.3.attn_q" in " ".join(problems), problems

    # An expert slot's block is the config's, not the slot index. Slot 1 sits at a block the config chooses;
    # naming it as the slot number is the same class of error.
    slot_entry = [e for e in converter.tensor_entries(model, cfg, expert_file_slot=1) if "ffn_gate_exps" in e.name][0]
    slot = int(slot_entry.module_path.split(".")[2])
    expected_block = cfg.expert_blocks()[slot]
    wrong = converter.Entry(
        module_path=slot_entry.module_path,
        name=f"blk.{slot}.ffn_gate_exps",
        shape=slot_entry.shape,
        quant_type=None,
        data=slot_entry.data,
    )
    problems = converter.check_entries([wrong], cfg)
    assert problems and str(expected_block) in " ".join(problems), problems

    # A block index the config does not have, for a path that is otherwise fine.
    out_of_range = converter.Entry(
        module_path=f"blocks.{cfg.n_blocks()}.attn.q.weight",
        name=f"blk.{cfg.n_blocks()}.attn_q",
        shape=good.shape,
        quant_type=None,
        data=good.data,
    )
    problems = converter.check_entries([out_of_range], cfg)
    assert any("outside" in p for p in problems), problems

    # A shape that is not the config's is caught even when the name is right.
    misshaped = converter.Entry(
        module_path="blocks.4.attn.q.weight",
        name=good.name,
        shape=(cfg.d_model, cfg.heads * cfg.head_dim // 2),
        quant_type=None,
        data=np.zeros((cfg.d_model, cfg.heads * cfg.head_dim // 2), dtype=np.float32),
    )
    problems = converter.check_entries([misshaped], cfg)
    assert any("shape" in p for p in problems), problems

    # A name the mapping does not cover, reported rather than raised: a report describes a checkpoint
    # completely, so the check must not stop at the first disagreement.
    unmapped = converter.Entry(
        module_path="blocks.4.attn.q_norm.weight",
        name="blk.4.attn_q_norm",
        shape=(cfg.d_model,),
        quant_type=None,
        data=np.zeros((cfg.d_model,), dtype=np.float32),
    )
    problems = converter.check_entries([bad, unmapped], cfg)
    assert len(problems) >= 2, problems
    assert any("no shape for this pattern" in p for p in problems), problems

    # Writing is refused, not warned: a file with a wrong name is not a smaller model, it is a different model.
    try:
        with tempfile.TemporaryDirectory() as tmp:
            converter.write_gguf(Path(tmp) / "bad.gguf", cfg, metadata_for(cfg), [bad])
    except ValueError as e:
        assert "refuses" in str(e), str(e)
    else:
        raise AssertionError("write_gguf must not write entries that check_entries reports")


def test_scalars_and_sub_block_parameters_cannot_be_quantized():
    """
    The rule, and the reason it is a format rule rather than a preference.

    A ggml block is 32 elements, so a one-element tensor cannot be quantized: the writer raises. That is why
    every scalar in the reference - the per-block reinjection gate, the per-slot output gate, the memory
    gate's bias - is written as F32, and why the harness's `tensor_force_quant` rule exists. The scalar is
    still written as a tensor, because tensor-info needs n_dims >= 1; the value is unchanged, only the shape
    fact.
    """
    cfg = TINY()

    for path in ["blocks.3.reinject_gate", "experts.1.0.gate_scale", "memory.gate_bias"]:
        shape = converter.expected_shapes(cfg)[converter.module_pattern(path)]
        assert shape == (1,), path
        assert converter.quant_type_for(path, shape) is None, path
        try:
            quants.quantize(np.array([0.5], dtype=np.float32), Quant.Q8_0)
        except Exception as e:
            assert "shape (1,)" in str(e), str(e)
        else:
            raise AssertionError("a scalar cannot be quantized; the writer must refuse it")

    # The smallest thing a block scheme can store is one block, so a vector shorter than 32 elements stays
    # F32 even though it is not a scalar.
    assert converter.can_quantize((32,)) is True
    assert converter.can_quantize((31,)) is False
    assert converter.can_quantize((1,)) is False

    # The gate key is a vector rather than a matrix, and is excluded for a semantic reason: the relevance
    # cosine is computed on it before projection, so it is stored at higher precision.
    key_shape = converter.expected_shapes(cfg)["memory.gate_key"]
    assert converter.quant_type_for("memory.gate_key", key_shape) is None
    assert "cosine" in harness_quant.EXCLUSION_REASONS["gate_key"]

    # An entry whose scheme says Q8_0 but whose shape the format cannot store is a problem the check reports,
    # because writing it would raise inside the writer rather than being refused before anything is written.
    entry = converter.Entry(
        module_path="blocks.3.reinject_gate",
        name="blk.3.reinject_gate",
        shape=(1,),
        quant_type=Quant.Q8_0,
        data=np.zeros((1,), dtype=np.uint8),
    )
    problems = converter.check_entries([entry], cfg)
    assert any("cannot be stored" in p for p in problems), problems


def test_quant_rule_matches_the_harness_and_the_embedding_is_never_q4():
    """
    The mirror, checked rather than assumed.

    `heronlm` is the reference and must not depend on the training-side harness, so the converter mirrors the
    rule instead of importing it - which is exactly the kind of copy that drifts. This test is what keeps the
    two equal, for every parameter path at TINY.

    The rule that has to survive the copy is the tied embedding's: it is the output head, so it is q8_0 and
    never Q4. A converter that quantized the head to Q4 would make the head cheaper than the dense weights it
    is supposed to be comparable to, and the number that gets measured would be the head's error.
    """
    cfg = TINY()
    model = tiny_model()

    for path, _ in model.named_parameters():
        mine = converter.scheme_for_path(path)
        theirs = harness_quant.scheme_for(path)
        assert mine == (theirs.name if theirs is not None else None), (path, mine, theirs)

    # The two copies of the exclusions, with their reasons, are the same facts.
    assert converter.EXCLUSION_REASONS == harness_quant.EXCLUSION_REASONS

    # q8_0 never becomes Q4 for the head, whatever the dense scheme is called.
    assert converter.scheme_for_path("embed.weight") == "q8_0"
    assert converter.quant_type_for("embed.weight", converter.expected_shapes(cfg)["embed.weight"]) == Quant.Q8_0
    # A dense scheme cheaper than the head's is the only case where the rule has teeth: the head stays q8_0.
    assert converter.quant_type_for("embed.weight", converter.expected_shapes(cfg)["embed.weight"], dense="q8_0") == Quant.Q8_0

    # The table's scheme has no ggml type, because the table is not a tensor. The scheme is still in the table
    # so that the scheme a parameter gets and the type it would be written as are one table.
    assert converter.scheme_for_path("memory.table.weight") == harness_quant.TABLE_INT8.name
    assert converter.SCHEMES["int8_per_row"] is None
    assert converter.quant_type_for("memory.table.weight", converter.expected_shapes(cfg)["memory.table.weight"]) is None

    # The block runs along the module's last dimension, which is the same axis `phase0.quant` blocks along:
    # ggml's fastest dimension is a weight's input dimension, and `DENSE_INT4.axis` is 1 on a [out, in] weight.
    # A dimension that is not an exact multiple of the block cannot be stored, which is the roadmap's open
    # question - and it is a question about which axis the width sits on, not about the number alone. At the
    # roadmap's depth, an expert's `ffn_gate` is fine because its second dimension is d_model (2048), while
    # its `ffn_down` is not, because its second dimension is the width (20345). Asserting both halves is what
    # stops "20345 is not a multiple of 32" from being read as "expert experts cannot be quantized".
    full = FULL()
    down = converter.expected_shapes(full)["experts.{eid}.{slot}.ffn.down.weight"]
    gate = converter.expected_shapes(full)["experts.{eid}.{slot}.ffn.gate.weight"]
    assert down[-1] == full.expert_ffn_width, down
    assert down[-1] % converter.MIN_QUANT_BLOCK != 0, down
    assert gate[-1] % converter.MIN_QUANT_BLOCK == 0, gate
    assert not converter.can_quantize(down), down
    assert converter.can_quantize(gate), gate
    assert converter.quant_type_for("experts.1.0.ffn.down.weight", down) is None, down
    assert converter.quant_type_for("experts.1.0.ffn.gate.weight", gate) == Quant.Q4_0, gate
    try:
        quants.quantize(np.zeros(down, dtype=np.float32), Quant.Q4_0)
    except Exception as e:
        # The library refuses, and the message names the shape rather than the rule: `Can't quantize tensor
        # with shape (2048, 20345) to Q4_0`. Checking the number is what keeps the failure attributable to the
        # width rather than to some other property of the tensor.
        assert "20345" in str(e), str(e)
    else:
        raise AssertionError("a row that is not a multiple of the block cannot be quantized")

    # The same rule, from the harness's side: `_blocks` raises for exactly the shapes `can_quantize` calls
    # unquantizable. The two are written in different modules for the reason recorded above, so the agreement
    # has to be asserted rather than assumed.
    dense = harness_quant.DENSE_INT4
    for pattern, shape in converter.expected_shapes(full).items():
        if harness_quant.scheme_for(pattern) is None or len(shape) < 2:
            continue
        try:
            harness_quant._blocks(torch.zeros(shape), dense)
            harness_ok = True
        except ValueError:
            harness_ok = False
        assert converter.can_quantize(shape) == harness_ok, (pattern, shape, harness_ok)


def test_the_converter_does_not_pre_normalize():
    """
    The parity rule, asserted on the bytes.

    nGPT normalization is applied at forward time, and the rule says it happens after dequantization. A
    converter that normalized a weight before quantizing it would store scales computed from magnitudes that
    normalization removes, so the file would describe weights that reproduce a different model - and the
    difference would show up as a loss change, not as an error.

    So the file's rows must not be unit-norm, and they must be the module's parameters, element for element.
    """
    cfg = TINY()
    model = tiny_model()
    params = params_of(model)
    md = metadata_for(cfg)
    assert md["general.arch.heron.weight_norm_order"] == harness_quant.PARITY_RULE
    assert "general.arch.heron.ngpt_weight_scale" in md

    for entry in converter.tensor_entries(model, cfg):
        path = entry.module_path
        ref = reference_bytes(params, path)
        if "[" in path:
            # A row of the shared tensor: the whole tensor is the parameter, and the row is its row.
            assert np.array_equal(entry.data, ref), entry.name
            continue
        if path.endswith(".weight") and len(entry.shape) == 2:
            row_norms = np.linalg.norm(entry.data, axis=1)
            assert not np.allclose(row_norms, 1.0), f"{entry.name}: rows are unit-norm, so something normalized first"
            assert np.allclose(row_norms, np.linalg.norm(ref, axis=1)), entry.name
            assert np.array_equal(entry.data, ref), entry.name

    # Quantization is where normalization order becomes visible, so it is checked after quantization: the
    # dequantized rows are not unit-norm either, and re-rounding against the stored scale reproduces the codes.
    quantized = converter.tensor_entries(model, cfg, quant=converter.quant_type_for)
    for entry in quantized:
        if entry.quant_type is None:
            continue
        values = dequantized(entry)
        assert not np.allclose(np.linalg.norm(values, axis=1), 1.0), (
            f"{entry.name}: dequantized rows are unit-norm, so the converter normalized before quantizing"
        )

    # A scalar stays a scalar, only with a shape fact added.
    gate = [e for e in converter.tensor_entries(model, cfg) if e.module_path == "blocks.3.reinject_gate"][0]
    assert gate.shape == (1,) and float(gate.data[0]) == float(params["blocks.3.reinject_gate"]), gate


def test_per_expert_file_contains_only_its_slots_and_records_which_slot_it_is():
    """
    One file per expert, and the file has to say which expert it is.

    An expert is a set of slots spread across blocks, so hot-swapping loads the set as a unit. A file that
    carried the core tensors as well could not be loaded as a unit, and a file that did not say which slot it
    was could not be resolved to a block. Slot 0 has no file at all, because the generalist is the block's base
    FFN - so the test asserts the refusal as well as the contents.

    The router does not travel with the file. It decides which expert to load, so it belongs to the core file;
    a per-expert file with its own copy of the prototypes would make two files state two different routers.
    """
    cfg = TINY()
    model = tiny_model()

    for slot in range(1, cfg.n_experts):
        entries = converter.tensor_entries(model, cfg, expert_file_slot=slot)
        assert entries, f"slot {slot} produced no tensors"
        for e in entries:
            expert = int(e.module_path.split(".")[1])
            assert expert == slot, f"{e.module_path} is not slot {slot}"
            block = int(e.name.split(".")[1])
            assert block in cfg.expert_blocks(), (e.name, cfg.expert_blocks())
            assert block == cfg.expert_blocks()[int(e.module_path.split(".")[2])], e.name

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"expert_{slot}.gguf"
            report = converter.write_gguf(path, cfg, metadata_for(cfg), entries, expert_file_slot=slot)
            read_back = converter.read_metadata(path)
            assert read_back["general.arch.heron.expert_file_slot"] == slot, read_back
            assert report["metadata_keys"]["general.arch.heron.expert_file_slot"] == "INT32"
            names = set(converter.read_tensors(path))
            assert names == {e.name for e in entries}, names
            # Nothing from the core file, and no router.
            assert not any("attn" in n for n in names), names
            assert "router_prototypes" not in names, "the router belongs to the file that decides routing"

    # Slot 0 is refused before anything is collected, so the failure is about the file rather than about a
    # path that happens to exist in a checkpoint.
    try:
        converter.tensor_entries(model, cfg, expert_file_slot=cfg.generalist_slot)
    except ValueError as e:
        assert "has no file" in str(e), str(e)
    else:
        raise AssertionError(f"slot {cfg.generalist_slot} must not have a file")

    # The core file has no expert tensors, so a loader can tell the two kinds of file apart by contents as
    # well as by the recorded slot.
    core = converter.tensor_entries(model, cfg)
    assert not any(e.module_path.startswith("experts.") for e in core)
    assert "router_prototypes" in {e.name for e in core}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "core.gguf"
        report = converter.write_gguf(path, cfg, metadata_for(cfg), core)
        read_back = converter.read_metadata(path)
        assert "general.arch.heron.expert_file_slot" not in read_back, read_back
        assert "general.arch.heron.expert_file_slot" not in report["metadata_keys"]
        assert not any("exps" in n for n in set(converter.read_tensors(path)))


def test_the_table_blob_is_codes_then_scales_and_is_not_a_tensor():
    """
    The blob, byte for byte against the harness's authority.

    A row is code x per-row scale, so the code dtype alone does not describe a row - that is why the metadata
    records both, and why the blob has an order. The test asserts the bytes the converter produces agree with
    `phase0.quant.quantize`, and that no tensor exists for the table in the file: there is no ggml probe op, so
    the table is an mmap'd host-side blob rather than a tensor.
    """
    cfg = TINY()
    model = tiny_model()
    table = model.memory.table.weight

    codes, scales = harness_quant.quantize(table, harness_quant.TABLE_INT8)
    blob = converter.write_table_blob(table, cfg)

    n_rows, width = cfg.ngram_memory_entries, cfg.ngram_memory_dim
    assert len(blob) == n_rows * width + n_rows * 2, len(blob)
    file_codes = np.frombuffer(blob[: n_rows * width], dtype=np.int8).reshape(n_rows, width)
    file_scales = np.frombuffer(blob[n_rows * width:], dtype=np.float16).reshape(n_rows)
    harness_codes, harness_scales = harness_row(codes, scales, n_rows, width)
    assert np.array_equal(file_codes, harness_codes), "codes disagree with the harness"
    assert np.array_equal(file_scales, harness_scales), "scales disagree with the harness"

    # The mirror of the code range, checked as a constant rather than as a divisor. This is the fact the file
    # records for a row: the codes are int8, the scale is float16, and the divisor is the largest magnitude in
    # the range. The test above compares bytes; this asserts the rule those bytes came from, which is what stops
    # a byte agreement from being a coincidence.
    assert converter.TABLE_CODE_RANGE == (harness_quant.TABLE_INT8.code_min, harness_quant.TABLE_INT8.code_max), (
        converter.TABLE_CODE_RANGE,
        harness_quant.TABLE_INT8.code_min,
        harness_quant.TABLE_INT8.code_max,
    )
    qmax = max(abs(converter.TABLE_CODE_RANGE[0]), abs(converter.TABLE_CODE_RANGE[1]))
    assert qmax == 128, qmax

    # What the asymmetric range costs, on a row that is controlled rather than random: the positive peak rounds
    # to 128 and is clamped to 127, so the stored row reproduces 127/128 of its own peak, while the negative
    # peak keeps a full code. That is the effective-bits question in the ROADMAP, and it is only visible because
    # the converter produces the harness's codes rather than a symmetric version of them.
    single = np.zeros((n_rows, width), dtype=np.float32)
    single[0, 0] = 1.0
    single[0, -1] = -1.0
    torch_single = torch.from_numpy(single)
    one_codes, one_scales = harness_row(*harness_quant.quantize(torch_single, harness_quant.TABLE_INT8), n_rows, width)
    row = converter.write_table_blob(torch_single, cfg)
    row_codes = np.frombuffer(row[: n_rows * width], dtype=np.int8).reshape(n_rows, width)
    row_scales = np.frombuffer(row[n_rows * width:], dtype=np.float16).reshape(n_rows)
    assert np.array_equal(row_codes, one_codes), "the controlled row disagrees with the harness"
    assert np.array_equal(row_scales, one_scales), "the controlled row's scale disagrees with the harness"
    assert row_codes[0, 0] == 127 and row_codes[0, -1] == -128, (row_codes[0, 0], row_codes[0, -1])
    assert abs(float(row_scales[0]) - 1.0 / 128) < 1e-6, float(row_scales[0])
    # Dequantized: the positive peak is one code short of what it was, the negative peak is exact.
    assert abs(float(row_codes[0, 0]) * float(row_scales[0]) - 127.0 / 128) < 1e-6, "the positive peak is one step short"
    assert abs(float(row_codes[0, -1]) * float(row_scales[0]) + 1.0) < 1e-6, "the negative peak is exact"

    # The order is what the metadata records, and it is what keeps a probe one contiguous read. A blob read in
    # the other order would return the scales first, which is the kind of disagreement that shows up as wrong
    # recalls rather than as an error.
    md = converter.blob_metadata(cfg)
    assert md["general.arch.heron.table.blob_order"] == ["codes", "scales"], md
    assert md["general.arch.heron.table.dtype"] == "int8"
    assert md["general.arch.heron.table.scale_dtype"] == "float16"
    assert TENSOR_NAMES["memory.table.weight"] is None

    # The stored scale is the file's fact, not a recomputed one: re-rounding the dequantized row against the
    # stored scale reproduces these codes, while re-quantizing from scratch does not.
    assert harness_quant.requantize_is_stable(table, harness_quant.TABLE_INT8)

    entries = converter.tensor_entries(model, cfg)
    assert not any(e.module_path == "memory.table.weight" for e in entries), "the table must not be a tensor entry"
    assert any(e.module_path == "memory.value_proj.weight" for e in entries), "the projection is a tensor"

    # An all-zero table stores scale 0.0 for every row, which is what an initialized table is and what ggml
    # does. The row count has to be the config's, because the blob is the whole table and the metadata states
    # that count as `table.entries`: a shorter blob is a different table, not a smaller view of the same one.
    # The refusal is asserted too, so the shape check cannot be read as a convenience.
    zero = torch.zeros(n_rows, width, dtype=torch.float32)
    zero_blob = converter.write_table_blob(zero, cfg)
    assert np.all(np.frombuffer(zero_blob[: n_rows * width], dtype=np.int8) == 0)
    assert np.all(np.frombuffer(zero_blob[n_rows * width:], dtype=np.float16) == 0.0)
    try:
        converter.write_table_blob(torch.zeros(64, width, dtype=torch.float32), cfg)
    except ValueError as e:
        assert "is not the config's" in str(e), str(e)
    else:
        raise AssertionError("a blob with a different number of rows is a different table")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "core.gguf"
        report = converter.write_gguf(path, cfg, metadata_for(cfg), entries)
        names = set(converter.read_tensors(path))
        assert "memory.table" not in " ".join(names), names
        assert report["tensors"] == names
        # The blob keys are in the file; the blob itself is not, because it is not a tensor.
        read_back = converter.read_metadata(path)
        for key in (
            "general.arch.heron.table.entries",
            "general.arch.heron.table.value_dim",
            "general.arch.heron.table.dtype",
            "general.arch.heron.table.scale_dtype",
            "general.arch.heron.table.blob_order",
            "general.arch.heron.table.storage_path",
        ):
            assert key in read_back, key
        assert read_back["general.arch.heron.table.blob_order"] == ["codes", "scales"]


def test_token_type_codes_still_equal_the_library_they_are_mirrored_from():
    """
    The codes live in the reference as numbers so that metadata does not depend on the gguf package, and this
    is the check that keeps the numbers from drifting away from the library they came from.

    A control token is what a chat turn is made of. If the codes and the vocabulary disagreed, a reader would
    decide a turn boundary was ordinary text, and the ids the n-gram table was built with would not be the ids
    the tokenizer produces.
    """
    assert int(TokenType.NORMAL) == TOKEN_TYPE_NORMAL, (int(TokenType.NORMAL), TOKEN_TYPE_NORMAL)
    assert int(TokenType.CONTROL) == TOKEN_TYPE_CONTROL, (int(TokenType.CONTROL), TOKEN_TYPE_CONTROL)

    cfg = TINY()
    tokens = placeholder_vocabulary(cfg.vocab_size)
    types = token_types(tokens)
    special = set(special_token_ids(tokens).values())
    assert len(types) == len(tokens)
    assert [i for i, t in enumerate(types) if t == TOKEN_TYPE_CONTROL] == sorted(special)
    assert any(t == TOKEN_TYPE_CONTROL for t in types), "the chat tags must be control tokens"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "core.gguf"
        entries = converter.tensor_entries(tiny_model(), cfg)
        converter.write_gguf(path, cfg, metadata_for(cfg), entries)
        read_back = converter.read_metadata(path)
        assert read_back["tokenizer.ggml.token_type"] == types, "the file disagrees about which tokens are tags"
        # The added tokens as the file writes them: in vocabulary-id order, because `token_type` next to it is
        # ordered by id and a reader pairs the two arrays position by position. For this fixture that order is the
        # same as the lookup's iteration order, so the case where the two differ has to be asserted on a
        # vocabulary that is not a fixture - see the pinned-tokenizer test.
        ids = special_token_ids(tokens)
        assert read_back["tokenizer.ggml.added_tokens"] == [SPECIAL_TAGS[n] for n in sorted(ids, key=ids.get)], (
            read_back["tokenizer.ggml.added_tokens"])
        # The ids in the file are the ids resolved from this vocabulary, not remembered numbers.
        for key, name in (("tokenizer.ggml.eos_token_id", "im_end"), ("tokenizer.ggml.bos_token_id", "endoftext")):
            assert read_back[key] == special_token_ids(tokens)[name], (key, read_back[key])
        # The byte-fallback value the file holds is the one the placeholder vocabulary can back, which is not the
        # one an earlier version of metadata() asserted. See `test_byte_fallback_is_recorded_from_what_can_back_it`.
        assert read_back["tokenizer.ggml.byte_fallback"] is False
        assert read_back["tokenizer.ggml.byte_fallback"] == byte_level_rule(tokens)[0]


def test_byte_fallback_is_recorded_from_what_can_back_it_and_the_cli_says_which_did():
    """
    The tokenizer key that the reference cannot decide by itself.

    `tokenizer.ggml.byte_fallback` is a rule a reader needs for a byte sequence the vocabulary does not have.
    gguf has no key constant for it and no writer helper for it, which is what makes it HeronLM's own claim -
    and a claim the reference writes has to be a claim something can back. There are two things that can back
    it, and they are not the same thing:

      - the vocabulary's shape: 256 distinct single-character symbols at ids 0..255, the positions a byte-level
        reader reads bytes from. Checkable from the vocabulary alone, and enough to say a vocabulary CANNOT have
        the rule when its first 256 entries are fake strings.
      - the tokenizer file's own claim, which is the only authority on whether the tokenizer uses those
        positions for bytes it cannot find. That is a decoder fact, not a symbol-set fact, so the reference
        cannot read it out of the vocabulary.

    The earlier version of this code asserted True as a constant and refused any tokenizer file that said
    otherwise. That is the shape of the bug this test is written against: a reference that enforces its own
    assumption instead of reporting the tokenizer's would have refused the pinned Qwen file, which states
    `model.byte_fallback: false` while carrying a byte-shaped base alphabet - its byte-level behaviour comes
    from its ByteLevel decoder, so the fallback key is not what makes it byte-level.

    The byte-shaped vocabulary below is built as one character per byte position rather than from a named byte
    table, because a table the reference cannot check is not a fact it can assert on.
    """
    cfg = TINY()
    filler = placeholder_vocabulary(cfg.vocab_size)
    byte_shaped = [chr(i) for i in range(BASE_ALPHABET_SIZE)] + filler[BASE_ALPHABET_SIZE:]

    # The key is the reference's own, not the library's: nothing outside this project can tell it what to write.
    tokenizer_keys = [v for v in vars(Keys.Tokenizer).values() if isinstance(v, str)]
    assert not any("byte_fallback" in v for v in tokenizer_keys), sorted(tokenizer_keys)
    assert not hasattr(GGUFWriter, "add_byte_fallback"), "the writer has no helper for this key"

    # The vocabulary is the authority when it can answer: a vocabulary whose first 256 ids are not one symbol
    # per byte has no byte rule to claim, and the value written is False for that reason.
    assert base_alphabet(filler) is None
    value, backed = byte_level_rule(filler)
    assert value is False and "base alphabet" in backed, (value, backed)
    assert metadata(cfg, filler, ["merge a b"])["tokenizer.ggml.byte_fallback"] is False

    # A byte-shaped vocabulary can hold the rule but cannot say the tokenizer uses it, so the reference has to
    # be told, and refusing is the honest answer when it is not.
    assert base_alphabet(byte_shaped) is not None
    try:
        byte_level_rule(byte_shaped)
    except ValueError as e:
        assert "Supply the file's claim" in str(e), e
    else:
        raise AssertionError("a byte-shaped vocabulary with no claim cannot be answered from the vocabulary")

    # The two authorities, kept separate so a divergence is visible instead of resolved by precedence.
    assert byte_level_rule(byte_shaped, claim=False) == (False, "the tokenizer file's claim")
    assert byte_level_rule(byte_shaped, claim=True) == (True, "the tokenizer file's claim")
    assert metadata(cfg, byte_shaped, ["merge a b"], claim=False)["tokenizer.ggml.byte_fallback"] is False

    # A claim the vocabulary cannot back is refused rather than written: this is the case where the reference
    # would be telling a reader there is a rule for bytes this vocabulary has no symbols for.
    try:
        byte_level_rule(filler, claim=True)
    except ValueError as e:
        assert "cannot be written" in str(e), e
    else:
        raise AssertionError("a byte-fallback claim a vocabulary cannot back must be refused")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # The CLI is where the two are reconciled, and the report has to say which one backed the value it wrote.
        report = run_cli([
            "--scale", "tiny", "--tokenizer", str(cli_tokenizer_file(tmp / "derived.json", cfg)),
            "--out-dir", str(tmp / "derived"), "--quant", "F32", "--allow-placeholder-prototypes",
        ])
        assert "byte_fallback: False <- the vocabulary" in report, report
        assert converter.read_metadata(tmp / "derived" / "heron_core.gguf")["tokenizer.ggml.byte_fallback"] is False

        report = run_cli([
            "--scale", "tiny",
            "--tokenizer", str(cli_tokenizer_file(tmp / "claimed.json", cfg, claim=True, tokens=byte_shaped)),
            "--out-dir", str(tmp / "claimed"), "--quant", "F32", "--allow-placeholder-prototypes",
        ])
        assert "byte_fallback: True <- the tokenizer file's claim" in report, report
        assert converter.read_metadata(tmp / "claimed" / "heron_core.gguf")["tokenizer.ggml.byte_fallback"] is True

        # The pinned case, reported as a divergence rather than as a contradiction: the file says no fallback,
        # the vocabulary is byte-shaped, and the decoder is what makes the tokenizer byte-level.
        report = run_cli([
            "--scale", "tiny",
            "--tokenizer", str(cli_tokenizer_file(tmp / "pinned_like.json", cfg, claim=False, tokens=byte_shaped)),
            "--out-dir", str(tmp / "pinned_like"), "--quant", "F32", "--allow-placeholder-prototypes",
        ])
        assert "byte_fallback: False <- the tokenizer file's claim" in report, report
        assert "the vocabulary's base alphabet is byte-shaped (True), the file claims False" in report, report

        # Both refusals happen before anything is written.
        cli_refuses([
            "--scale", "tiny",
            "--tokenizer", str(cli_tokenizer_file(tmp / "no_claim.json", cfg, tokens=byte_shaped)),
            "--out-dir", str(tmp / "refused_no_claim"), "--quant", "F32", "--allow-placeholder-prototypes",
        ], "Supply the file's claim", tmp / "refused_no_claim")
        cli_refuses([
            "--scale", "tiny",
            "--tokenizer", str(cli_tokenizer_file(tmp / "unbacked.json", cfg, claim=True)),
            "--out-dir", str(tmp / "refused_unbacked"), "--quant", "F32", "--allow-placeholder-prototypes",
        ], "cannot be written", tmp / "refused_unbacked")


def test_file_type_is_derived_from_what_is_written_and_not_claimed_by_metadata():
    """
    The one fact about the file that metadata() does not state.

    The file type is a fact about what the writer produced, not about the architecture, and the GGUF version is
    a header field with no metadata key at all. A metadata dict that claimed MOSTLY_F16 while writing F32
    tensors would describe a file it had not written, and that is the kind of claim that drifts. So the file
    type here is computed from the entries, and the test asserts the file agrees with the entries rather than
    with a statement.
    """
    cfg = TINY()
    model = tiny_model()
    md = metadata_for(cfg)
    assert "general.file_type" not in md and "general.ggversion" not in md, sorted(md)

    # An all-F32 file: every tensor is the reference's own bytes, which is the file the parity test is against.
    entries = converter.tensor_entries(model, cfg)
    assert converter.file_type_for(entries) == int(LlamaFileType.ALL_F32)
    assert all(e.quant_type is None for e in entries)

    quantized = converter.tensor_entries(model, cfg, quant=converter.quant_type_for)
    counts = {}
    for e in quantized:
        if e.quant_type is not None:
            counts[e.quant_type] = counts.get(e.quant_type, 0) + 1
    assert Quant.Q8_0 in counts, "the tied embedding is q8_0 by rule"
    assert Quant.Q4_0 in counts, "the dense weights get the dense scheme"
    # The scalars and sub-block parameters are still F32, so the file is mostly one type rather than purely it.
    assert any(e.quant_type is None for e in quantized), "scalars must stay F32"
    # The file is named by the scheme its bulk is stored under; a handful of F32 scalars do not make a file
    # mostly F32, because the format has no name for that - only ALL_F32, which describes an entirely-F32 file.
    dominant = max(counts, key=lambda t: counts[t])
    assert converter.file_type_for(quantized) == int(LlamaFileType[f"MOSTLY_{dominant.name}"]), (
        converter.file_type_for(quantized),
        dominant,
    )

    # A scheme the format has no name for is recorded as GUESSED rather than as a type the file has no name for.
    # No scheme this project uses is in that position, so this asserts the fallback's honesty, not a likely path.
    odd = [e for e in quantized if e.quant_type is not None][:1]
    odd = [converter.Entry(e.module_path, e.name, e.shape, Quant.Q8_1, e.data) for e in odd]
    assert any(m.name == "MOSTLY_Q4_0" for m in LlamaFileType) and not any(
        m.name == "MOSTLY_Q8_1" for m in LlamaFileType
    ), "the fallback is only reachable for a type with no name"
    assert converter.file_type_for(odd) == int(LlamaFileType.GUESSED), converter.file_type_for(odd)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "q.gguf"
        report = converter.write_gguf(path, cfg, md, quantized)
        read_back = converter.read_metadata(path)
        assert read_back["general.file_type"] == report["file_type"], (
            read_back["general.file_type"],
            report["file_type"],
        )
        assert read_back["general.file_type"] == converter.file_type_for(quantized)
        tensors = converter.read_tensors(path)
        for entry in quantized:
            # The entries that cannot be quantized are still in the file, as F32. Asserting only the quantized
            # ones would let a writer quietly convert an unquantizable entry into something else and still pass.
            if entry.quant_type is None:
                assert tensors[entry.name]["type"] == "F32", (entry.name, tensors[entry.name]["type"])
                assert tensors[entry.name]["n_bytes"] == entry.data.size * 4, (
                    entry.name,
                    tensors[entry.name]["n_bytes"],
                )
                continue
            assert tensors[entry.name]["type"] == entry.quant_type.name, entry.name
            assert tensors[entry.name]["n_bytes"] == quant_bytes(entry.shape, entry.quant_type), (
                entry.name,
                tensors[entry.name]["n_bytes"],
            )

        # A quantized file is not bit-exact with the reference, and the amount of error is attributed to the
        # scheme rather than hidden behind a tolerance. The head's error is smaller than the dense weights',
        # which is the same fact the scheme rule states in the other direction.
        by_type = {}
        for entry in quantized:
            if entry.quant_type is None:
                continue
            ref = reference_bytes(params_of(model), entry.module_path)
            error = float(np.max(np.abs(dequantized(entry) - ref)) / np.max(np.abs(ref)))
            by_type[entry.quant_type.name] = max(by_type.get(entry.quant_type.name, 0.0), error)
        assert 0.0 < by_type["Q8_0"] < 0.01, by_type
        assert 0.0 < by_type["Q4_0"] < 0.2, by_type
        assert by_type["Q8_0"] < by_type["Q4_0"], by_type

        # An F32 file is bit-exact with the reference, element for element. That is the property the parity test
        # needs, and it is the reason quantization is the caller's choice rather than this function's.
        f32 = converter.tensor_entries(model, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exact.gguf"
            converter.write_gguf(path, cfg, metadata_for(cfg), f32)
            read_back = converter.read_tensors(path)
            params = params_of(model)
            for entry in f32:
                ref = reference_bytes(params, entry.module_path)
                # Bit-exact means the bytes, not the shape fact: the file records dims reversed, so the same
                # bytes are the same tensor seen from ggml's side.
                assert np.array_equal(read_back[entry.name]["data"].reshape(-1), ref.reshape(-1)), entry.name


def cli_tokenizer_file(path: Path, cfg, claim: bool | None = None, tokens: list[str] | None = None) -> Path:
    """
    A tokenizer.json at the test's vocabulary, built from the same placeholder vocabulary metadata resolves.

    The CLI reads the vocabulary and the merges as JSON fields rather than through the `tokenizers` package, so
    the fixture is the JSON itself: the type, the vocab, the merges, the byte-fallback flag and the added tokens.
    `placeholder_vocabulary` is used rather than a hand-written list so that the special tokens sit where the
    name lookup says they sit - a fixture that hardcoded them would test the fixture, not the lookup.

    `claim` is the fixture's `model.byte_fallback`, and it is optional on purpose. The placeholder vocabulary is
    fake strings, so it is not byte-shaped, and the value metadata can write for it is the vocabulary's answer
    rather than a claim; passing True here is how a test exercises a claim the vocabulary cannot back. `tokens`
    is the fixture's vocabulary, so a test can hand the CLI a byte-shaped one and see what the reader claims for
    it - which is the other half of the rule, since a byte-shaped vocabulary can hold a byte rule without saying
    that the tokenizer uses it.
    """
    tokens = tokens or placeholder_vocabulary(cfg.vocab_size)
    model = {
        "type": "BPE",
        "vocab": {token: i for i, token in enumerate(tokens)},
        "merges": ["a b", "c d"],
    }
    if claim is not None:
        model["byte_fallback"] = claim
    fixture = {
        "model": model,
        "added_tokens": [{"id": i, "content": token} for i, token in enumerate(tokens)
                          if token in SPECIAL_TAGS.values()],
    }
    path.write_text(json.dumps(fixture), encoding="utf-8")
    return path


def run_cli(argv: list[str]) -> str:
    """Run the CLI with its report captured, so the report itself can be asserted on."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
        assert convert_to_gguf.main(argv) == 0
    return buffer.getvalue()


def cli_refuses(argv: list[str], expect: str, out_dir: Path) -> str:
    """
    Run the CLI expecting it to refuse, and return the words it refused with.

    The message is asserted rather than only the refusal, because a closed choice list or a vocabulary limit is
    only useful to a caller who is told which rule was broken - argparse's usage block names the accepted scheme
    names, which is the same closed set the test is checking, so checking the message checks the CLI's own claim
    instead of a constant written here.

    The empty output directory is asserted as part of the refusal: a CLI that wrote a file and then complained is
    a CLI that wrote a file. Two states are accepted because the checks sit at different depths. A bad choice is
    refused by argparse, so the directory is never created at all - the strongest form of "before anything is
    written". A tokenizer or a checkpoint problem is refused inside main, after `--out-dir` is created, so the
    directory may exist but must hold nothing.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()) as errors:
        try:
            code = convert_to_gguf.main(argv)
        except SystemExit as e:
            # argparse refuses by exiting. SystemExit is not an Exception, so it is caught as itself rather than
            # by the runner's generic handler - otherwise the whole test process would end on the first refusal.
            assert int(e.code) != 0, (argv, e.code)
            message = errors.getvalue()
        except ValueError as e:
            message = str(e)
        else:
            raise AssertionError(f"the CLI must refuse {argv}, expected {expect!r}; it exited {code}")

    assert expect in message, message
    written = sorted(out_dir.iterdir()) if out_dir.exists() else []
    assert not written, f"a refusal must happen before anything is written: {written}"
    return message


def test_the_cli_writes_the_files_the_metadata_names_and_no_others():
    """
    The names on disk are the names in the metadata.

    This is the only test that connects the metadata's naming keys to the files, which is why the CLI takes its
    output names from metadata rather than building them: `core_file`, `expert_files` and `table.storage_path`
    are the set of artifacts a loader has to open, and if the CLI named the core file for itself, the name would
    exist in one place only and nothing could check that the files written are the files claimed. Slot 0 is part
    of that set and has no file, so the test asserts the recorded name is not on disk.

    Every tensor the entries say for each file is checked against the file, and the blob is checked against the
    bytes the converter produces, so the test is about what landed on disk rather than about what the CLI was
    handed.
    """
    cfg = TINY()
    model = tiny_model()
    md = metadata_for(cfg)
    recorded = {
        "core": md["general.arch.heron.core_file"],
        "experts": md["general.arch.heron.expert_files"],
        "blob": md["general.arch.heron.table.storage_path"],
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tokenizer = cli_tokenizer_file(tmp / "tokenizer.json", cfg)
        out = tmp / "out"
        report = run_cli([
            "--scale", "tiny", "--tokenizer", str(tokenizer), "--out-dir", str(out),
            "--quant", "F32", "--allow-placeholder-prototypes",
        ])

        files = sorted(p.name for p in out.iterdir())
        expected = sorted([recorded["core"], recorded["blob"]] + [recorded["experts"][i] for i in range(1, cfg.n_experts)])
        assert files == expected, (files, expected)

        # Each file states the name it was written under, so the file and its own metadata agree.
        for name in [recorded["core"]] + [recorded["experts"][i] for i in range(1, cfg.n_experts)]:
            read_back = converter.read_metadata(out / name)
            slot = read_back.get("general.arch.heron.expert_file_slot")
            claimed = recorded["experts"][slot] if slot is not None else recorded["core"]
            assert name == claimed, (name, claimed)

        # What each file holds is what the entries say it holds - the core has the router, a per-expert file has
        # only its own slots.
        core_names = {e.name for e in converter.tensor_entries(model, cfg)}
        assert set(converter.read_tensors(out / recorded["core"])) == core_names
        for slot in range(1, cfg.n_experts):
            slot_names = {e.name for e in converter.tensor_entries(model, cfg, expert_file_slot=slot)}
            assert set(converter.read_tensors(out / recorded["experts"][slot])) == slot_names, slot

        blob = (out / recorded["blob"]).read_bytes()
        assert blob == converter.write_table_blob(model.memory.table.weight, cfg), "the blob on disk is not the blob the converter produces"

        # The report names every file it wrote by the recorded name, and names the one it did not write.
        for name in expected:
            assert name in report, (name, report)
        assert f"{recorded['experts'][cfg.generalist_slot]}  not written" in report, report

        # The vocabulary the file holds is the vocabulary the fixture stated, resolved by the CLI's own reader.
        read_back = converter.read_metadata(out / recorded["core"])
        fixture = json.loads(tokenizer.read_text(encoding="utf-8"))
        assert read_back["tokenizer.ggml.tokens"] == list(fixture["model"]["vocab"].keys())
        assert read_back["tokenizer.ggml.merges"] == fixture["model"]["merges"]


def test_the_cli_refuses_before_writing_when_it_cannot_say_where_the_router_directions_came_from():
    """
    The router asymmetry, enforced at the point where it would otherwise be invisible.

    `router.prototypes` is not a Parameter, so a checkpoint cannot carry it. A CLI that wrote whatever the model
    happened to have would produce a file that looks like a trained router and is a set of random directions -
    and gate P2.1 measures routing, so the number would be measuring the placeholder. So the CLI requires the
    source by name, and the refusal happens before any file is written.

    The three paths are all asserted: refused, with the reason the CLI prints as its own statement of the rule;
    written from a real source; and written as a placeholder only when the report is required to say it is one.
    """
    cfg = TINY()
    model = tiny_model()
    shape = (cfg.n_experts, cfg.d_model)

    # Directions that are not the model's, so "the file holds the source" is an assertion and not a tautology.
    directions = np.eye(shape[0], shape[1], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tokenizer = cli_tokenizer_file(tmp / "tokenizer.json", cfg)

        refused = tmp / "refused"
        cli_refuses(["--scale", "tiny", "--tokenizer", str(tokenizer), "--out-dir", str(refused)],
                    "not in a checkpoint's state_dict", refused)

        np.save(tmp / "prototypes.npy", directions)
        report = run_cli([
            "--scale", "tiny", "--tokenizer", str(tokenizer), "--out-dir", str(tmp / "written"),
            "--prototypes", str(tmp / "prototypes.npy"),
        ])
        assert "read from" in report, report

        read_back = converter.read_tensors(tmp / "written" / metadata_for(cfg)["general.arch.heron.core_file"])
        written = read_back[converter.TENSOR_NAMES["router.prototypes"]]
        name = converter.TENSOR_NAMES["router.prototypes"]
        flat = np.asarray(written["data"]).reshape(-1)
        assert np.allclose(flat, directions.reshape(-1)), "the file does not hold the directions it was given"
        assert not np.allclose(flat, model.router.prototypes.detach().numpy().reshape(-1)), (
            "the source did not replace the model's own directions, so a placeholder would be indistinguishable"
        )
        assert name in {e.name for e in converter.tensor_entries(model, cfg)}

        # The placeholder is allowed only when the report has to say it is one.
        report = run_cli([
            "--scale", "tiny", "--tokenizer", str(tokenizer), "--out-dir", str(tmp / "placeholder"),
            "--allow-placeholder-prototypes",
        ])
        assert "placeholder" in report, report
        assert "not directions that were trained" in report, report

        # A source whose shape is not the router's is refused rather than reshaped into the file.
        np.save(tmp / "wrong.npy", np.zeros((cfg.n_experts + 1, cfg.d_model), dtype=np.float32))
        wrong = tmp / "wrong"
        cli_refuses([
            "--scale", "tiny", "--tokenizer", str(tokenizer), "--out-dir", str(wrong),
            "--prototypes", str(tmp / "wrong.npy"),
        ], str(shape), wrong)


def test_the_cli_accepts_only_the_scheme_names_the_harness_records_and_refuses_bytes_it_cannot_name():
    """
    The closed scheme list, and the params a checkpoint carries that no file can hold.

    `--quant` rejects an invented spelling because a measured quality delta has to be attributable to the scheme
    rather than to the name: Q4_K_M is ROADMAP's phone tier and this converter cannot produce it, so there is no
    name for it to accept. The closed set is asserted from the CLI's own usage text rather than from a copy of
    the list written here, so the test checks the CLI's claim about what it accepts. The vocab limit is checked
    the same way - an added token above the embedding range is rejected rather than written, because every id in
    the file has to be an id the model can embed and the table was built with.

    The reference's checkpoint cannot carry slot 0, so a checkpoint that does is refused rather than written.
    Slot 0 is each expert block's base FFN, already named by blk.{bid}.ffn_*, so the bytes in a slot-0 key have no
    name - writing them would need a name the metadata does not claim. The accepted case and the refused case are
    both asserted, because they are the same rule seen from two sides: the reference's own state dict has nothing
    at slot 0, and a dict that has something there is a checkpoint from a different architecture.
    """
    cfg = TINY()
    model = tiny_model()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tokenizer = cli_tokenizer_file(tmp / "tokenizer.json", cfg)

        # The closed set, checked from the CLI's own usage text: every name it accepts has to appear in the
        # refusal, otherwise the refusal is a rejection of one spelling rather than a statement of the rule.
        for invented in ("Q4_K_M", "q4_0", "int8_per_row"):
            refused = tmp / "q"
            message = cli_refuses([
                "--scale", "tiny", "--tokenizer", str(tokenizer), "--out-dir", str(refused),
                "--allow-placeholder-prototypes", "--quant", invented,
            ], invented, refused)
            assert invented not in convert_to_gguf.QUANT_CHOICES, invented
            for accepted in convert_to_gguf.QUANT_CHOICES:
                assert accepted in message, (accepted, message)

        # A checkpoint recorded at the config being converted is accepted, and the reference's own state dict is
        # the proof that slot 0 can never appear in one: the report's line for slot 0 has to say the module is
        # outside the saved state, not that a file for it was skipped.
        slot0 = f"experts.{cfg.generalist_slot}."
        assert not any(k.startswith(slot0) for k in model.state_dict()), sorted(model.state_dict())
        matching = tmp / "matching.pt"
        torch.save({"format": 1, "run_key": {"cfg": dataclasses.asdict(cfg)},
                    "model": model.state_dict()}, matching)

        report = run_cli(["--scale", "tiny", "--tokenizer", str(tokenizer), "--out-dir", str(tmp / "out"),
                          "--checkpoint", str(matching), "--allow-placeholder-prototypes"])
        line = [ln for ln in report.splitlines() if f"slot {cfg.generalist_slot} is the generalist" in ln]
        assert len(line) == 1, report
        assert "excluded from state_dict()" in line[0], line[0]
        names = set(converter.read_tensors(tmp / "out" / metadata_for(cfg)["general.arch.heron.core_file"]))
        assert not any("exps" in n for n in names), names

        # A checkpoint trained at a different config is refused: the shapes would match at every field a loader
        # allocates from, while the metadata would describe an architecture these bytes were not trained in.
        # The state dict is the same bytes as the accepted one above, so the refusal is about the recorded run
        # key and not about the shapes - which is exactly the failure mode a loader would not notice.
        narrower = dataclasses.replace(cfg, d_model=cfg.d_model + 64)
        mismatched = tmp / "mismatched.pt"
        torch.save({"format": 1, "run_key": {"cfg": dataclasses.asdict(narrower)},
                    "model": model.state_dict()}, mismatched)
        refused = tmp / "out2"
        message = cli_refuses([
            "--scale", "tiny", "--tokenizer", str(tokenizer), "--out-dir", str(refused),
            "--checkpoint", str(mismatched), "--allow-placeholder-prototypes",
        ], "disagrees with --scale", refused)
        assert "d_model" in message, message

        # An added token above the embedding range is rejected, not stored: the metadata writes the vocabulary as
        # an ordered array, so a token the model cannot embed would shift every id after it.
        fixture = json.loads((tmp / "tokenizer.json").read_text(encoding="utf-8"))
        fixture["model"]["vocab"][SPECIAL_TAGS["im_end"]] = cfg.vocab_size
        too_big = tmp / "too_big.json"
        too_big.write_text(json.dumps(fixture), encoding="utf-8")
        refused = tmp / "out3"
        cli_refuses([
            "--scale", "tiny", "--tokenizer", str(too_big), "--out-dir", str(refused),
            "--allow-placeholder-prototypes",
        ], "above the model's vocabulary", refused)


def test_the_cli_reads_the_pinned_tokenizer_file_and_rejects_what_the_embedding_cannot_index():
    """
    The CLI's tokenizer reader against the file that is actually pinned, not against a fixture.

    A pin is only a pin if something checks it, so the hashes are read out of the tokenizer directory's own
    README - the single place the provenance is stated - rather than copied into this test, where a copy could
    drift away from the claim it is meant to be checking.

    The reader is asked for the vocabulary the model can embed: 151,646 entries, the base vocabulary plus the
    three chat tags. The pinned file carries 22 added tokens in total, and 19 of them sit above that limit -
    vision, tool-call, FIM and file separators. Those are rejected rather than stored, because the metadata
    writes the vocabulary as an ordered array and an entry the embedding cannot index would shift every id after
    it while still looking valid.

    This is the test where the rule about resolving special ids by name is checked against a real vocabulary. On
    the placeholder fixture it can only check the fixture, which is why the fixture is built from
    `placeholder_vocabulary` rather than from literals: the numbers asserted here are properties of the pinned
    file, so a test that asserted them and then read them from the file would be asserting the same thing twice.

    It is also the test where the byte-fallback rule meets a real file. The pinned file states
    `model.byte_fallback: false` while its first 256 ids are one distinct single-character symbol each, so the
    vocabulary is byte-shaped and the file claims no byte rule; the value written is the file's claim, and the
    divergence is reported rather than resolved. That is the case an earlier version of the reader refused, which
    is why the three facts are asserted separately here: the claim, what the vocabulary entails, and the value the
    metadata can therefore write.
    """
    cfg = FULL()
    pinned = ROOT / "tokenizer" / "qwen2.5"
    if not (pinned / "tokenizer.json").exists():
        raise AssertionError("the pinned tokenizer is missing; the provenance test has nothing to check")

    readme = (pinned / "README.md").read_text(encoding="utf-8")

    def recorded_hashes(text: str) -> dict[str, str]:
        """The hashes the README states, one per file bullet, as the file names the README gives them."""
        found = {}
        for block in text.split("\n- `")[1:]:
            name = block.split("`", 1)[0]
            match = re.search(r"SHA-256 `([0-9a-f]{64})`", block)
            if match:
                found[name] = match.group(1)
        return found

    hashes = recorded_hashes(readme)
    assert set(hashes) == {"tokenizer.json", "tokenizer_config.instruct.json"}, hashes
    for name, expected in hashes.items():
        actual = hashlib.sha256((pinned / name).read_bytes()).hexdigest()
        assert actual == expected, f"{name} is not the file the README pins: {actual} != {expected}"

    raw = json.loads((pinned / "tokenizer.json").read_text(encoding="utf-8"))
    tokens, merges, facts = convert_to_gguf.read_tokenizer(pinned / "tokenizer.json", cfg)

    # The counts the reader reports are the counts the file has, and the arithmetic the README states: 151,643
    # learned entries plus 22 added ones is 151,665, of which only the 3 chat tags are inside the embedding range.
    assert len(raw["model"]["vocab"]) == 151643, len(raw["model"]["vocab"])
    assert len(raw["added_tokens"]) == 22, len(raw["added_tokens"])
    assert 151643 + 22 == 151665, 151643 + 22
    assert facts["tokens"] == cfg.vocab_size == 151646, facts
    assert facts["merges"] == 151387, facts

    # Byte fallback, as three separate facts rather than as one value: the file's claim, what the vocabulary can
    # show on its own, and the value the metadata is allowed to write. The pinned file claims false, its base
    # alphabet is byte-shaped, and the value written is the claim - so the reference is reporting the tokenizer
    # rather than enforcing what it assumed about it. `decoder.type` is recorded because it is the fact the
    # vocabulary cannot show: ByteLevel is what makes this tokenizer byte-level, not the fallback key.
    assert facts["claim"] is False, facts
    assert facts["byte_level_entailed"] is True, facts
    assert facts["byte_fallback"] is False, facts
    assert facts["backed_by"] == "the tokenizer file's claim", facts
    assert facts["decoder_type"] == "ByteLevel", facts

    # The 256 single-character symbols the file carries are not the standard byte table: 221 of them are shared
    # with it and 35 are extra. That is the arithmetic reason `base_alphabet` says "byte-shaped" and nothing more -
    # the shape is checkable, the mapping to byte values is not, so the reference does not claim it.
    first = [None] * BASE_ALPHABET_SIZE
    for symbol, index in raw["model"]["vocab"].items():
        if index < BASE_ALPHABET_SIZE:
            first[index] = symbol
    assert len(set(first)) == BASE_ALPHABET_SIZE and all(len(symbol) == 1 for symbol in first), first[:8]
    assert base_alphabet(tokens) == first, (base_alphabet(tokens)[:8], first[:8])
    byte_table = {chr(256 + b) if (b < 33 or 127 <= b < 161) else chr(b) for b in range(256)}
    assert len(set(first) & byte_table) == 221 and len(set(first) - byte_table) == 35, (
        len(set(first) & byte_table), len(set(first) - byte_table))

    assert len(facts["added_tokens_written"]) == 3 and len(facts["added_tokens_rejected"]) == 19, facts

    # The three tags the reader stores are the tags the metadata resolves, and the ids are properties of this
    # file: the lookup has to return them, which is the only reason a literal here is a legitimate check.
    ids = special_token_ids(tokens)
    assert ids == {"endoftext": 151643, "im_start": 151644, "im_end": 151645}, ids
    for name, value in ids.items():
        assert tokens[value] == SPECIAL_TAGS[name], (name, tokens[value])
    assert sorted(facts["added_tokens_written"]) == sorted(SPECIAL_TAGS[name] for name in ids), facts

    # What the metadata claims about the tokenizer is what the file said, read through the reader rather than
    # through the same dict it is being compared to. `claim` is passed because this vocabulary is byte-shaped: the
    # rule cannot answer for it without being told, and telling it the file's claim is the only honest way to get
    # the value the CLI writes.
    md = metadata(cfg, tokens, merges, claim=facts["claim"])
    assert md["tokenizer.ggml.byte_fallback"] is False, md["tokenizer.ggml.byte_fallback"]
    assert md["tokenizer.ggml.tokens"] == tokens and md["tokenizer.ggml.merges"] == merges
    assert md["tokenizer.ggml.pre"] == "qwen2", md["tokenizer.ggml.pre"]
    assert md["tokenizer.ggml.eos_token_id"] == ids["im_end"], md["tokenizer.ggml.eos_token_id"]
    assert md["tokenizer.ggml.bos_token_id"] == ids["endoftext"], md["tokenizer.ggml.bos_token_id"]
    # The added-token array in vocabulary-id order, which for this vocabulary is endoftext, im_start, im_end. It
    # is not alphabetical (that would be endoftext, im_end, im_start), and the two orders only coincide for the
    # placeholder fixture, so asserting the order here is what keeps the metadata from writing whichever order
    # its own dict iteration happened to produce.
    assert md["tokenizer.ggml.added_tokens"] == [SPECIAL_TAGS[n] for n in sorted(ids, key=ids.get)], (
        md["tokenizer.ggml.added_tokens"], sorted(ids, key=ids.get))
    assert md["tokenizer.ggml.added_tokens"] != [SPECIAL_TAGS[n] for n in sorted(ids)], (
        "id order and name order are different for this vocabulary, which is why the order is asserted as an order")
    types = md["tokenizer.ggml.token_type"]
    assert len(types) == len(tokens)
    assert [i for i, t in enumerate(types) if t == TOKEN_TYPE_CONTROL] == sorted(ids.values()), types

    # The pinned template config carries the same vocabulary facts: it is the file the chat template came from,
    # so a disagreement between the two files would mean the template's tags are not the ids the metadata writes.
    # The policy keys - which tag is EOS, which is padding, that BOS is not added at encode time - are the
    # metadata's claims about the tokenizer, so they are compared against the tokenizer's own file. The tags
    # there are read out of that file rather than written in this test, which is what makes the comparison a
    # check against the source instead of a comparison of two constants.
    template_config = json.loads((pinned / "tokenizer_config.instruct.json").read_text(encoding="utf-8"))
    assert template_config["eos_token"] == SPECIAL_TAGS["im_end"], template_config["eos_token"]
    assert template_config["pad_token"] == SPECIAL_TAGS["endoftext"], template_config["pad_token"]
    assert template_config["bos_token"] is None and template_config["add_bos_token"] is False, template_config
    assert md["tokenizer.ggml.add_bos_token"] is False
    assert md["tokenizer.ggml.byte_fallback"] == facts["byte_fallback"], (md["tokenizer.ggml.byte_fallback"], facts)
    inside = {int(i): str(e["content"]) for i, e in template_config["added_tokens_decoder"].items()
              if int(i) < cfg.vocab_size}
    assert len(template_config["added_tokens_decoder"]) == 22, len(template_config["added_tokens_decoder"])
    assert inside == {ids[n]: SPECIAL_TAGS[n] for n in ids}, inside


def test_the_cli_writes_a_quantized_file_and_states_the_type_it_derived_and_the_exceptions():
    """
    The report a quantized run prints, checked against the file it wrote.

    Two claims the CLI makes in words and the file has to back up: the file type is the scheme its bulk is
    stored under, and a handful of parameters stay F32 because ggml's smallest block is 32 elements. A report
    that printed one type for the whole file while the file held exceptions would be describing a file the CLI
    did not write, and the exception count is the difference between a quality number being attributable to the
    scheme or to those exceptions.

    The embedding is the interesting case: the rule says it is never the dense scheme, so a file written under
    the dense scheme is mostly that scheme with one tensor under Q8_0 - which is only visible by reading the
    types back out of the file.
    """
    cfg = TINY()
    model = tiny_model()
    # The name the file writes the embedding under, taken from the single-source mapping rather than typed here:
    # this is the same dict the converter uses to name it, so the check below is about the file, not about a
    # spelling copied into the test. The key is the module path, `embed.weight`; the value is `token_embd`.
    embedding_name = converter.TENSOR_NAMES["embed.weight"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tokenizer = cli_tokenizer_file(tmp / "tokenizer.json", cfg)

        for scheme in ("q8_0", "q4_0_single_block_sim"):
            out = tmp / scheme
            report = run_cli([
                "--scale", "tiny", "--tokenizer", str(tokenizer), "--out-dir", str(out),
                "--quant", scheme, "--allow-placeholder-prototypes",
            ])
            entries = converter.tensor_entries(model, cfg, quant=convert_to_gguf.quant_rule(scheme))
            core = metadata_for(cfg)["general.arch.heron.core_file"]
            read_back = converter.read_metadata(out / core)
            tensors = converter.read_tensors(out / core)

            # The report's own claims: the scheme it was handed, the exception rule with the block size that
            # makes it, and the file type - printed as a name, so the reader can see what the header records.
            assert f"quant      : {scheme}" in report, report
            assert str(converter.MIN_QUANT_BLOCK) in report, report
            assert "stay F32" in report, report
            file_type = int(read_back["general.file_type"])
            assert convert_to_gguf.describe_type(file_type) in report, (convert_to_gguf.describe_type(file_type), report)
            assert file_type == converter.file_type_for(entries), (file_type, converter.file_type_for(entries))

            quantized = [e for e in entries if e.quant_type is not None]
            exceptions = [e for e in entries if e.quant_type is None]
            assert quantized and exceptions, (len(quantized), len(exceptions))

            # Every tensor's type as the file records it, exceptions included. Checking only the quantized ones
            # would let a writer quietly store an unquantizable entry as a quantized one and still pass. The
            # reader hands the type back as a name, so the name is what is compared, and then the name is tied
            # back to the value the header's dominant type is computed from - otherwise a check on names alone
            # could drift away from the numbers the file stores.
            for e in entries:
                expected_type = e.quant_type or Quant.F32
                recorded = tensors[e.name]["type"]
                assert recorded == expected_type.name, (e.name, recorded, e.quant_type)
                assert int(Quant[recorded]) == int(expected_type), (recorded, expected_type)

            # The embedding exception is not a block-size exception: the rule says it is never the dense scheme,
            # so a file written under the dense scheme is mostly that scheme with the embedding still Q8_0. The
            # header names the dominant type, so the exception is only visible by reading the types back.
            assert tensors[embedding_name]["type"] == Quant.Q8_0.name, tensors[embedding_name]["type"]
            if scheme == "q4_0_single_block_sim":
                assert file_type == int(LlamaFileType.MOSTLY_Q4_0), file_type
                assert int(Quant.Q8_0) in {int(e.quant_type) for e in quantized}, scheme
            else:
                assert file_type == int(LlamaFileType.MOSTLY_Q8_0), file_type

        # A per-expert file is the scheme plus the exceptions ggml cannot store: its weight tensors are all
        # quantizable, but its scalar gate is one element and ggml's smallest block is 32, so that entry stays
        # F32. The header therefore records a dominant type with an exception rather than the scheme, and the
        # exception is only visible by reading the types back out of the file - which is the difference between a
        # report that describes the file and one that describes the rule.
        report = run_cli([
            "--scale", "tiny", "--tokenizer", str(tokenizer), "--out-dir", str(tmp / "files"),
            "--quant", "q8_0", "--allow-placeholder-prototypes",
        ])
        md = metadata_for(cfg)
        for name, entries in [(md["general.arch.heron.core_file"],
                               converter.tensor_entries(model, cfg, quant=convert_to_gguf.quant_rule("q8_0")))]:
            line = f"{name}  {len(entries)} tensors  {convert_to_gguf.describe_type(converter.file_type_for(entries))}"
            assert line in report, (line, report)

        for slot in range(1, cfg.n_experts):
            name = md["general.arch.heron.expert_files"][slot]
            entries = converter.tensor_entries(model, cfg, expert_file_slot=slot,
                                               quant=convert_to_gguf.quant_rule("q8_0"))
            line = f"{name}  {len(entries)} tensors  {convert_to_gguf.describe_type(converter.file_type_for(entries))}"
            assert line in report, (line, report)

            path = tmp / "files" / name
            tensors = converter.read_tensors(path)
            assert len(tensors) == len(entries), (slot, len(tensors), len(entries))
            exceptions = [e for e in entries if e.quant_type is None]
            assert exceptions and all(e.shape[0] < converter.MIN_QUANT_BLOCK for e in exceptions), (
                slot, [e.shape for e in exceptions])
            for e in entries:
                expected_type = e.quant_type or Quant.F32
                assert tensors[e.name]["type"] == expected_type.name, (slot, e.name, tensors[e.name]["type"])
            assert int(converter.read_metadata(path)["general.file_type"]) == int(LlamaFileType.MOSTLY_Q8_0), slot


if __name__ == "__main__":
    from tests import run_tests

    raise SystemExit(1 if run_tests.run(globals()) else 0)
