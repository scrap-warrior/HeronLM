"""
Phase 4.1: the GGUF converter - weights plus metadata, written so the file describes the reference.

A GGUF file is a header, a block of key-value metadata, a block of tensor-info, then the tensor bytes.
Nothing in that layout says what an architecture *means*: it says a tensor is 2048 x 6144 at a name, and a
loader has to be told which block that tensor belongs to and what to do with it after it is loaded. For an
architecture with a loop, a sliding window, swappable experts and a host-side hash table, "the shapes are
obvious" is not true. So the converter's job is not copying bytes, it is making sure the two things it
writes - names and metadata - cannot disagree.

Four rules this file is deliberate about.

1. Names are not restated here. They come from `heronlm.metadata.TENSOR_NAMES`, which is also what the
   metadata records as data. A converter that kept its own copy of the names could write a tensor whose name
   the metadata does not claim, and the file would still open and still pass every shape check.
   `check_entries` is the direction that closes it: every entry's name is re-derived from the module path
   through the same dict.

2. Nothing is pre-processed. The converter writes the parameter bytes it was given. nGPT normalization is
   applied at forward time, and the parity rule says it happens AFTER dequantization, so a converter that
   normalized a weight before quantizing it would store scales computed from magnitudes that normalization
   removes - the file would describe weights that reproduce a different model. Rows are not unit-norm in the
   file, and that is what the tests assert.

3. Dims are recorded as the writer records them, not as the module names them. GGUF tensor-info stores dims
   fastest-first, so a PyTorch-shaped [out, in] weight appears in the file as [in, out] and its bytes stay in
   the module's own element order. That is ggml's indexing convention, so passing the array as-is is correct
   and passing `array.T` is the failure mode: it produces a transposed tensor with the right element count
   and a wrong meaning, which no shape check can see.

4. The n-gram table is not a tensor. There is no ggml op that probes a hash table, so it is an mmap'd blob
   read host-side, written here as codes then scales in the order the metadata records. A test asserts these
   bytes agree with `phase0.quant.quantize`, the harness's authority on what a stored row is.

The CLI is `scripts/convert_to_gguf.py`. This module does not read a tokenizer file: the vocabulary and
merges are arguments, so `heronlm` stays free of the `tokenizers` dependency.

Run the tests with `python tests/test_converter.py`.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from gguf import GGUFReader, GGUFWriter, LlamaFileType, quants
from gguf.constants import GGMLQuantizationType as Quant
from gguf.constants import GGUFValueType

from heronlm.config import HeronConfig
from heronlm.metadata import TENSOR_NAMES, metadata, placeholder_vocabulary

# ggml's smallest quantization block. A quantized tensor is stored as blocks along its input dimension, so a
# tensor smaller than one block cannot be quantized, and a dimension that is not a multiple of the block
# leaves a partial last block. This is not a rounding detail: it is why the writer refuses a tensor of shape
# (1,), and why a width like the roadmap's expert_ffn_width 20345 has to become a multiple of 32 before
# quantization is a decision rather than an exception.
MIN_QUANT_BLOCK = 32

# The int8 code range the table scheme states. `qmax` is the largest magnitude in it, which is 128 rather than
# the largest positive code, 127, because the range is not symmetric. That difference is what the effective-bits
# question is about, so it is written as a range rather than as a divisor: the range is the fact, the divisor is
# what falls out of it. tests/test_converter.py asserts this equals the harness's scheme, which is the only way
# the mirror stays honest.
TABLE_CODE_RANGE = (-128, 127)

# The scheme names the harness records as data, and the ggml type each one writes as. Kept as a
# name -> type table because these names are the ones in `phase0.quant`, and a converter that invented its
# own spelling for "the single-block int4 version that is simulated" would make a measured delta
# unattributable: the number could be coming from the scheme or from the name.
SCHEMES = {
    "q4_0_single_block_sim": Quant.Q4_0,
    "q8_0": Quant.Q8_0,
    # int8 per row is the table's scheme, and the table is not a tensor, so this entry is never used by the
    # writer. It is here so that the scheme a parameter gets and the type it would be written as are the same
    # table for every parameter, including the one that is not written as a tensor.
    "int8_per_row": None,
}

# Parameters that are never quantized as matrices, and why. Mirrored from `phase0.quant.EXCLUSION_REASONS`
# because that module is the authority on the training-side rule and `heronlm` must not depend on the harness
# - the dependency runs the other way. tests/test_converter.py asserts the two stay equal rather than
# trusting that the copy was correct.
EXCLUSION_REASONS = {
    "stream.read_weights": "the per-pass read mixture. Quantizing it would change which branch a block reads, which is the mechanism under test, not a rounding detail.",
    "reinject_gate": "one scalar per block, a ratio between the current state and the original token identity.",
    "gate_scale": "one scalar per expert slot, the amount that makes hot-swapping a slot safe.",
    "gate_bias": "the memory gate's bias, a single value.",
    "gate_key": "the gate's key direction, a vector rather than a matrix. The engine stores it at higher precision because the relevance cosine is computed on it before projection.",
}


@dataclass(frozen=True)
class Entry:
    """
    One tensor as the file will hold it.

    `module_path` is the reference's own name for the thing, kept because it is what `check_entries`
    re-derives the GGUF name from and what a parity failure quotes. `shape` is the PyTorch shape, not the
    shape as recorded in the file: the file's dims are the reverse, and the difference is the rule at the top
    of this file.
    """

    module_path: str
    name: str
    shape: tuple[int, ...]
    quant_type: Quant | None
    data: np.ndarray


def module_pattern(module_path: str) -> str:
    """
    A module path as a TENSOR_NAMES key: the indices replaced by placeholders.

    `blocks.12.attn.q.weight` -> `blocks.{bid}.attn.q.weight`. Doing this by substitution rather than by
    parsing is what makes the check work at any depth: a converter that special-cased "the first block" or
    "the last block" would pass a small test and mis-assign tensors at the roadmap's 32 blocks.

    A row of the shared read weights is written as `stream.read_weights[12]`. Brackets, not a dot, because
    the module has one tensor and the file has one tensor per block: the index is an element selection on a
    single Parameter, not a submodule, and it is the only place where the reference's path and the file's
    name are not the same thing.
    """
    out = re.sub(r"^experts\.\d+\.\d+\.", "experts.{eid}.{slot}.", module_path)
    out = re.sub(r"^blocks\.\d+\.", "blocks.{bid}.", out)
    out = re.sub(r"^stream\.read_weights\[\d+\]", "stream.read_weights", out)
    return out


def tensor_name(module_path: str, cfg: HeronConfig) -> str:
    """
    The name the file will use for this parameter, looked up rather than computed.

    Returns None for a parameter that is not a tensor in the file - the hash table. Raises for a path the
    mapping does not cover, so an unmapped parameter can neither be written under a name the file does not
    claim nor be quietly dropped from it.

    For a routed slot the block is not a placeholder: the config says which block holds slot {slot}, so a
    name for a slot cannot be written for a block the config does not put experts in. Slot 0 raises for the
    same reason in the other direction - the generalist is the block's base FFN, so its bytes are already
    named by the base FFN, and a name for slot 0 would be a second name for the same thing. The reference does
    instantiate a module at slot 0 for the harness, and holds its tensors as non-persistent buffers - the only
    form excluded from `state_dict()` in both directions - so a checkpoint produced by the reference cannot carry
    the bytes this function would have to name; where a checkpoint does carry them, it came from a model where
    slot 0 was an ordinary expert.

    A read-weight row needs the block index in the path, which is why `tensor_name_for_row` exists.
    """
    pattern = module_pattern(module_path)
    if pattern not in TENSOR_NAMES:
        raise ValueError(f"no GGUF name for {module_path!r} (pattern {pattern!r}); TENSOR_NAMES must cover it")
    base = TENSOR_NAMES[pattern]
    if base is None:
        return None

    parts = module_path.split(".")
    if pattern == "stream.read_weights":
        row = re.match(r"^stream\.read_weights\[(\d+)\]$", module_path)
        if row is None:
            raise ValueError(
                "stream.read_weights is one shared tensor and has no single name: the file records one row "
                "per block as blk.{bid}.residual_alpha, so use tensor_name_for_row(block_index, cfg)"
            )
        return tensor_name_for_row(int(row.group(1)), cfg)

    if pattern.startswith("experts."):
        eid, slot = int(parts[1]), int(parts[2])
        if eid == cfg.generalist_slot:
            raise ValueError(
                f"{module_path}: slot {cfg.generalist_slot} has no file. The generalist is each expert "
                "block's base FFN, so its bytes are already named by blk.{bid}.ffn_*. The reference keeps a "
                "module at the generalist index for the training harness and holds its tensors as "
                "non-persistent buffers, so a checkpoint does not carry these bytes at all; where it does, the "
                "bytes came from a model where slot 0 was an ordinary expert, and no name can be written for "
                "them."
            )
        bid = cfg.expert_blocks()[slot]
        return base.replace("{bid}", str(bid)).replace("{eid}", str(eid)).replace("{slot}", str(slot))
    if pattern.startswith("blocks."):
        return base.replace("{bid}", parts[1])
    return base


def tensor_name_for_row(block_index: int, cfg: HeronConfig) -> str:
    """The name for one block's read-weight row. The module has one shared tensor; the file has one row per
    block, so the name needs a block index that a lookup from `stream.read_weights` cannot supply."""
    return TENSOR_NAMES["stream.read_weights"].replace("{bid}", str(block_index))


def expected_shapes(cfg: HeronConfig) -> dict[str, tuple[int, ...]]:
    """
    What each pattern's shape must be, derived from the config rather than from the model.

    A check that read the shape from the tensor being checked could not fail, so this table is written from
    the roadmap's arithmetic. A module layout that disagrees with the config is caught here rather than in a
    file that looks fine and loads into the wrong blocks.
    """
    d = cfg.d_model
    q = cfg.heads * cfg.head_dim
    kv = cfg.kv_heads * cfg.head_dim
    w = cfg.ffn_width
    ew = cfg.expert_ffn_width
    md = cfg.ngram_memory_dim
    return {
        "embed.weight": (cfg.vocab_size, d),
        "blocks.{bid}.attn.q.weight": (q, d),
        "blocks.{bid}.attn.k.weight": (kv, d),
        "blocks.{bid}.attn.v.weight": (kv, d),
        "blocks.{bid}.attn.o.weight": (d, q),
        "blocks.{bid}.ffn.gate.weight": (w, d),
        "blocks.{bid}.ffn.up.weight": (w, d),
        "blocks.{bid}.ffn.down.weight": (d, w),
        "blocks.{bid}.reinject_gate": (1,),
        # One row of the shared tensor, as the file records it.
        "stream.read_weights": (cfg.max_loop, cfg.gr_branches),
        "experts.{eid}.{slot}.ffn.gate.weight": (ew, d),
        "experts.{eid}.{slot}.ffn.up.weight": (ew, d),
        "experts.{eid}.{slot}.ffn.down.weight": (d, ew),
        "experts.{eid}.{slot}.gate_scale": (1,),
        "router.prototypes": (cfg.n_experts, d),
        "memory.value_proj.weight": (d, md),
        "memory.gate_key": (md,),
        "memory.gate_bias": (1,),
        # Not a tensor, so nothing to check. It stays in the table so that the table covers every key in
        # TENSOR_NAMES, which is what tests/test_converter.py asserts.
        "memory.table.weight": (cfg.ngram_memory_entries, md),
    }


def can_quantize(shape: tuple[int, ...]) -> bool:
    """
    Whether ggml can store this shape under a block scheme.

    Two conditions, both from the format rather than from preference: the tensor needs at least one dimension
    (tensor-info requires n_dims >= 1, which is why scalars are written as one-element tensors), and the
    dimension the blocks run along must be an exact multiple of the block. A one-element tensor cannot be
    quantized because a block is 32 elements, so every scalar and sub-block parameter in the reference stays
    F32.
    """
    return len(shape) >= 1 and shape[-1] >= MIN_QUANT_BLOCK and shape[-1] % MIN_QUANT_BLOCK == 0


def scheme_for_path(module_path: str, dense: str = "q4_0_single_block_sim") -> str | None:
    """
    Which scheme a parameter gets, or None if it is excluded.

    The harness's rule, mirrored rather than imported, because `heronlm` is the reference and must not depend
    on the training-side harness. tests/test_converter.py asserts the mirror agrees for every parameter path.
    The rule that has to survive the mirror is the embedding's: the tied embedding is the output head, so it
    is q8_0 and never Q4.
    """
    for excluded in EXCLUSION_REASONS:
        if excluded in module_path:
            return None
    if "memory.table" in module_path:
        return "int8_per_row"
    if "embed" in module_path:
        # The head's scheme is never cheaper than the dense weights.
        return "q8_0"
    if "weight" in module_path:
        return dense
    return None


def quant_type_for(module_path: str, shape: tuple[int, ...], dense: str = "q4_0_single_block_sim") -> Quant | None:
    """The ggml type the file records for this parameter, or None for F32."""
    scheme = scheme_for_path(module_path, dense=dense)
    if scheme is None:
        return None
    q = SCHEMES[scheme]
    if q is None:
        # A scheme whose subject is not a tensor at all: the table goes to the blob, not to this file.
        return None
    if not can_quantize(shape):
        # A parameter the scheme would apply to but the format cannot store. Not left out silently: the
        # tensor is written as F32, which is what the file then records for it.
        return None
    return q


def as_bytes(tensor: torch.Tensor) -> np.ndarray:
    """
    The parameter as the file's bytes, in the module's own element order.

    A scalar becomes a one-element array because GGUF tensor-info needs n_dims >= 1: the value is unchanged,
    only the shape fact. Nothing else is done to the data.
    """
    array = tensor.detach().cpu().numpy()
    return array.reshape(1) if array.ndim == 0 else array


def tensor_entries(
    model: torch.nn.Module,
    cfg: HeronConfig,
    *,
    expert_file_slot: int | None = None,
    quant: Callable[[str, tuple[int, ...]], Quant | None] | None = None,
) -> list[Entry]:
    """
    Every tensor this file carries.

    `expert_file_slot` None means the core file: everything except the routed experts, because an expert is a
    set of slots spread across blocks and hot-swapping loads the set as a unit, so each expert is its own
    file. With a slot index, the entries are that expert's slots and nothing else - a per-expert file that
    also carried the core tensors could not be loaded as a unit. Slot 0 raises, as `tensor_name` says.

    `quant` is a rule from module path and shape to a ggml type. None, the default, writes F32: a file whose
    tensors are the reference's own bytes is bit-exact with the reference by construction, and that is the
    file the parity test is against. Quantization is a choice the caller makes, not something this function
    decides for itself.

    Two paths are handled apart from the parameter loop, and both are reasons this function exists rather
    than a loop over a state dict:

      - `stream.read_weights` is one shared parameter written as one row per block. `named_parameters`
        deduplicates the Parameter object, so it is counted once and emitted many times, under names that
        carry a block index.
      - `router.prototypes` is not a Parameter at all: the router stores it as a plain tensor, so
        `state_dict()` does not carry it. It is still data the file has to hold, which is why the test
        asserts the asymmetry instead of assuming parameters and tensors are the same thing.
    """
    rule = quant if quant is not None else lambda path, shape: None
    entries: list[Entry] = []

    # Refused before anything is collected, so the failure is about the file rather than about a path that
    # happens to exist in a checkpoint. Slot 0 has no file because the generalist is each expert block's base
    # FFN, already named by blk.{bid}.ffn_*. A reference model cannot reach this with slot 0 bytes to write at
    # all: its tensors at that index are non-persistent buffers, so they are not in `named_parameters()` and the
    # loop below never sees them, and `tensor_name` would refuse a name for them anyway. A file for slot 0 can
    # only be produced from a model where slot 0 was an ordinary expert, which is the case this refuses.
    if expert_file_slot == cfg.generalist_slot:
        raise ValueError(
            f"slot {cfg.generalist_slot} has no file. The generalist is each expert block's base FFN, so its "
            "tensors are already named by blk.{bid}.ffn_*, and the tensors at the generalist index have no "
            "destination in a file. Write the core file for the generalist."
        )

    def add(module_path: str, tensor: torch.Tensor):
        name = tensor_name(module_path, cfg)
        if name is None:
            # The hash table: recorded in TENSOR_NAMES as None so that "not a tensor" is a decision written in
            # the same place as the tensor names. It goes to the blob, not to this file.
            return
        array = as_bytes(tensor)
        q = rule(module_path, tuple(array.shape))
        data = quants.quantize(array, q) if q is not None else array
        entries.append(Entry(module_path, name, tuple(array.shape), q, np.asarray(data)))

    for path, param in model.named_parameters():
        pattern = module_pattern(path)
        if pattern.startswith("experts."):
            # Expert parameters belong to a per-expert file, never to the core file.
            if expert_file_slot is not None and int(path.split(".")[1]) == expert_file_slot:
                add(path, param)
            continue
        if expert_file_slot is not None:
            continue
        if pattern == "stream.read_weights":
            for bid in range(cfg.n_blocks()):
                add(f"stream.read_weights[{bid}]", param[bid])
            continue
        add(path, param)

    # The router's prototypes are not in named_parameters, so this is the only place they can be added. A
    # state-dict-driven converter would produce a file with no router tensors and no error.
    #
    # They belong to the core file only. The router decides which expert to load, so a per-expert file cannot
    # carry its own copy of the prototypes: two files would then state two different routers, and the loader
    # would have no way to know which one the routing decision was made with.
    if expert_file_slot is None:
        add("router.prototypes", getattr(model, "router").prototypes)

    return entries


def check_entries(entries: list[Entry], cfg: HeronConfig) -> list[str]:
    """
    Every fact an entry asserts, checked against the config rather than against the entry.

    Problems are returned rather than raised on the first one, because a converter run against a real
    checkpoint usually has more than one disagreement and a report that stops at the first makes the next run
    guess again. `write_gguf` refuses to write when this is not empty.

    What is checked:
      - the name is the name the mapping says for this module path;
      - the block index in the name is the block the config says for that path - which for an expert slot is
        `cfg.expert_blocks()[slot]`, not the slot index;
      - the shape is the shape the config says for the pattern;
      - a quantized tensor can be quantized, i.e. the format can store it;
      - no name is used twice.

    A path the mapping refuses - the generalist slot, a path with no key - is reported as a problem rather
    than raised, so a report describes a checkpoint completely.
    """
    problems: list[str] = []
    shapes = expected_shapes(cfg)
    blocks = set(range(cfg.n_blocks()))
    expert_blocks = set(cfg.expert_blocks())

    for entry in entries:
        pattern = module_pattern(entry.module_path)
        expected = shapes.get(pattern)
        if expected is None:
            problems.append(f"{entry.module_path}: no shape for this pattern")
            continue
        if entry.shape != expected:
            problems.append(f"{entry.module_path}: shape {entry.shape} is not the config's {expected}")

        try:
            mapping_name = tensor_name(entry.module_path, cfg)
        except ValueError as e:
            problems.append(str(e))
            continue
        if mapping_name is not None and mapping_name != entry.name:
            problems.append(
                f"{entry.module_path}: name {entry.name!r} is not the mapping's {mapping_name!r}"
            )

        # The block the entry claims. For a routed slot the config decides it, so the name cannot be the slot
        # index; for a read-weight row the path carries the block because the module has one shared tensor.
        claim = re.match(r"^blocks\.(\d+)\.", entry.module_path) or re.match(
            r"^stream\.read_weights\[(\d+)\]", entry.module_path
        )
        if claim is not None:
            bid = int(claim.group(1))
            if bid not in blocks:
                problems.append(f"{entry.module_path}: block {bid} is outside 0..{cfg.n_blocks() - 1}")

        if pattern.startswith("experts."):
            slot = int(entry.module_path.split(".")[2])
            bid = cfg.expert_blocks()[slot]
            if int(entry.name.split(".")[1]) != bid:
                problems.append(
                    f"{entry.module_path}: name puts it at block {entry.name.split('.')[1]}, the config puts "
                    f"slot {slot} at block {bid}"
                )
            if bid not in expert_blocks:
                problems.append(f"{entry.module_path}: block {bid} does not hold an expert slot")
        elif pattern.startswith("blocks."):
            bid = int(entry.module_path.split(".")[1])
            if bid not in blocks:
                problems.append(f"{entry.module_path}: block {bid} is outside 0..{cfg.n_blocks() - 1}")

        if entry.quant_type is not None:
            if not can_quantize(entry.shape):
                problems.append(
                    f"{entry.module_path}: {entry.quant_type.name} needs a last dimension that is a multiple "
                    f"of {MIN_QUANT_BLOCK}; {entry.shape} cannot be stored, so it must be written as F32"
                )
            if entry.data.dtype != np.uint8:
                problems.append(f"{entry.module_path}: a quantized tensor's bytes are uint8, got {entry.data.dtype}")
        elif entry.data.size != int(np.prod(entry.shape)):
            problems.append(f"{entry.module_path}: {entry.data.size} elements do not match shape {entry.shape}")

    names = [e.name for e in entries]
    if len(names) != len(set(names)):
        problems.append("duplicate tensor names in the file")
    return problems


def file_type_for(entries: list[Entry]) -> int:
    """
    The file type, derived from the tensors that are actually being written.

    This is the one fact about the file that metadata() deliberately does not state, because it is a fact
    about what the writer produced rather than about the architecture: a metadata dict that claimed MOSTLY_F16
    while writing F32 tensors would describe a file it had not produced.

    The name is the dominant storage type among the tensors that are quantized. F32 does not compete for the
    name: the format has a name for an entirely-F32 file (ALL_F32) but none for a mostly-F32 one, because F32
    is what a file type names as the storage rather than as the exception. A handful of scalars and sub-block
    parameters staying F32 inside a quantized file is the normal case, so the file is named by the scheme its
    bulk is stored under. Ties are broken by the type's value, so the same set of tensors always produces the
    same file type. If a type has no name for it, the file is recorded as GUESSED rather than as a type the
    format has no name for.
    """
    counts: dict[Quant, int] = {}
    for entry in entries:
        if entry.quant_type is not None:
            counts[entry.quant_type] = counts.get(entry.quant_type, 0) + 1
    if not counts:
        return int(LlamaFileType.ALL_F32)

    dominant = min(counts, key=lambda t: (-counts[t], int(t)))
    name = f"MOSTLY_{dominant.name}"
    for member in LlamaFileType:
        if member.name == name:
            return int(member.value)
    return int(LlamaFileType.GUESSED)


def write_table_blob(table_weight: torch.Tensor, cfg: HeronConfig) -> bytes:
    """
    The hash table as the blob: every code, then every scale.

    The order is not a style choice. It is what the metadata records as `table.blob_order`, and it is what
    keeps a probe a single contiguous read of value_dim bytes plus one half-precision load - the property
    `table.layout` was recorded for. Interleaving the two would make a probe read a strided row.

    One scale per row, so the whole row is one block: no subdivision, no imatrix, no decay. An all-zero row
    stores scale 0.0, which is what ggml does and what an initialized table is. A guard that quietly used
    scale 1 would produce the same dequantized value for a zero row while disagreeing with the file about what
    the scale is.

    The rounding is the harness's rounding. Codes plus scales are what the engine stores, and the parity
    invariant is about the stored scale: re-rounding the dequantized values against the stored scale
    reproduces these codes, while re-quantizing from scratch does not, because the recomputed max-abs is a
    floating-point product rather than the stored scale. tests/test_converter.py asserts these bytes agree
    with `phase0.quant.quantize` element for element.
    """
    md = blob_metadata(cfg)
    order = md["general.arch.heron.table.blob_order"]
    if order != ["codes", "scales"]:
        raise ValueError(f"the blob order the metadata records is {order}, not codes then scales")
    code_dtype = np.dtype(md["general.arch.heron.table.dtype"])
    scale_dtype = np.dtype(md["general.arch.heron.table.scale_dtype"])
    if code_dtype != np.int8 or scale_dtype != np.float16:
        raise ValueError(f"blob dtypes must be int8 codes and float16 scales, got {code_dtype} and {scale_dtype}")

    # The rounding is the harness's rounding, including its divisor. Codes plus scales are what the engine
    # stores, and the parity invariant is about the stored scale: re-rounding the dequantized values against the
    # stored scale reproduces these codes, while re-quantizing from scratch does not, because the recomputed
    # max-abs is a floating-point product rather than the stored scale. tests/test_converter.py asserts these
    # bytes agree with `phase0.quant.quantize` element for element.
    #
    # The divisor is the largest magnitude in the code range, not the largest positive code. The range is not
    # symmetric: -128..127, so qmax is 128 and a row whose peak is positive rounds to 128 and is clamped to 127,
    # which is one code step lost on the positive side. That is not a bug to fix here - it is the asymmetry the
    # effective-bits question is about, and the only way it stays visible is that the converter produces the
    # harness's codes rather than a plausible-looking version of them. Dividing by 127 instead changes codes on
    # rows near the peak while leaving every scale identical, so no scale comparison can see it.
    #
    # The arithmetic is done at the same precision the harness does it at, for the same reason: the scale is a
    # quotient, and a float64 max-abs rounded through a float32 scale can land on the other side of a half-code
    # boundary than a float32 max-abs divided in float32 does, which changes one code in a row.
    table = table_weight.detach().cpu().numpy()
    expected = (cfg.ngram_memory_entries, cfg.ngram_memory_dim)
    if table.shape != expected:
        raise ValueError(f"table shape {table.shape} is not the config's {expected}")

    code_min, code_max = TABLE_CODE_RANGE
    qmax = max(abs(code_min), abs(code_max))
    max_abs = np.abs(table).max(axis=1, keepdims=True)
    # An all-zero row stores scale 0.0, which is what ggml does and what an initialized table is. A guard that
    # quietly used scale 1 would produce the same dequantized value for a row that is all zeros while disagreeing
    # with the file about what the scale is.
    scale = np.where(max_abs > 0, max_abs / qmax, np.zeros_like(max_abs))
    # A zero row has no scale, so its codes are zero. Dividing by the stored scale rather than by a recomputed
    # one is the point: the file records the scale, and the codes have to be the codes that go with it.
    codes = np.where(max_abs > 0, np.round(table / np.where(scale > 0, scale, np.float32(1.0))), 0.0)
    codes = np.clip(codes, code_min, code_max).astype(np.int8)

    codes_bytes = codes.reshape(-1).tobytes(order="C")
    scale_bytes = scale.reshape(-1).astype(np.float16).reshape(-1).tobytes(order="C")
    return codes_bytes + scale_bytes


def blob_metadata(cfg: HeronConfig) -> dict[str, Any]:
    """
    The blob facts, read out of the metadata rather than written here as a second copy.

    `metadata()` needs a vocabulary because it resolves the special-token ids from it, and none of the blob
    keys depend on the vocabulary. The placeholder vocabulary is what the drivers use, for the same reason the
    tests use it: it exercises the lookup instead of skipping it.
    """
    return metadata(cfg, placeholder_vocabulary(cfg.vocab_size), [])


def write_gguf(
    path: Path,
    cfg: HeronConfig,
    md: dict[str, Any],
    entries: list[Entry],
    *,
    expert_file_slot: int | None = None,
) -> dict:
    """
    Write one file: metadata, tensor-info, bytes.

    `expert_file_slot` is recorded as `general.arch.heron.expert_file_slot` because a per-expert file is not
    the model: it is one slot set, and a loader has to know which slot it is opening to tell a code expert's
    file from a math expert's file. The core file records nothing for it, which is what None means here.

    The writer's order is fixed by the format: header, then key-value data, then tensor-info, then tensor
    bytes, then close. The tensor-info block is written after the tensors are known because it records each
    tensor's offset, so there is no way to write the parts in a different order.

    Returns what was written and what could not be. Skipping is reported rather than dropped: the writer
    cannot store an empty array, so a fact that cannot be written has to appear in the return value - a file
    with an empty merges list would otherwise look complete.

    `general.architecture` is not written here. It is the writer's own constructor argument, and the writer
    emits it before anything else, so writing it again produces a duplicated key - which is harmless in the
    file but means this function would be stating the same fact twice, from two different places. The value is
    read out of the metadata instead of being named here, so the architecture in the file is the architecture
    `heronlm.metadata` states.
    """
    problems = check_entries(entries, cfg)
    if problems:
        raise ValueError("converter refuses to write: " + "; ".join(problems))

    arch = md.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        raise ValueError("metadata must state general.architecture; the file records it as the architecture")

    writer = GGUFWriter(str(path), arch)
    written: dict[str, str] = {}
    skipped: list[str] = []
    for key, value in md.items():
        if key == "general.architecture":
            # Already in the file, by the writer. Recorded here as a key the file holds, so that the report
            # describes the file rather than the calls this function made.
            written[key] = "STRING"
            continue
        if isinstance(value, list):
            if len(value) == 0:
                # The writer cannot store an empty array. Reported rather than dropped, so that a file with an
                # empty merges list does not look complete.
                skipped.append(key)
                continue
            writer.add_key_value(key, value, GGUFValueType.ARRAY, sub_type=GGUFValueType.get_type(value[0]))
        else:
            writer.add_key_value(key, value, GGUFValueType.get_type(value))
        written[key] = GGUFValueType.get_type(value).name

    if expert_file_slot is not None:
        writer.add_key_value("general.arch.heron.expert_file_slot", int(expert_file_slot), GGUFValueType.INT32)
        written["general.arch.heron.expert_file_slot"] = "INT32"

    file_type = file_type_for(entries)
    writer.add_file_type(file_type)
    written["general.file_type"] = "INT32"

    for entry in entries:
        if entry.quant_type is None:
            writer.add_tensor(entry.name, entry.data)
        else:
            # raw_shape stays None: the array handed to the writer is the packed byte shape, and the writer
            # derives the recorded dims from it. Passing the logical shape instead is what raises, because
            # from the writer's point of view a quantized tensor's dims are bytes per row.
            writer.add_tensor(entry.name, entry.data, raw_dtype=entry.quant_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return {
        "path": str(path),
        # The keys the file actually holds, not the keys the metadata dict was given: an empty array is not a
        # fact in the file, and the file type is a fact the writer adds from what it wrote.
        "metadata_keys": written,
        "skipped": skipped,
        "tensors": {e.name for e in entries},
        "file_type": file_type,
    }


def _release(reader: GGUFReader) -> None:
    """
    Release the file the reader mapped.

    The reader is a `np.memmap`, and on Windows a live mapping keeps the file open: a test that reads a file
    inside a temporary directory cannot delete it while the mapping exists, which is a failure in the helper
    rather than in the file. Closing the mapping is the only part that matters, so it is done here - on the
    read side, where the reference is describing what the file contains - rather than in each test.
    """
    mapping = getattr(reader.data, "_mmap", None)
    if mapping is not None:
        mapping.close()


def read_metadata(path: Path) -> dict[str, Any]:
    """
    The metadata as the file states it, read back.

    Needed because gguf's reader exposes fields as a tuple of parts and indices: a field's `.data` holds the
    indices of the value's parts, not the value, so the value has to be decoded. Comparing metadata against
    the file is comparing it against what the file actually contains, which is the point - a writer that
    dropped an empty array, or stored a float at the precision the format happens to have, is only visible on
    the read side.
    """
    def element(part, vtype):
        if vtype == GGUFValueType.STRING:
            return bytes(part).decode("utf-8")
        if vtype == GGUFValueType.BOOL:
            return bool(part[0])
        return part[0].item() if len(part) == 1 else [int(x) for x in part]

    out: dict[str, Any] = {}
    reader = GGUFReader(str(path))
    try:
        for key, field in reader.fields.items():
            if key.startswith("GGUF."):
                continue
            parts = [field.parts[i] for i in field.data]
            out[key] = (
                [element(p, field.types[1]) for p in parts]
                if field.types[0] == GGUFValueType.ARRAY
                else element(parts[0], field.types[0])
            )
    finally:
        _release(reader)
    return out


def read_tensors(path: Path) -> dict[str, dict[str, Any]]:
    """
    The tensors as the file states them: name, dims as recorded, type, byte count, bytes.

    The dims come back reversed relative to the PyTorch shape that was written, which is the convention at the
    top of this file, and the bytes stay in the module's own element order. That pair is what a test asserts
    together: reversed dims plus the module's element order is what a ggml loader reads as the reference's
    matrix, while reversed dims plus a transposed element order is a wrong tensor that passes every shape
    check.

    The bytes are copied out of the mapping rather than handed back as a view. The reader's tensor data is a
    view on the mmap, and this function releases the mapping before returning, so a view would be a view on
    memory that no longer exists - which is not an error message, it is an access violation. A copy is the only
    form that survives the helper.
    """
    reader = GGUFReader(str(path))
    out: dict[str, dict[str, Any]] = {}
    try:
        for tensor in reader.tensors:
            out[tensor.name] = {
                "dims": tuple(int(x) for x in tensor.shape),
                "type": tensor.tensor_type.name,
                "n_bytes": int(tensor.n_bytes),
                "data": np.array(tensor.data, copy=True),
            }
    finally:
        _release(reader)
    return out
