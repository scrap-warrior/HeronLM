"""
Pre-flight measurement for the gate-scale run: what a PHASE0 arm costs in memory and in time.

Memory is arithmetic from the module shapes, so it is stated as arithmetic rather than as a measurement:
params, optimizer state, the logits tensor and the attention score tensor are each computed from the config.
The time is measured on this machine, which has no CUDA card, so it is not an estimate of a rented card - it
shows what the same code does without one, and it is labelled as CPU.

Run:
  python tests/preflight.py
"""

import ctypes
import ctypes.wintypes
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from heronlm.config import PHASE0
from heronlm.memory import slot_ids
from heronlm.model import HeronLM
from phase0.arms import build_arms
from phase0.train import device


def process_peak_bytes() -> int:
    """Peak resident set of this process. Windows-only; returns 0 if the call is unavailable."""
    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    # GetProcessMemoryInfo lives in psapi, not kernel32. If the call is unavailable the measurement is
    # reported as unavailable rather than as zero, so the arithmetic above stays the only memory claim.
    try:
        library = ctypes.windll.psapi
    except AttributeError:
        return 0
    if not library.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return 0
    return int(counters.PeakWorkingSetSize)


def main() -> int:
    cfg = PHASE0()
    arm = list(build_arms(cfg).values())[0][0]
    trained_k = arm.spec.train_k[0]
    model = HeronLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=arm.spec.lr, betas=(0.9, 0.95), weight_decay=arm.spec.wd)

    params = sum(p.numel() for p in model.parameters())
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
    state = sum(sum(v.numel() * v.element_size() for v in group["params"] if isinstance(v, torch.Tensor))
                for group in opt.param_groups)

    print(f"arm: {arm.name} ({arm.gate}) at gate scale "
          f"(d={cfg.d_model}, {cfg.input_blocks}+{cfg.loop_blocks}+{cfg.output_blocks} blocks, "
          f"{cfg.heads}/{cfg.kv_heads} heads, head_dim {cfg.head_dim}), device {device()}")
    print(f"  params                    {params:,} ({bytes_ / 1e9:.2f} GB at fp32)")
    print(f"  of which the tied vocab   {cfg.vocab_size * cfg.d_model * 4 / 1e9:.2f} GB "
          f"({cfg.vocab_size:,} x {cfg.d_model})")
    print(f"  optimizer state           {state / 1e9:.2f} GB (2 Adam moments per parameter)")
    print(f"  checkpoint on disk        {(bytes_ + state) / 1e9:.2f} GB per arm, before RNG states and "
          f"the recorded orders")

    print("\nper step, at the shapes a card would have to hold (fp32, everything kept for the backward pass):")
    for batch, seq_len in ((1, 1024), (2, 1024), (8, 2048)):
        blocks = cfg.input_blocks + cfg.loop_blocks * trained_k + cfg.output_blocks
        scores = batch * cfg.heads * seq_len * seq_len * 4
        logits = batch * seq_len * cfg.vocab_size * 4
        hidden = batch * seq_len * cfg.d_model * 4
        print(f"  batch {batch} seq {seq_len}, k={trained_k} ({blocks} blocks per pass):")
        print(f"    scores  {scores / 1e9:.2f} GB per block, {scores * blocks / 1e9:.2f} GB over the pass. "
              f"The reference materializes them because that is the tensor the parity test compares; a flash "
              f"kernel does not, and swapping one in changes the numbers.")
        print(f"    logits  {logits / 1e9:.2f} GB. Full vocabulary per position, because the loss is measured "
              f"over the tokenizer the design fixes.")
        print(f"    hidden  {hidden * blocks / 1e9:.2f} GB over the pass, before attention and MLP internals")

    batch, seq_len = 1, 1024
    ids = torch.arange(1, batch * seq_len + 1, dtype=torch.long).reshape(batch, seq_len) % cfg.vocab_size
    target = ids.roll(-1)
    target[:, -1] = 0
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        logits = model.forward(ids, trained_k, None, None, slot_ids(ids, cfg))
        F.cross_entropy(logits.reshape(-1, cfg.vocab_size), target.reshape(-1), ignore_index=-1).backward()
        opt.step()
    elapsed = 0.0
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        started = time.perf_counter()
        logits = model.forward(ids, trained_k, None, None, slot_ids(ids, cfg))
        F.cross_entropy(logits.reshape(-1, cfg.vocab_size), target.reshape(-1), ignore_index=-1).backward()
        opt.step()
        elapsed += time.perf_counter() - started
    rate = (batch * seq_len) / (elapsed / 3)
    print(f"\nmeasured here: {elapsed / 3:.1f} s per step at batch {batch} seq {seq_len} k={trained_k}, "
          f"{rate:.0f} tokens/s on CPU. No CUDA card on this machine, so this is not a rented-card estimate.")
    for budget in (1_000_000_000, 2_000_000_000):
        steps = budget // (batch * seq_len)
        days = steps * (elapsed / 3) / 86400
        print(f"  {budget:,} tokens is {steps:,} steps; at {rate:.0f} tokens/s that is {days:.0f} days "
              f"on this machine")
    peak = process_peak_bytes()
    if peak:
        print(f"\npeak process memory here: {peak / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
