# HeronLM

PyTorch reference implementation of the HeronLM design: looped core (~1.85B compute params), five
swappable domain experts, an n-gram memory table, Qwen's tokenizer, GGUF runtime on a phone.

`ROADMAP.md` is the design record and is treated as frozen. This repository is the reference and the
measurement harness, and the two are kept apart on purpose: the reference is the parity oracle, so the
harness cannot quietly change what the oracle means.

## Layout

| Path | Role |
|---|---|
| `heronlm/` | The reference. Readable and bit-exact: the arithmetic is written out rather than replaced by a fused kernel, and a library constant is not trusted when the reference can state the number itself. |
| `phase0/` | The measurement harness: arms, gates, corpus, training loop. Imports `heronlm`, never the other way round. |
| `scripts/` | Entry points - drivers, converter, tokenizer CLI, report. This is the only place that reads the pinned tokenizer file. |
| `tests/` | Parity and behaviour tests. `tests/preflight.py` is a measurement script rather than a test; it is what the memory and throughput numbers quoted in the driver come from. |
| `tokenizer/qwen2.5/` | The pinned tokenizer, tracked as source: vocab, merges, byte-fallback, special tokens, chat template, used exactly as-is. |
| `runs/` | Everything a run writes (shards, checkpoints, records). Created on demand, not committed. |

## Running

No install step is required; the tests keep a `sys.path` insertion so they run from a clone.

```
python tests/test_reference.py     # reference behaviour and parity
python tests/test_converter.py     # tensor-name and scheme mapping
python tests/test_loader.py        # file reading, quant schemes, metadata
python tests/test_phase0.py        # gates, arms, corpus, training
python tests/test_phase1.py        # Phase 1 gate plumbing
python tests/preflight.py          # measured params, memory, throughput
python scripts/phase0.py           # driver; --scale phase0 needs a GPU
python scripts/tokenize_corpus.py  # text -> uint32 shards plus manifest
```

## Conventions the code enforces

- **Three different numbers.** Active params (what computes per token), resident params (what a device
  holds in RAM), and capacity (what lives on storage) are never reported as one number.
- **The tokenizer is used as-is.** Special-token ids are resolved by name from the pinned file, never
  written as literals - in code, in comments, in docstrings and in prose.
- **Verdicts are states**, not booleans: `not stated`, `not measured`, `pending`, `indicative`, `pass`,
  `fail`. Only `pattern` thresholds can decide at any scale, so a short run cannot print a `fail`
  against a threshold it cannot resolve.
- **Refusals are lists, not exceptions.** A problem is reported as a list of problems so a driver can
  show every reason a run would be invalid rather than the first one.
- **Arms come from config.** Nothing hard-codes an arm: an arm is generated from the config it is
  measured against, which is what keeps a scale change from silently changing the comparison.
- **Tests are deterministic.** Seeds are explicit and orders are recorded with their own hash.
- **QAT stays in the training-side harness** - the reference does not simulate quantization.

## Dependencies

`torch` is the reference's only real dependency. `gguf` is declared for the converter and loader only.
`tokenizers` is deliberately not a dependency: the reference treats vocab, merges and the chat template
as data it reads, so a fact like a token type code stays a number the reference states and a test keeps
equal, rather than a library constant.

## Not committed

Run artifacts are evidence, not source: a recorded number is only meaningful with the key that produced
it, and that key lives in the record rather than in the tensor file. See `.gitignore` for the rules and
the reasons, and `.gitattributes` for why line endings are a repository decision rather than a machine
setting.
