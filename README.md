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
python tests/run_tests.py            # all five suites, one process per suite
python tests/test_reference.py       # reference behaviour and parity
python tests/test_converter.py       # tensor-name and scheme mapping
python tests/test_loader.py          # file reading, quant schemes, metadata
python tests/test_phase0.py          # gates, arms, corpus, training
python tests/test_phase1.py          # Phase 1 gate plumbing
python tests/preflight.py            # measured params, memory, throughput (not a test)
```

The runner uses one process per suite because the settings the tests depend on are process-wide: default
dtype, the TF32 flags the Phase 0 training tests set, and RNG state. Each module is still runnable on its
own, and the runner is only an aggregator.

The harness itself is run through the drivers:

```
python scripts/phase0.py           # driver; --scale phase0 needs a GPU
python scripts/phase1.py           # Phase 1 driver
python scripts/tokenize_corpus.py  # text or parquet rows -> uint32 shards, manifest, document index
python scripts/convert_to_gguf.py  # reference model -> GGUF
python scripts/report.py           # TINY-scale walkthrough report
bash scripts/deploy_3060.sh        # setup + pilot + run on the Linux card; its flags are the run key
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

`pyarrow` is not declared either, and is used by the corpus builder's CLI alone. `phase0.documents` holds the
document rules and takes `encode` as a function rather than importing a tokenizer, so the rules and their
tests run with an in-memory source and no parquet reader - which is what keeps CI runnable without it. The
builder reports a missing reader before the first read, because a run that cannot read the corpus is a run
that should not have started. `huggingface_hub` is not declared either: it is the download step, and the
reference has no reason to know that a corpus came from a hub.

Two things on the run machine are worth stating because both change what a number means. Install order:
PyPI's Linux wheel for `torch` is the CPU build, and `torch>=2.6` in `pyproject.toml` is satisfied by it, so
a plain install quietly installs the CPU wheel and a run trains on CPU while still printing a card name -
CUDA torch comes from the wheel index first, then `uv pip install --no-deps -e .`. And `tests/preflight.py`
measures peak memory through a Windows psapi call, so on Linux it prints no peak line and its memory figures
stay arithmetic: they are what the module shapes hold, not what a device was observed holding.
`scripts/deploy_3060.sh` samples the card with `nvidia-smi` during its pilot instead, and labels which of its
numbers is a sample and which is a measurement.

## License

Apache-2.0, in `LICENSE`. The scope is what this repository's own code and prose cover: the pinned
tokenizer files are third-party content and are not covered by it. Their provenance is recorded in
`tokenizer/qwen2.5/README.md` and the attribution is in `NOTICE`, because a license file cannot describe
content it did not cover, so the line between the two is stated rather than assumed.

## Not committed

Run artifacts are evidence, not source: a recorded number is only meaningful with the key that produced
it, and that key lives in the record rather than in the tensor file. See `.gitignore` for the rules and
the reasons, and `.gitattributes` for why line endings are a repository decision rather than a machine
setting.
