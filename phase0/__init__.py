"""
Training-side harness for the Phase 0 arms and the Phase 1 table gates.

The reference package defines semantics the GGUF fork has to reproduce bit-exactly. This package does
not: it trains, measures and decides. Keeping them separate is what keeps the parity oracle clean - if
training, fake quantization or measurement code lived inside the reference, the parity test would depend
on harness details (block layout, per-block scales, the order of dequantization relative to nGPT
normalization). The same argument covers the Phase 1 measurement code: the table's semantics live in
heronlm/ngram.py, and the numbers measured against those semantics belong here.

Contents:
  corpus.py  - generated tasks. Phase 0 measures mechanisms, not quality. `make_corpus` is the fixed split
               used by the tests and the Phase 1 builders; `SampleStream` is the endless deterministic
               stream over the synthetic families. Their task space is finite, so a gate-scale token budget
               needs the real-text path below.
  tokens.py  - uint32 token shards and a deterministic windowed stream over real text, with train/val splits
               that cannot overlap by construction. `scripts/tokenize_corpus.py` builds the shards.
  arms.py    - the arms, as single-variable config deltas.
  train.py   - training and evaluation at either scale, with per-arm checkpoint/resume and reuse of identical
               baselines across gates (the five baselines are one training run, not five).
  measure.py - per-branch magnitudes for gate G0.3.
  quant.py   - fake quantization, for the PTQ delta the QAT decision is gated on.
  gates.py   - the thresholds, as data, with the claims they came from.
  table.py   - the Phase 1 table gates: the disjointness guard, the recall axis as its own denominator, the
              length distribution, the memory path at the stated dims and after int8, and the decode path.

Run: python scripts/phase0.py, python scripts/phase1.py
"""
