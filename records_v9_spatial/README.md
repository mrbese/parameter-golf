# BESE v3 — Spatial Letter Encoding

**Author:** Omer Bese ([@mrbese](https://github.com/mrbese))
**Date:** 2026-04-30
**Track:** Non-record (novel tokenizer + structural embedding init)
**val_bpb:** `<TBD post-run>` (INT6 + LZMA + sliding window eval with n-gram tilt)
**Artifact size:** `<TBD post-run>` bytes

---

## The idea — one geometric prior, two jobs

v3 derives a 3D coordinate for every English letter from how often the letters co-occur in FineWeb (MDS embedding of the bigram structure, anchored on `e` at the origin). Letters that appear next to each other a lot end up close together; letters that almost never co-occur end up far apart.

The same coordinates are then used **twice**:

1. **BPE merge scoring.** Standard BPE picks the most-frequent pair to merge. v3 picks `freq × exp(−distance / (3·σ))^0.02` — a gentle multiplicative nudge that breaks ties in favor of spatially-close pairs without overriding the frequency signal. The `pow_0.02` shape is the winner of a local sweep on a 160K-char corpus where it beat standard BPE 76,070 vs 76,131 tokens.
2. **Embedding initialization.** The first 12 dims of each letter token's embedding row are seeded from the 12D version of the same coordinates (rest of the row is small Gaussian). The model starts training already knowing the bigram-cluster structure instead of having to learn it from scratch.

To our knowledge this is the first submission that uses one geometric prior to drive *both* tokenizer behavior and model embedding init. Previous submissions optimize either the tokenizer (custom BPE, BESE) or the embedding init (Spectral, OrthoInit) — never the same prior across both.

## Architecture

The architecture is **identical to PR #1666** (BESE record at 1.1531 BPB, 3-seed mean) so the v3 contribution is isolated to the tokenizer + embedding-init layer:

| Parameter | Value |
|---|---|
| `num_layers` | 12 |
| `model_dim` | 512 |
| `mlp_mult` | 3.5 |
| `num_heads` / `num_kv_heads` | 8 / 4 (GQA) |
| `depth_recurrence` | layers 3–5, 3 loops, activation at 35% |
| `parallel_residual_start` | layer 8 |
| `qk_gain_init` | 5.25 |
| `vocab_size` | **288** (47 v3-base + 241 BPE merges) |
| `quantization` | INT6 (mixed clipping per category) |
| `compression` | LZMA preset 9 |
| `eval` | sliding window stride 64 + n-gram tilt + score-first TTT |

`vocab_size` is held at 288 to make v3 directly comparable to v2 / PR #1666 — the only meaningful difference between the two is the tokenizer + init, not architecture or budget.

## v3 vocab layout

47 base tokens (vs v2's 40):

| Range | Token category |
|---|---|
| 0–3 | PAD, BOS, EOS, UNK (0 bytes) |
| 4–29 | 26 lowercase letters `a..z`, alphabetical (1 byte each) |
| 30–36 | space, period, comma, newline, ?, quote, OTHER_PUNCT (1 byte each) |
| 37–46 | digits `0..9` (1 byte each) |
| 47–287 | 241 BPE merges, byte count = sum of constituents |

There are **no group tokens, no position tokens** — every letter is a single token. This is the structural break from v2: vocabulary slots that used to encode "which group does this letter belong to" are repurposed to give every letter direct access, with the spatial geometry absorbing the disambiguation.

## BPB Correctness Proof

The competition rules require any tokenizer change to "prove with certainty that val_bpb is correctly calculated." We include the proof inline.

**Invariant:** for any input string `s`,

```
sum(BYTES_PER_TOKEN[t] for t in encode(s)) == len(s.encode("utf-8"))
```

so val_bpb shares its denominator with every SP1024/SP8192 submission.

**Per-token byte accounting** (`bese_v3_constants.py::build_bytes_per_token`):

| Category | Count | Bytes/token |
|---|---|---|
| Special tokens (PAD/BOS/EOS/UNK) | 4 | 0 |
| Letter tokens `a..z` | 26 | 1 |
| Punctuation tokens | 7 | 1 |
| Digit tokens | 10 | 1 |
| BPE merge tokens | 241 | recursive sum of constituents |

**Per-character argument** (`bese_v3_fast_bpe.py::_text_to_base_tokens`):

For each character `ch`:

- **Mapped path** — if `lower(ch)` is in `ENCODE_TABLE` and the mapped tokens' byte sum equals `len(ch.encode("utf-8"))`, emit those tokens. The runtime check guards against drift; if it ever fails we fall through to the byte-fallback path.
- **Byte-fallback path** — emit `OTHER_PUNCT_ID` exactly `len(ch.encode("utf-8"))` times. Each contributes 1 byte.

In both branches the per-character byte sum equals the UTF-8 byte length of that character, so the invariant holds for the full string by linearity.

**Merge byte count** is computed transitively when merges are loaded:

```python
bpt = np.zeros(vocab_size, dtype=np.int16)
bpt[:BASE_VOCAB_SIZE] = BYTES_PER_TOKEN
for pair, new_id in self.merges:
    bpt[new_id] = bpt[pair[0]] + bpt[pair[1]]
```

**Self-test** (run from this folder; <1s on CPU):

```python
from bese_v3_fast_bpe import BeseV3FastBPE
tok = BeseV3FastBPE.load("tokenizer.json")
bpt = tok.compute_bytes_per_token()
test_cases = [
    "the quick brown fox jumps over the lazy dog",
    "Hello, World! 1234567890",
    "naïve café résumé — emojis: 🚀✨🌍",
    "newlines\nand\ttabs and \"quotes\" and 'apostrophes'",
    "MixedCase HTML <p>tag</p> and JSON {\"a\": 1}",
]
for s in test_cases:
    ids = tok.encode(s)
    assert sum(int(bpt[t]) for t in ids) == len(s.encode("utf-8")), f"failed: {s!r}"
print("OK — v3 byte invariant holds")
```

The base tokenizer's `_self_test()` exercises this assertion by default; running it as a module (`python bese_v3_fast_bpe.py`) re-validates the invariant on every load.

## Training

| Stage | Value |
|---|---|
| Hardware | 8× NVIDIA H100 80GB SXM (RunPod) |
| Wallclock | 600 s (timed) |
| Optimizer | Muon (Newton-Schulz) for 2D matrices, AdamW for scalars/embeddings |
| EMA decay | 0.9965 |
| Warmdown | 5000 iters |
| SWA | activated at step 1200 |
| Sequence length | 2048 (train and eval) |
| Batch tokens / step | 786,432 (global) |

### Training Curve

`<TBD post-run>`

## Evaluation

| Stage | val_bpb |
|---|---|
| Raw (post-EMA) | `<TBD>` |
| INT6 roundtrip | `<TBD>` |
| **INT6 + Sliding Window + N-gram tilt** | **`<TBD>`** ← submission score |
| INT6 + Sliding Window + TTT | `<TBD>` |

> **Statistical-significance note.** This is a **single-seed** result. We submit under the non-record / novel-idea track which explicitly accepts in-progress and unoptimized solutions for novel directions; the headline number is meant to demonstrate that the geometric prior produces a comparable (or better) result against the v2 baseline (PR #1666 at 1.1531 BPB), not to claim leaderboard-worthy statistical significance. Three-seed validation is in *Ongoing Work* pending compute credits.

## Comparison to v2 (the control)

| | v2 (PR #1666) | v3 (this) |
|---|---|---|
| Base vocab | 40 (groups + positions) | 47 (flat 26 letters + special + punct + digits) |
| BPE scoring | Standard frequency | Spatial: `freq × exp(−d/3σ)^0.02` |
| Embedding init | Random gaussian (σ=0.005) | First 12 dims from spatial coords + rest random |
| Total vocab | 288 | 288 |
| Architecture | Identical | Identical |
| Quantization | Identical | Identical |
| Eval | Identical | Identical |
| BPB | **1.1531** (3-seed) | **`<TBD>`** (single-seed) |

The architecture, optimizer, quantization, and eval pipeline are pixel-identical to v2. Any BPB delta between the two is attributable to the tokenizer + embedding-init change.

## Predicted cross-language behavior

The v3 gain over standard BPE comes from the bigram matrix encoding real linguistic constraints. English vowel co-occurrence is roughly uniform, which is why the local-sweep improvement on Gutenberg English is modest (76,070 vs 76,131 tokens, −0.08%).

Languages with stricter co-occurrence rules — **Turkish, Finnish, Hungarian, Japanese, Korean** — should produce tighter MDS clusters and larger gains. Turkish's vowel-harmony rules in particular force every vowel in a word to agree on backness, so the 2-gram matrix encodes a *grammatical* constraint, not just a phonological tendency. The 12D embedding init would then carry that constraint into the model as a free inductive bias.

The current `letter_coords_v3.json` is FineWeb-derived and therefore English-specific. A Turkish-corpus-derived version would be a clean comparative test, predicted to show a larger v3-vs-v2 gap. **Untested but predicted; offered here as the natural next experiment, not as a claimed result.**

## Code Structure

| File | Bytes | Role |
|---|---|---|
| `train_gpt.py` | `<TBD>` | Self-contained training entry point — model definition, training loop, eval, quantization, compression. The `_apply_spatial_letter_init` helper lives here behind `SPATIAL_INIT_ENABLED=1`. |
| `bese_v3_fast_bpe.py` | `<TBD>` | v3 tokenizer encode/decode + spatial-scored BPE training |
| `bese_v3_constants.py` | `<TBD>` | 26-letter flat alphabet, `BYTES_PER_TOKEN`, spatial scoring helpers |
| `tokenizer.json` | `<TBD>` | Trained BPE merges + the 3D coords used to score them |
| `letter_coords_v3.json` | `<TBD>` | Fitted 3D + 12D letter coords (loaded by training for embedding init) |
| `submission.json` | — | Metadata |
| `train_log_run1.txt` | — | Single-seed training log |

`train_gpt.py` is a copy of `integration/train_gpt_bese.py` from the author's fork at the v9 commit. The same accounting pattern as PR #1665 applies: the bundled module bytes are accounted for in `submission.json::code_bytes`, and they could be inlined mechanically with no change to the artifact size.

## Reproduction

### Quick path — run from this records folder

```bash
# From the cloned upstream parameter-golf repo, in this folder:
cd records/track_non_record_16mb/2026-04-30_BESE_v3_Spatial

# Install the two extra packages (rest is in the official RunPod image):
pip install einops sentencepiece

# Validation needs the SP1024 cached FineWeb (used as the source text
# for the v3 re-encoding pipeline; the artifact does not ship SP):
python ../../../data/cached_challenge_fineweb.py --variant sp1024

# Run training + eval directly:
SPATIAL_INIT_ENABLED=1 \
SPATIAL_INIT_PATH=./letter_coords_v3.json \
SPATIAL_INIT_DIMS=12 \
TOKENIZER_PATH=./tokenizer.json \
VOCAB_SIZE=288 \
DATA_PATH=../../../data/datasets/fineweb10B_v3_spatial/ \
RUN_ID=v3_spatial_repro \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Full pipeline — rebuild from scratch

If you want to re-derive the v3 coordinates and re-encode the FineWeb shards:

```bash
git clone https://github.com/mrbese/parameter-golf-bese.git bese
cd bese
git checkout v9-spatial
pip install einops --break-system-packages
python scripts/runpod_v9_spatial.py --num-gpus 8
```

The orchestrator's Phase 0 covers (in order): (a) optimizing letter placement on FineWeb n-grams, (b) training spatial-scored BPE merges, (c) re-encoding training shards into v3 token IDs, (d) building the n-gram tilt table. Total Phase 0 wall time ≈ 30–50 min. Phase 1 training is the same 600s cap.

## About the author

Omer Bese ([@mrbese](https://github.com/mrbese)) — not an ML researcher by training. Came at this from a Turkish-speaker's intuition: Turkish forces every vowel in a word to obey vowel harmony, so I grew up reading text where letters cluster by phonological feature. The dual-use geometric prior in v3 is, in retrospect, a way of making that kind of language-structure explicit at the model level — testing whether the model can be given the structure for free instead of learning it from scratch over 600 s of training.

Previous BESE submissions:
- [PR #1666](https://github.com/openai/parameter-golf/pull/1666) — BESE 288-vocab record entry, 1.1531 BPB (3-seed mean). The control variable for v3.
- [PR #1665](https://github.com/openai/parameter-golf/pull/1665) — BESE + Mamba-3 SSD hybrid, 7.6 MB / 48% of cap. First byte-level tokenizer + state-space model in the challenge.
- [PR #1327](https://github.com/openai/parameter-golf/pull/1327) — Earlier BESE record at 1.1276 BPB (single-seed).

## Ongoing Work

- **Three-seed statistical validation** of v3 against v2 (control) at the same architecture
- **Turkish / vowel-harmony language test** — re-derive coords from a Turkish corpus, train at the same architecture, measure cross-language v3-vs-v2 delta
- **Higher-vocab variant** (1064 instead of 288) to see whether the spatial gain scales with merge budget
- **Combined with TTT and Triton-fixed Mamba** (PR #1665 path) for a full-stack v3 + SSM submission

## Acknowledgments

- PR #1666 (BESE 288-vocab) for the architecture stack used here unchanged
- PR #1493 (`bigbag`, 1.0810 SOTA) for the score-first TTT pattern
- The Mamba-2 SSD paper (Dao and Gu, 2024) — referenced for the byte-tokenizer + SSM combination in sister PR #1665
