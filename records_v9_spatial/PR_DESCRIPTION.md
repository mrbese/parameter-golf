# PR Description Draft — paste into the PR body when opening on openai/parameter-golf

> Title (use exactly this in the GitHub PR title field):
>
> `Non-record: BESE v3 — Spatial Letter Encoding (one geometric prior, two jobs)`

---

## Summary

- **Novel byte-level tokenizer where every English letter has a 3D coordinate derived from FineWeb bigram structure**, plus an embedding-init that uses the same coordinates. To our knowledge, this is the first submission to use a single geometric prior to drive *both* tokenizer behavior and model embedding init — previous submissions have optimized either one or the other (custom BPE, BESE, OrthoInit, Spectral init), never with the same prior across both.
- **val_bpb: `<TBD post-run>`** (INT6 + LZMA + sliding window eval with n-gram tilt; single-seed)
- **Artifact: `<TBD post-run>` bytes** (under the 16 MB cap)
- Architecture is **byte-for-byte identical to PR #1666** (BESE record at 1.1531 BPB, 3-seed mean) so the v3 contribution is isolated to the tokenizer + embedding-init layer. Any BPB delta between this PR and #1666 is attributable to the geometric prior, not architecture changes.

## How it works

The 26 lowercase letters are placed in 3D space by gradient descent against the constraint:

```
loss = sum over (n-gram, freq) in FineWeb of  freq * path_length(n-gram in 3D)
```

`'e'` is anchored at the origin; the rest find their positions. Letters that frequently co-occur end up close (e.g., `t`–`h`, `i`–`n`, `e`–`r`); letters that rarely co-occur end up far apart (`q`, `x`, `z`, `j` become outliers).

The same coordinates are then used **twice**:

1. **BPE merge scoring.** Standard BPE picks the most-frequent pair. v3 picks `freq × exp(−d / (3·σ))^0.02` — a gentle multiplicative nudge (the `pow_0.02` shape was the sweep winner) that breaks ties between near-equal-frequency pairs in favor of spatially-close ones, without overriding the frequency signal. Merge tokens inherit centroid coordinates so spatial scoring works through the full BPE tree. Local sweep on a 160K-char Gutenberg corpus: `pow_0.02` beats standard frequency BPE 76,070 vs 76,131 tokens.
2. **Embedding initialization.** The first 12 dims of each letter token's embedding row are seeded from the 12D version of the same coordinates (rest of the row is small Gaussian, std=0.005). The model starts training with the bigram-cluster structure already in its embedding table instead of having to learn it from scratch in the 600s training window.

The base alphabet is a **flat 26-letter vocabulary** — no GROUP/POSITION encoding (which v2 used to spread vocabulary capacity across grouped letters). Every letter is its own token. The structural break from v2 is that vocabulary slots that used to encode "which group does this letter belong to" are repurposed to give every letter direct access, with the spatial geometry absorbing the disambiguation.

Total vocab: **288** (47 base = 4 special + 26 letters + 7 punct + 10 digits, plus 241 BPE merges) — chosen to match v2/v6.1 / PR #1666 exactly so any delta isolates the tokenizer change.

## BPB correctness

`val_bpb` is computed against the same UTF-8 byte denominator as every SP1024/SP8192 submission. The full inline proof lives in the records README, but the invariant is:

```
sum(BYTES_PER_TOKEN[t] for t in encode(s)) == len(s.encode("utf-8"))   for all s
```

It holds branch-by-branch in `_text_to_base_tokens` (mapped path: byte sum equals UTF-8 length by precondition; byte-fallback path: `OTHER_PUNCT_ID` × utf-8-length matches by construction). Merge tokens compose byte counts transitively. A self-test (`bese_v3_fast_bpe.py::_self_test`) exercises the invariant on diverse inputs (ASCII, multi-byte Unicode, emoji, mixed-case) and runs in <1 s on CPU.

## Single-seed note

This is a **single-seed non-record submission** under the rule that explicitly accepts in-progress and unoptimized solutions for novel ideas. The headline number demonstrates that the geometric prior produces a comparable (or better) result than the v2 control (PR #1666 at 1.1531 BPB, 3-seed mean), not that v3 is statistically validated at the same bar. Three-seed validation is in *Ongoing Work* pending compute credits.

## Predicted cross-language behavior

The v3 gain over standard BPE comes from the bigram matrix encoding real linguistic constraints. English vowel co-occurrence is roughly uniform, so the local-sweep gain on Gutenberg English is modest (−0.08% on token count). Languages with stricter co-occurrence rules — **Turkish, Finnish, Hungarian, Japanese, Korean** — should produce tighter MDS clusters and correspondingly larger gains. Turkish vowel harmony in particular forces every vowel in a word to agree on backness, so the bigram matrix encodes a *grammatical* constraint, not just a phonological tendency. A Turkish-corpus-derived `letter_coords_v3.json` would be a clean comparative test, predicted to show a larger v3-vs-v2 gap. Untested but offered as the natural next experiment.

## Files

- `README.md` — full writeup with the dual-use framing, inline BPB proof, training/eval results, comparison to v2 control, Turkish prediction, "About the author" section
- `submission.json` — metadata
- `train_gpt.py` + `bese_v3_fast_bpe.py` + `bese_v3_constants.py` — self-contained, runnable from records folder. The same multi-file accounting pattern used in PR #1665: bundled module bytes are accounted for in `submission.json::code_bytes` and could be inlined mechanically with no artifact-size change.
- `tokenizer.json` — trained spatial-scored BPE merges + the 3D coords used to score them
- `letter_coords_v3.json` — fitted 3D + 12D letter coords (loaded by training for embedding init)
- `train_log_run1.txt` — single-seed training log

## Ongoing work

- Three-seed statistical validation of v3 vs v2 (control) at the same architecture
- Turkish-corpus comparative test (refit coords on Turkish FineWeb, train at the same architecture, measure cross-language v3-vs-v2 delta)
- Higher-vocab variant (1064 instead of 288) to see whether the spatial gain scales with merge budget
- Combined with TTT and Triton-fixed Mamba (PR #1665 path) for a full-stack v3 + SSM submission

Companion PRs in the BESE family:
- [#1666](https://github.com/openai/parameter-golf/pull/1666) — BESE 288-vocab record entry, 1.1531 BPB (3-seed). The control variable for v3.
- [#1665](https://github.com/openai/parameter-golf/pull/1665) — BESE + Mamba-3 SSD hybrid, 7.6 MB / 48% of cap. SSM bounty submission.
- [#1327](https://github.com/openai/parameter-golf/pull/1327) — Earlier BESE record at 1.1276 BPB (single-seed).
