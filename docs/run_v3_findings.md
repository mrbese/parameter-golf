# BESE v3 — First Full Run Findings
**Date:** April 13, 2026  
**Pod:** RunPod 8xH100 SXM (bare_white_chameleon, asv2tzoshecak4)  
**Branch:** bese-v3

---

## Summary

First end-to-end run of the BESE tokenizer with the full winning architecture (Muon param banks, SWA, late QAT, EMA, INT6+LZMA). The model achieved **1.1394 BPB** — only **0.025 above SOTA (1.1147)** — but failed the size check at **17.65 MB** vs the 16 MB submission limit.

---

## Pipeline Timing

| Step | Duration | Notes |
|------|----------|-------|
| SP shard decode (80 shards, 80 workers) | ~3 min | 6,293,016 train docs, 50,000 val docs |
| BPE training | skipped | Reused existing `bese_bpe_4000.json` |
| BESE shard export (64 workers) | ~34 min | Val: 90s. Train: ~2026s at 770 docs/s |
| Training (8xH100) | 10 min (600s cap) | 6302 steps |
| Eval (INT6 + sliding window) | ~5 min | INT6 roundtrip + 64-stride sliding window |
| **Total wall time** | **54.2 min** | |

---

## Data

- **Source:** 80 SP1024 shards × 100M tokens = 8B tokens decoded
- **Decoded docs:** 6,293,016 train (~78K docs/shard) + 50,000 val
- **Capped at:** 1,562,500 train docs (~500M tokens budget)
- **BESE train shards written:** 15 × ~100M tokens + 1 partial (59.4M) = ~1.559B BESE tokens
- **BESE val shard:** 45,533,991 tokens (vs 62,021,846 SP tokens — BESE is 26.6% more efficient on val set)
- **Shard location:** `/workspace/bese_shards_v3/` (persistent network volume, safe across restarts)

---

## Tokenizer

- **File:** `/workspace/bese/tokenizers/bese_bpe_4000.json`
- **Vocab:** 4040 tokens (40 BESE base chars + 4000 BPE merges)
- **Compression vs SP1024 val:** 45.5M vs 62.0M tokens = **26.6% fewer tokens** — better compression

---

## Training Config

```
VOCAB_SIZE=4040
NUM_LAYERS=11
MODEL_DIM=512
MLP_MULT=3
NUM_GPUS=8
MAX_WALLCLOCK_SECONDS=600
```

- **Model params:** 28,923,996 (~28.9M)
- **Embeddings:** tied (`tie_embeddings:True`) — embed and lm_head share weights
- **Attention:** GQA, 8 heads / 4 KV heads
- **XSA:** last 4 layers (7, 8, 9, 10)
- **Batch tokens:** 786,432 per step (seq_len=2048, 8 GPUs)
- **Optimizer:** Muon with param banks; embed_lr=0.035, matrix_lr=0.025, scalar_lr=0.025
- **SWA start:** step 5550
- **Late QAT enabled:** step 5732, scale=0.1499
- **Steps completed:** 6302 / 20000 (wall-capped at 599.9s)
- **Peak GPU memory:** 22,081 MiB allocated / 22,618 MiB reserved

---

## Val BPB Progression (during training)

| Step | Val BPB | Wall time |
|------|---------|-----------|
| 0 | 3.6123 | 0s |
| 500 | 1.3937 | 55s |
| 1000 | 1.3148 | 116s |
| 1500 | 1.2841 | 173s |
| 2000 | 1.2528 | 226s |
| 2500 | 1.2359 | 270s |
| 3000 | 1.2239 | 313s |
| 3500 | 1.2139 | 356s |
| 4000 | 1.2028 | 399s |
| 4500 | 1.1927 | 443s |
| 5000 | 1.1810 | 486s |
| 5500 | 1.1679 | 529s |
| 6000 | 1.1531 | 573s |
| 6302 | 1.1464 | 600s ← stopped |

**Loss was still clearly decreasing** — had the wall clock not hit, scores would continue to improve.

---

## Final Eval Results

| Metric | Value |
|--------|-------|
| Post-EMA val BPB (pre-quant) | **1.1454** |
| INT6 roundtrip val BPB | 1.1566 |
| INT6 sliding window BPB (stride=64) | **1.1394** ← submission score |
| INT8 zlib roundtrip BPB | 1.1394 (same as sliding) |
| **Current SOTA** | **1.1147** |
| **Gap to SOTA** | **+0.0247 BPB** |

---

## Size Problem

| Item | Size |
|------|------|
| Serialized model (FP32) | 110,018,998 bytes (110 MB) |
| Code size | 99,571 bytes (~97 KB) |
| Model INT6+LZMA | 17,550,160 bytes (17.55 MB) |
| **Total submission** | **17,649,731 bytes (17.65 MB)** |
| **Limit** | **16,000,000 bytes (16 MB)** |
| **Overage** | **1,649,731 bytes (1.65 MB)** |

**Root cause:** BESE vocab is 4040 vs SP1024's 1024 tokens. Even with tied embeddings, the embedding table is 4040 × 512 = 2,068,480 params vs 1024 × 512 = 524,288 — a difference of 1,544,192 params. At INT6 (0.75 bytes/param) before LZMA, that's ~1.16 MB extra. After LZMA compression overhead this accounts for most of the overage.

---

## Fix Options (in priority order)

### Option 1: Reduce model_dim 512 → 480 (recommended, no re-tokenization)
- Saves ~4M params across all transformer layers
- INT6+LZMA savings estimate: ~1.8 MB — enough to get under 16 MB
- BPB impact: minor (~0.005-0.010 increase), still competitive
- Can reuse existing BESE shards — just change `MODEL_DIM=480` and re-train

### Option 2: Reduce num_layers 11 → 10
- Saves ~3M params
- Similar savings, slightly more BPB impact than option 1

### Option 3: Reduce BPE merges (1000 merges → 1040 vocab)
- Nearly eliminates vocab overhead (1040 vs 1024 ≈ same size as SOTA)
- Requires re-running BPE training and re-exporting all shards (~40 min)
- May hurt BPB slightly (fewer merges = less compression)

### Option 4: Reduce num_merges to ~2500 (2540 vocab)
- Cuts vocab overhead roughly in half
- Still much better compression than SP1024
- Requires re-tokenization

---

## Key Observations

1. **BESE compression works.** Val set: 45.5M BESE tokens vs 62M SP tokens = 26.6% fewer tokens for the same text. This directly means the model sees more text per step.

2. **BPB is competitive.** 1.1394 vs SOTA 1.1147 — gap is only 0.025. With the size fix reducing model capacity slightly, we may land at ~1.14-1.15, still very close.

3. **Training was still improving at cutoff.** BPB was dropping ~0.007/500 steps near the end. More steps = better score.

4. **Embeddings are already tied** (`tie_embeddings:True` confirmed in logs). So the vocab overhead is from a single embedding matrix, not two.

5. **Step speed improved over training:** 146ms/step early → 95ms/step at end (JIT warmup + CUDA graph effects). This means ~6300 steps in 600s is roughly the right budget.

6. **SWA kicked in at step 5550, late QAT at step 5732** — both in the last ~750 steps. These are the upstream's final-phase tricks. They worked: val BPB went from 1.1679 (step 5500) to 1.1464 at stop, then 1.1394 post-sliding-window eval.

---

## Next Steps

1. **Change MODEL_DIM to 480** in `runpod_v3.py` default (or pass `--model-dim 480`)
2. **Rerun training only** — shards are already on the persistent volume, can `--skip-decode`
3. **Verify size < 16MB** before claiming a submission
4. If still competitive (BPB < 1.15), submit to leaderboard

---

## Infrastructure Notes

- `/workspace` is a persistent RunPod network volume (`mfs#ca-mtl-1.runpod.net`) — all shards, tokenizer, and logs survive pod restarts
- SSH key: `~/.runpod/ssh/RunPod-Key-Go`
- Pod ID: `asv2tzoshecak4` (may change if pod is stopped/restarted)
- Run log saved at: `/workspace/run_v3.log`
- tmux session: `bese`
