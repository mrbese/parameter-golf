# BESE v4 — Model Dim 448 Run Findings
**Date:** April 13, 2026  
**Pod:** RunPod 8xH100 SXM  
**Branch:** bese-v4  
**Log:** `/workspace/run_v3_448.log`

---

## Summary

Second full training run, focused on getting the model under the 16 MB submission limit. Reduced `model_dim` from 512 → 448 to fit INT6+LZMA size. **Succeeded on size (14.03 MB)** but paid a notable BPB cost: **1.1637 vs v3's 1.1394** — a regression of +0.024 BPB.

**Conclusion:** 448-dim was too aggressive a cut. Next target should be 512-dim + 10 layers, which loses less capacity and may still fit under 16 MB.

---

## What Changed vs v3

| Setting | v3 | v4 |
|---------|----|----|
| model_dim | 512 | **448** |
| num_layers | 11 | 11 |
| vocab | 4040 | 4040 |
| shards | re-exported | reused (--skip-decode) |
| BPE | reused | reused |

---

## Failed Attempt: model_dim=480

First attempt used `--model-dim 480`. Crashed immediately with:

```
RuntimeError: mha_fwd, head_size should be a multiple of 8
```

Flash Attention (Hopper) requires `head_dim % 8 == 0`. With 8 heads:
- `480 / 8 = 60` — not a multiple of 8 ✗
- `448 / 8 = 56` — multiple of 8 ✓ (56 = 7×8)
- `512 / 8 = 64` — multiple of 8 ✓

**Rule:** `model_dim` must be divisible by 64 when using 8 attention heads.

---

## Training Config

```
VOCAB_SIZE=4040
NUM_LAYERS=11
MODEL_DIM=448
MLP_MULT=3
NUM_GPUS=8
MAX_WALLCLOCK_SECONDS=600
```

- **Model params:** 22,567,388 (~22.6M vs v3's 28.9M — saved 6.3M params)
- **Embeddings:** tied (`tie_embeddings:True`)
- **Attention:** GQA, 8 heads / 4 KV heads, head_dim=56
- **XSA:** last 4 layers (7, 8, 9, 10)
- **Batch tokens:** 786,432 per step (seq_len=2048, 8 GPUs)
- **SWA start:** step 5550
- **Late QAT enabled:** step 5733, scale=0.1498
- **Steps completed:** 6313 / 20000 (wall-capped at 599.9s)
- **Peak GPU memory:** 19,398 MiB allocated / 20,130 MiB reserved

---

## Val BPB Progression (during training)

| Step | Val BPB | v3 BPB | Delta |
|------|---------|--------|-------|
| 0 | 3.6115 | 3.6123 | — |
| 500 | 1.4154 | 1.3937 | +0.022 |
| 1000 | 1.3352 | 1.3148 | +0.020 |
| 1500 | 1.3043 | 1.2841 | +0.020 |
| 2000 | 1.2724 | 1.2528 | +0.020 |
| 2500 | 1.2570 | 1.2359 | +0.021 |
| 3000 | 1.2450 | 1.2239 | +0.021 |
| 3500 | 1.2355 | 1.2139 | +0.022 |
| 4000 | 1.2247 | 1.2028 | +0.022 |
| 4500 | 1.2153 | 1.1927 | +0.023 |
| 5000 | 1.2036 | 1.1810 | +0.023 |
| 5500 | 1.1910 | 1.1679 | +0.023 |
| 6000 | 1.1768 | 1.1531 | +0.024 |
| 6313 | 1.1701 | 1.1464 | +0.024 |

The gap is consistent (~0.02-0.024 BPB) and slightly widening — smaller model capacity hurts more as training progresses.

---

## Final Eval Results

| Metric | v4 (448d) | v3 (512d) |
|--------|-----------|-----------|
| Post-EMA val BPB (pre-quant) | 1.1691 | 1.1454 |
| INT6 roundtrip BPB | 1.1810 | 1.1566 |
| INT6 sliding window BPB (stride=64) | **1.1637** | **1.1394** |
| **Current SOTA** | **1.1147** | **1.1147** |
| **Gap to SOTA** | **+0.049 BPB** | **+0.025 BPB** |

---

## Size Results

| Item | v4 (448d) | v3 (512d) |
|------|-----------|-----------|
| Serialized model (FP32) | 85,109,686 bytes (85.1 MB) | 110,018,998 bytes |
| Code size | 99,571 bytes (~97 KB) | 99,571 bytes |
| Model INT6+LZMA | **13,926,488 bytes (13.93 MB)** | 17,550,160 bytes (17.55 MB) |
| **Total submission** | **14,026,059 bytes (14.03 MB) ✓** | 17,649,731 bytes (17.65 MB) ✗ |
| **Limit** | **16,000,000 bytes** | **16,000,000 bytes** |
| **Headroom** | **+1.97 MB under limit** | −1.65 MB over limit |

**Size is solved.** 14.03 MB with 1.97 MB to spare. But we over-corrected — 6.3M fewer params was too much.

---

## Key Observations

1. **Size solved, BPB regressed.** 448 dim gets us comfortably under 16 MB (14.03 MB) but costs +0.024 BPB vs v3. That's too much to pay.

2. **Consistent ~0.02 gap throughout training.** The gap between 448 and 512 dim appeared immediately at step 500 and held steady. This is pure capacity loss, not a warm-up artifact.

3. **1.97 MB of unused budget.** With 14.03 MB we have room to add back capacity. The ideal target is a model that uses ~15.5-15.9 MB.

4. **480 dim is off the table** (Flash Attention head_dim constraint). The only valid options with 8 heads are multiples of 64: 384, 448, 512, 576...

5. **Step time improved:** 110ms early → 95ms at end. Same JIT/CUDA graph warmup pattern as v3. Similar step count (6313 vs 6302).

6. **SWA and late QAT timings identical** (steps 5550 and 5733) — the architecture triggers these at fixed thresholds independent of model_dim.

---

## Next Run Strategy

The right fix is **512 dim + 10 layers** (not 448 dim + 11 layers):

| Approach | Estimated params | Estimated size | BPB impact |
|----------|-----------------|----------------|------------|
| v3: 512d × 11L | 28.9M | 17.65 MB ✗ | baseline |
| v4: 448d × 11L | 22.6M | 14.03 MB ✓ | +0.024 |
| v5: 512d × 10L | ~26.5M | ~16.1 MB (~±) | ~+0.005 |
| v5b: 512d × 10L + smaller vocab | ~25.5M | ~15.4 MB ✓ | ~+0.005 |

**Recommendation for v5:** `--model-dim 512 --num-layers 10 --skip-decode`

- Saves 1 full transformer layer (~2.4M params)
- Estimated INT6+LZMA savings: ~1.43 MB → lands at ~16.1 MB (borderline)
- If still over: reduce MLP mult 3→2.5 or trim vocab slightly
- BPB expected: ~1.142-1.145 (much better than v4's 1.164)
- Can reuse existing shards — training only, ~11 min

**Flash Attention constraint reminder:** `model_dim % 64 == 0` required. Valid dims: 384, 448, 512 (for 8 heads).

---

## Infrastructure Notes

- SSH: `ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no root@63.141.33.78 -p 22144`
- All shards reused from v3: `/workspace/bese_shards_v3/` (15 train + 1 val, persistent)
- Tokenizer reused: `/workspace/bese/tokenizers/bese_bpe_4000.json`
- Run log: `/workspace/run_v3_448.log`
- Wall time: 14.8 min (training + eval, no data prep)
