# BESE v5 Run Findings — 2026-04-14

## Hardware
- Pod: 8x H100 80GB HBM3, 208 CPUs
- RunPod network volume (MFS filesystem)

## Run Config
- Vocab: 288 (40 base + 248 BPE merges)
- Layers: 11, dim: 512, MLP mult: 3, heads: 8, KV heads: 4
- Depth recurrence: layers 3-5, 3 loops, activated after 35.6% progress
- Parallel residuals: start layer 7
- SWA, late QAT (scale 0.1499), EMA decay 0.9965
- Warmdown iters: 5000
- SLOT eval: enabled (but not reached due to crash)
- N-gram tilt: enabled (beta 0.3, max-n 4)
- Data: 80 SP1024 shards → decoded → filtered → BESE re-encoded (44 shards output)

## Phase 0 Timing (untimed)
| Step | Time |
|------|------|
| 0.1 SP decode (80 workers) | 187s (3.1 min) |
| 0.2 Quality filter (6.29M → 6.24M docs, single-threaded) | 389s (6.5 min) |
| 0.3 BPE train 248 merges on 50K docs | 740s (12.3 min) — slow due to memory pressure |
| 0.4 Curriculum sort | ~300s est. |
| 0.5 Encode (128 pre-spawned workers, 44 shards) | 404s |
| 0.6 N-gram table build (pure Python, max-n 4) | ~600s est. |
| **Phase 0 total** | **~40 min** |

## Training Results (600s cap)
| Checkpoint | val_bpb |
|------------|---------|
| step 500 | 1.4692 |
| step 1000 | 1.3679 |
| step 1500 | 1.3274 |
| step 2000 | 1.2691 |
| step 2500 | 1.2341 |
| step 3000 | 1.2075 |
| step 3500 | 1.1819 |
| **step 3688 (600s stop)** | **1.1757** |
| post-EMA | 1.1765 |

Depth recurrence activated at step 1654 (35.6% progress).
SWA started at step 2800. Late QAT enabled at step 3036.

## Eval: CRASHED
**Error:** `torch._dynamo.exc.FailOnRecompileLimitHit` on all 8 ranks during INT6 roundtrip eval.

**Root cause:** `compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)` on the freshly INT6-quantized model. The RoPE module's `_cos_cached` is None on the new model instance, triggering recompilation on every sequence length. Default cache limit is 8 — exceeded immediately.

**Fix applied (commit 33873d7):**
```python
torch._dynamo.config.cache_size_limit = 64
compiled_eval = torch.compile(eval_model, dynamic=True, fullgraph=False)
```

No INT6, SLOT, or sliding window BPB obtained this run.

## Artifact Size: FAIL
| Component | Size |
|-----------|------|
| Serialized model (raw) | 105,216,438 bytes (100 MB) |
| Code | 109,543 bytes |
| N-gram table (raw) | 10,905,476 bytes (10.4 MB) |
| **INT6+LZMA compressed total** | **23,284,219 bytes (23.3 MB)** |
| Limit | 16,000,000 bytes (16 MB) |
| **Over by** | **7.3 MB** |

**Main culprit:** N-gram table is 10.9 MB raw. Even with LZMA it doesn't compress well enough. INT6+LZMA model alone is ~12.4 MB, which would fit. N-gram table needs to be either removed or drastically reduced.

## Issues to Fix Before Next Run
1. **Eval crash** — fixed in 33873d7 (dynamic=True, cache_size_limit=64)
2. **N-gram table size** — 10.9 MB is too large. Options:
   - Reduce `--max-n` from 4 to 3
   - Reduce `--top-k` (currently 1 = only unigrams? check script)
   - Drop n-gram tilt entirely (saves ~10 MB, costs ~0.01 BPB estimate)
3. **Phase 0 speed** — BPE training took 12.3 min (vs ~40s expected). Investigate.
   Likely cause: memory pressure from 34 GB of decoded docs in RAM during BPE on 50K sample.
   Consider clearing/reducing in-memory data before BPE step.

## Comparison to Prior Runs
| Run | BPB | Artifact | Notes |
|-----|-----|----------|-------|
| v3 | 1.1394 | 17.65 MB | SP1024 tokenizer, size too large |
| v4 | 1.1637 | 14.03 MB | Reduced model, size OK |
| **v5 (this run)** | **1.1757** | **23.3 MB** | BESE 288 vocab, eval crashed, size too large |

v5 BPB is worse than v3/v4 — BESE 288-vocab tokenizer is generating longer token sequences per byte, hurting compression efficiency relative to the larger SP1024 vocab. Need to investigate whether the tokenizer choice is helping or hurting net BPB.
