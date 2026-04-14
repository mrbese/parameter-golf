# v5.3 Execution Plan

## Goal

Beat v5.1 BPB (1.1897) and ideally v4 BPB (1.1637). All improvements from raw 600s training only — SLOT+L-BFGS ruled non-compliant (PR #675).

## Changes vs v5.1

| # | Change | File | Why |
|---|--------|------|-----|
| 1 | Restore `DEPTH_RECURRENCE_LOOPS` 2→3 | `scripts/runpod_v5.py` | v5.1's reduction caused 0.014 BPB regression. 100-run autopsy identifies depth recurrence as the single biggest architectural win. Don't trade it for steps. |
| 2 | Restore `DEPTH_RECURRENCE_ACTIVATION_FRAC` 0.50→0.35 | `scripts/runpod_v5.py` | v5.1's delay gave recurrent layers less time to specialize. Revert to v5's proven value. |
| 3 | Re-enable n-gram tilt with `max-n=3` | `scripts/runpod_v5.py` | max-n=4 raw table was 10.9 MB and blew 16 MB budget. max-n=3 estimated ~2-3 MB raw. Recovers ~0.01 BPB benefit. |
| 4 | Add `NGRAM_TILT_MAX_N=3` + `NGRAM_PRIOR_PATH` to TRAIN_ENV | `scripts/runpod_v5.py` | Pass n-gram config to training subprocess so the pre-built table is loaded and the tilt is applied during eval. |
| 5 | `torch.compile(zeropower_via_newtonschulz5)` | `integration/train_gpt_bese.py` | Fuses the 5-iteration bmm loop into a single CUDA kernel. Upstream has this; our submission didn't. Free step recovery. |
| 6 | Batched EMA via `_foreach_mul_` + `_foreach_add_` | `integration/train_gpt_bese.py` | Replaces Python for-loop over state dict with a single fused CUDA call. ~20 extra training steps free per 600s budget. |
| 7 | INT6 eval: drop `torch.compile`, run eager | `integration/train_gpt_bese.py` | Already applied (v5.2 fix). `torch.compile(dynamic=True)` wrapped INT6 scale tensors as SymFloat proxies → `AttributeError: 'SymFloat' has no attribute 'size'`. Eager mode is sufficient for eval. |

## Expected Outcomes

| Change | Expected BPB delta |
|--------|-------------------|
| Restore recurrence (3 loops, 35% activation) | −0.014 (recover v5.1 regression) |
| N-gram tilt max-n=3 | −0.005 to −0.010 |
| Batched EMA + compiled NS5 | −0.001 to −0.003 (via extra steps) |
| INT6 eval fix | 0 BPB (correctness fix, not a training change) |

Target: beat v5.1 (1.1897) by ≥0.02, aiming for ≤1.1750 (matches v5) or better.

## Artifact Size Budget

| Component | v5.1 | v5.3 estimate |
|-----------|------|---------------|
| Model INT6+LZMA | 10.9 MB | ~10.9 MB (unchanged arch) |
| N-gram table (max-n=3, raw) | 0 MB (disabled) | ~2–3 MB raw → ~0.3–0.5 MB LZMA |
| Code | ~0.1 MB | ~0.1 MB |
| **Total** | **11.0 MB** | **~11.3–11.5 MB** |
| Limit | 16.0 MB | 16.0 MB |
| Margin | 5.0 MB | ~4.5 MB |

If the n-gram table comes in larger than expected: reduce `--top-k` from 1 to 0 (disables counting, just stores presence) or lower `--max-tokens`.

## Files Changed

| File | Change |
|------|--------|
| `scripts/runpod_v5.py` | TRAIN_ENV: loops=3, frac=0.35, ngram enabled, NGRAM_TILT_MAX_N=3, NGRAM_PRIOR_PATH. Build cmd: `--max-n 3` |
| `integration/train_gpt_bese.py` | `torch.compile(zeropower_via_newtonschulz5)` after fn def; batched EMA with `_foreach_*`; INT6 eager eval (already applied) |

## Files NOT Changed

- Tokenizer (same BESE 288 vocab)
- Model arch (same 11-layer, dim=512, MLP×3, 8/4 GQA)
- Shards: reuse v5 shards with `--skip-prep`
- SWA, QAT, EMA decay: unchanged

## Run Command (on pod)

```bash
cd /workspace/bese
python scripts/runpod_v5.py --skip-prep --num-gpus 8 2>&1 | tee /workspace/run_v53.log
```

The `--skip-prep` flag reuses the v5 shards and tokenizer. Phase 0 will only rebuild the n-gram table (since it's now enabled and `max-n` changed).

## Risks

1. **N-gram table size**: max-n=3 estimated ~2-3 MB raw but depends on token distribution. The existing warning threshold in `runpod_v5.py` fires at 500 KB. If it exceeds ~4 MB LZMA, the total artifact will be tight. Check the table size log line before proceeding.
2. **NS5 compile overhead**: first-step compile adds ~30-60s warmup. This happens before the 600s clock starts (compile fires on first optimizer step, which is inside the timed region). If it adds >1 step of overhead, the gain may not materialize. Monitor step timing in logs.
3. **_foreach_* with non-float buffers**: if any model buffer is non-floating-point (unlikely given init), `_foreach_add_` will raise. The existing for-loop coerced everything to float. The new code also does `.float()` on model vals and ema_state was initialized as float, so this should be safe.
