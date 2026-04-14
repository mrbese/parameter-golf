# BESE v5.1 Run Findings — 2026-04-14

## Hardware
- Pod: 8x H100 80GB HBM3, 208 CPUs
- RunPod network volume (MFS filesystem)

## Run Config
- Vocab: 288 (40 base + 248 BPE merges)
- Layers: 11, dim: 512, MLP mult: 3, heads: 8, KV heads: 4
- Depth recurrence: layers 3-5, **2 loops** (v5: 3), activated after **50%** progress (v5: 35.6%)
- Parallel residuals: start layer 7
- SWA, late QAT (scale 0.1499), EMA decay 0.9965
- Warmdown iters: 5000
- **N-gram tilt: disabled** (v5: enabled, was 10.9 MB, blew 16 MB budget)
- Data: reused v5 shards (`--skip-prep`) — 44 BESE-encoded shards

## Changes vs v5
| Parameter | v5 | v5.1 | Reason |
|-----------|-----|-------|--------|
| `DEPTH_RECURRENCE_LOOPS` | 3 | 2 | Recover ~1000 training steps from recurrence overhead |
| `DEPTH_RECURRENCE_ACTIVATION_FRAC` | 0.35 | 0.50 | Let model converge more before recurrence kicks in |
| `NGRAM_TILT_ENABLED` | 1 | 0 | N-gram table was 10.9 MB raw, blew 16 MB artifact limit |
| INT6 eval compile | `dynamic=False, fullgraph=True` → `dynamic=True, fullgraph=False` | (kept from v5 fix) | Dynamo recompile crash |

## Training Results (600s cap)
| Checkpoint | val_bpb |
|------------|---------|
| step 500 | 1.4749 |
| step 1000 | 1.3673 |
| step 1500 | 1.3207 |
| step 2000 | 1.2715 |
| step 2500 | 1.2353 |
| step 3000 | 1.2088 |
| **step 3474 (600s stop)** | **1.1897** |
| post-EMA | 1.1903 |

Depth recurrence activated at step 1781 (51.2% progress).
SWA started at step 2300. Late QAT enabled at step 2602.

## Eval: CRASHED (new error)
**Error:** `AttributeError: 'SymFloat' object has no attribute 'size'` on all 8 ranks during INT6 roundtrip eval.

**Root cause:** `torch.compile(eval_model, dynamic=True, fullgraph=False)` traces through the INT6-quantized model. The quantization scale tensors become `SymFloat` proxies under dynamic-shape tracing. PyTorch's `statically_known_true(sym_eq(val1.size(), val2.size()))` then calls `.size()` on a `SymFloat`, which has no such attribute.

**Fix applied (this session, not yet committed):**
```python
# v5.2: run INT6 eval in eager mode — torch.compile(dynamic=True) wraps INT6
# scale tensors as SymFloat proxies, causing AttributeError on .size() calls
q_val_loss, q_val_bpb = eval_val(
    args, eval_model, ...  # eval_model instead of compiled_eval
)
```
Remove `torch.compile` from INT6 eval entirely. Eager mode is sufficient — eval speed is not a bottleneck.

No INT6 or sliding window BPB obtained this run.

## Artifact Size: PASS
| Component | Size |
|-----------|------|
| Serialized model (raw) | 105,216,438 bytes (100 MB) |
| Code | 109,590 bytes |
| Model INT6+LZMA | 11,387,304 bytes (10.9 MB) |
| **Total INT6+LZMA** | **11,496,894 bytes (11.0 MB)** |
| Limit | 16,000,000 bytes (16 MB) |
| **Margin** | **4.5 MB under limit** |

Disabling n-gram tilt recovered ~10.9 MB. Artifact now comfortably under budget.

## BPB Regression vs v5
v5.1 BPB of 1.1897 is **0.014 worse** than v5's 1.1757. The config changes intended to recover training steps may have hurt:
- Reducing recurrence loops 3→2 weakens the recurrent depth signal
- Delaying activation to 50% means the recurrent layers have less time to specialize
- Net effect: fewer effective gradient updates through the recurrent path

## Issues to Fix Before Next Run
1. **INT6 eval crash** — fixed in this session: drop `torch.compile` from INT6 eval, run eager mode
2. **BPB regression** — need to reconsider recurrence config. Options:
   - Restore loops to 3, accept the step cost
   - Keep 2 loops but activate earlier (e.g. 40%)
   - Try asymmetric activation: activate earlier but ramp up loops gradually
3. **SLOT ruled non-compliant** — PR #675 closed April 10. L-BFGS was optimizing on scored tokens. All BPB improvements must come from raw training only.

## Comparison to Prior Runs
| Run | BPB | Artifact | Notes |
|-----|-----|----------|-------|
| v3 | 1.1394 | 17.65 MB | SP1024 tokenizer, size too large |
| v4 | 1.1637 | 14.03 MB | Reduced model, size OK |
| v5 | 1.1757 | 23.3 MB | BESE 288 vocab, size too large |
| **v5.1** | **1.1897** | **11.0 MB** | BESE 288 vocab, size OK, BPB regressed |

v5.1 is the first BESE run that fits the artifact budget. BPB needs improvement to beat v4 (1.1637). The gap to v3 (1.1394) is 0.05 BPB — significant.
