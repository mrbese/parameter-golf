# Plan: v5 End-to-End — Data Pipeline + Artifact + Submission

## Context

The v5 code (architecture, eval stack, hyperparams) is committed and reviewed on `bese-v5`. All 8 agent-found bugs are fixed in code. But two things block an end-to-end run → leaderboard submission:

1. **Data pipeline broken:** `runpod_v5.py` Phase 0 expects a JSONL file that doesn't exist on the pod
2. **Submission not assembled:** The submission file (`submission/train_gpt.py`) is still the v2 version — needs v5 code inlined
3. **N-gram prior not in artifact:** The compressed n-gram table needs to be bundled inside `final_model.int6.ptz`

## Part 1: Fix Data Pipeline — In-Memory Decode

### Modify `scripts/runpod_v5.py` — rewrite `phase0_data_prep()`

Replace the JSONL-based flow with v3's proven in-memory decode approach:

```
Step 1: Parallel decode SP1024 shards → text strings in memory
        - Reuse v3's _decode_shard() (multiprocessing.Pool, 80 workers)
        - Source: /workspace/parameter-golf/data/datasets/fineweb10B_sp1024/*.bin
        - SP model: /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
        - ~6M docs, ~6GB RAM (pod has 2TB)

Step 2: Quality filter in-memory
        - is_high_value(): word count ≥ 50, vocab richness ≥ 0.25, no boilerplate

Step 3: Train BESE BPE (248 merges) directly on text list
        - Call train_bpe_merges_fast() from bese_fast_bpe module
        - Save to /workspace/bese/tokenizers/bese_bpe_248_v5.json
        - Skip if tokenizer already exists

Step 4: Curriculum sort in-memory (sort by difficulty_score)

Step 5: Export BESE shards from in-memory texts
        - Encode with FastBESEBPETokenizer, write binary shards
        - Output: /workspace/bese_shards_v5/

Step 6: Build n-gram table from first shard
        - subprocess: python scripts/build_ngram_table.py
```

Add to top of file:
```python
SP_MODEL = PG_DIR / "data/tokenizers/fineweb_1024_bpe.model"
SP_SHARD_DIR = PG_DIR / "data/datasets/fineweb10B_sp1024"
```

Remove: `DECODED_JSONL`, `CURRICULUM_OUTPUT` constants and JSONL references.

**Time estimate:** ~25-30 min (vs 1+ hours with JSONL)

### Code to reuse

- `_decode_shard()` from `scripts/runpod_v3.py:47-73` — parallel SP decode
- `is_high_value()` from `scripts/export_shards.py:41-55` — quality filter
- `difficulty_score()` from `scripts/curriculum_sort.py:14-20` — curriculum sort
- `train_bpe_merges_fast()` from `tokenizer/bese_fast_bpe.py` — BPE training
- `FastBESEBPETokenizer` from `tokenizer/bese_fast_bpe.py` — encoding
- `write_shard()` from `scripts/export_shards.py:41-54` — shard binary format

---

## Part 2: Bundle N-gram Prior in Artifact

### Modify `integration/train_gpt_bese.py` — artifact save (~line 2164)

Currently saves: `{"w": quant_result, "m": quant_meta}`

Change to also bundle the n-gram table if it exists:
```python
save_dict = {"w": quant_result, "m": quant_meta}
ngram_path = os.environ.get("NGRAM_PRIOR_PATH", "")
if ngram_path and os.path.exists(ngram_path):
    with open(ngram_path, "rb") as nf:
        save_dict["ngram"] = nf.read()
torch.save(save_dict, quant_buf)
```

Update the load side (artifact reload + eval_model construction) to extract and write the n-gram table to a temp file for `NgramTilt.load_prior()`.

Also update the size logging to include ngram bytes.

---

## Part 3: Assemble Submission File

### Create `scripts/assemble_submission_v5.py`

This script generates `submission/train_gpt.py` by:

1. Reading `integration/train_gpt_bese.py` as the base
2. Inlining the BESE tokenizer classes from `tokenizer/bese_fast_bpe.py`
3. Inlining the tokenizer JSON data (merges, base vocab) as a Python constant
4. Removing multi-file imports, replacing with inlined code
5. Stripping MTP heads (not needed for submission, saves code size)
6. Adding the artifact load path that extracts n-gram table

The output is a single self-contained Python file that the competition eval server can run.

### Update `submission/submission.json`

Update metadata with v5 BPB, technique list, artifact size.

---

## Files to Modify

| File | Change |
|------|--------|
| `scripts/runpod_v5.py` | Rewrite `phase0_data_prep()` with in-memory decode |
| `integration/train_gpt_bese.py` | Bundle n-gram table in artifact (~line 2164) |
| `scripts/assemble_submission_v5.py` | **New** — generate single-file submission |
| `submission/train_gpt.py` | **Regenerated** by assembly script |
| `submission/submission.json` | Updated metadata after run |

## Files NOT Modified

- `scripts/export_shards.py` — keep as standalone tool
- `scripts/curriculum_sort.py` — keep as standalone tool
- `scripts/build_ngram_table.py` — called as subprocess, unchanged

---

## End-to-End Flow

```
1. python scripts/runpod_v5.py          # One command on RunPod
   ├── Phase 0: Decode → Filter → BPE → Sort → Shards → N-gram  (25-30 min)
   ├── Phase 1: 8-GPU training, 600s cap                         (11 min)
   ├── Phase 2: SLOT + tilt eval, quantize, compress, save       (11 min)  
   └── Phase 3: Size check + summary                             (1 min)
   
   Output: final_model.int6.ptz (with bundled n-gram table)
   Output: BPB number in logs

2. If BPB beats SOTA:
   python scripts/assemble_submission_v5.py                      (local, 10 sec)
   
   Output: submission/train_gpt.py (single file, all code inlined)

3. Submit to leaderboard (manual upload)
```

---

## Verification

1. SSH to pod: `cd /workspace/bese && python scripts/runpod_v5.py`
2. Phase 0 completes in ~25-30 min, shards in `/workspace/bese_shards_v5/`
3. Training completes within 600s, logs show step count and loss
4. Eval logs show `val_bpb` with SLOT + tilt
5. `final_model.int6.ptz` exists and < 16 MB
6. Locally: `python scripts/assemble_submission_v5.py` produces valid `submission/train_gpt.py`
7. `python -c "import ast; ast.parse(open('submission/train_gpt.py').read())"` passes
