# BESE v2 Data Prep — Cowork Briefing

**Date:** April 4, 2026
**Branch:** `optimize-bese-alphabet` (already committed + pushed)
**Goal:** Retrain BPE merges and re-encode all shards for the BESE v2 alphabet

---

## What Changed (v1 → v2)

Promoted 3 letters from 2-token group encoding to 1-token single-letter:
- **h** (6.1% frequency) — was more frequent than r which already had its own token
- **d** (4.3%)
- **l** (4.0%)

Remaining 15 letters regrouped into 4 groups (was 5), positions reordered by frequency.

```
v2 Layout (BASE_VOCAB_SIZE = 40):
  Tokens 0-3:   PAD, BOS, EOS, UNK
  Tokens 4-14:  Single letters: e t a o i n s r h d l  (11 letters)
  Tokens 15-18: Groups: [c,w,v,j] [u,f,b,z] [m,y,k,x] [g,p,q]
  Tokens 19-22: Position tokens P1-P4
  Tokens 23-29: Space, period, comma, newline, ?, quote, other_punct
  Tokens 30-39: Digits 0-9
```

**Impact:** ~15.6% fewer tokens in the base encoding. BPE merges go from 250 → 248 (to keep vocab size at 288).

---

## What Needs to Happen

### Step 1: Spin up a CPU pod
- 32-vCPU pod on RunPod (~$0.20/hr, ~50 min total)
- Needs: Python 3.10+, numpy, sentencepiece
- The original SP-1024 shards are at `/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/` (if using the same network volume) or need to be synced

### Step 2: Upload updated tokenizer code
SCP these files to the pod:
```
tokenizer/bese_constants.py       ← v2 alphabet (40 base tokens)
tokenizer/bese_bpe_tokenizer.py   ← unchanged BPE logic
tokenizer/bese_fast_bpe.py        ← fast encoder (updated for v2)
scripts/runpod_all_in_one.py      ← needs NUM_MERGES changed to 248
scripts/export_shards.py          ← shard exporter
scripts/train_bpe_jsonl.py        ← BPE trainer
```

### Step 3: Run the pipeline
The all-in-one script does it in 3 steps:
1. **Decode** SP-1024 binary shards back to text (needs `sentencepiece`)
2. **Train** 248 BPE merges on decoded text (produces `bese_bpe_248.json`)
3. **Export** binary shards encoded with the new tokenizer

**Important:** Change `NUM_MERGES = 250` → `NUM_MERGES = 248` in `runpod_all_in_one.py`

### Step 4: Get shards to training pod
Either sync to RunPod network volume or SCP to the GPU pod when ready.

---

## Existing Scripts

| Script | What it does | Location |
|--------|-------------|----------|
| `scripts/runpod_all_in_one.py` | Full pipeline: decode → train BPE → export shards | Runs on pod |
| `scripts/train_bpe_jsonl.py` | Standalone BPE training from JSONL | Can run locally if you have JSONL |
| `scripts/export_shards.py` | Standalone shard export | Needs tokenizer JSON + JSONL input |

The all-in-one expects these pod paths:
```
/workspace/parameter-golf/          ← upstream repo with SP shards
/workspace/bese/                    ← our code
/workspace/bese/tokenizer/          ← tokenizer source
/workspace/bese/tokenizers/         ← output tokenizer JSON
/workspace/bese_shards/             ← output binary shards
```

---

## Key Notes

- Old BPE merge files (like `tokenizers/bese_bpe_250.json`) are INVALID for v2 — different token IDs
- Last time data prep took ~50 minutes on 32-vCPU
- The `runpod_all_in_one.py` uses `MAX_DOCS = 10,000` for BPE training — the full pipeline uses 50K+ docs via `export_shards.py`
- SSH key for RunPod: `~/.ssh/id_runpod`
- Previous CPU pod SSH pattern: `ssh root@<ip> -p <port> -i ~/.ssh/id_runpod`

---

## After Data Prep

Once shards are ready, training command on 8xH100:
```bash
DATA_PATH=data/shards VOCAB_SIZE=288 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Target: beat the v1 result of 1.1276 BPB. The 15.6% sequence length reduction should help significantly.
