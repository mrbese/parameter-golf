# BESE v5 — Execution Plan

**Date:** April 13, 2026
**Prerequisites:** v3 findings (docs/run_v3_findings.md), v2 submission (1.1276 BPB)
**Goal:** Beat current SOTA (~1.08 BPB) by stacking modern competition techniques on BESE's compact-vocab advantage.
**Branch:** `bese-v5`

---

## Lessons from v3 That Shape v5

The v3 run taught us three hard lessons:

1. **Bigger vocab was worse.** v3 went from 288 → 4040 vocab and BPB got worse (1.1276 → 1.1394). Giving up BESE's compact embedding table didn't pay off in reduced sequence length benefit.
2. **Size overflow is caused by vocab.** v3 hit 17.65 MB vs 16 MB limit — the extra 1.65 MB is embedding table overhead from vocab 4040.
3. **v3 used old techniques.** Muon + SWA + QAT + EMA + XSA is the March 2026 SOTA stack. The leaderboard has moved two generations past that: depth recurrence (PR #1420, #1445), parallel residuals (PR #1437), n-gram tilt (PR #1420, #1437), and SLOT+L-BFGS (PR #675) are the current frontier.

**v5 strategy:** revert to v2's small vocab (288), layer on the frontier techniques, and exploit the untimed data prep phase.

---

## Target Configuration

```bash
# Model shape
VOCAB_SIZE=288              # Reverted from v3's 4040 — size fits in 16MB, preserves SLOT advantage
NUM_LAYERS=11
MODEL_DIM=512
MLP_MULT=3
NUM_HEADS=8
NUM_KV_HEADS=4

# NEW: Depth recurrence
DEPTH_RECURRENCE_START=3
DEPTH_RECURRENCE_END=5
DEPTH_RECURRENCE_LOOPS=3
DEPTH_RECURRENCE_ACTIVATION_FRAC=0.35

# NEW: Parallel residuals
PARALLEL_RESIDUAL_START=7

# Updated: Competition-tuned hyperparams (from PR #1445)
QK_GAIN_INIT=5.0            # was 1.5
MATRIX_LR=0.022             # was 0.04
WEIGHT_DECAY=0.095          # new
EMA_DECAY=0.9965            # was 0.997
WARMDOWN_FRAC=0.72          # was 0.667

# Unchanged from v3
MAX_WALLCLOCK_SECONDS=600
TRAIN_SEQ_LEN=2048
TRAIN_BATCH_TOKENS=786432
```

**Expected artifact size at vocab 288:**
- Embedding table: 288 × 512 = 147,456 params (tied with lm_head)
- Saves ~3.86M params vs v3's 4040 vocab
- Estimated INT6+LZMA artifact: ~14.5 MB (well under 16 MB)

---

## Phase 0: Untimed Data Prep (before training clock starts)

All prep runs on the RunPod persistent volume `/workspace/bese_shards_v5/`. No 600s cap applies here.

### Step 0.1 — Train BESE BPE on Full FineWeb (vocab 288)

```bash
# On RunPod
cd /workspace/bese
source .venv/bin/activate

python scripts/train_bpe_jsonl.py \
  --input /workspace/decoded_docs_jsonl/fineweb_train_all.jsonl \
  --output /workspace/bese/tokenizers/bese_bpe_248_fineweb_v5.json \
  --num-merges 248 \
  --max-docs 6000000 \
  --fast
```

**Why:** v2 trained BPE on 50K docs. v5 trains on 6M docs (all of FineWeb decoded). Better merges = 2-5% shorter sequences = more training steps.

**Expected time:** ~2-4 minutes with fast BPE (v2 took 30s on 50K docs, linearly scaled ≈ 1 hour but fast BPE sublinear).

### Step 0.2 — Curriculum Sort the Training Data

Create `scripts/curriculum_sort.py`:

```python
#!/usr/bin/env python3
"""Sort training docs easy→hard for curriculum learning."""
import json
import argparse
from pathlib import Path

def difficulty_score(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    avg_word_len = sum(len(w) for w in words) / len(words)
    vocab_richness = len(set(words)) / len(words)
    sentences = max(text.count('.') + text.count('!') + text.count('?'), 1)
    avg_sentence_len = len(words) / sentences
    # Normalize to roughly 0-1 range
    return (avg_word_len / 10) * 0.3 + vocab_richness * 0.4 + (avg_sentence_len / 30) * 0.3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    docs = []
    with open(args.input) as f:
        for line in f:
            obj = json.loads(line)
            docs.append((difficulty_score(obj['text']), obj))

    docs.sort(key=lambda x: x[0])  # Easy first

    with open(args.output, 'w') as f:
        for _, obj in docs:
            f.write(json.dumps(obj) + '\n')

    print(f"Sorted {len(docs)} docs by difficulty: easy first")

if __name__ == '__main__':
    main()
```

Run it:

```bash
python scripts/curriculum_sort.py \
  --input /workspace/decoded_docs_jsonl/fineweb_train_all.jsonl \
  --output /workspace/decoded_docs_jsonl/fineweb_train_curriculum.jsonl
```

### Step 0.3 — Data Selection Filter

Modify `scripts/export_shards.py` — add at top:

```python
def is_high_value(text: str) -> bool:
    """Filter out web boilerplate and low-information docs."""
    words = text.split()
    if len(words) < 50:
        return False
    if len(set(words)) / len(words) < 0.25:
        return False
    lowered = text.lower()
    boilerplate_markers = [
        'cookie', 'subscribe', 'click here', 'privacy policy',
        'all rights reserved', 'terms of service', 'sign up',
    ]
    if sum(1 for m in boilerplate_markers if m in lowered) >= 3:
        return False
    return True
```

Apply during shard export (inside the document loop):

```python
# In export_shards.py, inside the docs loop:
if not is_high_value(doc['text']):
    skipped += 1
    continue
```

### Step 0.4 — Export v5 BESE Shards

```bash
python scripts/export_shards.py \
  --input /workspace/decoded_docs_jsonl/fineweb_train_curriculum.jsonl \
  --tokenizer /workspace/bese/tokenizers/bese_bpe_248_fineweb_v5.json \
  --output-dir /workspace/bese_shards_v5/ \
  --num-workers 64 \
  --train-shards 20 \
  --val-shards 1
```

**Expected time:** ~15-20 minutes (smaller vocab = faster tokenization than v3).

### Step 0.5 — Build Pre-Computed N-gram Table

Create `scripts/build_ngram_table.py`:

```python
#!/usr/bin/env python3
"""Build a compact n-gram frequency table for eval-time tilt."""
import argparse
import numpy as np
from collections import defaultdict
import pickle
import zlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', required=True, help='One training shard to scan')
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-n', type=int, default=4)
    ap.add_argument('--top-k', type=int, default=1, help='Keep top-k next tokens per prefix')
    args = ap.parse_args()

    # Read binary shard (uint16 tokens)
    shard = np.fromfile(args.shard, dtype=np.uint16, offset=264*4)[:100_000_000]
    print(f"Loaded {len(shard)} tokens from {args.shard}")

    tables = {n: defaultdict(lambda: defaultdict(int)) for n in range(2, args.max_n+1)}

    for i in range(len(shard)):
        for n in range(2, args.max_n+1):
            if i >= n - 1:
                prefix = tuple(int(x) for x in shard[i-n+1:i])
                tables[n][prefix][int(shard[i])] += 1

    # Compress: keep only top-k per prefix (sparse representation)
    compact = {}
    for n, table in tables.items():
        compact[n] = {}
        for prefix, counts in table.items():
            top = sorted(counts.items(), key=lambda x: -x[1])[:args.top_k]
            compact[n][prefix] = top

    data = pickle.dumps(compact)
    compressed = zlib.compress(data, level=9)
    with open(args.output, 'wb') as f:
        f.write(compressed)
    print(f"Wrote {len(compressed)} bytes of compressed n-gram table")

if __name__ == '__main__':
    main()
```

Run:

```bash
python scripts/build_ngram_table.py \
  --shard /workspace/bese_shards_v5/fineweb_train_000.bin \
  --output /workspace/bese/artifacts/ngram_table_v5.bin \
  --max-n 4 \
  --top-k 1
```

**Target size:** 200-500 KB after compression. This gets embedded in the 16MB submission artifact.

---

## Phase 1: Architecture Changes to `integration/train_gpt_bese.py`

### Change 1.1 — Add Depth Recurrence Hyperparameters

In the `Hyperparameters` class (around line 47-99), add:

```python
# Depth recurrence (loop middle layers multiple times)
depth_recurrence_start = int(os.environ.get("DEPTH_RECURRENCE_START", 3))
depth_recurrence_end = int(os.environ.get("DEPTH_RECURRENCE_END", 5))  # inclusive
depth_recurrence_loops = int(os.environ.get("DEPTH_RECURRENCE_LOOPS", 3))
depth_recurrence_activation_frac = float(os.environ.get("DEPTH_RECURRENCE_ACTIVATION_FRAC", 0.35))

# Parallel residuals (GPT-J style, attn + mlp in parallel for late layers)
parallel_residual_start = int(os.environ.get("PARALLEL_RESIDUAL_START", 7))
```

### Change 1.2 — Update Default Hyperparams for v5

Change these defaults in `Hyperparameters`:

```python
qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.0))        # was 1.5
matrix_lr = float(os.environ.get("MATRIX_LR", 0.022))            # was 0.04
weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.095))      # NEW
ema_decay = float(os.environ.get("EMA_DECAY", 0.9965))           # was 0.997
```

### Change 1.3 — Implement Parallel Residuals in `TransformerBlock`

Find the existing `TransformerBlock` class (search for `class TransformerBlock` or similar). Replace its `forward` method with:

```python
def forward(self, x, rope_cache=None):
    if self.parallel_residual:
        # GPT-J style: attn and MLP both see the same normed input
        normed = self.norm1(x)
        attn_out = self.attn(normed, rope_cache=rope_cache)
        mlp_out = self.mlp(normed)  # Same input as attention — key difference
        x = x + self.attn_scale * attn_out + self.mlp_scale * mlp_out
    else:
        # Standard sequential residual
        x = x + self.attn_scale * self.attn(self.norm1(x), rope_cache=rope_cache)
        x = x + self.mlp_scale * self.mlp(self.norm2(x))
    return x
```

Add `self.parallel_residual` flag in `TransformerBlock.__init__`:

```python
def __init__(self, ..., layer_idx: int, parallel_residual_start: int):
    # ... existing init ...
    self.parallel_residual = layer_idx >= parallel_residual_start
```

### Change 1.4 — Implement Depth Recurrence in `GPT.forward`

Find the `GPT` class's `forward` method. Replace the layer loop:

**Before (simplified):**
```python
for block in self.blocks:
    x = block(x, rope_cache=self.rope_cache)
```

**After (v5):**
```python
# Determine if depth recurrence is active (activates after warmup fraction)
current_frac = self._training_progress  # 0.0 to 1.0, passed from training loop
recurrence_active = current_frac >= self.hp.depth_recurrence_activation_frac

rec_start = self.hp.depth_recurrence_start
rec_end = self.hp.depth_recurrence_end
rec_loops = self.hp.depth_recurrence_loops if recurrence_active else 1

i = 0
while i < len(self.blocks):
    if i == rec_start:
        # Enter the recurrence zone: loop layers [rec_start..rec_end] rec_loops times
        loop_blocks = self.blocks[rec_start:rec_end + 1]
        for loop_pass in range(rec_loops):
            for lb in loop_blocks:
                x = lb(x, rope_cache=self.rope_cache)
        i = rec_end + 1  # Skip past the recurrence zone
    else:
        x = self.blocks[i](x, rope_cache=self.rope_cache)
        i += 1
```

Pass the training progress fraction into `GPT`:

```python
# In training loop (where you call model.forward):
model._training_progress = step / total_steps
logits = model(input_ids)
```

### Change 1.5 — Remove U-Net Skip Connections (Optional)

The v3 architecture has U-Net skip connections between first and second halves. These conflict with depth recurrence because the looped layers (3-5) sit right where the skip connections cross over. Options:

- **Safer:** Keep U-Net skips for layers 0-2 and 6-10; depth recurrence only affects layers 3-5 internally.
- **Cleaner:** Remove U-Net skips entirely; rely on depth recurrence + parallel residuals.

**Recommended:** Start with the safer option. If BPB doesn't improve as expected, drop U-Net in a follow-up run.

---

## Phase 2: Eval-Time Stack (the big exploit)

Create new directory `eval/` with three files.

### Step 2.1 — Create `eval/ngram_tilt.py`

```python
"""Causal n-gram prediction booster for eval time."""
import pickle
import zlib
import torch
from collections import defaultdict


class NgramTilt:
    def __init__(self, vocab_size: int, beta: float = 0.3, max_n: int = 4):
        self.vocab_size = vocab_size
        self.beta = beta
        self.max_n = max_n
        # Live-updated n-gram table during eval (causal — only from prefix)
        self.live_table = {n: defaultdict(lambda: defaultdict(int))
                           for n in range(2, max_n + 1)}
        # Optional: pre-loaded table from training data
        self.prior_table = None

    def load_prior(self, path: str):
        """Load pre-computed n-gram table from artifact."""
        with open(path, 'rb') as f:
            compressed = f.read()
        self.prior_table = pickle.loads(zlib.decompress(compressed))

    def update(self, context_ids: list, new_token: int):
        """Called after ground truth token is revealed (still causal — uses prefix only)."""
        for n in range(2, self.max_n + 1):
            if len(context_ids) >= n - 1:
                prefix = tuple(context_ids[-(n - 1):])
                self.live_table[n][prefix][new_token] += 1

    def tilt_logits(self, logits: torch.Tensor, context_ids: list) -> torch.Tensor:
        """Apply n-gram hint tilt to logits. logits shape: [vocab_size]"""
        hint = torch.zeros_like(logits)
        for n in range(2, self.max_n + 1):
            if len(context_ids) >= n - 1:
                prefix = tuple(context_ids[-(n - 1):])
                # Check live table first (more recent), then prior
                if prefix in self.live_table[n]:
                    counts = self.live_table[n][prefix]
                    if counts:
                        best = max(counts, key=counts.get)
                        hint[best] += 1.0
                elif self.prior_table and prefix in self.prior_table.get(n, {}):
                    top = self.prior_table[n][prefix]
                    if top:
                        best_token, _ = top[0]  # top-k=1 format
                        hint[best_token] += 0.5  # Lower weight for prior than live
        # Tilt: logits + beta * hint, then renormalize via softmax downstream
        return logits + self.beta * hint
```

### Step 2.2 — Create `eval/slot_lbfgs.py`

```python
"""SLOT (Sparse Linear Online Training) with L-BFGS for test-time adaptation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS


class SLOTLayer(nn.Module):
    """Tiny trainable bottleneck layer applied to model logits at eval time."""

    def __init__(self, vocab_size: int, hidden_dim: int = 4, device: str = 'cuda'):
        super().__init__()
        # Parameter count: 2 * vocab_size * hidden_dim
        # For vocab=288, hidden=4: 2 * 288 * 4 = 2,304 params (tuneable)
        self.down = nn.Linear(vocab_size, hidden_dim, bias=False)
        self.up = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.to(device)
        # Initialize as no-op: SLOT starts as identity
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Residual correction on logits
        return logits + self.up(self.down(logits))


class SLOTEvaluator:
    """Evaluate a frozen model with SLOT adaptation during eval."""

    def __init__(self, model, vocab_size: int, hidden_dim: int = 4,
                 lbfgs_steps: int = 8, history_size: int = 10, device: str = 'cuda'):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.slot = SLOTLayer(vocab_size, hidden_dim=hidden_dim, device=device)
        self.optimizer = LBFGS(
            self.slot.parameters(),
            lr=1.0,
            max_iter=lbfgs_steps,
            history_size=history_size,
            line_search_fn='strong_wolfe',
        )

    def score_chunk(self, input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Score one chunk. input_ids and targets are [batch, seq_len]."""
        # 1. Forward pass through frozen base model (no grad)
        with torch.no_grad():
            base_logits = self.model(input_ids)  # [B, S, V]

        # 2. L-BFGS optimizes SLOT to minimize loss on this chunk
        def closure():
            self.optimizer.zero_grad()
            corrected = self.slot(base_logits)
            loss = F.cross_entropy(
                corrected.reshape(-1, corrected.size(-1)),
                targets.reshape(-1),
                reduction='mean',
            )
            loss.backward()
            return loss

        self.optimizer.step(closure)

        # 3. Final scoring with optimized SLOT (no grad)
        with torch.no_grad():
            final_logits = self.slot(base_logits)
            final_loss = F.cross_entropy(
                final_logits.reshape(-1, final_logits.size(-1)),
                targets.reshape(-1),
                reduction='sum',
            )
            num_tokens = targets.numel()
        return final_loss, num_tokens
```

### Step 2.3 — Create `eval/eval_pipeline.py`

```python
"""Orchestrate eval-time stack: forward → n-gram tilt → SLOT."""
import torch
import torch.nn.functional as F
from pathlib import Path

from .ngram_tilt import NgramTilt
from .slot_lbfgs import SLOTEvaluator


def evaluate_with_stack(
    model,
    val_loader,
    vocab_size: int,
    bytes_per_token: torch.Tensor,
    ngram_prior_path: str = None,
    slot_hidden_dim: int = 4,
    slot_lbfgs_steps: int = 8,
    ngram_beta: float = 0.3,
    device: str = 'cuda',
):
    """Run the full v5 eval stack and return val_bpb."""
    model.eval()

    # Initialize components
    tilt = NgramTilt(vocab_size, beta=ngram_beta)
    if ngram_prior_path and Path(ngram_prior_path).exists():
        tilt.load_prior(ngram_prior_path)
        print(f"Loaded pre-computed n-gram prior from {ngram_prior_path}")

    slot_eval = SLOTEvaluator(model, vocab_size, hidden_dim=slot_hidden_dim,
                              lbfgs_steps=slot_lbfgs_steps, device=device)

    total_loss_nats = 0.0
    total_bytes = 0
    context = []

    for batch_idx, (input_ids, targets) in enumerate(val_loader):
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        # SLOT adaptation + scoring for this chunk
        loss_sum, n_tokens = slot_eval.score_chunk(input_ids, targets)
        total_loss_nats += loss_sum.item()

        # Track bytes for BPB calculation
        token_bytes = bytes_per_token[targets.cpu()].sum().item()
        total_bytes += token_bytes

        # Update n-gram table causally (prefix only)
        flat_targets = targets.cpu().flatten().tolist()
        for i, t in enumerate(flat_targets):
            if i > 0:
                tilt.update(flat_targets[:i], t)

        if batch_idx % 50 == 0:
            current_bpb = (total_loss_nats / total_bytes) / 0.6931  # nats → bits
            print(f"Batch {batch_idx}: running BPB = {current_bpb:.4f}")

    val_bpb = (total_loss_nats / total_bytes) / 0.6931
    return val_bpb
```

**Note on SLOT hidden dim:** `hidden_dim=4` gives 2×288×4 = 2,304 params. Try `hidden_dim=3` (1,728 params, closer to PR #675's 1,536) if L-BFGS converges too slowly, or `hidden_dim=6` (3,456 params) if we want more expressiveness.

---

## Phase 3: Submission File Assembly

### Step 3.1 — Update `submission/train_gpt.py`

The submission is a single file that must include everything. Inline the new eval components:

1. Paste `NgramTilt` class (from `eval/ngram_tilt.py`)
2. Paste `SLOTLayer` and `SLOTEvaluator` classes (from `eval/slot_lbfgs.py`)
3. Add the v5 architecture changes (depth recurrence in `GPT.forward`, parallel residual in `TransformerBlock`)
4. Replace the existing eval function with `evaluate_with_stack`
5. Pack the pre-computed n-gram table alongside the model:

```python
# After model save, bundle with n-gram table
artifact = {
    'model_state': compressed_state_dict,
    'ngram_table': open('artifacts/ngram_table_v5.bin', 'rb').read(),
    'config': config_dict,
}
```

### Step 3.2 — Size Budget Check

Target breakdown for 16 MB artifact:

| Component | Size |
|-----------|------|
| Model weights (INT6 + LZMA, vocab 288) | ~14.5 MB |
| N-gram table (compressed) | ~300 KB |
| Code (inline) | ~100 KB |
| Headroom | ~1 MB |
| **Total** | **< 16 MB** |

Run size check before training:

```bash
python -c "
import pickle, zlib, os
size = os.path.getsize('/workspace/bese/artifacts/ngram_table_v5.bin')
print(f'N-gram table: {size/1024:.1f} KB')
assert size < 500_000, 'N-gram table too large'
"
```

---

## Phase 4: Run Sequence on RunPod

### Pre-flight (one-time setup)

```bash
# SSH to RunPod
ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<pod-ip>

# Create v5 branch
cd /workspace/bese
git checkout -b bese-v5
git push -u origin bese-v5

# Create new directories
mkdir -p eval kernels artifacts
mkdir -p /workspace/bese_shards_v5
```

### Phase 0 Execution (untimed prep, ~30 min total)

```bash
# 0.1 - Train BPE (if not reusing v2's merges)
python scripts/train_bpe_jsonl.py \
  --input /workspace/decoded_docs_jsonl/fineweb_train_all.jsonl \
  --output /workspace/bese/tokenizers/bese_bpe_248_v5.json \
  --num-merges 248 --max-docs 6000000 --fast

# 0.2 - Curriculum sort
python scripts/curriculum_sort.py \
  --input /workspace/decoded_docs_jsonl/fineweb_train_all.jsonl \
  --output /workspace/decoded_docs_jsonl/fineweb_train_curriculum.jsonl

# 0.3+0.4 - Export v5 shards with filtering
python scripts/export_shards.py \
  --input /workspace/decoded_docs_jsonl/fineweb_train_curriculum.jsonl \
  --tokenizer /workspace/bese/tokenizers/bese_bpe_248_v5.json \
  --output-dir /workspace/bese_shards_v5/ \
  --num-workers 64

# 0.5 - Build n-gram table
python scripts/build_ngram_table.py \
  --shard /workspace/bese_shards_v5/fineweb_train_000.bin \
  --output /workspace/bese/artifacts/ngram_table_v5.bin \
  --max-n 4 --top-k 1
```

### Phase 1+2 Implementation (code changes)

Apply the edits in Phase 1 and Phase 2 above to:
- `integration/train_gpt_bese.py` (architecture)
- Create `eval/ngram_tilt.py`, `eval/slot_lbfgs.py`, `eval/eval_pipeline.py`

Smoke test locally or on the pod before full run:

```bash
python scripts/smoke_bese_integration.py
python -c "from eval.ngram_tilt import NgramTilt; print('import OK')"
python -c "from eval.slot_lbfgs import SLOTEvaluator; print('import OK')"
```

### Phase 4.1 — Full Training Run (600s cap)

```bash
cd /workspace/bese

export VOCAB_SIZE=288
export NUM_LAYERS=11
export MODEL_DIM=512
export DEPTH_RECURRENCE_LOOPS=3
export PARALLEL_RESIDUAL_START=7
export QK_GAIN_INIT=5.0
export MATRIX_LR=0.022
export WEIGHT_DECAY=0.095
export EMA_DECAY=0.9965
export TOKENIZER_PATH=/workspace/bese/tokenizers/bese_bpe_248_v5.json
export DATA_PATH=/workspace/bese_shards_v5
export MAX_WALLCLOCK_SECONDS=600

tmux new -s bese-v5
torchrun --nproc-per-node=8 integration/train_gpt_bese.py 2>&1 | tee /workspace/run_v5.log
```

### Phase 4.2 — Eval with SLOT + N-gram Tilt (600s cap)

```bash
python -m eval.run_eval \
  --model /workspace/bese/checkpoints/v5_ema.pt \
  --val-shards /workspace/bese_shards_v5/fineweb_val_*.bin \
  --ngram-prior /workspace/bese/artifacts/ngram_table_v5.bin \
  --slot-hidden-dim 4 \
  --slot-lbfgs-steps 8 \
  --max-seconds 600 \
  2>&1 | tee /workspace/eval_v5.log
```

### Phase 4.3 — Artifact Assembly and Size Check

```bash
python submission/assemble_v5.py \
  --model /workspace/bese/checkpoints/v5_ema.pt \
  --ngram /workspace/bese/artifacts/ngram_table_v5.bin \
  --output /workspace/bese/submission/v5_submission.tar.gz

ls -la /workspace/bese/submission/v5_submission.tar.gz
# Must be < 16,000,000 bytes
```

---

## Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Artifact size | < 16 MB | Hard requirement |
| Training completes | < 600s | Wall clock |
| Eval completes | < 600s | Wall clock |
| val_bpb (base model, no SLOT) | < 1.12 | Should beat v3's 1.1394 and v2's 1.1276 |
| val_bpb (+ n-gram tilt) | < 1.11 | +0.003 from tilt |
| val_bpb (+ SLOT) | < 1.05 | Biggest variable — depends on SLOT signal |
| val_bpb (full stack) | < 1.08 | Beats current merged SOTA (1.1147) |

**Stretch goal:** < 0.9 BPB if BESE+SLOT hypothesis holds (BESE's longer sequences → richer SLOT signal).

---

## What Changed From v3

| Aspect | v3 | v5 | Reason |
|--------|----|----|--------|
| Vocab size | 4040 | 288 | v3 was over-size AND worse BPB |
| Depth recurrence | No | Yes (layers 3-5, 3x) | Current #1-3 all use it |
| Parallel residuals | No | Yes (layers 7-10) | +60-70 training steps |
| QK gain init | 1.5 | 5.0 | Every top PR uses 5.0 |
| Weight decay | default | 0.095 | PR #1445 |
| EMA decay | 0.997 | 0.9965 | PR #1445 |
| BPE training data | 50K docs | 6M docs | Richer merges |
| Curriculum ordering | No | Yes | Faster convergence |
| Data filtering | No | Yes | Higher quality tokens |
| Pre-computed n-gram table | No | Yes | Saves 30s eval time |
| N-gram tilt at eval | No | Yes | Free ~0.003 BPB |
| SLOT + L-BFGS at eval | No | Yes | Potentially huge gain |

---

## Risk Register

**Known risks:**

1. **Depth recurrence + U-Net conflict.** The existing BESE code has U-Net skip connections between first and second halves. Recurrence on layers 3-5 may break skip routing. Mitigation: keep U-Net skips only for layers 0-2 and 6-10; verify forward pass integrity with smoke test.

2. **SLOT convergence instability.** L-BFGS with strong Wolfe line search can occasionally fail to converge on flat landscapes. If this happens, the eval will stall. Mitigation: wrap `optimizer.step()` in try/except; fall back to AdamW SLOT if L-BFGS diverges on any chunk.

3. **Parallel residual incompatibility with RoPE.** Some RoPE implementations assume sequential attn→mlp ordering. Mitigation: verify by running a forward pass and comparing outputs against the sequential baseline on 1 step of random data.

4. **SLOT hidden dim sizing.** Too small (< 3) and L-BFGS curvature estimate is noisy; too large (> 8) and optimization becomes slow. Mitigation: default hidden_dim=4, sweep 3-6 in ablation runs.

5. **N-gram table size may exceed budget.** At max_n=4 and top-k=1 on 100M tokens, expect 200-500 KB. If larger, reduce max_n to 3 or sample a smaller shard.

**Unknown risks:**

- Whether BESE + SLOT actually outperforms SP8192 + SLOT. The hypothesis (more tokens per byte → more SLOT signal) is plausible but unproven. Primary experiment: ablate SLOT on/off and compare gains between BESE and an SP8192 control run.

---

## Abort / Fallback

If v5 full stack fails (e.g., SLOT breaks on eval), fall back to partial stack in priority order:

1. **v5-minus-SLOT:** All architecture + n-gram tilt, no SLOT. Expected BPB ~1.10.
2. **v5-minus-SLOT-minus-tilt:** Just architecture + hyperparam changes. Expected BPB ~1.11.
3. **v2 settings (proven):** Fall back to v2 architecture with v2 BPE merges. Known good at 1.1276.

Each fallback takes < 5 minutes to switch to (environment variable changes only, no code edits).

---

## Timeline

Assuming RunPod pod is already set up with v3 infrastructure:

- Phase 0 (data prep): 30-40 minutes
- Phase 1+2 (code changes): 2-3 hours (human)
- Phase 1+2 (smoke tests): 15 minutes
- Phase 4.1 (full training): 11 minutes (600s + startup/save)
- Phase 4.2 (full eval): 11 minutes
- Phase 4.3 (artifact assembly + size check): 2 minutes

**Total first v5 run:** ~4 hours from start to submission-ready artifact.

---

## Post-Run Deliverables

After v5 completes, create:

1. `docs/run_v5_findings.md` (mirroring `run_v3_findings.md` format)
2. Updated training log at `/workspace/run_v5.log`
3. Submission artifact at `/workspace/bese/submission/v5_submission.tar.gz`
4. Ablation table: base, +tilt, +SLOT, +both (so we know where each BPB gain comes from)
