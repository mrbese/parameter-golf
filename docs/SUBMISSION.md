# Non-record PR Checklist (Parameter Golf + BESE)

Use this when submitting to [openai/parameter-golf](https://github.com/openai/parameter-golf) on the **non-record** track with the BESE+BPE tokenizer.

## Upstream requirements (from openai/parameter-golf README)

All submissions — including non-record — must be a PR that **only adds a new folder** to the appropriate `/records` subfolder and includes:

1. **`README.md`** — explains the submission in reasonable detail
2. **`submission.json`** — name, GitHub ID, `val_bpb`, and related metadata
3. **Train log** — automatically produced by the training script; demonstrate statistical significance (typically 3 runs averaged)
4. **`train_gpt.py`** (or equivalent) + any dependencies — must compile and run **from within the records folder**

Additional rules:
- **Artifact size = code bytes + compressed model bytes ≤ 16,000,000 bytes** (decimal 16 MB, not 16 MiB)
- **No external downloads or network calls during evaluation** — the artifact must be fully self-contained
- **Tokenizer changes get extra scrutiny** — must prove with certainty that `val_bpb` is correctly calculated
- **Cannot access validation data during training**

## Submission folder structure

Your PR should add a single folder under `records/track_non_record_16mb/`:

```
records/track_non_record_16mb/YYYY-MM-DD_BESE_NovelTokenizer/
├── README.md              # Detailed explanation (see template below)
├── submission.json        # Metadata (see template below)
├── train_gpt.py           # Self-contained training script (BESE logic inlined or bundled)
├── bese_constants.py      # Tokenizer: shared alphabet constants
├── bese_tokenizer.py      # Tokenizer: base 38-token encoder
├── bese_bpe_tokenizer.py  # Tokenizer: BESE + BPE system
├── bese_fast_bpe.py       # Tokenizer: fast BPE training/encoding
├── tokenizer.json         # Pre-trained BESE+BPE tokenizer (250 merges)
├── export_shards.py       # Script to reproduce data shards from FineWeb
├── train_bpe_jsonl.py     # Script to reproduce BPE merges from FineWeb
├── requirements.txt       # Any extra pip dependencies beyond upstream
├── train_log_run1.txt     # Training log (run 1)
├── train_log_run2.txt     # Training log (run 2)
└── train_log_run3.txt     # Training log (run 3)
```

**Critical:** `train_gpt.py` must run from this folder without referencing paths outside it. The tokenizer code must be co-located (not via `BESE_TOKENIZER_ROOT` pointing elsewhere).

## submission.json template

```json
{
  "author": "Omer Bese",
  "github_id": "mrbese",
  "val_bpb": 0.0,
  "date": "YYYY-MM-DD",
  "summary": "BESE two-layer tokenizer (38 base + 250 BPE merges) shrinks embedding table by ~276KB, funding extra transformer layers",
  "hardware": "8xH100 SXM",
  "training_time_seconds": 0,
  "compressed_model_bytes": 0,
  "code_bytes": 0,
  "total_artifact_bytes": 0,
  "vocab_size": 288,
  "num_layers": 11,
  "model_dim": 512,
  "tokenizer": "BESE+BPE (custom)",
  "notes": "Novel tokenizer submission — non-record track"
}
```

Fill in `val_bpb`, timing, and size fields after the final run.

## README.md template (inside submission folder)

```markdown
# BESE: Base-Efficient Subword Encoding for Parameter Golf

## Summary
Two-layer tokenizer: 38 structured base tokens + 250 BPE merges (288 total vocab).
Saves ~276KB in Int6 embeddings vs SP1024 baseline, enabling extra transformer layers
within the 16MB budget.

## Motivation
Standard SP1024 spends ~384KB on embeddings. BESE reduces this to ~108KB by encoding
the 8 most frequent letters as single tokens and grouping the remaining 18 into 5
context-disambiguated groups (2-token codes). BPE merges on top recover sequence length.

## BPB Calculation
Every token maps to a deterministic byte count via a lookup table:
- Single-token letters (e,t,a,o,i,n,s,r): 1 byte
- Group tokens (12-16): 0 bytes (incomplete character)
- Position markers (17-20): 1 byte (completes character)
- Punctuation, space, digits: 1 byte each
- Multi-byte UTF-8: 1 OTHER_PUNCT token per byte
- BPE merge tokens: sum of constituent token bytes

Total bytes always equals the original UTF-8 byte count. Verified on 100+ FineWeb
validation documents with zero mismatches.

## Training Config
- Dataset: FineWeb (10 train shards, re-exported with BESE tokenizer)
- Vocab: 288 (38 base + 250 BPE merges)
- Architecture: [layers]L / [dim]d / [heads] heads / [mlp_mult]x MLP
- Hardware: 8xH100 SXM
- Wall clock: [time]s
- Compressed model: [size] bytes (int8 + zstd)

## Results
| Run | val_bpb | Training time |
|-----|---------|---------------|
| 1   |         |               |
| 2   |         |               |
| 3   |         |               |
| **Mean** | | |

Baseline SP1024 comparison: [val_bpb] (same architecture, same hardware)

## Reproducing
1. Train BPE merges: `python train_bpe_jsonl.py --input <fineweb_jsonl> --merges 250 --output tokenizer.json`
2. Export shards: `python export_shards.py --tokenizer tokenizer.json --output ./data/`
3. Train model: `torchrun --standalone --nproc_per_node=8 train_gpt.py`

## License
MIT
```

## Before you open the PR

1. **Fork** upstream `openai/parameter-golf` (if not already done).
2. **Branch** from `main` with a descriptive name, e.g. `bese-novel-tokenizer`.
3. **Create the submission folder** under `records/track_non_record_16mb/`.
4. **Bundle all code** into the folder — `train_gpt.py` must import tokenizer files from the same directory (use relative imports or `sys.path.insert`).
5. **Re-export FineWeb shards** with `export_shards.py` using the exact `tokenizer.json` you'll submit.
6. **Run 3 training runs** on 8xH100 SXM, saving train logs.
7. **Verify BPB**: run byte checks on validation text to confirm UTF-8 byte count matches.
8. **Check artifact size**: `code_bytes + compressed_model_bytes ≤ 16,000,000`.
9. **Fill in** `submission.json` and `README.md` with final numbers.
10. **Test from scratch**: clone your fork, `cd` into the records folder, and confirm `train_gpt.py` runs without errors.

## Review expectations

- **Tokenizer submissions get extra OpenAI scrutiny.** The byte accounting must be transparent, reproducible, and provably correct.
- Non-record track accepts novel approaches even if they don't beat SOTA, but submissions must still run successfully and be well-justified.
- Broken scripts will not be accepted.
