#!/usr/bin/env bash
# Commit + push the BESE v3 (Spatial Letter Encoding) work to the
# v9-spatial branch on mrbese/parameter-golf-bese.
#
# Run this from /Users/mrbese/Projects/parameter-golf-bese AFTER the
# local build (build_v3_locally.py) has finished and produced:
#   artifacts/letter_coords_v3.json
#   tokenizers/bese_v3_bpe_241_v9.json
#
# Usage:
#   bash scripts/commit_v9_spatial.sh
#   # or:
#   bash scripts/commit_v9_spatial.sh --no-push    # commit only, don't push

set -euo pipefail

cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# 1. Sanity checks
# ---------------------------------------------------------------------------
echo "==> Sanity checks"

REQUIRED_FILES=(
  "tokenizer/bese_v3_constants.py"
  "tokenizer/bese_v3_fast_bpe.py"
  "scripts/build_v3_locally.py"
  "scripts/runpod_v9_spatial.py"
  "records_v9_spatial/README.md"
  "records_v9_spatial/submission.json"
  "records_v9_spatial/requirements.txt"
  "integration/train_gpt_bese.py"
)
MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "  MISSING: $f"
    MISSING=1
  fi
done
if [[ $MISSING -ne 0 ]]; then
  echo "  Required v3 source files are missing. Aborting."
  exit 1
fi
echo "  All v3 source files present."

# Build outputs are optional but recommended (so RunPod can skip Phase 0a/0b)
BUILD_OUTPUTS=(
  "artifacts/letter_coords_v3.json"
  "tokenizers/bese_v3_bpe_241_v9.json"
)
HAVE_BUILD=1
for f in "${BUILD_OUTPUTS[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "  Build output missing: $f"
    HAVE_BUILD=0
  fi
done
if [[ $HAVE_BUILD -eq 1 ]]; then
  echo "  Local build outputs present (RunPod can skip Phase 0a + 0b)."
else
  echo "  Local build outputs NOT present — RunPod will run Phase 0a + 0b itself (~50 min)."
  echo "  This is fine; just slower. Continuing."
fi

# Syntax check
echo ""
echo "==> Syntax check on all v3 Python files"
for f in tokenizer/bese_v3_constants.py tokenizer/bese_v3_fast_bpe.py scripts/runpod_v9_spatial.py scripts/build_v3_locally.py integration/train_gpt_bese.py; do
  python3 -c "import ast; ast.parse(open('$f').read())" \
    && echo "  OK  $f" \
    || { echo "  FAIL $f"; exit 1; }
done

# Tokenizer self-test
echo ""
echo "==> Tokenizer self-test"
( cd tokenizer && python3 bese_v3_fast_bpe.py 2>&1 | tail -2 )

# ---------------------------------------------------------------------------
# 2. Branch setup
# ---------------------------------------------------------------------------
echo ""
echo "==> Branch setup"

CURRENT_BRANCH="$(git branch --show-current)"
echo "  Current branch: $CURRENT_BRANCH"

# Create v9-spatial branch from main if it doesn't exist
if git show-ref --verify --quiet refs/heads/v9-spatial; then
  echo "  v9-spatial branch already exists; switching to it."
  git checkout v9-spatial
else
  echo "  Creating v9-spatial branch from $CURRENT_BRANCH..."
  git checkout -b v9-spatial
fi

# ---------------------------------------------------------------------------
# 3. Stage only v3 files (be explicit; the working tree may have other dirt)
# ---------------------------------------------------------------------------
echo ""
echo "==> Staging v3 files"

git add tokenizer/bese_v3_constants.py
git add tokenizer/bese_v3_fast_bpe.py
git add scripts/build_v3_locally.py
git add scripts/runpod_v9_spatial.py
git add scripts/commit_v9_spatial.sh
git add scripts/verify_v3_coords.py
git add records_v9_spatial/    # README.md, submission.json, requirements.txt, PR_DESCRIPTION.md
git add integration/train_gpt_bese.py

# Build outputs (only if they exist). artifacts/ is in .gitignore by default
# but the v3 coords are an explicit ship-with-branch artifact so RunPod can
# skip Phase 0a/0b — force-add it.
if [[ -f artifacts/letter_coords_v3.json ]]; then
  git add -f artifacts/letter_coords_v3.json
fi
if [[ -f tokenizers/bese_v3_bpe_241_v9.json ]]; then
  git add tokenizers/bese_v3_bpe_241_v9.json
fi

echo "  Staged files:"
git diff --cached --name-only | sed 's/^/    /'

# ---------------------------------------------------------------------------
# 4. Commit
# ---------------------------------------------------------------------------
echo ""
echo "==> Commit"

git diff --cached --stat | tail -5

if git diff --cached --quiet; then
  echo "  No changes staged. Nothing to commit."
  exit 0
fi

git commit -m "v9: BESE v3 (Spatial Letter Encoding) — dual-use geometric prior

Adds a new BESE variant where every English letter has a 3D coordinate
derived from FineWeb bigram MDS. The same coordinates do double duty:

  1. TOKENIZER: BPE merge scoring uses freq * exp(-d/3sigma)^0.02
     (the pow_0.02 sweep winner) to prefer spatially-close pairs.
  2. MODEL: First 12 dims of each letter token's embedding row are
     initialized from the 12D version of the same coordinates.

Architecture is identical to PR #1666 — only the tokenizer + embedding
init change. Same 12L / dim=512 / mlp_mult=3.5 / depth recurrence /
parallel residuals / INT6 + LZMA preset 9 / 600s training cap.

Files:
  tokenizer/bese_v3_constants.py     — flat 26-letter alphabet, 47-token
                                       base vocab, byte-accounting table,
                                       spatial scoring helpers
  tokenizer/bese_v3_fast_bpe.py      — spatial-scored BPE training,
                                       centroid-tracked merges, encode/
                                       decode, save/load, self-test
  integration/train_gpt_bese.py      — _apply_spatial_letter_init helper
                                       gated by SPATIAL_INIT_ENABLED=1
  scripts/build_v3_locally.py        — Phase 0a + 0b runnable on Mac;
                                       produces letter_coords_v3.json
                                       and the v3 BPE merges
  scripts/runpod_v9_spatial.py       — full orchestration (Phase 0a-d
                                       prep + Phase 1 train + Phase 2/3)
  scripts/commit_v9_spatial.sh       — this commit script
  records_v9_spatial/                — submission folder template:
                                         README.md (with BPB proof,
                                         single-seed disclaimer, Turkish
                                         prediction, About-the-author)
                                         submission.json, requirements.txt
  artifacts/letter_coords_v3.json    — fitted coords (3D + 12D), if local
                                       build was run before this commit
  tokenizers/bese_v3_bpe_241_v9.json — trained spatial BPE merges, if
                                       local build was run before this commit

Vocab: 47 base (4 special + 26 letters + 7 punct + 10 digits) + 241 BPE
merges = 288 total, matching v2/v6.1 for direct comparability.

Byte invariant verified: sum(BYTES_PER_TOKEN[t] for t in encode(s)) ==
len(s.encode('utf-8')) holds before AND after BPE merges. Self-test in
bese_v3_fast_bpe.py exercises this on diverse strings."

echo ""
echo "==> Commit done"
git log -1 --stat | head -20

# ---------------------------------------------------------------------------
# 5. Push (unless --no-push)
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--no-push" ]]; then
  echo ""
  echo "==> --no-push: skipping push. Run \`git push origin v9-spatial\` when ready."
  exit 0
fi

echo ""
echo "==> Pushing to origin/v9-spatial"
git push -u origin v9-spatial

echo ""
echo "==> Done"
echo "  Branch:  v9-spatial"
echo "  Remote:  $(git remote get-url origin)"
echo ""
echo "  When credits land, on a RunPod 8xH100 SXM pod:"
echo "    cd /workspace"
echo "    git clone https://github.com/mrbese/parameter-golf-bese.git bese"
echo "    cd bese && git checkout v9-spatial"
echo "    pip install einops --break-system-packages"
echo "    python scripts/runpod_v9_spatial.py --num-gpus 8"
echo ""
echo "  If you committed letter_coords_v3.json + the v3 BPE JSON, the"
echo "  orchestrator will skip Phase 0a + 0b automatically."
