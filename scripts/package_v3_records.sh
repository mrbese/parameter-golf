#!/usr/bin/env bash
# Package the v3 submission for the openai/parameter-golf PR.
#
# Creates a CLEAN branch from upstream/main with ONLY the canonical
# records/track_non_record_16mb/2026-04-30_BESE_v3_Spatial/ folder,
# so the PR diff is minimal and reviewable.
#
# Run AFTER the pod produces final_model.int6.ptz and the train log.
#
# Usage:
#   FINAL_MODEL=/path/to/final_model.int6.ptz \
#   TRAIN_LOG=/path/to/train_log_run1.txt \
#     bash scripts/package_v3_records.sh
#
# Or to also auto-push and open the PR:
#   FINAL_MODEL=... TRAIN_LOG=... PUSH=1 PR=1 bash scripts/package_v3_records.sh

set -euo pipefail

cd "$(dirname "$0")/.."
REPO=$(pwd)

FINAL_MODEL="${FINAL_MODEL:-}"
TRAIN_LOG="${TRAIN_LOG:-}"
PR_BRANCH="${PR_BRANCH:-v9-spatial-pr}"
DEST_REL="records/track_non_record_16mb/2026-04-30_BESE_v3_Spatial"

# Use a worktree so we don't disturb the v9-spatial working tree
WORKTREE_DIR="${WORKTREE_DIR:-/tmp/parameter-golf-pr-worktree}"

echo "==> Setting up clean worktree at $WORKTREE_DIR (off upstream/main)"
git fetch upstream main --depth=1
rm -rf "$WORKTREE_DIR"

# Create the worktree on a new branch that starts from upstream/main.
if git show-ref --verify --quiet "refs/heads/$PR_BRANCH"; then
  echo "    branch $PR_BRANCH already exists locally; deleting and recreating"
  git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
  git branch -D "$PR_BRANCH"
fi
git worktree add -b "$PR_BRANCH" "$WORKTREE_DIR" upstream/main

cd "$WORKTREE_DIR"
mkdir -p "$DEST_REL"

echo ""
echo "==> Copying records folder content into $WORKTREE_DIR/$DEST_REL/"
cp "$REPO/records_v9_spatial/README.md" "$DEST_REL/README.md"
cp "$REPO/records_v9_spatial/submission.json" "$DEST_REL/submission.json"
cp "$REPO/records_v9_spatial/requirements.txt" "$DEST_REL/requirements.txt"

echo "==> Copying code (counted in submission.json::bytes_code)"
cp "$REPO/integration/train_gpt_bese.py" "$DEST_REL/train_gpt.py"
cp "$REPO/tokenizer/bese_v3_constants.py" "$DEST_REL/bese_v3_constants.py"
cp "$REPO/tokenizer/bese_v3_fast_bpe.py" "$DEST_REL/bese_v3_fast_bpe.py"

echo "==> Copying tokenizer + coords"
cp "$REPO/tokenizers/bese_v3_bpe_241_v9.json" "$DEST_REL/tokenizer.json"
cp "$REPO/artifacts/letter_coords_v3.json" "$DEST_REL/letter_coords_v3.json"

if [[ -n "$FINAL_MODEL" ]] && [[ -f "$FINAL_MODEL" ]]; then
  echo "==> Copying final model artifact"
  cp "$FINAL_MODEL" "$DEST_REL/final_model.int6.ptz"
fi
if [[ -n "$TRAIN_LOG" ]] && [[ -f "$TRAIN_LOG" ]]; then
  echo "==> Copying train log"
  cp "$TRAIN_LOG" "$DEST_REL/train_log_run1.txt"
fi

CODE_BYTES=$(stat -f%z "$DEST_REL"/*.py 2>/dev/null | awk '{s+=$1} END {print s}')
TOTAL_BYTES=$(stat -f%z "$DEST_REL"/* 2>/dev/null | awk '{s+=$1} END {print s}')

echo ""
echo "==> Files in $DEST_REL (size on disk, NOT compressed):"
ls -lah "$DEST_REL" | awk 'NR>1 {printf "    %10s  %s\n", $5, $NF}'
echo ""
echo "    Total .py code:  $CODE_BYTES bytes (= submission.json::bytes_code)"
echo "    Total folder:    $TOTAL_BYTES bytes"

echo ""
echo "==> Staging + committing"
git add "$DEST_REL/"
git commit -m "Non-record: BESE v3 — Spatial Letter Encoding (one geometric prior, two jobs)

Adds records/track_non_record_16mb/2026-04-30_BESE_v3_Spatial/.

Novel byte-level tokenizer where every English letter has a 3D
coordinate derived from FineWeb bigram MDS structure. The same
coordinates do double duty:

  1. BPE merge scoring uses freq * exp(-d / 3sigma)^0.02 to gently
     prefer spatially-close letter pairs (pow_0.02 sweep winner).
  2. First 12 dims of each letter token's embedding row are seeded
     from the 12D version of the same coordinates.

To our knowledge, first submission to use one geometric prior across
both tokenizer and embedding init. Architecture is byte-for-byte
identical to PR #1666 (BESE record at 1.1531 BPB, 3-seed mean) so
the v3 contribution is isolated to the tokenizer + embedding-init
layer.

Single-seed non-record submission. Inline byte-invariant proof in
records README. Three-seed validation, Turkish-corpus comparative
test, and higher-vocab variants are pending compute credits."

echo ""
echo "==> Worktree state:"
git log --oneline -3
echo ""
git diff --stat HEAD~1

if [[ "${PUSH:-0}" == "1" ]]; then
  echo ""
  echo "==> Pushing $PR_BRANCH to origin"
  git push -u origin "$PR_BRANCH"
fi

if [[ "${PR:-0}" == "1" ]]; then
  echo ""
  echo "==> Opening PR via gh"
  gh pr create \
    --title "Non-record: BESE v3 — Spatial Letter Encoding (one geometric prior, two jobs)" \
    --body-file "$REPO/records_v9_spatial/PR_DESCRIPTION.md" \
    --base main \
    --head "mrbese:$PR_BRANCH" \
    --repo openai/parameter-golf
fi

echo ""
echo "================================================================"
echo "  PR worktree ready at $WORKTREE_DIR"
echo "  Branch: $PR_BRANCH"
echo ""
if [[ "${PUSH:-0}" != "1" ]]; then
  echo "  To push:    cd $WORKTREE_DIR && git push -u origin $PR_BRANCH"
fi
if [[ "${PR:-0}" != "1" ]]; then
  echo "  To open PR: gh pr create --base main --head mrbese:$PR_BRANCH \\"
  echo "                            --repo openai/parameter-golf \\"
  echo "                            --title 'Non-record: BESE v3 — Spatial Letter Encoding (one geometric prior, two jobs)' \\"
  echo "                            --body-file $REPO/records_v9_spatial/PR_DESCRIPTION.md"
fi
echo "================================================================"
