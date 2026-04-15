#!/usr/bin/env python3
"""
Train BESE BPE tokenizer locally (e.g. on M4 Mac) and save to a JSON file.

Streams 100K docs from HuggingFace FineWeb — no local data needed.
Output tokenizer can be uploaded to the RunPod network volume.

Usage:
  pip install numpy sentencepiece datasets
  python3 scripts/train_bpe_local.py
  # Then upload:
  scp -P 19292 tokenizers/bese_bpe_1024_v7.json root@64.247.206.95:/workspace/tokenizers/
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

# Add tokenizer module to path
BESE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BESE_DIR / "tokenizer"))

from bese_fast_bpe import train_bpe_merges_fast, FastBESEBPETokenizer

NUM_MERGES = 1024
NUM_DOCS = 100_000
MIN_WORDS = 50
OUTPUT = BESE_DIR / "tokenizers" / "bese_bpe_1024_v7.json"


def _is_high_value(text: str) -> bool:
    words = text.split()
    if len(words) < MIN_WORDS:
        return False
    if len(set(words)) / len(words) < 0.25:
        return False
    lowered = text.lower()
    boilerplate = ['cookie', 'subscribe', 'click here', 'privacy policy',
                   'all rights reserved', 'terms of service', 'sign up']
    if sum(1 for m in boilerplate if m in lowered) >= 3:
        return False
    return True


def stream_fineweb_docs(n: int) -> list[str]:
    """Stream n docs from HuggingFace FineWeb (no local download needed)."""
    from datasets import load_dataset
    print(f"Streaming {n:,} docs from HuggingFace FineWeb...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    docs = []
    t0 = time.time()
    for sample in ds:
        text = sample.get("text", "")
        if _is_high_value(text):
            docs.append(text)
            if len(docs) % 10_000 == 0:
                elapsed = time.time() - t0
                print(f"  {len(docs):,}/{n:,} docs ({elapsed:.1f}s)")
        if len(docs) >= n:
            break
    print(f"  Done: {len(docs):,} docs in {time.time() - t0:.1f}s")
    return docs


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists():
        print(f"Tokenizer already exists at {OUTPUT}, skipping.")
        return

    # Stream docs
    docs = stream_fineweb_docs(NUM_DOCS)

    # Train BPE
    print(f"\nTraining BPE with {NUM_MERGES} merges on {len(docs):,} docs...")
    t0 = time.time()
    merges = train_bpe_merges_fast(docs, num_merges=NUM_MERGES, verbose=True)
    elapsed = time.time() - t0
    print(f"\nBPE training done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    tok = FastBESEBPETokenizer(merges=merges)
    print(f"Vocab size: {tok.vocab_size}")
    tok.save(str(OUTPUT))
    print(f"Saved to {OUTPUT}")
    print(f"\nUpload with:")
    print(f"  scp -P 19292 {OUTPUT} root@64.247.206.95:/workspace/tokenizers/")


if __name__ == "__main__":
    main()
