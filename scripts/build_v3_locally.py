#!/usr/bin/env python3
"""
Build BESE v3 spatial artifacts locally (no GPU needed).

Produces two files that will be uploaded to RunPod:
  artifacts/letter_coords_v3.json    — 3D + 12D letter coordinates
  tokenizers/bese_v3_bpe_241_v9.json — spatial-scored BPE merges

This is Phase 0a + 0b of runpod_v9_spatial.py extracted as a standalone
script that runs on any machine with Python + numpy + datasets installed.

Usage:
  pip install numpy datasets
  python scripts/build_v3_locally.py            # smoke test (5K docs, 500 steps)
  python scripts/build_v3_locally.py --docs 100000 --steps 5000   # production
  python scripts/build_v3_locally.py --output-dir /tmp/v3prep     # custom output dir

Outputs are deterministic given a fixed --seed. Re-running with the same
args will produce identical coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Step 1: stream FineWeb (HuggingFace)
# ---------------------------------------------------------------------------

def stream_fineweb(n_docs: int, min_words: int = 50, verbose: bool = True) -> list[str]:
    """Stream n_docs from HuggingFace FineWeb (sample-10BT). Filters out
    low-quality docs (too short, too repetitive)."""
    from datasets import load_dataset
    if verbose:
        print(f"[stream] requesting {n_docs:,} FineWeb docs...", flush=True)
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    docs: list[str] = []
    t0 = time.time()
    for sample in ds:
        text = sample.get("text", "")
        words = text.split()
        if len(words) < min_words:
            continue
        if len(set(words)) / len(words) < 0.25:
            continue
        docs.append(text)
        if verbose and len(docs) % 5000 == 0:
            print(f"[stream]   {len(docs):,}/{n_docs:,} ({time.time() - t0:.1f}s)", flush=True)
        if len(docs) >= n_docs:
            break
    if verbose:
        total_chars = sum(len(d) for d in docs)
        print(f"[stream] done — {len(docs):,} docs, {total_chars:,} chars in {time.time() - t0:.1f}s",
              flush=True)
    return docs


# ---------------------------------------------------------------------------
# Step 2: count letter n-grams (within-word, letters only)
# ---------------------------------------------------------------------------

def count_ngrams(docs: list[str], max_n: int = 5, verbose: bool = True) -> dict[int, Counter]:
    if verbose:
        print(f"[ngrams] counting 2..{max_n}-grams...", flush=True)
    counters = {n: Counter() for n in range(2, max_n + 1)}
    t0 = time.time()
    for di, doc in enumerate(docs):
        words: list[list[int]] = []
        cur: list[int] = []
        for ch in doc:
            c = ord(ch)
            if 65 <= c <= 90:
                c += 32
            if 97 <= c <= 122:
                cur.append(c - 97)
            else:
                if cur:
                    words.append(cur)
                    cur = []
        if cur:
            words.append(cur)
        for w in words:
            wlen = len(w)
            for s in range(wlen):
                for n in range(2, min(max_n + 1, wlen - s + 1)):
                    counters[n][tuple(w[s:s + n])] += 1
        if verbose and (di + 1) % 20000 == 0:
            print(f"[ngrams]   {di + 1:,}/{len(docs):,} docs ({time.time() - t0:.1f}s)", flush=True)
    if verbose:
        for n in range(2, max_n + 1):
            print(f"[ngrams]   {n}-grams: {len(counters[n]):,} distinct, "
                  f"total_freq={sum(counters[n].values()):,}", flush=True)
    return counters


# ---------------------------------------------------------------------------
# Step 3: optimize letter placement so frequent sequences = short paths
# ---------------------------------------------------------------------------

def optimize_placement(
    counters: dict[int, Counter],
    n_dims: int,
    n_steps: int = 5000,
    lr: float = 0.0001,
    top_k: int = 50_000,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    """Place 26 letters in n_dims so that loss = sum_seq(freq * path_length)
    is minimized. Returns (positions, sigma) where sigma is the median
    inter-letter distance."""
    all_ngrams: list[tuple[int, tuple]] = []
    for n, counter in counters.items():
        for ngram, freq in counter.items():
            all_ngrams.append((freq, ngram))
    all_ngrams.sort(reverse=True)
    constraints = all_ngrams[:top_k]

    if verbose:
        total_freq = sum(f for f, _ in all_ngrams)
        constraint_freq = sum(f for f, _ in constraints)
        print(f"[opt-{n_dims}D] {len(constraints):,}/{len(all_ngrams):,} constraints "
              f"({constraint_freq / max(total_freq, 1) * 100:.1f}% of frequency mass)",
              flush=True)

    c_freqs = np.array([f for f, _ in constraints], dtype=np.float64)
    c_seqs = [list(ng) for _, ng in constraints]
    c_freqs = c_freqs / c_freqs.max()

    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((26, n_dims)) * 0.5
    e_idx = ord('e') - ord('a')

    best_loss = float('inf')
    best_pos = pos.copy()
    t0 = time.time()

    for step in range(n_steps):
        grad = np.zeros_like(pos)
        total_loss = 0.0
        for ci in range(len(constraints)):
            seq = c_seqs[ci]
            freq = c_freqs[ci]
            for edge in range(len(seq) - 1):
                a, b = seq[edge], seq[edge + 1]
                diff = pos[a] - pos[b]
                dist = math.sqrt(float(np.sum(diff ** 2))) + 1e-8
                total_loss += freq * dist
                direction = diff / dist
                grad[a] += freq * direction
                grad[b] -= freq * direction
        pos -= lr * grad
        pos -= pos[e_idx]  # anchor 'e' at origin
        if total_loss < best_loss:
            best_loss = total_loss
            best_pos = pos.copy()
        if verbose and step > 0 and step % max(1, n_steps // 20) == 0:
            elapsed = time.time() - t0
            eta = (elapsed / step) * (n_steps - step)
            print(f"[opt-{n_dims}D]   step {step:5d}/{n_steps}: loss={total_loss:.4f} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)

    pos = best_pos
    dists = []
    for i in range(26):
        for j in range(i + 1, 26):
            dists.append(float(np.sqrt(float(np.sum((pos[i] - pos[j]) ** 2)))))
    sigma = float(np.median(dists)) if dists else 1.0

    if verbose:
        letters = "abcdefghijklmnopqrstuvwxyz"
        elapsed = time.time() - t0
        print(f"[opt-{n_dims}D] done in {elapsed:.0f}s (final loss={best_loss:.4f}, sigma={sigma:.4f})",
              flush=True)
        # Show neighborhoods of e, t, h
        for target in ["e", "t", "h", "i", "n", "s"]:
            ti = ord(target) - ord("a")
            ld = [(np.sqrt(np.sum((pos[ti] - pos[j]) ** 2)), letters[j])
                  for j in range(26) if j != ti]
            ld.sort()
            print(f"[opt-{n_dims}D]   {target} near: {', '.join(f'{l}({d:.3f})' for d, l in ld[:5])}",
                  flush=True)

    return pos, sigma


# ---------------------------------------------------------------------------
# Step 4: train v3 BPE merges using the fitted coordinates
# ---------------------------------------------------------------------------

def train_v3_bpe(
    docs: list[str],
    coords_3d: dict[str, tuple],
    sigma: float,
    num_merges: int = 241,
    verbose: bool = True,
):
    """Train v3 BPE on the streamed docs using the fitted spatial coords."""
    here = Path(__file__).parent.parent / "tokenizer"
    sys.path.insert(0, str(here))
    from bese_v3_fast_bpe import BeseV3FastBPE, train_bpe_merges_v3
    if verbose:
        print(f"[bpe] training {num_merges} merges on {len(docs):,} docs...", flush=True)
    merges = train_bpe_merges_v3(
        docs,
        num_merges=num_merges,
        coords_3d=coords_3d,
        sigma=sigma,
        decay=3.0,
        shape=0.02,  # pow_0.02 sweep winner
        verbose=verbose,
    )
    return BeseV3FastBPE(merges, coords_3d=coords_3d, sigma=sigma)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build BESE v3 prep artifacts locally")
    parser.add_argument("--docs", type=int, default=5000,
                        help="FineWeb docs to use (default: 5000 — smoke test)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Optimization steps (default: 500 — smoke test)")
    parser.add_argument("--top-k", type=int, default=50_000,
                        help="Top-K n-gram constraints (default: 50000)")
    parser.add_argument("--num-merges", type=int, default=241,
                        help="BPE merges to learn (default: 241 -> 288 vocab)")
    parser.add_argument("--bpe-docs", type=int, default=0,
                        help="Use this many docs for BPE (default: same as --docs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent.parent),
                        help="Project root (writes to <root>/artifacts/ and <root>/tokenizers/)")
    parser.add_argument("--skip-bpe", action="store_true",
                        help="Build coords only, skip BPE training")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    coords_path = out_root / "artifacts" / "letter_coords_v3.json"
    bpe_path = out_root / "tokenizers" / "bese_v3_bpe_241_v9.json"
    coords_path.parent.mkdir(parents=True, exist_ok=True)
    bpe_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  BESE v3 local prep")
    print(f"  docs={args.docs:,}  steps={args.steps:,}  top_k={args.top_k:,}  merges={args.num_merges}")
    print(f"  outputs:")
    print(f"    coords -> {coords_path}")
    print(f"    bpe    -> {bpe_path}")
    print("=" * 72, flush=True)

    t_total = time.time()

    # Stream + count
    docs = stream_fineweb(args.docs, min_words=50, verbose=True)
    counters = count_ngrams(docs, max_n=5, verbose=True)

    # Optimize 3D + 12D
    pos_3d, sigma_3d = optimize_placement(
        counters, n_dims=3, n_steps=args.steps,
        top_k=args.top_k, seed=args.seed, verbose=True,
    )
    pos_12d, _ = optimize_placement(
        counters, n_dims=12, n_steps=max(1, args.steps // 2),
        top_k=args.top_k, seed=args.seed, verbose=True,
    )

    letters = "abcdefghijklmnopqrstuvwxyz"
    coords_3d = {ch: [float(x) for x in pos_3d[i]] for i, ch in enumerate(letters)}
    coords_emb = {ch: [float(x) for x in pos_12d[i]] for i, ch in enumerate(letters)}

    blob = {
        "version": 3,
        "coords": coords_3d,
        "coords_emb": coords_emb,
        "sigma": sigma_3d,
        "letter_to_token_id": {ch: 4 + i for i, ch in enumerate(letters)},
        "n_docs_used": len(docs),
        "n_optimization_steps": args.steps,
        "top_k_constraints": args.top_k,
        "seed": args.seed,
    }
    with open(coords_path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"[save] coords -> {coords_path} ({coords_path.stat().st_size:,} bytes)", flush=True)

    # BPE training
    if not args.skip_bpe:
        bpe_doc_count = args.bpe_docs or args.docs
        bpe_docs = docs if bpe_doc_count <= len(docs) else docs + stream_fineweb(
            bpe_doc_count - len(docs), min_words=50, verbose=True
        )
        coords_3d_tup = {k: tuple(v) for k, v in coords_3d.items()}
        tok = train_v3_bpe(
            bpe_docs, coords_3d_tup, sigma_3d,
            num_merges=args.num_merges, verbose=True,
        )
        tok.save(bpe_path)
        print(f"[save] bpe    -> {bpe_path} ({bpe_path.stat().st_size:,} bytes)", flush=True)

    elapsed = time.time() - t_total
    print("=" * 72)
    print(f"  total wall: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
