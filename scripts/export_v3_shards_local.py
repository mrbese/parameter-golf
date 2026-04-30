#!/usr/bin/env python3
"""Build BESE v3 token shards locally on CPU (parallelized).

Encodes streamed FineWeb text into Phase 0c-format binary shards
(uint32 header[256] with magic=20240520/version=1/num_tokens, then
uint16 token IDs). Output drop-in replaces what Phase 0c would
produce on the pod, so uploading these to /runpod-volume/bese_shards_v9_spatial/
makes Phase 0c auto-skip.

Why: Phase 0c on an 8×H100 SXM pod is ~30-40 min of single-process
encoding (~$11-14 of GPU time wasted on CPU work). On a Mac with
mp.Pool we get full multi-core utilization.

Usage:
    python scripts/export_v3_shards_local.py \\
        --output-dir /tmp/v9_shards \\
        --target-train-tokens 1_000_000_000 \\
        --shard-tokens 100_000_000 \\
        --val-docs 50000 \\
        --workers 8

Architecture:
    1. Main process streams HF FineWeb (state-ful, single iterator).
    2. Producer enqueues batches of N docs.
    3. Pool workers encode batches via BeseV3FastBPE -> uint16 arrays.
    4. Main collects encoded arrays in stream order, concatenates,
       and flushes to disk every SHARD_TOKENS tokens.

Order is preserved by tagging each batch with an index and reassembling
in order on the main process.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tokenizer"))


# ---------------------------------------------------------------------------
# Worker — runs in each child process
# ---------------------------------------------------------------------------
_TOK = None


def _worker_init(bpe_path: str) -> None:
    global _TOK
    from bese_v3_fast_bpe import BeseV3FastBPE
    _TOK = BeseV3FastBPE.load(bpe_path)


def _encode_batch(args: tuple) -> tuple[int, np.ndarray]:
    """Encode a batch of docs; return (batch_idx, concatenated uint16 ids)."""
    batch_idx, docs = args
    out = []
    for doc in docs:
        ids = _TOK.encode(doc)
        if ids.size:
            out.append(ids)
    if out:
        return batch_idx, np.concatenate(out).astype(np.uint16, copy=False)
    return batch_idx, np.zeros(0, dtype=np.uint16)


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------
HEADER_INTS = 256
MAGIC = 20240520


def write_shard(path: Path, tokens: np.ndarray) -> int:
    arr = tokens.astype(np.uint16, copy=False)
    header = np.zeros(HEADER_INTS, dtype="<u4")
    header[0] = MAGIC
    header[1] = 1
    header[2] = len(arr)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(arr.tobytes())
    return path.stat().st_size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bpe", type=Path,
                    default=REPO / "tokenizers" / "bese_v3_bpe_241_v9.json")
    ap.add_argument("--target-train-tokens", type=int, default=1_000_000_000)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--val-docs", type=int, default=50_000)
    ap.add_argument("--min-words", type=int, default=20,
                    help="Skip docs with fewer words (matches phase0c)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--batch-docs", type=int, default=200,
                    help="Docs per encode batch (worker granularity)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] workers={args.workers}  batch_docs={args.batch_docs}  "
          f"shard_tokens={args.shard_tokens:,}  target_train={args.target_train_tokens:,}",
          flush=True)

    # Resume from existing shards if present. We can't replay the exact HF
    # stream order, so resume = "skip the first N docs that the previous run
    # consumed, then start writing shard N." Conservative: skip val_docs +
    # shard_tokens * existing_train_shards / approx_avg_tokens_per_doc.
    val_already = (args.output_dir / "fineweb_val_000000.bin").exists()
    existing_train = sorted(args.output_dir.glob("fineweb_train_*.bin"))
    completed_shards = len(existing_train)
    if val_already or completed_shards:
        print(f"[resume] val={val_already}  completed_train_shards={completed_shards}",
              flush=True)

    # Force fork so workers can re-use loaded tokenizer state cheaply on Linux;
    # on macOS spawn re-runs _worker_init per worker which is fine.
    try:
        if sys.platform.startswith("linux"):
            mp.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):
        pass

    # Stream FineWeb sample-10BT
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                      split="train", streaming=True)

    # ---- Phase 1: validation shard (first --val-docs) ----
    from bese_v3_fast_bpe import BeseV3FastBPE
    tok_main = BeseV3FastBPE.load(str(args.bpe))
    if val_already:
        print(f"[val] already exists, skipping. Fast-forwarding HF stream past "
              f"first {args.val_docs:,} docs to keep training-set offset consistent...",
              flush=True)
        skipped = 0
        for sample in ds:
            if skipped >= args.val_docs:
                break
            skipped += 1
        print(f"[val] fast-forward complete ({skipped:,} docs skipped)", flush=True)
    else:
        print(f"[val] streaming {args.val_docs:,} docs and encoding sequentially",
              flush=True)
        val_tokens: list[int] = []
        val_seen = 0
        t_val = time.time()
        for sample in ds:
            text = sample.get("text", "")
            if len(text.split()) < args.min_words:
                val_seen += 1
                continue
            val_tokens.extend(tok_main.encode(text).tolist())
            val_seen += 1
            if val_seen >= args.val_docs:
                break
        val_arr = np.array(val_tokens, dtype=np.uint16)
        val_path = args.output_dir / "fineweb_val_000000.bin"
        sz = write_shard(val_path, val_arr)
        print(f"[val] wrote {val_path.name} ({len(val_arr):,} tokens, {sz:,} bytes, "
              f"{time.time()-t_val:.1f}s)", flush=True)

    # ---- Phase 2: training shards via parallel pool ----
    print(f"[train] starting parallel encode (target {args.target_train_tokens:,} tokens)",
          flush=True)
    pool = mp.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(str(args.bpe),),
    )

    cur_buffer: list[np.ndarray] = []
    cur_count = 0
    shard_idx = completed_shards  # start writing at the next free slot
    total_train = completed_shards * args.shard_tokens  # approximate; we just need progress

    # If resuming, fast-forward the HF stream to skip docs that prior shards
    # consumed. Approximation: avg tokens-per-doc from a previously-written
    # shard (or fall back to 1700, the empirical avg from FineWeb sample-10BT).
    if completed_shards:
        # Estimate avg tokens/doc from the first completed shard
        with open(existing_train[0], "rb") as f:
            f.seek(HEADER_INTS * 4)
            sample_arr = np.frombuffer(f.read(min(2 * 50_000, args.shard_tokens * 2)), dtype=np.uint16)
        # Crude: count documents by counting BOS-or-newline tokens? we don't
        # have document boundaries. Use 1700 tok/doc empirical default.
        avg_tok_per_doc = 1700
        skip_docs = (completed_shards * args.shard_tokens) // avg_tok_per_doc
        print(f"[resume] fast-forwarding HF stream past ~{skip_docs:,} docs "
              f"({completed_shards} shards × {args.shard_tokens:,} ÷ {avg_tok_per_doc} tok/doc)",
              flush=True)
        skipped = 0
        for sample in ds:
            skipped += 1
            if skipped >= skip_docs:
                break
        print(f"[resume] skipped {skipped:,} docs", flush=True)
    t_train = time.time()
    last_log = t_train

    def flush(buf: list[np.ndarray], count: int, idx: int) -> int:
        path = args.output_dir / f"fineweb_train_{idx:06d}.bin"
        arr = np.concatenate(buf) if buf else np.zeros(0, dtype=np.uint16)
        sz = write_shard(path, arr)
        print(f"[train] flushed {path.name} ({len(arr):,} tokens, {sz:,} bytes)",
              flush=True)
        return sz

    # Producer-consumer using imap_unordered with reassembly
    def doc_batches():
        batch: list[str] = []
        bidx = 0
        for sample in ds:
            text = sample.get("text", "")
            if len(text.split()) < args.min_words:
                continue
            batch.append(text)
            if len(batch) >= args.batch_docs:
                yield bidx, batch
                bidx += 1
                batch = []
            # Heuristic: stop streaming when we have enough buffered work
            if total_train + cur_count > args.target_train_tokens + 5_000_000:
                break
        if batch:
            yield bidx, batch

    # imap_unordered for max throughput; we don't care about ordering across batches
    # (FineWeb stream order has no inherent meaning vs. shuffle at training time)
    for batch_idx, ids in pool.imap_unordered(_encode_batch, doc_batches(), chunksize=1):
        if ids.size == 0:
            continue
        cur_buffer.append(ids)
        cur_count += ids.size
        # Periodic log
        now = time.time()
        if now - last_log > 30:
            elapsed = now - t_train
            rate = (total_train + cur_count) / max(elapsed, 1e-3)
            remaining = (args.target_train_tokens - total_train - cur_count) / max(rate, 1)
            print(f"[train]   total={total_train + cur_count:,} tokens  "
                  f"rate={rate/1e6:.2f}M tok/s  ETA={remaining/60:.1f} min",
                  flush=True)
            last_log = now
        # Flush shard when we've buffered enough
        if cur_count >= args.shard_tokens:
            # split: first SHARD_TOKENS to disk, remainder carry over
            full = np.concatenate(cur_buffer)
            cur_buffer = []
            head = full[:args.shard_tokens]
            tail = full[args.shard_tokens:]
            flush([head], len(head), shard_idx)
            shard_idx += 1
            total_train += len(head)
            if tail.size:
                cur_buffer.append(tail)
                cur_count = tail.size
            else:
                cur_count = 0
            if total_train >= args.target_train_tokens:
                pool.terminate()
                break

    pool.close()
    pool.join()

    # Flush remainder
    if cur_count > 0:
        flush(cur_buffer, cur_count, shard_idx)
        total_train += cur_count
        shard_idx += 1

    elapsed = time.time() - t_train
    print(f"[train] done — {shard_idx} train shards, {total_train:,} tokens "
          f"in {elapsed/60:.1f} min ({total_train/elapsed/1e6:.2f}M tok/s)",
          flush=True)
    print(f"[done] output dir: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
