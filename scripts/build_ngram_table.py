#!/usr/bin/env python3
"""Build a compact n-gram frequency table for eval-time tilt.

Scans a binary training shard in parallel across multiple CPUs and
produces a compressed pickle containing the top-k next token for each
observed n-gram prefix. The output is small enough to embed in the
16 MB artifact.
"""
import argparse
import multiprocessing as mp
import numpy as np
import pickle
import zlib
from pathlib import Path

HEADER_INTS = 256

# Set as a module-level global before Pool creation so forked workers
# inherit it via copy-on-write — avoids pickling 200 MB over IPC.
_shard: "np.ndarray | None" = None


def _process_chunk(args: tuple) -> dict:
    """Count n-gram continuations for a contiguous slice of the shard."""
    global _shard
    global_start, global_end, max_n = args
    tables: dict[int, dict] = {n: {} for n in range(2, max_n + 1)}
    for i in range(global_start, global_end):
        for n in range(2, max_n + 1):
            if i >= n - 1:
                prefix = tuple(int(x) for x in _shard[i - n + 1:i])
                tbl = tables[n]
                if prefix not in tbl:
                    tbl[prefix] = {}
                cnt = tbl[prefix]
                token = int(_shard[i])
                cnt[token] = cnt.get(token, 0) + 1
    return tables


def main():
    ap = argparse.ArgumentParser(description="Build compressed n-gram table from a shard")
    ap.add_argument('--shard', required=True, help='One training shard (.bin) to scan')
    ap.add_argument('--output', required=True, help='Output compressed n-gram table')
    ap.add_argument('--max-n', type=int, default=4, help='Maximum n-gram order')
    ap.add_argument('--top-k', type=int, default=1, help='Keep top-k next tokens per prefix')
    ap.add_argument('--max-tokens', type=int, default=100_000_000,
                    help='Max tokens to scan from shard')
    ap.add_argument('--workers', type=int, default=None,
                    help='Number of parallel workers (default: all CPUs)')
    args = ap.parse_args()

    global _shard
    header_bytes = HEADER_INTS * 4
    _shard = np.fromfile(args.shard, dtype=np.uint16, offset=header_bytes)[:args.max_tokens]
    print(f"Loaded {len(_shard):,} tokens from {args.shard}", flush=True)

    num_workers = min(args.workers or mp.cpu_count(), len(_shard))
    print(f"Using {num_workers} workers", flush=True)

    chunk_size = (len(_shard) + num_workers - 1) // num_workers
    work_items = [
        (k * chunk_size, min((k + 1) * chunk_size, len(_shard)), args.max_n)
        for k in range(num_workers)
        if k * chunk_size < len(_shard)
    ]

    # Fork after _shard is set — workers inherit it read-only (copy-on-write),
    # no pickling of the array over IPC.
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_process_chunk, work_items)

    print(f"Merging {len(results)} partial tables...", flush=True)
    merged: dict[int, dict] = {n: {} for n in range(2, args.max_n + 1)}
    for partial in results:
        for n in range(2, args.max_n + 1):
            for prefix, counts in partial[n].items():
                if prefix not in merged[n]:
                    merged[n][prefix] = {}
                for token, count in counts.items():
                    merged[n][prefix][token] = merged[n][prefix].get(token, 0) + count

    # Compress: keep only top-k per prefix
    compact: dict[int, dict] = {}
    for n, table in merged.items():
        compact[n] = {}
        for prefix, counts in table.items():
            top = sorted(counts.items(), key=lambda x: -x[1])[:args.top_k]
            compact[n][prefix] = top
        print(f"  n={n}: {len(compact[n]):,} prefixes", flush=True)

    data = pickle.dumps(compact)
    compressed = zlib.compress(data, level=9)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        f.write(compressed)
    print(
        f"Wrote {len(compressed):,} bytes ({len(compressed) / 1024 / 1024:.2f} MB) "
        f"to {args.output}",
        flush=True,
    )


if __name__ == '__main__':
    main()
