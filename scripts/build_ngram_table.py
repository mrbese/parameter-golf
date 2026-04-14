#!/usr/bin/env python3
"""Build a compact n-gram frequency table for eval-time tilt.

Scans a binary training shard and produces a compressed pickle
containing the top-k next token for each observed n-gram prefix.
The output is small enough (~200-500 KB) to embed in the 16 MB artifact.
"""
import argparse
import numpy as np
from collections import defaultdict
import pickle
import zlib

HEADER_INTS = 256


def main():
    ap = argparse.ArgumentParser(description="Build compressed n-gram table from a shard")
    ap.add_argument('--shard', required=True, help='One training shard (.bin) to scan')
    ap.add_argument('--output', required=True, help='Output compressed n-gram table')
    ap.add_argument('--max-n', type=int, default=4, help='Maximum n-gram order')
    ap.add_argument('--top-k', type=int, default=1, help='Keep top-k next tokens per prefix')
    ap.add_argument('--max-tokens', type=int, default=100_000_000,
                    help='Max tokens to scan from shard')
    args = ap.parse_args()

    # Read binary shard: skip 256 x int32 header
    header_bytes = HEADER_INTS * 4
    shard = np.fromfile(args.shard, dtype=np.uint16, offset=header_bytes)[:args.max_tokens]
    print(f"Loaded {len(shard):,} tokens from {args.shard}")

    tables = {n: defaultdict(lambda: defaultdict(int)) for n in range(2, args.max_n + 1)}

    for i in range(len(shard)):
        for n in range(2, args.max_n + 1):
            if i >= n - 1:
                prefix = tuple(int(x) for x in shard[i - n + 1:i])
                tables[n][prefix][int(shard[i])] += 1

        if i % 10_000_000 == 0 and i > 0:
            print(f"  scanned {i:,} / {len(shard):,} tokens")

    # Compress: keep only top-k per prefix
    compact = {}
    for n, table in tables.items():
        compact[n] = {}
        for prefix, counts in table.items():
            top = sorted(counts.items(), key=lambda x: -x[1])[:args.top_k]
            compact[n][prefix] = top
        print(f"  n={n}: {len(compact[n]):,} prefixes")

    data = pickle.dumps(compact)
    compressed = zlib.compress(data, level=9)

    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        f.write(compressed)
    print(f"Wrote {len(compressed):,} bytes of compressed n-gram table to {args.output}")


if __name__ == '__main__':
    main()
