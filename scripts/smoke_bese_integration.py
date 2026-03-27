#!/usr/bin/env python3
"""
Smoke test (no GPU required): tokenizer JSON round-trip, byte identity, shard header I/O.

Tests BOTH the old BESEBPETokenizer and the fast FastBESEBPETokenizer to ensure
the production codepath (fast) is exercised.

Run from repo root:
  .venv/bin/python scripts/smoke_bese_integration.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOK_DIR = ROOT / "tokenizer"
sys.path.insert(0, str(TOK_DIR))

from bese_bpe_tokenizer import BESEBPETokenizer, train_bpe_merges  # noqa: E402
from bese_fast_bpe import FastBESEBPETokenizer, train_bpe_merges_fast  # noqa: E402


def test_shard_roundtrip(shard_path: Path) -> None:
    import numpy as np

    header = np.fromfile(shard_path, dtype="<i4", count=256)
    assert int(header[0]) == 20240520, "magic"
    assert int(header[1]) == 1, "version"
    n = int(header[2])
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens = np.fromfile(shard_path, dtype="<u2", count=n, offset=header_bytes)
    assert tokens.size == n, "token count"


def main() -> int:
    sample = ROOT / "fixtures" / "sample_docs.jsonl"
    texts = []
    with sample.open(encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    # --- Test 1: Old (slow) tokenizer ---
    print("Testing BESEBPETokenizer (slow)...")
    merges = train_bpe_merges(texts * 50, num_merges=32, verbose=False)
    tok = BESEBPETokenizer(merges=merges)
    bpt = tok.get_bytes_per_token_lut()
    for t in texts:
        enc = tok.encode(t)
        assert sum(bpt[x] for x in enc) == len(t.encode("utf-8")), "BPB bytes (slow)"
    print("  BESEBPETokenizer: OK")

    # --- Test 2: Fast tokenizer (production path) ---
    print("Testing FastBESEBPETokenizer (fast)...")
    fast_merges = train_bpe_merges_fast(texts * 50, num_merges=32, verbose=False)
    fast_tok = FastBESEBPETokenizer(merges=fast_merges)
    fast_bpt = fast_tok.get_bytes_per_token_lut()
    for t in texts:
        enc = fast_tok.encode(t)
        tb = int(sum(fast_bpt[x] for x in enc))
        ub = len(t.encode("utf-8"))
        assert tb == ub, f"BPB bytes (fast): token_bytes={tb} utf8={ub}"
    print("  FastBESEBPETokenizer: OK")

    # --- Test 3: Fast tokenizer edge cases ---
    print("Testing edge cases...")
    # Empty text
    enc_empty = fast_tok.encode("")
    assert len(enc_empty) == 0, "empty text should produce empty tokens"

    # Single character
    enc_single = fast_tok.encode("a")
    assert int(sum(fast_bpt[x] for x in enc_single)) == 1, "single char byte count"

    # Multi-byte UTF-8
    for ch in ["é", "ñ", "ü", "中"]:
        enc_mb = fast_tok.encode(ch)
        assert int(sum(fast_bpt[x] for x in enc_mb)) == len(ch.encode("utf-8")), f"multi-byte {ch}"

    # Round-trip: save/load fast tokenizer
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        json_path = tdir / "fast_tok.json"
        fast_tok.save(json_path)
        fast_tok2 = FastBESEBPETokenizer.load(json_path)
        assert fast_tok2.vocab_size == fast_tok.vocab_size, "vocab_size mismatch after load"
        for t in texts:
            enc1 = list(fast_tok.encode(t))
            enc2 = list(fast_tok2.encode(t))
            assert enc1 == enc2, f"encode mismatch after save/load for: {t[:40]}"
    print("  Edge cases: OK")

    # --- Test 4: Shard export with fast tokenizer ---
    print("Testing shard export...")
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        json_path = tdir / "tok.json"
        fast_tok.save(json_path)

        out = tdir / "ds"
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "export_shards.py"),
                "--input",
                str(sample),
                "--tokenizer",
                str(json_path),
                "--output-dir",
                str(out),
                "--val-docs",
                "2",
                "--shard-tokens",
                "500",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            return r.returncode
        val_bin = out / "fineweb_val_0.bin"
        assert val_bin.is_file(), "val shard"
        test_shard_roundtrip(val_bin)
        train_bins = list(out.glob("fineweb_train_*.bin"))
        assert train_bins, "train shards"
        test_shard_roundtrip(train_bins[0])
    print("  Shard export: OK")

    print("\nsmoke_bese_integration: ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
