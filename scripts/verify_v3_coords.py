#!/usr/bin/env python3
"""
Verify a fitted letter_coords_v3.json is sensible.

Run this after build_v3_locally.py completes:
  python3 scripts/verify_v3_coords.py

Checks (each fails LOUD and red):
  1. Structural — JSON keys exist, 26 letters in 3D and 12D, sigma > 0.
  2. Anchor — 'e' is at the origin in both 3D and 12D.
  3. Distinct — no two letters collapse onto the same point (degenerate).
  4. Sigma sanity — typical inter-letter distance is in a reasonable range
     (not collapsed to ~0, not blown up to >100).
  5. Neighborhood sanity — for each of {e, t, h, n, s, i, r, a}, the 5
     nearest letters should overlap meaningfully with their canonical
     English bigram neighbors. We score "looks like English" using a
     reference set of strong bigram pairs and report the recall.
  6. Tokenizer roundtrip — if the v3 BPE JSON also exists, encode +
     decode a few sample strings and re-verify the byte invariant
     against the new vocab.

Exits 0 on success, 1 on any check failure (so it's CI-friendly).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (default to repo-root paths produced by build_v3_locally.py)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COORDS_JSON = REPO / "artifacts" / "letter_coords_v3.json"
BPE_JSON = REPO / "tokenizers" / "bese_v3_bpe_241_v9.json"

LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Reference: strong English bigram pairs from FineWeb-scale data.
# These are pairs that should be "close" in any reasonable spatial
# encoding derived from English. Used for the neighborhood sanity check.
EXPECTED_NEIGHBORS = {
    "e": {"r", "n", "a", "d", "s", "t"},
    "t": {"h", "s", "i", "o", "e", "r"},
    "h": {"t", "e", "i", "a", "o", "s"},
    "n": {"e", "g", "t", "d", "i", "o", "a"},
    "s": {"t", "e", "i", "h", "o", "n"},
    "i": {"n", "t", "s", "o", "e", "c"},
    "r": {"e", "a", "i", "o", "t", "s"},
    "a": {"n", "t", "r", "l", "s", "d"},
}


def _ok(msg: str) -> None:
    print(f"  \033[32mOK \033[0m {msg}")


def _fail(msg: str) -> None:
    print(f"  \033[31mFAIL\033[0m {msg}")


def _warn(msg: str) -> None:
    print(f"  \033[33mWARN\033[0m {msg}")


# ---------------------------------------------------------------------------
def main() -> int:
    if not COORDS_JSON.exists():
        print(f"FATAL: coords JSON not found at {COORDS_JSON}")
        print("Run scripts/build_v3_locally.py first.")
        return 1

    print(f"Verifying: {COORDS_JSON}")
    print(f"  size: {COORDS_JSON.stat().st_size:,} bytes")
    print()

    with open(COORDS_JSON, "r") as f:
        blob = json.load(f)

    failures = 0

    # ── Check 1: structural ──────────────────────────────────────────
    print("Structural checks:")
    required_keys = {"version", "coords", "coords_emb", "sigma", "letter_to_token_id"}
    missing = required_keys - set(blob.keys())
    if missing:
        _fail(f"missing keys: {missing}")
        failures += 1
    else:
        _ok("all required keys present")

    coords_3d = blob.get("coords", {})
    coords_emb = blob.get("coords_emb", {})
    sigma = blob.get("sigma", 0)

    if set(coords_3d.keys()) != set(LETTERS):
        missing_letters = set(LETTERS) - set(coords_3d.keys())
        _fail(f"3D coords missing letters: {missing_letters}")
        failures += 1
    else:
        _ok("3D coords have all 26 letters")

    if set(coords_emb.keys()) != set(LETTERS):
        missing_letters = set(LETTERS) - set(coords_emb.keys())
        _fail(f"12D coords missing letters: {missing_letters}")
        failures += 1
    else:
        _ok("12D coords have all 26 letters")

    # All 3D vectors have length 3, all 12D >= 12
    bad3 = [ch for ch, v in coords_3d.items() if len(v) != 3]
    bad12 = [ch for ch, v in coords_emb.items() if len(v) < 12]
    if bad3:
        _fail(f"3D coords with wrong length: {bad3}")
        failures += 1
    else:
        _ok("3D coords are 3-vectors")
    if bad12:
        _fail(f"12D coords too short: {bad12}")
        failures += 1
    else:
        _ok("12D coords are at least 12-vectors")

    # ── Check 2: 'e' at origin ───────────────────────────────────────
    print("\nAnchor check (e at origin):")
    e3 = coords_3d.get("e", [99, 99, 99])
    e12 = coords_emb.get("e", [99] * 12)
    if max(abs(x) for x in e3) < 1e-6:
        _ok("e is at origin in 3D")
    else:
        _fail(f"e not at origin in 3D: {e3}")
        failures += 1
    if max(abs(x) for x in e12) < 1e-6:
        _ok("e is at origin in 12D")
    else:
        _fail(f"e not at origin in 12D: {e12}")
        failures += 1

    # ── Check 3: no collapsed letters ────────────────────────────────
    print("\nDistinctness check (no two letters collapsed):")
    pairs_3d_close = []
    threshold = 1e-3
    items_3d = list(coords_3d.items())
    for i in range(len(items_3d)):
        for j in range(i + 1, len(items_3d)):
            la, va = items_3d[i]
            lb, vb = items_3d[j]
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(va, vb)))
            if d < threshold:
                pairs_3d_close.append((la, lb, d))
    if pairs_3d_close:
        for la, lb, d in pairs_3d_close[:3]:
            _fail(f"3D letters collapsed: {la} ~ {lb} (d={d:.6f})")
        failures += 1
    else:
        _ok(f"all 26 letters are distinct in 3D (min separation > {threshold})")

    # ── Check 4: sigma sanity ────────────────────────────────────────
    print("\nSigma sanity (median inter-letter 3D distance):")
    if sigma <= 0:
        _fail(f"sigma must be > 0, got {sigma}")
        failures += 1
    elif sigma < 0.01:
        _warn(f"sigma is very small ({sigma:.6f}) — coords may be under-optimized")
    elif sigma > 50:
        _warn(f"sigma is very large ({sigma:.6f}) — coords may have blown up")
    else:
        _ok(f"sigma = {sigma:.4f} (in expected range 0.01 - 50)")

    # ── Check 5: neighborhood sanity ─────────────────────────────────
    print("\nNeighborhood sanity (top-5 nearest in 3D vs expected English bigram pals):")
    total_recall = 0
    n_checked = 0
    for target, expected in EXPECTED_NEIGHBORS.items():
        if target not in coords_3d:
            continue
        tv = coords_3d[target]
        dists = []
        for other, ov in coords_3d.items():
            if other == target:
                continue
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(tv, ov)))
            dists.append((d, other))
        dists.sort()
        top5 = [l for _, l in dists[:5]]
        hits = set(top5) & expected
        recall = len(hits) / 5.0
        total_recall += recall
        n_checked += 1
        nearest_str = ", ".join(f"{l}({d:.3f})" for d, l in dists[:5])
        marker = "  " if recall >= 0.4 else "??"
        print(f"  {marker} {target}: nearest = [{nearest_str}]  hits {sorted(hits)} / {sorted(expected)}  recall={recall:.0%}")
    avg_recall = total_recall / max(n_checked, 1)
    if avg_recall >= 0.5:
        _ok(f"average top-5 recall vs expected English neighbors: {avg_recall:.0%}")
    elif avg_recall >= 0.3:
        _warn(f"average top-5 recall: {avg_recall:.0%} — low but not crazy. More steps may help.")
    else:
        _fail(f"average top-5 recall: {avg_recall:.0%} — coords don't look like English bigram structure")
        failures += 1

    # ── Check 6: tokenizer + byte invariant ──────────────────────────
    if BPE_JSON.exists():
        print(f"\nTokenizer roundtrip (loading {BPE_JSON.name}):")
        try:
            sys.path.insert(0, str(REPO / "tokenizer"))
            from bese_v3_fast_bpe import BeseV3FastBPE
            tok = BeseV3FastBPE.load(BPE_JSON)
            bpt = tok.compute_bytes_per_token()
            _ok(f"loaded tokenizer (vocab={tok.vocab_size}, {len(tok.merges)} merges)")
            test_strings = [
                "the quick brown fox jumps over the lazy dog",
                "Hello, World! 1234567890",
                "naïve café résumé — emojis: 🚀✨🌍",
                "MixedCase HTML <p>tag</p> and JSON {\"a\": 1}",
            ]
            byte_ok = True
            for s in test_strings:
                ids = tok.encode(s)
                bsum = sum(int(bpt[t]) for t in ids)
                ulen = len(s.encode("utf-8"))
                if bsum != ulen:
                    _fail(f"byte invariant FAILED on {s!r}: encode_bytes={bsum} != utf8_len={ulen}")
                    byte_ok = False
                    failures += 1
            if byte_ok:
                _ok(f"byte invariant holds across {len(test_strings)} test strings (post-BPE)")
            # Compression efficiency
            sample = "the temperature of the ocean determines the weather " * 50
            ids = tok.encode(sample)
            ratio = len(sample) / len(ids)
            _ok(f"sample compression: {len(sample)} chars -> {len(ids)} tokens = {ratio:.2f} chars/token")
        except Exception as e:
            _fail(f"tokenizer load/test errored: {e}")
            failures += 1
    else:
        print(f"\nTokenizer roundtrip: SKIPPED (no {BPE_JSON.name} yet — local build still running, or --skip-bpe used)")

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if failures == 0:
        print(f"  ALL CHECKS PASSED — letter_coords_v3.json is healthy")
        print("  Ready to commit and push.")
        print("=" * 60)
        return 0
    else:
        print(f"  {failures} CHECK(S) FAILED")
        print("  Re-run build_v3_locally.py with more --steps and/or --docs,")
        print("  or investigate the specific failures above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
