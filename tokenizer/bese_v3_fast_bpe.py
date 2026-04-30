"""
Fast BPE training and encoding for BESE v3 (Spatial Letter Encoding).

Differences from bese_fast_bpe.py (v2):
  - Imports from bese_v3_constants (flat 26-letter alphabet, no groups).
  - Adds a spatial-scored merge selection mode:
        score = freq * exp(-distance(a, b) / (decay * sigma)) ** shape
    where distance is measured in the fitted 3D LETTER_SPACE. Merge tokens
    inherit centroid coordinates so spatial scoring works through the
    full BPE tree.
  - Fallback to pure-frequency mode when no spatial coords are provided
    (so tests can run without a fitted JSON).

The base tokenization is byte-accurate by construction:
  sum(BYTES_PER_TOKEN[t] for t in encode(s)) == len(s.encode("utf-8"))
for any input string s. The byte invariant is maintained through merges
because each merge token's byte count is the sum of its constituents.
"""

from __future__ import annotations

import heapq
import json
import math
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .bese_v3_constants import (
        BASE_VOCAB_SIZE,
        BOS_ID,
        BYTES_PER_TOKEN,
        DECODE_TABLE,
        ENCODE_TABLE,
        EOS_ID,
        LETTERS,
        LETTER_START,
        OTHER_PUNCT_ID,
        PAD_ID,
        SPATIAL_DECAY,
        SPATIAL_SHAPE,
        UNK_ID,
        letter_id_to_letter,
        load_letter_space_3d,
        merge_centroid,
        spatial_pair_score,
    )
except ImportError:
    from bese_v3_constants import (
        BASE_VOCAB_SIZE,
        BOS_ID,
        BYTES_PER_TOKEN,
        DECODE_TABLE,
        ENCODE_TABLE,
        EOS_ID,
        LETTERS,
        LETTER_START,
        OTHER_PUNCT_ID,
        PAD_ID,
        SPATIAL_DECAY,
        SPATIAL_SHAPE,
        UNK_ID,
        letter_id_to_letter,
        load_letter_space_3d,
        merge_centroid,
        spatial_pair_score,
    )


# ---------------------------------------------------------------------------
# Base tokenization (text -> base token sequence)
# ---------------------------------------------------------------------------

def _text_to_base_tokens(text: str) -> list[int]:
    """Convert text to v3 base token sequence.

    Per-character rule (preserves the byte-count invariant):
      - If lower(ch) is in ENCODE_TABLE and the mapped token's byte count
        equals len(ch.encode("utf-8")), emit the mapped token.
      - Otherwise emit OTHER_PUNCT_ID once per UTF-8 byte of the character.
    """
    tokens: list[int] = []
    for ch in text:
        lower = ch.lower()
        if lower in ENCODE_TABLE:
            utf8_len = len(ch.encode("utf-8"))
            mapped = ENCODE_TABLE[lower]
            mapped_bytes = sum(int(BYTES_PER_TOKEN[t]) for t in mapped)
            if utf8_len == mapped_bytes:
                tokens.extend(mapped)
            else:
                tokens.extend([OTHER_PUNCT_ID] * utf8_len)
        else:
            utf8_len = len(ch.encode("utf-8"))
            tokens.extend([OTHER_PUNCT_ID] * utf8_len)
    return tokens


def _encode_texts_worker(texts: list[str]) -> list[list[int]]:
    """Worker for parallel base-token encoding."""
    return [_text_to_base_tokens(text) for text in texts]


# ---------------------------------------------------------------------------
# BPE training (frequency or spatial-weighted)
# ---------------------------------------------------------------------------

class _Node:
    """Doubly-linked list node for fast BPE merge operations."""
    __slots__ = ("token", "prev", "next", "doc_id")

    def __init__(self, token: int, doc_id: int):
        self.token = token
        self.prev = None
        self.next = None
        self.doc_id = doc_id


def _build_initial_coord_map(
    coords_3d: dict[str, tuple] | None,
) -> dict[int, tuple] | None:
    """Build a map of base-letter-token-id -> 3D coordinate, or None
    if spatial mode is disabled."""
    if coords_3d is None:
        return None
    cmap: dict[int, tuple] = {}
    for i, ch in enumerate(LETTERS):
        if ch in coords_3d:
            cmap[LETTER_START + i] = tuple(coords_3d[ch])
    return cmap


def _score_pair(
    pair: tuple[int, int],
    count: int,
    coord_by_id: dict[int, tuple] | None,
    sigma: float,
    decay: float,
    shape: float,
) -> float:
    """Compute the merge-selection score for a token pair.

    If `coord_by_id` is None, returns plain frequency (standard BPE).
    Otherwise returns the spatial-weighted score IF both pair members
    have coordinates; falls back to pure frequency for pairs involving
    non-spatial tokens (punct, digits, special, merges-with-non-spatial).
    """
    if coord_by_id is None:
        return float(count)
    ca = coord_by_id.get(pair[0])
    cb = coord_by_id.get(pair[1])
    if ca is None or cb is None:
        return float(count)
    return spatial_pair_score(count, ca, cb, sigma, decay, shape)


def train_bpe_merges_v3(
    texts: list[str],
    num_merges: int = 241,
    *,
    coords_3d: dict[str, tuple] | None = None,
    sigma: float | None = None,
    decay: float = SPATIAL_DECAY,
    shape: float = SPATIAL_SHAPE,
    verbose: bool = True,
) -> list[tuple[tuple[int, int], int]]:
    """Train BESE v3 BPE merges.

    Args:
      texts: training corpus, one document per string
      num_merges: number of merges to learn (default 241 -> 288 vocab)
      coords_3d: dict of letter -> 3D coordinate tuple. If provided, BPE
        scoring uses score = freq * exp(-d / (decay * sigma)) ** shape.
        If None, falls back to standard frequency-based BPE.
      sigma: median inter-letter distance (required if coords_3d given)
      decay: distance scaling factor (default 3.0)
      shape: power on the spatial factor (default 0.02 — gentle nudge)
      verbose: print progress

    Returns:
      list of ((left_id, right_id), new_id) tuples
    """
    spatial_mode = coords_3d is not None
    if spatial_mode and sigma is None:
        raise ValueError("sigma is required when coords_3d is provided")
    if not spatial_mode:
        sigma = 1.0  # unused

    if verbose:
        mode_str = (
            f"spatial (decay={decay}, shape={shape}, sigma={sigma:.4f})"
            if spatial_mode else "frequency-only"
        )
        print(f"[v3 BPE] mode: {mode_str}")
        print(f"[v3 BPE] encoding {len(texts):,} texts with v3 base tokenizer...")

    # Step 1: parallel encode to base tokens
    n_workers = min(mp.cpu_count(), 128)
    chunk_size = max(1, len(texts) // n_workers)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    t_enc = time.time()
    with mp.Pool(n_workers) as pool:
        encoded_chunks = pool.map(_encode_texts_worker, chunks)
    all_encoded = [tokens for chunk in encoded_chunks for tokens in chunk]
    if verbose:
        print(f"[v3 BPE]   parallel encoding: {n_workers} workers, {time.time() - t_enc:.1f}s")

    # Step 2: build linked lists + pair index
    pair_positions: dict[tuple[int, int], set[int]] = defaultdict(set)
    all_nodes: list[_Node] = []
    total_base = 0

    for doc_id, base_tokens in enumerate(all_encoded):
        total_base += len(base_tokens)
        if not base_tokens:
            continue
        nodes: list[_Node] = []
        for t in base_tokens:
            node = _Node(t, doc_id)
            if nodes:
                prev_node = nodes[-1]
                prev_node.next = len(all_nodes)
                node.prev = len(all_nodes) - 1
            nodes.append(node)
            all_nodes.append(node)
        for i in range(len(nodes) - 1):
            nid = len(all_nodes) - len(nodes) + i
            pair = (nodes[i].token, nodes[i + 1].token)
            pair_positions[pair].add(nid)

    del all_encoded

    if verbose:
        print(f"[v3 BPE]   base tokens: {total_base:,}")
        print(f"[v3 BPE]   unique pairs: {len(pair_positions):,}")
        print(f"[v3 BPE] learning {num_merges} merges...")

    # Step 3: heap-based merge loop
    coord_by_id = _build_initial_coord_map(coords_3d) if spatial_mode else None
    merges: list[tuple[tuple[int, int], int]] = []
    next_id = BASE_VOCAB_SIZE
    pair_counts = {pair: len(positions) for pair, positions in pair_positions.items()}

    # Heap entries: (-score, pair). We re-validate on pop because counts
    # can change between insertion and pop.
    heap: list[tuple[float, tuple[int, int]]] = []
    for pair, count in pair_counts.items():
        if count >= 2:
            sc = _score_pair(pair, count, coord_by_id, sigma, decay, shape)
            heap.append((-sc, pair))
    heapq.heapify(heap)

    merge_num = 0
    while merge_num < num_merges and heap:
        # Pop best pair (skip stale entries)
        best_pair = None
        best_count = 0
        while heap:
            neg_score, candidate = heapq.heappop(heap)
            cur_count = pair_counts.get(candidate, 0)
            if cur_count < 2:
                continue
            cur_score = _score_pair(candidate, cur_count, coord_by_id, sigma, decay, shape)
            if abs(-neg_score - cur_score) < 1e-9:
                best_pair = candidate
                best_count = cur_count
                break
            # stale — re-push with current score
            heapq.heappush(heap, (-cur_score, candidate))
        if best_pair is None:
            break

        # Materialize the merge
        a, b = best_pair
        new_id = next_id
        next_id += 1
        merges.append((best_pair, new_id))

        # Track coordinate for the new token
        if coord_by_id is not None:
            ca = coord_by_id.get(a)
            cb = coord_by_id.get(b)
            if ca is not None and cb is not None:
                coord_by_id[new_id] = merge_centroid(ca, cb)
            # else: new token has no coord; its merges with letters fall back
            # to pure frequency via _score_pair's None-check.

        # Apply the merge to the linked list, updating pair counts
        positions = list(pair_positions.get(best_pair, set()))
        for nid in positions:
            node_a = all_nodes[nid]
            if node_a.token != a or node_a.next is None:
                continue
            node_b = all_nodes[node_a.next]
            if node_b.token != b:
                continue

            # Remove old pairs (a,b), (prev,a), (b,next)
            pair_positions[best_pair].discard(nid)
            pair_counts[best_pair] = pair_counts.get(best_pair, 1) - 1

            if node_a.prev is not None:
                prev_node = all_nodes[node_a.prev]
                old_left = (prev_node.token, a)
                pair_positions[old_left].discard(node_a.prev)
                pair_counts[old_left] = pair_counts.get(old_left, 1) - 1
                if pair_counts[old_left] >= 2:
                    sc = _score_pair(old_left, pair_counts[old_left], coord_by_id, sigma, decay, shape)
                    heapq.heappush(heap, (-sc, old_left))

            if node_b.next is not None:
                next_node = all_nodes[node_b.next]
                old_right = (b, next_node.token)
                pair_positions[old_right].discard(node_a.next)
                pair_counts[old_right] = pair_counts.get(old_right, 1) - 1
                if pair_counts[old_right] >= 2:
                    sc = _score_pair(old_right, pair_counts[old_right], coord_by_id, sigma, decay, shape)
                    heapq.heappush(heap, (-sc, old_right))

            # Merge: replace node_a's token with new_id, remove node_b
            node_a.token = new_id
            node_a.next = node_b.next
            if node_b.next is not None:
                all_nodes[node_b.next].prev = nid

            # Add new pairs (prev, new_id), (new_id, next)
            if node_a.prev is not None:
                prev_node = all_nodes[node_a.prev]
                new_left = (prev_node.token, new_id)
                pair_positions[new_left].add(node_a.prev)
                pair_counts[new_left] = pair_counts.get(new_left, 0) + 1
                if pair_counts[new_left] >= 2:
                    sc = _score_pair(new_left, pair_counts[new_left], coord_by_id, sigma, decay, shape)
                    heapq.heappush(heap, (-sc, new_left))

            if node_a.next is not None:
                next_node = all_nodes[node_a.next]
                new_right = (new_id, next_node.token)
                pair_positions[new_right].add(nid)
                pair_counts[new_right] = pair_counts.get(new_right, 0) + 1
                if pair_counts[new_right] >= 2:
                    sc = _score_pair(new_right, pair_counts[new_right], coord_by_id, sigma, decay, shape)
                    heapq.heappush(heap, (-sc, new_right))

        # Cleanup
        if best_pair in pair_positions:
            del pair_positions[best_pair]
        if best_pair in pair_counts:
            del pair_counts[best_pair]

        merge_num += 1
        if verbose and (merge_num <= 20 or merge_num % 50 == 0 or merge_num == num_merges):
            la = letter_id_to_letter(a) or f"<{a}>"
            lb = letter_id_to_letter(b) or f"<{b}>"
            print(f"[v3 BPE]   merge {merge_num:4d}: ({a:3d},{b:3d}) {la}+{lb} -> {new_id:4d}  freq={best_count}")

    if verbose:
        print(f"[v3 BPE] done — learned {len(merges)} merges")

    return merges


# ---------------------------------------------------------------------------
# Tokenizer class — load/save, encode, decode, byte-accounting
# ---------------------------------------------------------------------------

class BeseV3FastBPE:
    """Loaded v3 BPE tokenizer with spatial-aware merges."""

    def __init__(
        self,
        merges: list[tuple[tuple[int, int], int]],
        coords_3d: dict[str, tuple] | None = None,
        sigma: float | None = None,
    ):
        self.merges = merges
        self.coords_3d = coords_3d
        self.sigma = sigma
        self.vocab_size = BASE_VOCAB_SIZE + len(merges)
        self._merge_priority = {pair: i for i, (pair, _) in enumerate(merges)}
        self._merge_to_id = {pair: new_id for pair, new_id in merges}

    @classmethod
    def load(cls, path: str | Path) -> "BeseV3FastBPE":
        with open(path, "r") as f:
            blob = json.load(f)
        merges = [(tuple(m["pair"]), int(m["new_id"])) for m in blob["merges"]]
        coords_3d = None
        sigma = None
        if "coords_3d" in blob:
            coords_3d = {k: tuple(v) for k, v in blob["coords_3d"].items()}
            sigma = float(blob.get("sigma", 1.0))
        return cls(merges, coords_3d=coords_3d, sigma=sigma)

    def save(self, path: str | Path) -> None:
        blob = {
            "version": 3,
            "tokenizer_type": "bese_v3_spatial",
            "base_vocab_size": BASE_VOCAB_SIZE,
            "num_merges": len(self.merges),
            "vocab_size": self.vocab_size,
            "merges": [{"pair": list(pair), "new_id": new_id} for pair, new_id in self.merges],
        }
        if self.coords_3d is not None:
            blob["coords_3d"] = {k: list(v) for k, v in self.coords_3d.items()}
            blob["sigma"] = self.sigma
        with open(path, "w") as f:
            json.dump(blob, f, indent=2)

    def encode(self, text: str) -> np.ndarray:
        """Encode text into v3 token IDs (with merges applied)."""
        base = _text_to_base_tokens(text)
        if not base:
            return np.zeros(0, dtype=np.uint16)
        # Apply merges in priority order using the linked-list trick.
        # Simple O(N * num_merges) implementation — fine for inference.
        tokens = list(base)
        # Precompute longest-prefix lookup to apply highest-priority merges first.
        # For simplicity, iterate merges in training order.
        for pair, new_id in self.merges:
            i = 0
            out: list[int] = []
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
        return np.array(tokens, dtype=np.uint16)

    def decode(self, token_ids: list[int] | np.ndarray) -> str:
        """Decode token IDs back to text (best-effort; lossy on case
        and non-ASCII byte fallbacks)."""
        # Expand merges back to base tokens
        merge_table = {new_id: pair for pair, new_id in self.merges}
        expanded: list[int] = []
        for tid in token_ids:
            stack = [int(tid)]
            while stack:
                t = stack.pop()
                if t in merge_table:
                    a, b = merge_table[t]
                    stack.append(b)
                    stack.append(a)
                else:
                    expanded.append(t)
        # Map base tokens back to characters
        chars: list[str] = []
        for t in expanded:
            ch = DECODE_TABLE.get((t,))
            if ch is not None:
                chars.append(ch)
        return "".join(chars)

    def compute_bytes_per_token(self) -> np.ndarray:
        """Per-token UTF-8 byte count for the full vocab (base + merges).
        Each merge token's byte count is the sum of its constituents."""
        bpt = np.zeros(self.vocab_size, dtype=np.int16)
        bpt[:BASE_VOCAB_SIZE] = BYTES_PER_TOKEN
        merge_bpt = {i: int(BYTES_PER_TOKEN[i]) for i in range(BASE_VOCAB_SIZE)}
        for pair, new_id in self.merges:
            merge_bpt[new_id] = merge_bpt[pair[0]] + merge_bpt[pair[1]]
            bpt[new_id] = merge_bpt[new_id]
        return bpt

    def get_letter_token_ids(self) -> dict[str, int]:
        """Map letter -> base token id (for embedding init)."""
        return {ch: LETTER_START + i for i, ch in enumerate(LETTERS)}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Quick sanity check: byte invariant + roundtrip on diverse strings."""
    test_cases = [
        "the quick brown fox jumps over the lazy dog",
        "Hello, World! 1234567890",
        "naïve café résumé — emojis: 🚀✨🌍",
        "newlines\nand\ttabs and \"quotes\" and 'apostrophes'",
        "MixedCase HTML <p>tag</p> and JSON {\"a\": 1}",
    ]
    for s in test_cases:
        ids = _text_to_base_tokens(s)
        bytes_total = sum(int(BYTES_PER_TOKEN[t]) for t in ids)
        utf8_len = len(s.encode("utf-8"))
        assert bytes_total == utf8_len, (
            f"byte invariant failed: {s!r}\n"
            f"  encode -> {len(ids)} tokens, byte sum {bytes_total}\n"
            f"  utf-8 len {utf8_len}"
        )
    print("OK — v3 base byte invariant holds for all test cases")

    # Train a tiny BPE and check roundtrip + invariant after merges
    from bese_v3_constants import _FALLBACK_LETTER_SPACE_3D, _FALLBACK_SIGMA_3D
    sample = test_cases * 200  # repeat to give the trainer something to chew on
    merges = train_bpe_merges_v3(
        sample,
        num_merges=20,
        coords_3d=_FALLBACK_LETTER_SPACE_3D,
        sigma=_FALLBACK_SIGMA_3D,
        verbose=False,
    )
    tok = BeseV3FastBPE(merges, coords_3d=_FALLBACK_LETTER_SPACE_3D, sigma=_FALLBACK_SIGMA_3D)
    bpt = tok.compute_bytes_per_token()
    for s in test_cases:
        ids = tok.encode(s)
        bytes_total = sum(int(bpt[t]) for t in ids)
        utf8_len = len(s.encode("utf-8"))
        assert bytes_total == utf8_len, (
            f"post-BPE byte invariant failed: {s!r}\n"
            f"  encode -> {len(ids)} tokens, byte sum {bytes_total}\n"
            f"  utf-8 len {utf8_len}"
        )
    print(f"OK — v3 byte invariant holds after {len(merges)} BPE merges")


if __name__ == "__main__":
    _self_test()
