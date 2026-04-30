"""
BESE v3: Spatial Letter Encoding constants and lookup tables.

Design intent (vs v2):
  - NO groups, NO position tokens. Every English letter is its own token.
  - Each letter has a 3D coordinate derived from MDS embedding of bigram
    co-occurrence (the LETTER_SPACE dict, loaded from JSON at runtime).
  - The same coordinates do double duty:
      1. TOKENIZER: BPE merge scoring favors spatially-close letter pairs
         (score = freq * exp(-distance / (DECAY * SIGMA)) ** SHAPE)
      2. MODEL: Embedding init for letter tokens uses the 12D version of
         the same coordinates (LETTER_SPACE_EMB), filling the first 12 dims
         of each letter row in the embedding table.

Vocab layout (47 base tokens + 241 BPE merges = 288 total, same as v2):
   0    PAD
   1    BOS
   2    EOS
   3    UNK
   4-29 26 lowercase letters a..z (1 byte each)
   30   SPACE
   31   PERIOD
   32   COMMA
   33   NEWLINE
   34   QUESTION
   35   QUOTE
   36   OTHER_PUNCT (used for byte-fallback on unknown chars)
   37-46 digits 0..9
   47+  BPE merges (built at training time)

Byte-per-token invariant:
  sum(BYTES_PER_TOKEN[t] for t in encode(s)) == len(s.encode("utf-8"))
  for any input string s. Same proof structure as v2 — see bese_v3_fast_bpe.py
  for the per-character argument.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# ─── Special tokens ────────────────────────────────────────────────────
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# ─── 26-letter flat alphabet ───────────────────────────────────────────
# No groups, no position tokens. Every letter is its own token, in
# alphabetical order so the index is deterministic and language-agnostic.
LETTERS = "abcdefghijklmnopqrstuvwxyz"
LETTER_START = 4  # 'a' = 4, 'b' = 5, ..., 'z' = 29

# ─── Punctuation ───────────────────────────────────────────────────────
SPACE_ID = 30
PERIOD_ID = 31
COMMA_ID = 32
NEWLINE_ID = 33
QUESTION_ID = 34
QUOTE_ID = 35
OTHER_PUNCT_ID = 36

# ─── Digits ────────────────────────────────────────────────────────────
DIGIT_START = 37  # '0' = 37, '1' = 38, ..., '9' = 46

BASE_VOCAB_SIZE = 47

# ─── BPE merge scoring parameters ──────────────────────────────────────
# Defaults match the local pow_0.02 sweep winner.
# These can be overridden at training time via env var or argument.
SPATIAL_DECAY = 3.0    # 'sigma multiplier' in score = exp(-d / (DECAY * SIGMA))
SPATIAL_SHAPE = 0.02   # power applied to the spatial factor; 0.02 = gentle nudge


# ─── Encode / decode tables ────────────────────────────────────────────

def build_encode_table() -> dict[str, list[int]]:
    """Map character -> list of base token ids. Lowercase ASCII only;
    uppercase is normalized to lowercase before lookup."""
    table: dict[str, list[int]] = {}
    for i, ch in enumerate(LETTERS):
        table[ch] = [LETTER_START + i]
    table[" "] = [SPACE_ID]
    table["."] = [PERIOD_ID]
    table[","] = [COMMA_ID]
    table["\n"] = [NEWLINE_ID]
    table["?"] = [QUESTION_ID]
    for ch in ["'", '"', "‘", "’", "“", "”"]:
        table[ch] = [QUOTE_ID]
    for d in range(10):
        table[str(d)] = [DIGIT_START + d]
    return table


def build_decode_table() -> dict[tuple[int, ...], str]:
    """Map base-token-id-tuple -> single character (best-effort)."""
    table: dict[tuple[int, ...], str] = {}
    for i, ch in enumerate(LETTERS):
        table[(LETTER_START + i,)] = ch
    table[(SPACE_ID,)] = " "
    table[(PERIOD_ID,)] = "."
    table[(COMMA_ID,)] = ","
    table[(NEWLINE_ID,)] = "\n"
    table[(QUESTION_ID,)] = "?"
    table[(QUOTE_ID,)] = "'"
    table[(OTHER_PUNCT_ID,)] = "?"
    for d in range(10):
        table[(DIGIT_START + d,)] = str(d)
    return table


def build_bytes_per_token() -> np.ndarray:
    """UTF-8 bytes each base token represents (BPB-critical).

    PAD/BOS/EOS/UNK are 0 bytes. Every other base token is 1 byte
    (each letter, each punct token, each digit, OTHER_PUNCT)."""
    bpt = np.zeros(BASE_VOCAB_SIZE, dtype=np.int16)
    # Letters: 1 byte each
    for i in range(26):
        bpt[LETTER_START + i] = 1
    # Punct: 1 byte each
    for tid in (
        SPACE_ID, PERIOD_ID, COMMA_ID, NEWLINE_ID,
        QUESTION_ID, QUOTE_ID, OTHER_PUNCT_ID,
    ):
        bpt[tid] = 1
    # Digits: 1 byte each
    for d in range(10):
        bpt[DIGIT_START + d] = 1
    return bpt


# Eager singletons (import cost is tiny; avoids recomputation)
ENCODE_TABLE = build_encode_table()
DECODE_TABLE = build_decode_table()
BYTES_PER_TOKEN = build_bytes_per_token()


# ─── Spatial coordinates ───────────────────────────────────────────────
# These are loaded from a JSON file produced by scripts/build_v3_space.py
# (which optimizes letter positions on FineWeb data). A fallback set of
# coordinates derived from English bigram MDS is provided for tests when
# no fitted JSON is available.

# Fallback (English MDS-derived) — replace at runtime with a fitted set.
_FALLBACK_LETTER_SPACE_3D = {
    'e': (0.0, 0.0, 0.0),
    't': (-0.18251, 0.39466, -0.28084),
    'a': (-0.00126, -0.00324, -0.00433),
    'o': (0.25313, -0.01499, -0.20366),
    'i': (0.08368, 0.02714, -0.21579),
    'n': (-0.15106, 0.00651, -0.04990),
    's': (-0.34651, 0.17576, -0.28248),
    'h': (-0.31677, 0.47292, -0.56814),
    'r': (0.02722, 0.00366, 0.13977),
    'd': (-0.14434, -0.55401, -0.01074),
    'l': (-0.07709, -0.24700, 0.48101),
    'c': (-0.43504, 0.38334, -0.09540),
    'u': (0.08257, -0.31985, -0.00352),
    'm': (0.71219, -0.08518, 0.09997),
    'w': (-0.22226, 0.44677, -0.60879),
    'f': (0.05454, -0.05543, 0.77213),
    'g': (-0.24255, -0.50033, 0.25487),
    'y': (-0.33136, -0.03332, 0.56934),
    'p': (0.51927, 0.23063, 0.04933),
    'b': (0.52747, -0.42374, 0.29052),
    'v': (0.79851, 0.08412, -0.88280),
    'k': (-0.68960, 0.25024, 0.39231),
    'j': (0.38994, -0.96616, -0.13745),
    'x': (0.66487, 0.78043, -0.15170),
    'q': (-0.75272, -0.60605, -0.87817),
    'z': (0.31615, -0.61373, -1.00000),
}
_FALLBACK_SIGMA_3D = 0.926124


def load_letter_space_3d(path: Path | str | None = None) -> tuple[dict[str, tuple[float, float, float]], float]:
    """Load fitted 3D letter coordinates and sigma.

    If `path` is None or the file doesn't exist, returns the fallback
    English MDS coordinates. Returned tuple is (coords_dict, sigma).
    """
    if path is None:
        return dict(_FALLBACK_LETTER_SPACE_3D), _FALLBACK_SIGMA_3D
    p = Path(path)
    if not p.exists():
        return dict(_FALLBACK_LETTER_SPACE_3D), _FALLBACK_SIGMA_3D
    with open(p, "r") as f:
        blob = json.load(f)
    coords = {k: tuple(v) for k, v in blob["coords"].items() if len(k) == 1}
    sigma = float(blob.get("sigma", _FALLBACK_SIGMA_3D))
    return coords, sigma


def load_letter_space_emb(path: Path | str | None = None, n_dims: int = 12) -> np.ndarray:
    """Load the higher-dimensional letter coordinates used for embedding init.

    Returns an array of shape (26, n_dims). Rows are in alphabetical order
    matching `LETTERS`. When no fitted file is provided, falls back to
    using the 3D coords zero-padded out to `n_dims` (degraded but usable).
    """
    if path is not None and Path(path).exists():
        with open(path, "r") as f:
            blob = json.load(f)
        if "coords_emb" in blob:
            mat = np.zeros((26, n_dims), dtype=np.float32)
            for i, ch in enumerate(LETTERS):
                if ch in blob["coords_emb"]:
                    vec = blob["coords_emb"][ch]
                    k = min(n_dims, len(vec))
                    mat[i, :k] = vec[:k]
            return mat
    # Fallback: 3D coords zero-padded
    coords, _ = load_letter_space_3d(path)
    mat = np.zeros((26, n_dims), dtype=np.float32)
    for i, ch in enumerate(LETTERS):
        if ch in coords:
            mat[i, :3] = coords[ch]
    return mat


# ─── Spatial scoring helpers ───────────────────────────────────────────

def letter_id_to_letter(token_id: int) -> str | None:
    """Return the letter for a base letter-token-id, or None."""
    if LETTER_START <= token_id < LETTER_START + 26:
        return LETTERS[token_id - LETTER_START]
    return None


def merge_centroid(coord_a: tuple, coord_b: tuple) -> tuple:
    """Centroid of two coordinates, used to give merge tokens a position."""
    return tuple((a + b) / 2.0 for a, b in zip(coord_a, coord_b))


def spatial_pair_score(
    freq: int,
    coord_a: tuple,
    coord_b: tuple,
    sigma: float,
    decay: float = SPATIAL_DECAY,
    shape: float = SPATIAL_SHAPE,
) -> float:
    """Spatial BPE merge score = freq * exp(-d / (decay * sigma)) ** shape.

    The pow_0.02 sweep winner uses shape=0.02 — a gentle multiplicative
    nudge that breaks ties between near-equal-frequency pairs in favor
    of spatially close ones, without dominating the frequency signal.
    """
    diff = tuple(a - b for a, b in zip(coord_a, coord_b))
    dist = math.sqrt(sum(d * d for d in diff))
    spatial = math.exp(-dist / (decay * sigma))
    return float(freq) * (spatial ** shape)
