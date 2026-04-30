#!/usr/bin/env python3
"""
RunPod v9: BESE v3 Spatial Letter Encoding submission pipeline (8xH100).

The v3 idea: every English letter gets a 3D coordinate derived from its
bigram co-occurrence structure on FineWeb. The same coordinates do
double duty:
  1. TOKENIZER: BPE merge scoring favors spatially-close letter pairs
     (score = freq * exp(-d / (3 * sigma)) ** 0.02 — pow_0.02 sweep winner).
  2. MODEL: First 12 dims of each letter-token's embedding row are
     initialized from the same coordinates (rest is random gaussian).

Vocab: 47 base tokens (4 special + 26 letters + 7 punct + 10 digits) +
241 BPE merges = 288 total (matches v2/v6.1 for direct comparability).

Architecture: same v6.1 transformer stack proven in PR #1666 — 12 layers,
dim=512, mlp_mult=3.5, depth recurrence on layers 3-5 (3 loops),
parallel residuals on layers 8-11. Only the tokenizer + embedding init
change; everything else is the v6.1 control variable.

Phase 0 (untimed, ~30-50 min):
  0a. Build 3D + 12D letter coords by optimizing letter placement on
      FineWeb n-grams (calls into v3 path-optimization solver).
  0b. Train v3 BPE merges using spatial scoring on FineWeb sample.
  0c. Re-encode FineWeb training shards into v3 token IDs.
  0d. Build n-gram tilt table over the v3 token vocab.

Phase 1 (timed, 600s):
  Training with SPATIAL_INIT_ENABLED=1, pointing at the v3 coords JSON.

Phase 2 (timed, ~7-10 min):
  Sliding-window eval with INT6 + LZMA + n-gram tilt. Optional TTT.

Phase 3 (untimed):
  Artifact assembly, size check, copy to /runpod-volume/checkpoints/v9/.

Usage on RunPod:
  cd /workspace && git clone https://github.com/mrbese/parameter-golf-bese.git bese
  cd /workspace/bese && python scripts/runpod_v9_spatial.py --num-gpus 8

  # If v3 prep already done on this network volume:
  python scripts/runpod_v9_spatial.py --skip-prep --num-gpus 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (network volume for persistence; pod-local for code)
# ---------------------------------------------------------------------------
# NET_VOL defaults to the RunPod network volume mount; override to
# /workspace (or any pod-local path) when running on a pod without a
# network volume attached, e.g., when the 8xH100 SXM you grabbed is in
# a region different from your existing volume.
#   NET_VOL=/workspace python scripts/runpod_v9_spatial.py ...
NET_VOL = Path(os.environ.get("NET_VOL", "/runpod-volume"))
BESE_DIR = Path(os.environ.get("BESE_DIR", "/workspace/bese"))
WORK_DIR = Path("/workspace")
PG_DIR = Path(os.environ.get("PG_DIR", "/workspace/parameter-golf"))

# Upstream SP1024 — used as the *source* text for v3 BPE training and
# shard re-encoding. We don't ship the SP tokenizer in the artifact.
SP_MODEL = PG_DIR / "data/tokenizers/fineweb_1024_bpe.model"
SP_SHARD_DIR = PG_DIR / "data/datasets/fineweb10B_sp1024"

# v9 outputs
COORDS_JSON = NET_VOL / "artifacts" / "letter_coords_v3.json"
BPE_OUTPUT = NET_VOL / "tokenizers" / "bese_v3_bpe_241_v9.json"
SHARD_DIR = NET_VOL / "bese_shards_v9_spatial"
NGRAM_TABLE = NET_VOL / "artifacts" / "ngram_table_v9.bin"
LOGFILE = NET_VOL / "logs" / "run_v9_spatial.log"

TRAIN_SCRIPT = BESE_DIR / "integration" / "train_gpt_bese.py"

# Ensure persistent directories exist
for d in [
    NET_VOL / "checkpoints" / "v9",
    NET_VOL / "logs",
    NET_VOL / "artifacts",
    NET_VOL / "tokenizers",
    SHARD_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Training environment (v9 = v6.1 stack + v3 tokenizer + spatial init)
# ---------------------------------------------------------------------------
TRAIN_ENV = {
    "VOCAB_SIZE": "288",
    "NUM_LAYERS": "12",
    "MODEL_DIM": "512",
    "MLP_MULT": "3.5",
    "NUM_HEADS": "8",
    "NUM_KV_HEADS": "4",
    "DEPTH_RECURRENCE_START": "3",
    "DEPTH_RECURRENCE_END": "5",
    "DEPTH_RECURRENCE_LOOPS": "3",
    "DEPTH_RECURRENCE_ACTIVATION_FRAC": "0.35",
    "PARALLEL_RESIDUAL_START": "8",
    "QK_GAIN_INIT": "5.25",
    "MATRIX_LR": "0.026",
    "MUON_WD": "0.095",
    "ADAM_WD": "0.095",
    "EMA_DECAY": "0.9965",
    "WARMDOWN_ITERS": "5000",
    "VE_LAYERS": "10,11",
    "EVAL_STRIDE": "64",
    "TOKENIZER_PATH": str(BPE_OUTPUT),
    "DATA_PATH": str(SHARD_DIR),
    "MAX_WALLCLOCK_SECONDS": "600",
    "NGRAM_TILT_ENABLED": "1",
    "NGRAM_TILT_MAX_N": "3",
    "NGRAM_PRIOR_PATH": str(NGRAM_TABLE),
    # --- v3 Spatial (the only new thing vs v8.2) ---
    "SPATIAL_INIT_ENABLED": "1",
    "SPATIAL_INIT_PATH": str(COORDS_JSON),
    "SPATIAL_INIT_DIMS": "12",
    # --- Disable v8 experiments (stick to clean v6.1 baseline) ---
    "QAT_ENABLED": "0",
    "NOISY_QAT_ENABLED": "0",
    "BIGRAM_PRIOR_ENABLED": "0",
    "LATE_QAT_THRESHOLD": "0",
    # --- TTT (proven in v8.2 with dtype fix) ---
    "TTT_ENABLED": "1",
    "TTT_LR": "0.005",
    "TTT_MOMENTUM": "0.9",
    "TTT_EPOCHS": "1",
    "TTT_GRAD_CLIP": "1.0",
    "TTT_CHUNK_SIZE": "32768",
    # --- v3 tokenizer wiring ---
    "BESE_TOKENIZER_VARIANT": "v3",
    "BESE_TOKENIZER_ROOT": str(BESE_DIR / "tokenizer"),
}

SIZE_LIMIT_BYTES = 16_000_000  # 16 MB hard limit (decimal, per upstream FAQ)

# How much FineWeb to use for v3 prep (untimed, not the 10-min training cap)
PREP_DOCS_FOR_BPE = int(os.environ.get("V9_PREP_DOCS", "100000"))
PREP_DOCS_FOR_COORDS = int(os.environ.get("V9_COORD_DOCS", "100000"))
PREP_OPT_STEPS = int(os.environ.get("V9_OPT_STEPS", "5000"))

# ---------------------------------------------------------------------------
# Logging helpers (match runpod_v8.py)
# ---------------------------------------------------------------------------
_log_fh = None


def _open_log():
    global _log_fh
    LOGFILE.parent.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOGFILE, "a", encoding="utf-8")


def _close_log():
    global _log_fh
    if _log_fh:
        _log_fh.close()
        _log_fh = None


def banner(msg: str) -> None:
    line = f"\n{'=' * 72}\n  {msg}\n{'=' * 72}"
    print(line, flush=True)
    if _log_fh:
        _log_fh.write(line + "\n")
        _log_fh.flush()


def log(msg: str) -> None:
    print(msg, flush=True)
    if _log_fh:
        _log_fh.write(msg + "\n")
        _log_fh.flush()


def run_cmd(cmd: list[str], *, env: dict | None = None, cwd=None, label: str = "", timeout=None) -> str:
    if label:
        log(f"  [{label}] Running: {' '.join(str(c) for c in cmd)}")
    else:
        log(f"  Running: {' '.join(str(c) for c in cmd)}")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    t0 = time.time()
    output_lines = []
    proc = subprocess.Popen(
        cmd, env=merged_env, cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if _log_fh:
                _log_fh.write(line)
                _log_fh.flush()
            output_lines.append(line)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise RuntimeError(f"Command timed out after {timeout}s")
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log(f"  ERROR: exit code {proc.returncode} after {elapsed:.1f}s")
        raise RuntimeError(f"Command failed (exit {proc.returncode})")
    log(f"  Completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return "".join(output_lines)


def detect_gpus() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
        )
        return len([l for l in result.stdout.strip().split("\n") if "GPU" in l])
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# Auto-seed: copy committed prep artifacts from the repo onto /runpod-volume
# ---------------------------------------------------------------------------
def _seed_from_repo() -> None:
    """If the user pre-built v3 coords + BPE locally and committed them to
    the repo (under <repo>/artifacts/ and <repo>/tokenizers/), copy them
    onto the network volume so Phase 0a + 0b auto-skip.

    No-op when the repo doesn't have the files, or when the network volume
    already has them. Safe to call unconditionally."""
    seeds = [
        (BESE_DIR / "artifacts" / "letter_coords_v3.json", COORDS_JSON),
        (BESE_DIR / "tokenizers" / "bese_v3_bpe_241_v9.json", BPE_OUTPUT),
    ]
    seeded = 0
    for src, dst in seeds:
        if not src.exists():
            continue
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        log(f"  [seed] copied {src.name} from repo to {dst}")
        seeded += 1
    if seeded:
        log(f"  [seed] {seeded} pre-built artifact(s) seeded from repo — "
            f"matching Phase 0 step(s) will auto-skip")


# ---------------------------------------------------------------------------
# Phase 0a: Build v3 spatial coordinates (3D + 12D) from FineWeb
# ---------------------------------------------------------------------------
def phase0a_build_coords() -> None:
    """Optimize 26-letter placement so that frequent n-gram sequences in
    FineWeb correspond to short paths. Saves the resulting coords (3D for
    BPE scoring, 12D for embedding init) to COORDS_JSON.
    """
    if COORDS_JSON.exists():
        log(f"  v3 coords already exist: {COORDS_JSON}")
        return

    banner("PHASE 0a: Build v3 spatial coordinates from FineWeb")

    import numpy as np

    # Stream a sample of FineWeb (use the SP1024 source data — we only
    # care about the raw text, not the SP tokenization)
    log(f"  Streaming {PREP_DOCS_FOR_COORDS:,} FineWeb docs for n-gram counts...")
    docs = _stream_fineweb_text(PREP_DOCS_FOR_COORDS, min_words=50)
    log(f"  Got {len(docs):,} docs, {sum(len(d) for d in docs):,} chars")

    # Count letter n-grams (2-5) within words; non-letters break the chain
    log("  Counting 2- to 5-grams (letters only)...")
    counters = {n: Counter() for n in range(2, 6)}
    for doc in docs:
        words = []
        cur = []
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
                for n in range(2, min(6, wlen - s + 1)):
                    counters[n][tuple(w[s:s + n])] += 1
    for n in range(2, 6):
        log(f"    {n}-grams: {len(counters[n]):,} distinct, total_freq={sum(counters[n].values()):,}")

    # Optimize 3D positions so that loss = sum_seq(freq * path_length) is minimized
    pos_3d, sigma_3d = _optimize_placement(counters, n_dims=3, n_steps=PREP_OPT_STEPS)
    log(f"  3D placement done — sigma={sigma_3d:.4f}")

    # Optimize 12D positions for embedding init
    log("  Building 12D coords for embedding init...")
    pos_12d, _ = _optimize_placement(counters, n_dims=12, n_steps=PREP_OPT_STEPS // 2)

    letters = "abcdefghijklmnopqrstuvwxyz"
    coords_3d = {ch: [float(x) for x in pos_3d[i]] for i, ch in enumerate(letters)}
    coords_emb = {ch: [float(x) for x in pos_12d[i]] for i, ch in enumerate(letters)}

    COORDS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(COORDS_JSON, "w") as f:
        json.dump({
            "version": 3,
            "coords": coords_3d,           # 3D for BPE scoring
            "coords_emb": coords_emb,      # 12D for embedding init
            "sigma": sigma_3d,
            "letter_to_token_id": {ch: 4 + i for i, ch in enumerate(letters)},
            "n_docs_used": len(docs),
            "n_optimization_steps": PREP_OPT_STEPS,
        }, f, indent=2)
    log(f"  Saved coords to {COORDS_JSON}")


def _stream_fineweb_text(n_docs: int, min_words: int = 50) -> list[str]:
    """Stream raw text from HuggingFace FineWeb. Skips low-quality docs."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    docs = []
    for sample in ds:
        text = sample.get("text", "")
        words = text.split()
        if len(words) < min_words:
            continue
        if len(set(words)) / len(words) < 0.25:
            continue
        docs.append(text)
        if len(docs) >= n_docs:
            break
    return docs


def _optimize_placement(counters: dict[int, Counter], n_dims: int, n_steps: int, lr: float = 0.0001, top_k: int = 50_000):
    """Place 26 letters in n_dims so that sum_seq(freq * path_length) is minimized."""
    import numpy as np
    all_ngrams = []
    for n, counter in counters.items():
        for ngram, freq in counter.items():
            all_ngrams.append((freq, ngram))
    all_ngrams.sort(reverse=True)
    constraints = all_ngrams[:top_k]
    c_freqs = np.array([f for f, _ in constraints], dtype=np.float64)
    c_seqs = [list(ng) for _, ng in constraints]
    c_freqs = c_freqs / c_freqs.max()
    np.random.seed(42)
    pos = np.random.randn(26, n_dims) * 0.5
    e_idx = ord('e') - ord('a')
    best_loss, best_pos = float('inf'), pos.copy()
    for step in range(n_steps):
        grad = np.zeros_like(pos)
        total_loss = 0.0
        for ci in range(len(constraints)):
            seq = c_seqs[ci]
            freq = c_freqs[ci]
            for edge in range(len(seq) - 1):
                a, b = seq[edge], seq[edge + 1]
                diff = pos[a] - pos[b]
                dist = np.sqrt(np.sum(diff ** 2)) + 1e-8
                total_loss += freq * dist
                direction = diff / dist
                grad[a] += freq * direction
                grad[b] -= freq * direction
        pos -= lr * grad
        pos -= pos[e_idx]  # anchor 'e' at origin
        if total_loss < best_loss:
            best_loss = total_loss
            best_pos = pos.copy()
    pos = best_pos
    dists = []
    for i in range(26):
        for j in range(i + 1, 26):
            dists.append(float(np.sqrt(np.sum((pos[i] - pos[j]) ** 2))))
    sigma = float(np.median(dists)) if dists else 1.0
    return pos, sigma


# ---------------------------------------------------------------------------
# Phase 0b: Train v3 BPE merges using spatial scoring
# ---------------------------------------------------------------------------
def phase0b_train_bpe() -> None:
    if BPE_OUTPUT.exists():
        log(f"  v3 BPE already trained: {BPE_OUTPUT}")
        return

    banner("PHASE 0b: Train v3 BPE merges (spatial scoring)")

    sys.path.insert(0, str(BESE_DIR / "tokenizer"))
    from bese_v3_constants import load_letter_space_3d
    from bese_v3_fast_bpe import BeseV3FastBPE, train_bpe_merges_v3

    coords_3d, sigma = load_letter_space_3d(COORDS_JSON)
    log(f"  Loaded coords for {len(coords_3d)} letters, sigma={sigma:.4f}")

    log(f"  Streaming {PREP_DOCS_FOR_BPE:,} FineWeb docs for BPE training...")
    docs = _stream_fineweb_text(PREP_DOCS_FOR_BPE, min_words=50)
    log(f"  Got {len(docs):,} docs")

    merges = train_bpe_merges_v3(
        docs,
        num_merges=241,
        coords_3d=coords_3d,
        sigma=sigma,
        decay=3.0,
        shape=0.02,  # pow_0.02 sweep winner
        verbose=True,
    )
    tok = BeseV3FastBPE(merges, coords_3d=coords_3d, sigma=sigma)
    tok.save(BPE_OUTPUT)
    log(f"  Saved {len(merges)} merges to {BPE_OUTPUT}")


# ---------------------------------------------------------------------------
# Phase 0c: Re-encode FineWeb training shards into v3 token IDs
# ---------------------------------------------------------------------------
def phase0c_export_shards() -> None:
    """Re-encode FineWeb text into v3 token bins. Output format matches the
    upstream SP shards (HEADER_INTS=256 uint32 header followed by uint16
    token IDs) so train_gpt_bese.py can load them unchanged."""
    existing = list(SHARD_DIR.glob("fineweb_train_*.bin"))
    if existing:
        log(f"  v3 shards already exist ({len(existing)} files in {SHARD_DIR})")
        return

    banner("PHASE 0c: Re-encode FineWeb shards into v3 token IDs")

    import numpy as np
    sys.path.insert(0, str(BESE_DIR / "tokenizer"))
    from bese_v3_fast_bpe import BeseV3FastBPE

    tok = BeseV3FastBPE.load(BPE_OUTPUT)
    log(f"  Loaded v3 tokenizer (vocab={tok.vocab_size})")

    # Source: HuggingFace FineWeb stream (we don't have raw text shards on disk;
    # we'll stream and write our own shards).
    HEADER_INTS = 256
    SHARD_TOKENS = 100_000_000  # 100M tokens per shard

    log("  Streaming FineWeb and writing v3 shards...")
    shard_idx = 0
    cur_tokens: list[int] = []
    n_written = 0
    val_written = False

    def flush_shard(tokens: list[int], idx: int, name: str):
        path = SHARD_DIR / f"fineweb_{name}_{idx:06d}.bin"
        arr = np.array(tokens, dtype=np.uint16)
        header = np.zeros(HEADER_INTS, dtype=np.uint32)
        header[0] = 20240520  # magic
        header[1] = 1          # version
        header[2] = len(arr)
        with open(path, "wb") as f:
            f.write(header.tobytes())
            f.write(arr.tobytes())
        log(f"    wrote {path.name} ({len(arr):,} tokens, {path.stat().st_size:,} bytes)")

    # Validation shard first (matches upstream split)
    log("  Streaming validation split...")
    val_docs = _stream_fineweb_text(50_000, min_words=20)  # first 50K docs
    val_tokens: list[int] = []
    for doc in val_docs:
        ids = tok.encode(doc)
        val_tokens.extend(ids.tolist())
    flush_shard(val_tokens, 0, "val")

    log("  Streaming training split...")
    # Use a HUGE doc count for training; cap by total tokens
    target_train_tokens = SHARD_TOKENS * 10  # 1B tokens
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    seen_val = 0
    for sample in ds:
        # skip first 50K validation docs
        if seen_val < 50_000:
            seen_val += 1
            continue
        text = sample.get("text", "")
        ids = tok.encode(text)
        cur_tokens.extend(ids.tolist())
        if len(cur_tokens) >= SHARD_TOKENS:
            flush_shard(cur_tokens[:SHARD_TOKENS], shard_idx, "train")
            cur_tokens = cur_tokens[SHARD_TOKENS:]
            shard_idx += 1
            n_written += SHARD_TOKENS
            if n_written >= target_train_tokens:
                break
    if cur_tokens:
        flush_shard(cur_tokens, shard_idx, "train")
        n_written += len(cur_tokens)
    log(f"  Wrote {shard_idx + 1} training shards, ~{n_written:,} tokens total")


# ---------------------------------------------------------------------------
# Phase 0d: Build n-gram tilt table over the v3 token vocab
# ---------------------------------------------------------------------------
def phase0d_build_ngram() -> None:
    if NGRAM_TABLE.exists():
        log(f"  n-gram table already exists: {NGRAM_TABLE}")
        return
    banner("PHASE 0d: Build n-gram tilt table")
    # build_ngram_table.py takes a single shard file (not a directory) and
    # has no --vocab-size flag — it scans whatever IDs are in the shard.
    train_shards = sorted(SHARD_DIR.glob("fineweb_train_*.bin"))
    if not train_shards:
        raise RuntimeError(
            f"No fineweb_train_*.bin shards in {SHARD_DIR}; "
            "Phase 0c must run before Phase 0d."
        )
    shard_for_ngram = train_shards[0]
    log(f"  Scanning {shard_for_ngram.name} for n-grams (n=2..3)")
    cmd = [
        "python", str(BESE_DIR / "scripts" / "build_ngram_table.py"),
        "--shard", str(shard_for_ngram),
        "--output", str(NGRAM_TABLE),
        "--max-n", "3",
        "--top-k", "1",
    ]
    run_cmd(cmd, label="ngram", timeout=2400)


# ---------------------------------------------------------------------------
# Phase 1: Training
# ---------------------------------------------------------------------------
def phase1_training(num_gpus: int) -> str:
    banner("PHASE 1: Training (600s wallclock cap)")
    env = dict(TRAIN_ENV)
    env["RUN_ID"] = "v9_spatial"
    env["BESE_TOKENIZER_ROOT"] = str(BESE_DIR / "tokenizer")
    env["WORLD_SIZE"] = str(num_gpus)
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        str(TRAIN_SCRIPT),
    ]
    return run_cmd(cmd, env=env, cwd=BESE_DIR, label="train", timeout=2400)


# ---------------------------------------------------------------------------
# Phase 2: Eval (parse metrics from training output)
# ---------------------------------------------------------------------------
def phase2_eval(train_output: str) -> dict:
    banner("PHASE 2: Parse eval metrics")
    metrics = {}
    for line in train_output.strip().split("\n"):
        if "final_int6_lzma_roundtrip_exact" in line and "val_bpb:" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["sliding_bpb"] = float(part.split(":")[1])
        if "final_int6_roundtrip_exact" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["int6_bpb"] = float(part.split(":")[1])
        if "final_ttt_sliding_window_exact" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["ttt_bpb"] = float(part.split(":")[1])
        if "DIAGNOSTIC post_ema" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["raw_bpb"] = float(part.split(":")[1])
        if "Total submission size" in line and "bytes" in line:
            m = re.search(r"(\d+)\s*bytes", line)
            if m:
                metrics["size_bytes"] = int(m.group(1))
    metrics["best_bpb"] = (
        metrics.get("ttt_bpb") or metrics.get("sliding_bpb") or metrics.get("int6_bpb")
    )
    log(f"  Parsed: raw={metrics.get('raw_bpb')}, int6={metrics.get('int6_bpb')}, "
        f"sliding={metrics.get('sliding_bpb')}, ttt={metrics.get('ttt_bpb')}, "
        f"size={metrics.get('size_bytes')}")
    return metrics


# ---------------------------------------------------------------------------
# Phase 3: Artifact assembly
# ---------------------------------------------------------------------------
def phase3_artifact(metrics: dict) -> None:
    banner("PHASE 3: Artifact assembly")
    size_bytes = metrics.get("size_bytes")
    if size_bytes:
        size_mb = size_bytes / 1_000_000
        ok = size_bytes < SIZE_LIMIT_BYTES
        log(f"  Artifact size: {size_bytes:,} bytes ({size_mb:.2f} MB) — {'PASS' if ok else 'FAIL'}")
    artifact_src = BESE_DIR / "final_model.int6.ptz"
    artifact_dst = NET_VOL / "checkpoints" / "v9" / "final_model.int6.ptz"
    if artifact_src.exists():
        artifact_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact_src, artifact_dst)
        log(f"  Saved artifact to {artifact_dst}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(metrics: dict, total_elapsed: float) -> None:
    banner("RUN SUMMARY")
    log(f"  raw_bpb:     {metrics.get('raw_bpb', 'N/A')}")
    log(f"  int6_bpb:    {metrics.get('int6_bpb', 'N/A')}")
    log(f"  sliding_bpb: {metrics.get('sliding_bpb', 'N/A')}")
    log(f"  ttt_bpb:     {metrics.get('ttt_bpb', 'N/A')}")
    log(f"  best_bpb:    {metrics.get('best_bpb', 'N/A')}")
    sb = metrics.get("size_bytes")
    log(f"  size:        {sb / 1_000_000:.2f} MB" if sb else "  size: N/A")
    log(f"\n  Total wall: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    log(f"  Logfile:    {LOGFILE}")
    log(f"\n  v9 config:")
    log(f"    vocab=288 (47 base + 241 BPE)  layers=12  dim=512  mlp_mult=3.5")
    log(f"    spatial_init: ENABLED ({COORDS_JSON.name}, first 12 dims)")
    log(f"    bpe_scoring: spatial pow_0.02 (decay=3, sigma from FineWeb)")
    log(f"    ttt: enabled (1 epoch, 32K chunk)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="BESE v9: v3 Spatial submission pipeline")
    parser.add_argument("--skip-prep", action="store_true",
                        help="Skip ALL Phase 0 steps (coords/BPE/shards/ngram must already exist)")
    parser.add_argument("--skip-coords", action="store_true",
                        help="Skip 0a (coords already built)")
    parser.add_argument("--skip-bpe", action="store_true",
                        help="Skip 0b (BPE already trained)")
    parser.add_argument("--skip-shards", action="store_true",
                        help="Skip 0c (shards already encoded)")
    parser.add_argument("--skip-ngram", action="store_true",
                        help="Skip 0d (ngram table already built)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip Phase 1 (training); read metrics from existing log")
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="Number of GPUs for torchrun (default: 8)")
    args = parser.parse_args()

    _open_log()
    t_start = time.time()

    banner("BESE v9 (v3 Spatial Letter Encoding) Pipeline")
    log(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Workdir: {BESE_DIR}")
    log(f"  Net vol: {NET_VOL}")
    log(f"  Logfile: {LOGFILE}")

    detected = detect_gpus()
    if detected > 0 and detected != args.num_gpus:
        log(f"  Detected {detected} GPUs (requested {args.num_gpus}), using {detected}")
        args.num_gpus = detected

    # Auto-seed: if the user pre-built coords + BPE locally and committed
    # them to the repo, copy them onto the network volume so Phase 0a/0b
    # auto-skip via their existing-file checks.
    _seed_from_repo()

    # Phase 0: v3 prep
    if not args.skip_prep:
        if not args.skip_coords:
            phase0a_build_coords()
        if not args.skip_bpe:
            phase0b_train_bpe()
        if not args.skip_shards:
            phase0c_export_shards()
        if not args.skip_ngram:
            phase0d_build_ngram()
    else:
        log("  --skip-prep: skipping all Phase 0 steps")

    # Sanity check: prep outputs must exist before Phase 1
    for required in [COORDS_JSON, BPE_OUTPUT, NGRAM_TABLE]:
        if not required.exists():
            raise FileNotFoundError(f"Required prep output missing: {required}")
    train_shards = list(SHARD_DIR.glob("fineweb_train_*.bin"))
    if not train_shards:
        raise FileNotFoundError(f"No training shards in {SHARD_DIR}")
    log(f"  Prep ready: coords + BPE + ngram + {len(train_shards)} train shards")

    # Phase 1: Training
    if args.skip_train:
        log("  --skip-train: reading metrics from existing log")
        train_output = LOGFILE.read_text(encoding="utf-8", errors="replace") if LOGFILE.exists() else ""
    else:
        train_output = phase1_training(args.num_gpus)

    metrics = phase2_eval(train_output)
    phase3_artifact(metrics)
    print_summary(metrics, time.time() - t_start)
    _close_log()


if __name__ == "__main__":
    main()
