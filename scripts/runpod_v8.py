#!/usr/bin/env python3
"""
RunPod v8: End-to-end BESE submission pipeline for Parameter Golf.

Targets 8xH100 pod. Runs the v8 pipeline:
  Phase 0 (untimed): Build bigram prior matrix (reuses v5/v6 shards + tokenizer)
  Phase 1 (timed):   600s wallclock training with torchrun on 8 GPUs
  Phase 2 (timed):   Eval with n-gram tilt + TTT
  Phase 3 (untimed): Artifact assembly — quantize + compress + size check

v8 changes vs v6.1:
  - 12 layers (was 13) with mlp_mult=3.5 (was 3) — proven v6.1 config
  - Noisy QAT: Gaussian noise calibrated to INT6 quantization error (replaces STE QAT)
  - Bigram prior: frozen 288x288 log-prob matrix as logit bias during training + inference
  - TTT fixed: works properly because Noisy QAT collapses INT6 gap
  - Reuses v5/v6 shards and tokenizer (skip data prep entirely)

Storage layout:
  /runpod-volume/           — Network volume (persists across pod restarts)
    bese_shards_v5/         — v5/v6/v7 training shards (REUSED)
    tokenizers/bese_bpe_248_v5.json  — 288 vocab tokenizer (REUSED)
    artifacts/ngram_table_v6.bin     — ngram table (REUSED)
    artifacts/bigram_prior_v8.pt     — NEW: 288x288 bigram log-prob matrix
    checkpoints/v8/         — NEW: v8 artifacts
    logs/run_v8.log         — NEW: v8 run log

Usage (on the RunPod pod):
  cd /workspace && git clone https://github.com/mrbese/parameter-golf-bese.git bese
  cd /workspace/bese && python scripts/runpod_v8.py --skip-shards --num-gpus 8

  # If shards don't exist yet (first run on fresh pod):
  python scripts/runpod_v8.py --num-gpus 8
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (v8: network volume for persistence)
# ---------------------------------------------------------------------------
NET_VOL = Path("/runpod-volume")
BESE_DIR = Path(os.environ.get("BESE_DIR", "/workspace/bese"))
WORK_DIR = Path("/workspace")
PG_DIR = Path(os.environ.get("PG_DIR", "/workspace/parameter-golf"))

SP_MODEL = PG_DIR / "data/tokenizers/fineweb_1024_bpe.model"
SP_SHARD_DIR = PG_DIR / "data/datasets/fineweb10B_sp1024"

BPE_OUTPUT = NET_VOL / "tokenizers" / "bese_bpe_248_v5.json"       # REUSE v5 tokenizer
SHARD_DIR = NET_VOL / "bese_shards_v5"                              # REUSE v5/v6 shards
NGRAM_TABLE = NET_VOL / "artifacts" / "ngram_table_v6.bin"          # REUSE v6 ngram table
TRAIN_SCRIPT = BESE_DIR / "integration" / "train_gpt_bese.py"
LOGFILE = NET_VOL / "logs" / "run_v8.2.log"

# Ensure persistent directories exist
for d in [NET_VOL / "checkpoints" / "v8", NET_VOL / "logs", NET_VOL / "artifacts"]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Training environment (v8 configuration)
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
    # --- ALL v8 experiments DISABLED ---
    "QAT_ENABLED": "0",
    "NOISY_QAT_ENABLED": "0",
    "BIGRAM_PRIOR_ENABLED": "0",
    "LATE_QAT_THRESHOLD": "0",
    # --- TTT (the only new thing) ---
    "TTT_ENABLED": "1",
    "TTT_LR": "0.005",
    "TTT_MOMENTUM": "0.9",
    "TTT_EPOCHS": "1",
    "TTT_GRAD_CLIP": "1.0",
    "TTT_CHUNK_SIZE": "32768",
}

SIZE_LIMIT_BYTES = 16_000_000  # 16 MB hard limit


# ---------------------------------------------------------------------------
# Helpers
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


def run_cmd(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    label: str = "",
    timeout: int | None = None,
) -> str:
    """Run a subprocess, stream output to console + logfile, return full output."""
    if label:
        log(f"  [{label}] Running: {' '.join(str(c) for c in cmd)}")
    else:
        log(f"  Running: {' '.join(str(c) for c in cmd)}")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    t0 = time.time()
    output_lines: list[str] = []
    proc = subprocess.Popen(
        cmd,
        env=merged_env,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
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
        raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(str(c) for c in cmd)}")

    elapsed = time.time() - t0
    if proc.returncode != 0:
        log(f"  ERROR: exit code {proc.returncode} after {elapsed:.1f}s")
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): {' '.join(str(c) for c in cmd)}"
        )
    log(f"  Completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return "".join(output_lines)


def detect_gpus() -> int:
    """Return the number of available NVIDIA GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
        )
        gpus = [l for l in result.stdout.strip().split("\n") if "GPU" in l]
        return len(gpus)
    except Exception:
        return 1


def extract_metrics(output: str) -> dict:
    """Parse training/eval output for key metrics."""
    metrics: dict = {}
    for line in output.strip().split("\n"):
        # Sliding window BPB
        if "final_int6_lzma_roundtrip_exact" in line and "val_bpb:" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["sliding_bpb"] = float(part.split(":")[1])
                if part.startswith("val_loss:"):
                    metrics["sliding_loss"] = float(part.split(":")[1])
        # INT6 roundtrip
        if "final_int6_roundtrip_exact" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["int6_bpb"] = float(part.split(":")[1])
        # TTT result
        if "final_ttt_sliding_window_exact" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["ttt_bpb"] = float(part.split(":")[1])
        # Raw BPB (DIAGNOSTIC post_ema)
        if "DIAGNOSTIC post_ema" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["raw_bpb"] = float(part.split(":")[1])
        # Submission size
        if "Total submission size" in line and "bytes" in line:
            m = re.search(r"(\d+)\s*bytes", line)
            if m:
                metrics["size_bytes"] = int(m.group(1))
        # Serialized model size
        if "Serialized model" in line and "bytes" in line:
            m = re.search(r"(\d+)\s*bytes", line)
            if m:
                metrics["model_bytes"] = int(m.group(1))

    # Best BPB: ttt > sliding > int6
    metrics["best_bpb"] = (
        metrics.get("ttt_bpb")
        or metrics.get("sliding_bpb")
        or metrics.get("int6_bpb")
    )
    return metrics


# ---------------------------------------------------------------------------
# v8: Build bigram log-probability prior matrix
# ---------------------------------------------------------------------------
def build_bigram_prior():
    """Build a 288x288 bigram log-probability matrix from the first training shard.

    Entry [i, j] = log P(next=j | current=i), estimated from token bigram counts.
    Used as a frozen logit bias during training so the model learns residuals.
    """
    import torch
    import numpy as np

    if BIGRAM_PRIOR.exists():
        log(f"  Bigram prior already exists: {BIGRAM_PRIOR}")
        return

    banner("Building bigram prior matrix (288x288)")

    vocab_size = int(TRAIN_ENV["VOCAB_SIZE"])

    # Load first training shard
    shard_files = sorted(SHARD_DIR.glob("fineweb_train_*.bin"))
    if not shard_files:
        raise FileNotFoundError(f"No training shards in {SHARD_DIR}")

    HEADER_INTS = 256
    header_bytes = HEADER_INTS * 4
    tokens = np.fromfile(str(shard_files[0]), dtype=np.uint16, offset=header_bytes)
    # Use up to 50M tokens (more than enough for bigram stats)
    tokens = tokens[:50_000_000]
    log(f"  Loaded {len(tokens):,} tokens from {shard_files[0].name}")

    # Count bigrams
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    prev = tokens[:-1]
    curr = tokens[1:]
    # Filter to valid token range
    mask = (prev < vocab_size) & (curr < vocab_size)
    np.add.at(counts, (prev[mask], curr[mask]), 1.0)

    # Convert to log probabilities with Laplace smoothing
    alpha = 0.01  # small smoothing constant
    counts += alpha
    row_sums = counts.sum(axis=1, keepdims=True)
    log_probs = np.log(counts / row_sums).astype(np.float32)

    # Save as torch tensor
    bigram_tensor = torch.from_numpy(log_probs)
    BIGRAM_PRIOR.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bigram_tensor, str(BIGRAM_PRIOR))

    size_kb = BIGRAM_PRIOR.stat().st_size / 1024
    log(f"  Bigram prior: {vocab_size}x{vocab_size} = {vocab_size**2:,} entries, {size_kb:.1f} KB")
    log(f"  Saved to {BIGRAM_PRIOR}")


# ---------------------------------------------------------------------------
# N-gram table build (reused from v6)
# ---------------------------------------------------------------------------
def _build_ngram_table() -> None:
    """Build (or rebuild) the n-gram frequency table."""
    if TRAIN_ENV.get("NGRAM_TILT_ENABLED", "0") == "0":
        log("  N-gram tilt disabled — skipping table build")
        return

    max_n = TRAIN_ENV.get("NGRAM_TILT_MAX_N", "3")

    if NGRAM_TABLE.exists():
        log(f"  N-gram table already exists: {NGRAM_TABLE}")
        return

    banner(f"Build n-gram frequency table (max-n={max_n})")
    first_shard = sorted(SHARD_DIR.glob("fineweb_train_*.bin"))
    if not first_shard:
        raise FileNotFoundError(f"No training shards found in {SHARD_DIR}")
    run_cmd(
        [
            sys.executable,
            str(BESE_DIR / "scripts" / "build_ngram_table.py"),
            "--shard", str(first_shard[0]),
            "--output", str(NGRAM_TABLE),
            "--max-n", max_n,
            "--top-k", "1",
        ],
        cwd=BESE_DIR,
        label="ngram table",
    )
    if NGRAM_TABLE.exists():
        ngram_size = NGRAM_TABLE.stat().st_size
        log(f"  N-gram table size: {ngram_size:,} bytes ({ngram_size / 1024 / 1024:.2f} MB)")


# ---------------------------------------------------------------------------
# Phase 1: Training (timed, 600s wallclock cap)
# ---------------------------------------------------------------------------
def phase1_training(num_gpus: int) -> str:
    banner(f"PHASE 1: TRAINING (600s wallclock, {num_gpus} GPUs)")

    # Pre-flight checks
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")
    if not BPE_OUTPUT.exists():
        raise FileNotFoundError(f"Tokenizer not found: {BPE_OUTPUT}")
    train_shards = list(SHARD_DIR.glob("fineweb_train_*.bin"))
    val_shards = list(SHARD_DIR.glob("fineweb_val_*.bin"))
    if not train_shards:
        raise FileNotFoundError(f"No training shards in {SHARD_DIR}")
    if not val_shards:
        raise FileNotFoundError(f"No validation shards in {SHARD_DIR}")
    log(f"  Shards: {len(train_shards)} train, {len(val_shards)} val")

    env = TRAIN_ENV.copy()
    env["RUN_ID"] = "bese_v8.2"
    env["BESE_TOKENIZER_ROOT"] = str(BESE_DIR / "tokenizer")
    env["VAL_LOSS_EVERY"] = "500"
    env["TRAIN_LOG_EVERY"] = "100"
    env["EVAL_STRIDE"] = "64"

    log("  Training env:")
    for k in sorted(env):
        log(f"    {k}={env[k]}")

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        str(TRAIN_SCRIPT),
    ]

    output = run_cmd(cmd, env=env, cwd=BESE_DIR, label="torchrun", timeout=1800)
    return output


# ---------------------------------------------------------------------------
# Phase 2: Evaluation
# ---------------------------------------------------------------------------
def phase2_eval(train_output: str) -> dict:
    """Extract eval metrics from the training output."""
    banner("PHASE 2: EVALUATION")

    metrics = extract_metrics(train_output)

    if metrics:
        log("  Extracted metrics from training output:")
        for k, v in sorted(metrics.items()):
            log(f"    {k}: {v}")
    else:
        log("  WARNING: No metrics could be extracted from training output.")
        log("  Check the training log for errors.")

    return metrics


# ---------------------------------------------------------------------------
# Phase 3: Artifact assembly + copy to network volume
# ---------------------------------------------------------------------------
def phase3_artifact(metrics: dict) -> None:
    banner("PHASE 3: ARTIFACT ASSEMBLY")

    size_bytes = metrics.get("size_bytes") or metrics.get("model_bytes")
    if size_bytes:
        size_mb = size_bytes / 1_000_000
        under_limit = size_bytes < SIZE_LIMIT_BYTES
        status = "PASS" if under_limit else "FAIL"
        log(f"  Artifact size: {size_bytes:,} bytes ({size_mb:.2f} MB)")
        log(f"  Size check: {status} (limit: {SIZE_LIMIT_BYTES / 1_000_000:.0f} MB)")
        if not under_limit:
            log(f"  WARNING: Artifact is {size_bytes - SIZE_LIMIT_BYTES:,} bytes over the 16 MB limit!")
    else:
        log("  WARNING: Could not determine artifact size from training output.")

    # Copy artifact to persistent network volume
    artifact_src = BESE_DIR / "final_model.int6.ptz"
    artifact_dst = NET_VOL / "checkpoints" / "v8" / "final_model.int6.ptz"
    if artifact_src.exists():
        shutil.copy2(artifact_src, artifact_dst)
        log(f"  Saved artifact to {artifact_dst}")
    else:
        log(f"  WARNING: artifact not found at {artifact_src}")

    # Also copy the raw model if available
    raw_src = BESE_DIR / "final_model.pt"
    raw_dst = NET_VOL / "checkpoints" / "v8" / "final_model.pt"
    if raw_src.exists():
        shutil.copy2(raw_src, raw_dst)
        log(f"  Saved raw model to {raw_dst}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(metrics: dict, total_elapsed: float) -> None:
    banner("RUN SUMMARY")

    raw = metrics.get("raw_bpb", "N/A")
    best = metrics.get("best_bpb", "N/A")
    sliding = metrics.get("sliding_bpb", "N/A")
    int6 = metrics.get("int6_bpb", "N/A")
    ttt = metrics.get("ttt_bpb", "N/A")
    size_bytes = metrics.get("size_bytes") or metrics.get("model_bytes")
    size_str = f"{size_bytes / 1_000_000:.2f} MB" if size_bytes else "N/A"

    log(f"  raw_bpb:     {raw}")
    log(f"  int6_bpb:    {int6}")
    log(f"  sliding_bpb: {sliding}")
    log(f"  ttt_bpb:     {ttt}")
    log(f"  best_bpb:    {best}")
    log(f"  total_size:  {size_str}")

    # v8: INT6 gap analysis
    if isinstance(raw, float) and isinstance(int6, float):
        gap = int6 - raw
        log(f"  int6_gap:    {gap:.4f} (target: <0.010)")

    if size_bytes:
        under = size_bytes < SIZE_LIMIT_BYTES
        log(f"  under 16 MB: {'YES' if under else 'NO'}")
    else:
        log("  under 16 MB: UNKNOWN")

    log(f"\n  Total wall time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    log(f"  Log file: {LOGFILE}")

    # Key config summary
    log("\n  v8 config:")
    log(f"    vocab_size={TRAIN_ENV['VOCAB_SIZE']}  layers={TRAIN_ENV['NUM_LAYERS']}"
        f"  dim={TRAIN_ENV['MODEL_DIM']}  mlp_mult={TRAIN_ENV['MLP_MULT']}")
    log(f"    depth_recurrence: layers {TRAIN_ENV['DEPTH_RECURRENCE_START']}-{TRAIN_ENV['DEPTH_RECURRENCE_END']}"
        f" x{TRAIN_ENV['DEPTH_RECURRENCE_LOOPS']} loops"
        f" (active after {float(TRAIN_ENV['DEPTH_RECURRENCE_ACTIVATION_FRAC']) * 100:.0f}% of training)")
    log(f"    parallel_residual: start={TRAIN_ENV['PARALLEL_RESIDUAL_START']}")
    log(f"    noisy_qat: disabled")
    log(f"    bigram_prior: disabled")
    log(f"    ttt: enabled={TRAIN_ENV['TTT_ENABLED']} "
        f"lr={TRAIN_ENV['TTT_LR']} epochs={TRAIN_ENV['TTT_EPOCHS']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="BESE v8: Full Parameter Golf submission pipeline (8xH100)"
    )
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip all data prep (shards + ngram table must already exist)",
    )
    parser.add_argument(
        "--skip-shards",
        action="store_true",
        help="Skip shard prep (reuse v5/v6 shards), rebuild ngram table if needed",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip Phase 1 (training) if checkpoint already exists",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs for torchrun (default: 8, auto-detected)",
    )
    args = parser.parse_args()

    _open_log()
    t_start = time.time()

    banner("BESE v8 Pipeline")
    log(f"  Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Working directory: {BESE_DIR}")
    log(f"  Network volume: {NET_VOL}")
    log(f"  Log file: {LOGFILE}")

    # Auto-detect GPUs
    detected = detect_gpus()
    if detected > 0:
        if args.num_gpus != detected:
            log(f"  Detected {detected} GPUs (requested {args.num_gpus}), using {detected}")
            args.num_gpus = detected
        else:
            log(f"  Detected {detected} GPUs")
    else:
        log(f"  GPU detection failed, using {args.num_gpus}")

    # Verify shards exist (v8 reuses v5/v6 data)
    if not BPE_OUTPUT.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {BPE_OUTPUT}. "
            "v8 requires existing v5/v6 tokenizer on the network volume."
        )
    train_shards = list(SHARD_DIR.glob("fineweb_train_*.bin"))
    if not train_shards:
        raise FileNotFoundError(
            f"No training shards in {SHARD_DIR}. "
            "v8 requires existing v5/v6 shards on the network volume."
        )
    log(f"  Found tokenizer: {BPE_OUTPUT}")
    log(f"  Found {len(train_shards)} training shards in {SHARD_DIR}")

    # Build ngram table if needed (reuses v6 table if it exists)
    if not args.skip_prep:
        _build_ngram_table()

    # Phase 1: Training
    train_output = ""
    if args.skip_train:
        log("\n  --skip-train: Skipping Phase 1 (training)")
        if LOGFILE.exists():
            log(f"  Will attempt to extract metrics from {LOGFILE}")
            train_output = LOGFILE.read_text(encoding="utf-8", errors="replace")
        else:
            log("  WARNING: No log file found; metrics will be unavailable.")
    else:
        train_output = phase1_training(args.num_gpus)

    # Phase 2: Evaluation
    metrics = phase2_eval(train_output)

    # Phase 3: Artifact assembly + copy to network volume
    phase3_artifact(metrics)

    # Summary
    total_elapsed = time.time() - t_start
    print_summary(metrics, total_elapsed)

    _close_log()


if __name__ == "__main__":
    main()
