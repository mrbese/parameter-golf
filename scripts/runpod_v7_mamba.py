#!/usr/bin/env python3
"""
RunPod v7-mamba: Mamba-3 + Attention hybrid for Parameter Golf.

Non-record submission for the unlimited compute track.
Architecture: 7 Mamba-3 blocks + 1 Attention block, BESE 288 vocab.
Thesis: BESE's 2x token density is an advantage with O(n) SSMs.

Uses existing v5/v6 shards and tokenizer — no data prep needed.

Usage:
  cd /workspace/bese
  pip install einops
  python scripts/runpod_v7_mamba.py --skip-shards --num-gpus 8
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — reuse v5/v6 persistent data
# ---------------------------------------------------------------------------
NET_VOL = Path("/runpod-volume")
BESE_DIR = Path(os.environ.get("BESE_DIR", "/workspace/bese"))
WORK_DIR = Path("/workspace")
PG_DIR = Path(os.environ.get("PG_DIR", "/workspace/parameter-golf"))

BPE_OUTPUT = NET_VOL / "tokenizers" / "bese_bpe_248_v5.json"
SHARD_DIR = NET_VOL / "bese_shards_v5"
NGRAM_TABLE = NET_VOL / "artifacts" / "ngram_table_v6.bin"

# Mamba uses the same training script as the transformer — we swap the model class
# via an env var. The training script checks MODEL_TYPE to decide which model to build.
TRAIN_SCRIPT = BESE_DIR / "integration" / "train_gpt_bese.py"
LOGFILE = NET_VOL / "logs" / "run_v7_mamba.log"

# ---------------------------------------------------------------------------
# Training environment
# ---------------------------------------------------------------------------
TRAIN_ENV = {
    # Model architecture
    "MODEL_TYPE": "mamba_hybrid",           # NEW: tells training script to use HybridMambaGPT
    "VOCAB_SIZE": "288",
    "NUM_LAYERS": "8",                       # 7 Mamba + 1 Attention = 8 physical layers
    "MODEL_DIM": "512",
    "MLP_MULT": "3.0",                      # for the attention layer's MLP
    "NUM_HEADS": "8",                        # for the attention layer
    "NUM_KV_HEADS": "4",                     # for the attention layer
    # Mamba-specific
    "D_STATE": "64",
    "MAMBA_EXPAND": "2",
    "MAMBA_HEADDIM": "64",
    "MAMBA_CHUNK_SIZE": "64",
    "ATTN_LAYER_POS": "4",                  # attention layer at position 4 (middle)
    # Depth recurrence on Mamba layers 2-4
    "DEPTH_RECURRENCE_START": "2",
    "DEPTH_RECURRENCE_END": "4",
    "DEPTH_RECURRENCE_LOOPS": "3",
    "DEPTH_RECURRENCE_ACTIVATION_FRAC": "0.35",
    # Training
    "QK_GAIN_INIT": "5.25",
    "MATRIX_LR": "0.026",
    "MUON_WD": "0.095",
    "ADAM_WD": "0.095",
    "EMA_DECAY": "0.9965",
    "WARMDOWN_ITERS": "5000",
    "TRAIN_SEQ_LEN": "2048",                # match transformer; 4096 maxes VRAM at 99%
    "EVAL_SEQ_LEN": "2048",
    "EVAL_STRIDE": "64",
    "TOKENIZER_PATH": str(BPE_OUTPUT),
    "DATA_PATH": str(SHARD_DIR),
    "MAX_WALLCLOCK_SECONDS": "600",
    # N-gram tilt
    "NGRAM_TILT_ENABLED": "1",
    "NGRAM_TILT_MAX_N": "3",
    "NGRAM_PRIOR_PATH": str(NGRAM_TABLE),
    # Disable v8 experiments
    "QAT_ENABLED": "0",
    "NOISY_QAT_ENABLED": "0",
    "BIGRAM_PRIOR_ENABLED": "0",
    "LATE_QAT_THRESHOLD": "0",
    # TTT
    "TTT_ENABLED": "1",
    "TTT_LR": "0.005",
    "TTT_MOMENTUM": "0.9",
    "TTT_EPOCHS": "1",
    "TTT_GRAD_CLIP": "1.0",
    "TTT_CHUNK_SIZE": "32768",
}

SIZE_LIMIT_BYTES = 16_000_000

# ---------------------------------------------------------------------------
# Utilities (shared with runpod_v6.py)
# ---------------------------------------------------------------------------

_LOG_FH = None

def _open_log():
    global _LOG_FH
    LOGFILE.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FH = open(LOGFILE, "w", encoding="utf-8")

def _close_log():
    global _LOG_FH
    if _LOG_FH:
        _LOG_FH.close()
        _LOG_FH = None

def log(msg: str = ""):
    print(msg, flush=True)
    if _LOG_FH:
        _LOG_FH.write(msg + "\n")
        _LOG_FH.flush()

def banner(msg: str):
    sep = "=" * 72
    log(f"\n{sep}\n  {msg}\n{sep}")

def run_cmd(cmd, env=None, cwd=None, label="cmd", timeout=1800):
    full_env = {**os.environ, **(env or {})}
    log(f"\n  [{label}] Running: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    output_lines = []
    proc = subprocess.Popen(
        cmd, env=full_env, cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if _LOG_FH:
                _LOG_FH.write(line)
                _LOG_FH.flush()
            output_lines.append(line)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise RuntimeError(f"Command timed out after {timeout}s")
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log(f"  [{label}] FAILED (exit {proc.returncode}) after {elapsed:.1f}s")
    else:
        log(f"  [{label}] Completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return "".join(output_lines)

def detect_gpus() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return len(result.stdout.strip().split("\n"))
    except Exception:
        return 0

def extract_metrics(output: str) -> dict:
    metrics = {}
    patterns = {
        "best_bpb": r"best_val_bpb:([0-9.]+)",
        "sliding_bpb": r"sliding.*?val_bpb:([0-9.]+)",
        "int6_bpb": r"final_int6.*?val_bpb:([0-9.]+)",
        "ttt_bpb": r"final_ttt.*?val_bpb:([0-9.]+)",
        "size_bytes": r"Total submission size.*?:\s*(\d+)",
        "model_bytes": r"Serialized model.*?:\s*(\d+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, output, re.IGNORECASE)
        if m:
            val = m.group(1)
            metrics[key] = float(val) if "." in val else int(val)
    return metrics


# ---------------------------------------------------------------------------
# Pre-flight: install einops
# ---------------------------------------------------------------------------
def install_deps():
    banner("Installing dependencies")
    run_cmd([sys.executable, "-m", "pip", "install", "einops", "--quiet"],
            label="pip-einops")


# ---------------------------------------------------------------------------
# Phase 1: Training
# ---------------------------------------------------------------------------
def phase1_training(num_gpus: int) -> str:
    banner(f"PHASE 1: MAMBA HYBRID TRAINING (600s wallclock, {num_gpus} GPUs)")

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
    env["RUN_ID"] = "bese_v7_mamba"
    env["BESE_TOKENIZER_ROOT"] = str(BESE_DIR / "tokenizer")
    env["PYTHONPATH"] = str(BESE_DIR)  # so `from integration.mamba3_ssd import ...` works
    env["VAL_LOSS_EVERY"] = "500"
    env["TRAIN_LOG_EVERY"] = "100"

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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BESE v7-mamba: Mamba-3 + Attention hybrid (non-record)"
    )
    parser.add_argument("--skip-shards", action="store_true",
                        help="Skip shard prep (use existing v5/v6 shards)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (use existing log)")
    parser.add_argument("--num-gpus", type=int, default=8)
    args = parser.parse_args()

    _open_log()
    t_start = time.time()

    banner("BESE v7-mamba: Mamba-3 + Attention Hybrid")
    log(f"  Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Architecture: 7 Mamba-3 + 1 Attention, dim=512, BESE 288 vocab")
    log(f"  Thesis: BESE's 2x token density is an advantage with O(n) SSMs")

    # Install deps
    install_deps()

    # Check shards
    if not args.skip_shards:
        log("  WARNING: No data prep for Mamba — reusing v5/v6 shards.")
        log("  If shards don't exist, run v6 first to generate them.")

    train_shards = list(SHARD_DIR.glob("fineweb_train_*.bin"))
    if not train_shards:
        raise FileNotFoundError(
            f"No training shards in {SHARD_DIR}. "
            "Run v6 first to generate shards, then use --skip-shards."
        )
    log(f"  Found {len(train_shards)} training shards")

    # Detect GPUs
    detected = detect_gpus()
    if detected > 0:
        args.num_gpus = detected
        log(f"  Detected {detected} GPUs")

    # Train
    train_output = ""
    if args.skip_train:
        log("\n  --skip-train: Skipping training")
        if LOGFILE.exists():
            train_output = LOGFILE.read_text(encoding="utf-8", errors="replace")
    else:
        train_output = phase1_training(args.num_gpus)

    # Metrics
    metrics = extract_metrics(train_output)

    banner("RUN SUMMARY")
    for k, v in sorted(metrics.items()):
        log(f"  {k}: {v}")

    total_elapsed = time.time() - t_start
    log(f"\n  Total wall time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    # Copy artifact to network volume
    artifact_src = BESE_DIR / "final_model.int6.ptz"
    artifact_dst = NET_VOL / "checkpoints" / "v7_mamba" / "final_model.int6.ptz"
    if artifact_src.exists():
        artifact_dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(artifact_src, artifact_dst)
        log(f"  Saved artifact to {artifact_dst}")

    _close_log()


if __name__ == "__main__":
    main()
