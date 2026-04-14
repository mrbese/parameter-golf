#!/usr/bin/env python3
"""
RunPod v5: End-to-end BESE submission pipeline for Parameter Golf.

Targets 8xH100 pod. Runs the full v5 pipeline:
  Phase 0 (untimed): BPE training, curriculum sort, data filtering + shard export, n-gram table
  Phase 1 (timed):   600s wallclock training with torchrun on 8 GPUs
  Phase 2 (timed):   Eval with SLOT + n-gram tilt
  Phase 3 (untimed): Artifact assembly — quantize + compress + size check

Usage (on the RunPod pod):
  cd /workspace && git clone https://github.com/mrbese/parameter-golf-bese.git bese
  cd /workspace/bese && python scripts/runpod_v5.py

  # Resume after data prep:
  python scripts/runpod_v5.py --skip-prep

  # Resume after training:
  python scripts/runpod_v5.py --skip-prep --skip-train
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
# Paths
# ---------------------------------------------------------------------------
BESE_DIR = Path(os.environ.get("BESE_DIR", "/workspace/bese"))
WORK_DIR = Path("/workspace")

BPE_INPUT = WORK_DIR / "decoded_docs_jsonl" / "fineweb_train_all.jsonl"
BPE_OUTPUT = BESE_DIR / "tokenizers" / "bese_bpe_248_v5.json"
CURRICULUM_OUTPUT = WORK_DIR / "decoded_docs_jsonl" / "fineweb_train_curriculum.jsonl"
SHARD_DIR = Path("/workspace/bese_shards_v5")
NGRAM_TABLE = BESE_DIR / "artifacts" / "ngram_table_v5.bin"
TRAIN_SCRIPT = BESE_DIR / "integration" / "train_gpt_bese.py"
LOGFILE = WORK_DIR / "run_v5.log"

# ---------------------------------------------------------------------------
# Training environment (v5 target configuration)
# ---------------------------------------------------------------------------
TRAIN_ENV = {
    "VOCAB_SIZE": "288",
    "NUM_LAYERS": "11",
    "MODEL_DIM": "512",
    "MLP_MULT": "3",
    "NUM_HEADS": "8",
    "NUM_KV_HEADS": "4",
    "DEPTH_RECURRENCE_START": "3",
    "DEPTH_RECURRENCE_END": "5",
    "DEPTH_RECURRENCE_LOOPS": "3",
    "DEPTH_RECURRENCE_ACTIVATION_FRAC": "0.35",
    "PARALLEL_RESIDUAL_START": "7",
    "QK_GAIN_INIT": "5.0",
    "MATRIX_LR": "0.022",
    "MUON_WD": "0.095",
    "ADAM_WD": "0.095",
    "EMA_DECAY": "0.9965",
    "WARMDOWN_ITERS": "5000",  # ~72% of ~7000 steps (was WARMDOWN_FRAC in plan, code uses WARMDOWN_ITERS)
    "TOKENIZER_PATH": str(BPE_OUTPUT),
    "DATA_PATH": str(SHARD_DIR),
    "MAX_WALLCLOCK_SECONDS": "600",
    "SLOT_ENABLED": "1",
    # v5: N-gram tilt at eval time
    "NGRAM_TILT_ENABLED": "1",
    "NGRAM_TILT_BETA": "0.3",
    "NGRAM_TILT_MAX_N": "4",
    "NGRAM_PRIOR_PATH": str(NGRAM_TABLE),
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
        if "final_int8_zlib_roundtrip_exact" in line and "val_bpb:" in line:
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
        # SLOT
        if "slot_lbfgs_exact" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["slot_bpb"] = float(part.split(":")[1])
        # TTT
        if "legal_ttt_exact" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["ttt_bpb"] = float(part.split(":")[1])
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

    # Best BPB: TTT > SLOT > sliding > int6
    metrics["best_bpb"] = (
        metrics.get("ttt_bpb")
        or metrics.get("slot_bpb")
        or metrics.get("sliding_bpb")
        or metrics.get("int6_bpb")
    )
    return metrics


# ---------------------------------------------------------------------------
# Phase 0: Data Preparation (untimed)
# ---------------------------------------------------------------------------
def phase0_data_prep() -> None:
    banner("PHASE 0: DATA PREPARATION (untimed)")
    t0 = time.time()

    # Step 0.1 — Train BESE BPE (248 merges on full FineWeb)
    if BPE_OUTPUT.exists():
        log(f"  Step 0.1: BPE tokenizer already exists at {BPE_OUTPUT}, skipping")
    else:
        banner("Step 0.1: Train BESE BPE (248 merges)")
        if not BPE_INPUT.exists():
            raise FileNotFoundError(
                f"BPE input not found: {BPE_INPUT}\n"
                "Decode SP shards to JSONL first (see docs/run_v5_plan.md)."
            )
        run_cmd(
            [
                sys.executable,
                str(BESE_DIR / "scripts" / "train_bpe_jsonl.py"),
                "--input", str(BPE_INPUT),
                "--output", str(BPE_OUTPUT),
                "--num-merges", "248",
                "--max-docs", "6000000",
            ],
            cwd=BESE_DIR,
            label="BPE train",
        )

    # Step 0.2 — Curriculum sort
    if CURRICULUM_OUTPUT.exists():
        log(f"  Step 0.2: Curriculum-sorted JSONL already exists at {CURRICULUM_OUTPUT}, skipping")
    else:
        banner("Step 0.2: Curriculum sort (easy-to-hard)")
        if not BPE_INPUT.exists():
            raise FileNotFoundError(f"Curriculum input not found: {BPE_INPUT}")
        run_cmd(
            [
                sys.executable,
                str(BESE_DIR / "scripts" / "curriculum_sort.py"),
                "--input", str(BPE_INPUT),
                "--output", str(CURRICULUM_OUTPUT),
            ],
            cwd=BESE_DIR,
            label="curriculum sort",
        )

    # Step 0.3+0.4 — Export v5 BESE shards (with quality filter)
    train_shards = list(SHARD_DIR.glob("fineweb_train_*.bin"))
    if train_shards:
        log(f"  Step 0.3: Found {len(train_shards)} existing train shards in {SHARD_DIR}, skipping export")
    else:
        banner("Step 0.3: Export v5 BESE shards (filtered + curriculum-sorted)")
        # Use curriculum-sorted input if available, otherwise raw
        shard_input = CURRICULUM_OUTPUT if CURRICULUM_OUTPUT.exists() else BPE_INPUT
        if not shard_input.exists():
            raise FileNotFoundError(f"Shard input not found: {shard_input}")
        run_cmd(
            [
                sys.executable,
                str(BESE_DIR / "scripts" / "export_shards.py"),
                "--input", str(shard_input),
                "--tokenizer", str(BPE_OUTPUT),
                "--output-dir", str(SHARD_DIR),
                "--filter",
                "--num-workers", "64",
            ],
            cwd=BESE_DIR,
            label="shard export",
        )

    # Step 0.5 — Build n-gram table
    if NGRAM_TABLE.exists():
        log(f"  Step 0.5: N-gram table already exists at {NGRAM_TABLE}, skipping")
    else:
        banner("Step 0.5: Build n-gram frequency table")
        first_shard = sorted(SHARD_DIR.glob("fineweb_train_*.bin"))
        if not first_shard:
            raise FileNotFoundError(f"No training shards found in {SHARD_DIR}")
        run_cmd(
            [
                sys.executable,
                str(BESE_DIR / "scripts" / "build_ngram_table.py"),
                "--shard", str(first_shard[0]),
                "--output", str(NGRAM_TABLE),
                "--max-n", "4",
                "--top-k", "1",
            ],
            cwd=BESE_DIR,
            label="ngram table",
        )
        # Verify size fits in artifact budget
        if NGRAM_TABLE.exists():
            ngram_size = NGRAM_TABLE.stat().st_size
            log(f"  N-gram table size: {ngram_size:,} bytes ({ngram_size / 1024:.1f} KB)")
            if ngram_size > 500_000:
                log("  WARNING: N-gram table exceeds 500 KB target. Consider --max-n 3 or fewer tokens.")

    elapsed = time.time() - t0
    log(f"\n  Phase 0 total: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


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
    env["RUN_ID"] = "bese_v5"
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

    output = run_cmd(cmd, env=env, cwd=BESE_DIR, label="torchrun")
    return output


# ---------------------------------------------------------------------------
# Phase 2: Evaluation (timed, with SLOT + n-gram tilt)
# ---------------------------------------------------------------------------
def phase2_eval(train_output: str) -> dict:
    """Extract eval metrics from the training output.

    In the current pipeline, eval runs inline at the end of training
    (the training script handles SLOT and eval itself when SLOT_ENABLED=1).
    If a separate eval pass is needed in the future, add it here.
    """
    banner("PHASE 2: EVALUATION (SLOT + n-gram tilt)")

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
# Phase 3: Artifact assembly (quantize + compress + size check)
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
        log("  Check checkpoints manually in /workspace/bese/checkpoints/")

    # Verify n-gram table was included if it exists
    if NGRAM_TABLE.exists():
        ngram_size = NGRAM_TABLE.stat().st_size
        log(f"  N-gram table: {ngram_size:,} bytes ({ngram_size / 1024:.1f} KB)")
    else:
        log("  N-gram table: not found (eval ran without pre-computed priors)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(metrics: dict, total_elapsed: float) -> None:
    banner("RUN SUMMARY")

    best = metrics.get("best_bpb", "N/A")
    sliding = metrics.get("sliding_bpb", "N/A")
    slot = metrics.get("slot_bpb", "N/A")
    ttt = metrics.get("ttt_bpb", "N/A")
    int6 = metrics.get("int6_bpb", "N/A")
    size_bytes = metrics.get("size_bytes") or metrics.get("model_bytes")
    size_str = f"{size_bytes / 1_000_000:.2f} MB" if size_bytes else "N/A"
    model_bytes = metrics.get("model_bytes")
    model_str = f"{model_bytes / 1_000_000:.2f} MB" if model_bytes else "N/A"

    log(f"  best_bpb:    {best}")
    log(f"  sliding_bpb: {sliding}")
    log(f"  slot_bpb:    {slot}")
    log(f"  ttt_bpb:     {ttt}")
    log(f"  int6_bpb:    {int6}")
    log(f"  model_size:  {model_str}")
    log(f"  total_size:  {size_str}")

    if size_bytes:
        under = size_bytes < SIZE_LIMIT_BYTES
        log(f"  under 16 MB: {'YES' if under else 'NO'}")
    else:
        log("  under 16 MB: UNKNOWN")

    log(f"\n  Total wall time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    log(f"  Log file: {LOGFILE}")

    # Key config summary
    log("\n  v5 config:")
    log(f"    vocab_size={TRAIN_ENV['VOCAB_SIZE']}  layers={TRAIN_ENV['NUM_LAYERS']}"
        f"  dim={TRAIN_ENV['MODEL_DIM']}  mlp_mult={TRAIN_ENV['MLP_MULT']}")
    log(f"    depth_recurrence: layers {TRAIN_ENV['DEPTH_RECURRENCE_START']}-{TRAIN_ENV['DEPTH_RECURRENCE_END']}"
        f" x{TRAIN_ENV['DEPTH_RECURRENCE_LOOPS']} loops"
        f" (active after {float(TRAIN_ENV['DEPTH_RECURRENCE_ACTIVATION_FRAC']) * 100:.0f}% of training)")
    log(f"    parallel_residual: start={TRAIN_ENV['PARALLEL_RESIDUAL_START']}")
    log(f"    slot_enabled={TRAIN_ENV['SLOT_ENABLED']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="BESE v5: Full Parameter Golf submission pipeline (8xH100)"
    )
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip Phase 0 (data prep) if shards already exist",
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

    banner("BESE v5 Pipeline")
    log(f"  Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Working directory: {BESE_DIR}")
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

    # Phase 0: Data Preparation
    if args.skip_prep:
        log("\n  --skip-prep: Skipping Phase 0 (data preparation)")
        # Verify critical files exist
        if not BPE_OUTPUT.exists():
            raise FileNotFoundError(
                f"--skip-prep requires tokenizer at {BPE_OUTPUT}. "
                "Run without --skip-prep first."
            )
        train_shards = list(SHARD_DIR.glob("fineweb_train_*.bin"))
        if not train_shards:
            raise FileNotFoundError(
                f"--skip-prep requires training shards in {SHARD_DIR}. "
                "Run without --skip-prep first."
            )
        log(f"  Found tokenizer: {BPE_OUTPUT}")
        log(f"  Found {len(train_shards)} training shards in {SHARD_DIR}")
    else:
        phase0_data_prep()

    # Phase 1: Training
    train_output = ""
    if args.skip_train:
        log("\n  --skip-train: Skipping Phase 1 (training)")
        # Try to load training log for metrics extraction
        if LOGFILE.exists():
            log(f"  Will attempt to extract metrics from {LOGFILE}")
            train_output = LOGFILE.read_text(encoding="utf-8", errors="replace")
        else:
            log("  WARNING: No log file found; metrics will be unavailable.")
    else:
        train_output = phase1_training(args.num_gpus)

    # Phase 2: Evaluation
    metrics = phase2_eval(train_output)

    # Phase 3: Artifact assembly + size check
    phase3_artifact(metrics)

    # Summary
    total_elapsed = time.time() - t_start
    print_summary(metrics, total_elapsed)

    _close_log()


if __name__ == "__main__":
    main()
