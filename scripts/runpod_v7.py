#!/usr/bin/env python3
"""
RunPod v7: End-to-end BESE submission pipeline for Parameter Golf.

Targets 8xH100 pod. Runs the full v7 pipeline:
  Phase 0 (untimed): BPE training, curriculum sort, data filtering + shard export, n-gram table
  Phase 1 (timed):   600s wallclock training with torchrun on 8 GPUs
  Phase 2 (timed):   Eval with n-gram tilt + Legal TTT
  Phase 3 (untimed): Artifact assembly — quantize + compress + size check

v7 changes vs v6.1:
  - 1064 vocab (40 base + 1024 BPE merges) — was 288 (40 + 248 merges)
  - 12 layers, mlp_mult=3.5 — proven in v6.1
  - Persistent data on /workspace network volume (survives pod restarts)
  - Legal TTT at eval with chunk-based scoring (fast ~7 min)
  - LZMA preset 9

Usage (on the RunPod pod):
  cd /workspace && git clone https://github.com/mrbese/parameter-golf-bese.git bese
  cd /workspace/bese && python scripts/runpod_v7.py --num-gpus 8

  # Resume with existing shards (rebuilds ngram table):
  python scripts/runpod_v7.py --skip-shards --num-gpus 8

  # Resume after data prep:
  python scripts/runpod_v7.py --skip-prep --num-gpus 8

  # Resume after training:
  python scripts/runpod_v7.py --skip-prep --skip-train --num-gpus 8
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
# Paths — /workspace IS the network volume (persistent across pod restarts)
# Container root filesystem (/) is ephemeral (80GB overlay).
# ---------------------------------------------------------------------------
NET_VOL = Path("/workspace")  # network volume mount point

BESE_DIR = Path(os.environ.get("BESE_DIR", "/workspace/bese"))
WORK_DIR = Path("/workspace")
PG_DIR = Path(os.environ.get("PG_DIR", "/workspace/parameter-golf"))
TRAIN_SCRIPT = BESE_DIR / "integration" / "train_gpt_bese.py"

# Upstream SP data (read-only)
SP_MODEL = PG_DIR / "data/tokenizers/fineweb_1024_bpe.model"
SP_SHARD_DIR = PG_DIR / "data/datasets/fineweb10B_sp1024"

# Persistent data paths (on network volume, outside git repo)
BPE_OUTPUT = NET_VOL / "tokenizers" / "bese_bpe_1024_v7.json"
SHARD_DIR = NET_VOL / "bese_shards_v7"
NGRAM_TABLE = NET_VOL / "artifacts" / "ngram_table_v7.bin"
LOGFILE = NET_VOL / "logs" / "run_v7.log"
DECODED_CACHE = NET_VOL / "artifacts" / "decoded_docs_v7.pkl"  # Step 0.1 checkpoint

# ---------------------------------------------------------------------------
# Training environment (v7 configuration)
# ---------------------------------------------------------------------------
TRAIN_ENV = {
    "VOCAB_SIZE": "1064",                         # v7: 40 base + 1024 BPE merges (was 288)
    "NUM_LAYERS": "12",                           # v7: 12 layers (was 13) — proven in v6.1
    "MODEL_DIM": "512",
    "MLP_MULT": "3.5",                            # v7: 3.5x (was 3) — proven in v6.1
    "NUM_HEADS": "8",
    "NUM_KV_HEADS": "4",
    "DEPTH_RECURRENCE_START": "3",
    "DEPTH_RECURRENCE_END": "5",
    "DEPTH_RECURRENCE_LOOPS": "3",
    "DEPTH_RECURRENCE_ACTIVATION_FRAC": "0.35",
    "PARALLEL_RESIDUAL_START": "8",               # v7: last 4 of 12 layers (was 9 for 13 layers)
    "QK_GAIN_INIT": "5.25",
    "MATRIX_LR": "0.026",
    "MUON_WD": "0.095",
    "ADAM_WD": "0.095",
    "EMA_DECAY": "0.9965",
    "WARMDOWN_ITERS": "5000",
    "VE_LAYERS": "10,11",                         # v7: last 2 of 12 layers (was 11,12)
    "EVAL_STRIDE": "2048",                         # v7: skip slow SW eval — TTT is the primary eval
    "TOKENIZER_PATH": str(BPE_OUTPUT),
    "DATA_PATH": str(SHARD_DIR),
    "MAX_WALLCLOCK_SECONDS": "600",
    "NGRAM_TILT_ENABLED": "1",
    "NGRAM_TILT_MAX_N": "3",
    "NGRAM_PRIOR_PATH": str(NGRAM_TABLE),
    "TTT_ENABLED": "1",                           # Legal TTT at eval
    "TTT_LR": "0.005",
    "TTT_MOMENTUM": "0.9",
    "TTT_EPOCHS": "1",                            # 1 epoch (fast ~7 min). Try 2-3 if time allows.
    "TTT_GRAD_CLIP": "1.0",
    "TTT_CHUNK_SIZE": "32768",                    # 32K tokens per TTT chunk
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
        # TTT BPB
        if "final_ttt_sliding_window_exact" in line and "val_bpb:" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["ttt_bpb"] = float(part.split(":")[1])

    # Best BPB: ttt > sliding > int6
    metrics["best_bpb"] = (
        metrics.get("ttt_bpb")
        or metrics.get("sliding_bpb")
        or metrics.get("int6_bpb")
    )
    return metrics


# ---------------------------------------------------------------------------
# SP shard decode worker (reused from v3)
# ---------------------------------------------------------------------------
def _decode_shard(args):
    """Decode a single SP shard — runs in a worker process.

    Also applies quality filter and difficulty scoring IN-PROCESS so we
    never accumulate 6M unfiltered docs in the parent, avoiding the
    single-threaded bottleneck that cost 389s + 300s in v5.
    """
    shard_file, sp_model_path, min_len = args
    import sentencepiece as spm
    import numpy as np
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    bos = sp.bos_id()
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(shard_file, dtype="<i4", count=256)
    n = int(header[2])
    tokens = np.fromfile(shard_file, dtype="<u2", count=n, offset=header_bytes)
    scored_docs = []
    current = []
    for t in tokens:
        if t == bos:
            if current:
                text = sp.decode(current)
                if len(text.strip()) > min_len and _is_high_value(text):
                    scored_docs.append((_difficulty_score(text), text))
            current = []
        else:
            current.append(int(t))
    if current:
        text = sp.decode(current)
        if len(text.strip()) > min_len and _is_high_value(text):
            scored_docs.append((_difficulty_score(text), text))
    return str(shard_file.name), n, scored_docs


def _is_high_value(text: str) -> bool:
    """Filter out web boilerplate and low-information docs."""
    words = text.split()
    if len(words) < 50:
        return False
    if len(set(words)) / len(words) < 0.25:
        return False
    lowered = text.lower()
    boilerplate_markers = [
        'cookie', 'subscribe', 'click here', 'privacy policy',
        'all rights reserved', 'terms of service', 'sign up',
    ]
    if sum(1 for m in boilerplate_markers if m in lowered) >= 3:
        return False
    return True


def _difficulty_score(text: str) -> float:
    """Curriculum difficulty: composite of word length, vocab richness, sentence length."""
    words = text.split()
    if not words:
        return 0.0
    avg_word_len = sum(len(w) for w in words) / len(words)
    vocab_richness = len(set(words)) / len(words)
    sentences = max(text.count('.') + text.count('!') + text.count('?'), 1)
    avg_sentence_len = len(words) / sentences
    return (avg_word_len / 10) * 0.3 + vocab_richness * 0.4 + (avg_sentence_len / 30) * 0.3


# ---------------------------------------------------------------------------
# Parallel worker functions (must be top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------
def _encode_chunk(args: tuple) -> "np.ndarray":
    """Encode a chunk of docs with the BESE tokenizer — runs in a worker process."""
    docs, tok_path, bese_tok_root = args
    import sys
    import numpy as np
    sys.path.insert(0, bese_tok_root)
    from bese_fast_bpe import FastBESEBPETokenizer
    tok = FastBESEBPETokenizer.load(tok_path)
    arrays = [np.asarray(tok.encode(text), dtype=np.uint16) for text in docs]
    return np.concatenate(arrays) if arrays else np.array([], dtype=np.uint16)


# ---------------------------------------------------------------------------
# N-gram table build (called from both full prep and --skip-prep path)
# ---------------------------------------------------------------------------
def _build_ngram_table() -> None:
    """Build (or rebuild) the n-gram frequency table.

    Always deletes any existing table before rebuilding so a stale max-n value
    (e.g. a max-n=4 table left over from v5) cannot be picked up by v5.3.
    No-ops when NGRAM_TILT_ENABLED=0.
    """
    if TRAIN_ENV.get("NGRAM_TILT_ENABLED", "0") == "0":
        log("  N-gram tilt disabled — skipping table build")
        return

    max_n = TRAIN_ENV.get("NGRAM_TILT_MAX_N", "3")

    # Always remove existing table — it may have been built with a different max-n
    if NGRAM_TABLE.exists():
        NGRAM_TABLE.unlink()
        log(f"  Removed existing n-gram table (rebuilding with max-n={max_n})")

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
        # Model INT6+LZMA is ~11 MB; artifact budget is 16 MB → ~4.9 MB headroom.
        # LZMA ratio for ngram tables is roughly 5-10x, so flag raw tables > 4 MB.
        if ngram_size > 4_000_000:
            log(
                f"  WARNING: N-gram table is {ngram_size / 1024 / 1024:.1f} MB raw. "
                "LZMA ~5-10x compression; verify artifact total < 16 MB after run."
            )


# Phase 0: Data Preparation (untimed) — in-memory decode pipeline
# ---------------------------------------------------------------------------
def phase0_data_prep() -> None:
    import multiprocessing as mp
    import numpy as np

    banner("PHASE 0: DATA PREPARATION (untimed)")
    t0 = time.time()

    sys.path.insert(0, str(BESE_DIR / "tokenizer"))

    # Pre-spawn encode pool NOW while the process is still tiny.
    # CRITICAL: Never create a Pool after loading large data — fork() copies the
    # parent's page table for every worker. At 100 GB RSS that's ~200 MB of PTEs
    # × N workers = minutes of OS work with CPUs sitting idle.
    ENCODE_WORKERS = min(mp.cpu_count(), 224)
    ENCODE_CHUNK = 5000
    bese_tok_root = str(BESE_DIR / "tokenizer")
    tok_path = str(BPE_OUTPUT)
    log(f"  Pre-spawning {ENCODE_WORKERS} encode workers (process is still small)...")
    encode_pool = mp.Pool(ENCODE_WORKERS)

    # ------------------------------------------------------------------
    # Step 0.1 — Parallel decode SP shards → text strings in memory
    # ------------------------------------------------------------------
    banner("Step 0.1: Decode SP shards → in-memory text")

    import pickle as _pickle
    if DECODED_CACHE.exists():
        log(f"  Loading decoded docs from cache: {DECODED_CACHE}")
        with open(DECODED_CACHE, "rb") as _f:
            scored_train, val_docs = _pickle.load(_f)
        log(f"  Loaded {len(scored_train):,} train docs, {len(val_docs):,} val docs from cache")
    else:
        if not SP_MODEL.exists():
            raise FileNotFoundError(f"SentencePiece model not found: {SP_MODEL}")

        train_shard_files = sorted(SP_SHARD_DIR.glob("fineweb_train_*.bin"))
        val_shard_files = sorted(SP_SHARD_DIR.glob("fineweb_val_*.bin"))
        if not train_shard_files:
            raise FileNotFoundError(f"No SP train shards in {SP_SHARD_DIR}")
        log(f"  Found {len(train_shard_files)} train + {len(val_shard_files)} val SP shards")

        sp_model_path = str(SP_MODEL)
        num_workers = min(mp.cpu_count(), len(train_shard_files))
        log(f"  Using {num_workers} workers for parallel decode")

        scored_train = []
        train_args = [(f, sp_model_path, 50) for f in train_shard_files]
        with mp.Pool(num_workers) as pool:
            for name, n, scored_docs in pool.imap(_decode_shard, train_args):
                scored_train.extend(scored_docs)
                if len(scored_train) % 500_000 < len(scored_docs):
                    log(f"    {name}: {n:,} tokens → {len(scored_docs):,} docs (total: {len(scored_train):,})")

        val_docs = []
        val_args = [(f, sp_model_path, 50) for f in val_shard_files]
        with mp.Pool(min(num_workers, max(len(val_shard_files), 1))) as pool:
            for name, n, scored_docs in pool.imap(_decode_shard, val_args):
                val_docs.extend(doc for _, doc in scored_docs)

        elapsed_decode = time.time() - t0
        log(f"  Decoded + filtered + scored: {len(scored_train):,} train docs, {len(val_docs):,} val docs")
        log(f"  Decode+filter+score time: {elapsed_decode:.1f}s ({elapsed_decode / 60:.1f} min)")

        log(f"  Saving decoded docs cache to {DECODED_CACHE} ...")
        DECODED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(DECODED_CACHE, "wb") as _f:
            _pickle.dump((scored_train, val_docs), _f, protocol=_pickle.HIGHEST_PROTOCOL)
        log(f"  Cache saved ({DECODED_CACHE.stat().st_size / 1e9:.1f} GB)")

    # ------------------------------------------------------------------
    # Step 0.3 — Train BESE BPE (1024 merges) directly on text list
    # ------------------------------------------------------------------
    if BPE_OUTPUT.exists():
        log(f"  Step 0.3: BPE tokenizer already exists at {BPE_OUTPUT}, skipping")
        from bese_fast_bpe import FastBESEBPETokenizer
        tok = FastBESEBPETokenizer.load(str(BPE_OUTPUT))
    else:
        banner("Step 0.3: Train BESE BPE (1024 merges)")
        from bese_fast_bpe import train_bpe_merges_fast, FastBESEBPETokenizer

        # Free encode workers before BPE — the linked list needs ~40-60 GB and
        # 224 pre-spawned workers occupy ~16 GB.  We recreate the pool after BPE.
        log("  Releasing encode pool to free RAM for BPE linked list...")
        encode_pool.terminate()
        encode_pool.join()
        del encode_pool
        import gc; gc.collect()

        t_bpe = time.time()
        # Extract just the text from scored_train for BPE training (100K docs for deeper merges)
        bpe_sample = [doc for _, doc in scored_train[:100000]]

        # Prefer HF tokenizers (Rust, ~100x faster).  Fall back to pure Python
        # if the library is missing — pip install tokenizers to enable.
        try:
            from bese_fast_bpe import train_bpe_merges_hf
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "tokenizers", "-q", "--break-system-packages"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            from bese_fast_bpe import train_bpe_merges_hf
            merges = train_bpe_merges_hf(bpe_sample, num_merges=1024, verbose=True)
        except Exception as _hf_err:
            log(f"  HF tokenizers unavailable ({_hf_err}), falling back to pure-Python BPE")
            merges = train_bpe_merges_fast(bpe_sample, num_merges=1024, verbose=True)
        del bpe_sample
        tok = FastBESEBPETokenizer(merges=merges)
        log(f"  Vocab size: {tok.vocab_size}")

        BPE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        tok.save(BPE_OUTPUT)
        log(f"  Saved tokenizer to {BPE_OUTPUT} ({time.time() - t_bpe:.1f}s)")

        # Recreate encode workers now that BPE is done.
        # Workers fork after the corpus is loaded — PTEs are ~120 MB each at 60 GB RSS,
        # but COW means only written pages are physically copied (encode workers only write
        # their own output arrays, not the corpus), so the overhead is acceptable.
        log(f"  Re-spawning {ENCODE_WORKERS} encode workers...")
        encode_pool = mp.Pool(ENCODE_WORKERS)

    # ------------------------------------------------------------------
    # Step 0.4 — Curriculum sort (scores pre-computed in workers)
    # Scoring was moved into _decode_shard so it runs on 80 CPUs.
    # Only the final sort (Python timsort on floats) is single-threaded.
    # ------------------------------------------------------------------
    banner("Step 0.4: Curriculum sort (easy → hard)")
    t_sort = time.time()
    scored_train.sort()  # sort by pre-computed difficulty score
    train_docs = [doc for _, doc in scored_train]
    del scored_train
    log(f"  Sorted {len(train_docs):,} docs by difficulty ({time.time() - t_sort:.1f}s)")

    # ------------------------------------------------------------------
    # Step 0.5 — Export BESE shards from in-memory texts
    # ------------------------------------------------------------------
    # Consistency check: shards and tokenizer must both exist or both be absent.
    # A partial prior run could leave one without the other, causing silent mismatch.
    train_shards_exist = list(SHARD_DIR.glob("fineweb_train_*.bin"))
    if train_shards_exist and not BPE_OUTPUT.exists():
        log(f"  WARNING: Found {len(train_shards_exist)} stale shards but no tokenizer — deleting stale shards")
        for f in train_shards_exist:
            f.unlink()
        for f in SHARD_DIR.glob("fineweb_val_*.bin"):
            f.unlink()
        train_shards_exist = []
    if not train_shards_exist and BPE_OUTPUT.exists():
        log(f"  NOTE: Tokenizer exists but no shards found — will re-encode with existing tokenizer")
    if train_shards_exist:
        log(f"  Step 0.5: Found {len(train_shards_exist)} existing train shards in {SHARD_DIR}, skipping")
    else:
        banner("Step 0.5: Export BESE shards")
        t_export = time.time()
        SHARD_DIR.mkdir(parents=True, exist_ok=True)
        HEADER_INTS = 256
        SHARD_SIZE = 100_000_000

        def write_shard(path, tokens):
            header = np.zeros(HEADER_INTS, dtype="<i4")
            header[0] = 20240520
            header[1] = 1
            header[2] = int(tokens.shape[0])
            with open(path, "wb") as f:
                f.write(header.tobytes())
                f.write(tokens.astype("<u2").tobytes())
            return int(tokens.shape[0])

        # Encode + write val shard using pre-spawned pool
        log(f"  Encoding {len(val_docs):,} validation docs (parallel, {ENCODE_WORKERS} workers)...")
        val_enc_chunks = [val_docs[i:i + ENCODE_CHUNK] for i in range(0, len(val_docs), ENCODE_CHUNK)]
        val_enc_args = [(c, tok_path, bese_tok_root) for c in val_enc_chunks]
        val_arrays = encode_pool.map(_encode_chunk, val_enc_args)
        if val_arrays:
            val_tokens = np.concatenate(val_arrays)
            write_shard(SHARD_DIR / "fineweb_val_0.bin", val_tokens)
            log(f"    Val shard: {val_tokens.shape[0]:,} tokens")
            del val_tokens, val_arrays

        # Cap train docs to ~500M tokens
        max_train_tokens = 500_000_000
        est_docs = int(max_train_tokens / 400 * 1.25)
        if len(train_docs) > est_docs:
            log(f"  Capping train docs: {len(train_docs):,} → {est_docs:,}")
            train_docs = train_docs[:est_docs]

        # Encode train docs using pre-spawned pool, write shards as buffer fills
        log(f"  Encoding {len(train_docs):,} training docs (parallel, {ENCODE_WORKERS} workers)...")
        train_enc_chunks = [train_docs[i:i + ENCODE_CHUNK] for i in range(0, len(train_docs), ENCODE_CHUNK)]
        train_enc_args = [(c, tok_path, bese_tok_root) for c in train_enc_chunks]
        shard_idx = 0
        buffer = []
        buffer_tokens = 0
        for chunk_idx, arr in enumerate(encode_pool.imap(_encode_chunk, train_enc_args)):
            buffer.append(arr)
            buffer_tokens += len(arr)
            if (chunk_idx + 1) % 20 == 0:
                log(f"    Encoded {(chunk_idx+1)*ENCODE_CHUNK:,}/{len(train_docs):,} docs, {buffer_tokens:,} tokens buffered")
            while buffer_tokens >= SHARD_SIZE:
                combined = np.concatenate(buffer)
                shard_tokens = combined[:SHARD_SIZE]
                remainder = combined[SHARD_SIZE:]
                path = SHARD_DIR / f"fineweb_train_{shard_idx:06d}.bin"
                n = write_shard(path, shard_tokens)
                log(f"    Train shard {shard_idx}: {n:,} tokens")
                shard_idx += 1
                buffer = [remainder] if len(remainder) > 0 else []
                buffer_tokens = len(remainder)
        if buffer and buffer_tokens > 0:
            shard_tokens = np.concatenate(buffer)
            path = SHARD_DIR / f"fineweb_train_{shard_idx:06d}.bin"
            n = write_shard(path, shard_tokens)
            log(f"    Train shard {shard_idx}: {n:,} tokens")
            shard_idx += 1

        log(f"  Exported {shard_idx} train shards + 1 val shard ({time.time() - t_export:.1f}s)")

    # Shut down encode pool
    encode_pool.close()
    encode_pool.join()

    # Free memory
    del train_docs, val_docs

    # ------------------------------------------------------------------
    # Step 0.6 — Build n-gram table
    # ------------------------------------------------------------------
    _build_ngram_table()

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
    env["RUN_ID"] = "bese_v7"
    env["BESE_TOKENIZER_ROOT"] = str(BESE_DIR / "tokenizer")
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

    output = run_cmd(cmd, env=env, cwd=BESE_DIR, label="torchrun", timeout=900)
    return output


# ---------------------------------------------------------------------------
# Phase 2: Evaluation (timed, with n-gram tilt)
# ---------------------------------------------------------------------------
def phase2_eval(train_output: str) -> dict:
    """Extract eval metrics from the training output.

    In the current pipeline, eval runs inline at the end of training.
    If a separate eval pass is needed in the future, add it here.
    """
    banner("PHASE 2: EVALUATION (n-gram tilt)")

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
    int6 = metrics.get("int6_bpb", "N/A")
    size_bytes = metrics.get("size_bytes") or metrics.get("model_bytes")
    size_str = f"{size_bytes / 1_000_000:.2f} MB" if size_bytes else "N/A"
    model_bytes = metrics.get("model_bytes")
    model_str = f"{model_bytes / 1_000_000:.2f} MB" if model_bytes else "N/A"

    ttt = metrics.get("ttt_bpb", "N/A")

    log(f"  best_bpb:    {best}")
    log(f"  ttt_bpb:     {ttt}")
    log(f"  sliding_bpb: {sliding}")
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
    log("\n  v7 config:")
    log(f"    vocab_size={TRAIN_ENV['VOCAB_SIZE']}  layers={TRAIN_ENV['NUM_LAYERS']}"
        f"  dim={TRAIN_ENV['MODEL_DIM']}  mlp_mult={TRAIN_ENV['MLP_MULT']}")
    log(f"    depth_recurrence: layers {TRAIN_ENV['DEPTH_RECURRENCE_START']}-{TRAIN_ENV['DEPTH_RECURRENCE_END']}"
        f" x{TRAIN_ENV['DEPTH_RECURRENCE_LOOPS']} loops"
        f" (active after {float(TRAIN_ENV['DEPTH_RECURRENCE_ACTIVATION_FRAC']) * 100:.0f}% of training)")
    log(f"    parallel_residual: start={TRAIN_ENV['PARALLEL_RESIDUAL_START']}")
    log(f"    ttt: enabled={TRAIN_ENV.get('TTT_ENABLED', '0')}"
        f"  epochs={TRAIN_ENV.get('TTT_EPOCHS', 'N/A')}"
        f"  chunk_size={TRAIN_ENV.get('TTT_CHUNK_SIZE', 'N/A')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="BESE v7: Full Parameter Golf submission pipeline (8xH100)"
    )
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip Phase 0 (data prep) if shards already exist",
    )
    parser.add_argument(
        "--skip-shards",
        action="store_true",
        help="Skip shard prep (steps 0.1-0.5) but rebuild ngram table",
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

    # Ensure persistent directories exist on network volume
    for d in [NET_VOL / "tokenizers", NET_VOL / "artifacts", NET_VOL / "logs",
              NET_VOL / "checkpoints" / "v7"]:
        d.mkdir(parents=True, exist_ok=True)

    _open_log()
    t_start = time.time()

    banner("BESE v7 Pipeline")
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
        # Always rebuild ngram table even under --skip-prep: v6 uses a new path
        _build_ngram_table()
    elif args.skip_shards:
        log("\n  --skip-shards: Skipping shard prep (steps 0.1-0.5), rebuilding ngram table")
        # Verify shards exist from a prior run (e.g. v5)
        if not BPE_OUTPUT.exists():
            raise FileNotFoundError(
                f"--skip-shards requires tokenizer at {BPE_OUTPUT}. "
                "Run without --skip-shards first."
            )
        train_shards = list(SHARD_DIR.glob("fineweb_train_*.bin"))
        if not train_shards:
            raise FileNotFoundError(
                f"--skip-shards requires training shards in {SHARD_DIR}. "
                "Run without --skip-shards first."
            )
        log(f"  Found tokenizer: {BPE_OUTPUT}")
        log(f"  Found {len(train_shards)} training shards in {SHARD_DIR}")
        _build_ngram_table()
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

    # Save artifact to network volume
    import shutil
    artifact_src = BESE_DIR / "final_model.int6.ptz"
    artifact_dst = NET_VOL / "checkpoints" / "v7" / "final_model.int6.ptz"
    if artifact_src.exists():
        shutil.copy2(artifact_src, artifact_dst)
        log(f"  Saved artifact to {artifact_dst}")

    # Summary
    total_elapsed = time.time() - t_start
    print_summary(metrics, total_elapsed)

    _close_log()


if __name__ == "__main__":
    main()
