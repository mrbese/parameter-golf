#!/usr/bin/env python3
"""
RunPod v3: End-to-end BESE submission pipeline for Parameter Golf.

Targets 8xH100 pod. Decodes SP shards -> trains BESE BPE -> exports BESE shards
-> trains with winning architecture (param banks, Muon, INT6, TTT) -> evaluates.

Usage (on the RunPod pod):
  cd /workspace && git clone https://github.com/mrbese/parameter-golf-bese.git bese
  cd /workspace && python3 bese/scripts/runpod_v3.py

  # BESE only (skip baseline):
  python3 bese/scripts/runpod_v3.py --bese-only

  # Custom config:
  python3 bese/scripts/runpod_v3.py --num-merges 4000 --num-layers 12 --ttt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Paths (override with env vars if needed)
PG_DIR = Path(os.environ.get("PG_DIR", "/workspace/parameter-golf"))
BESE_DIR = Path(os.environ.get("BESE_DIR", "/workspace/bese"))
SP_MODEL = PG_DIR / "data/tokenizers/fineweb_1024_bpe.model"
SHARD_DIR = PG_DIR / "data/datasets/fineweb10B_sp1024"
TOK_DIR = BESE_DIR / "tokenizers"
BESE_SHARD_DIR = Path("/workspace/bese_shards_v3")

sys.path.insert(0, str(BESE_DIR / "tokenizer"))


def step(msg):
    print(f"\n{'='*70}\n  {msg}\n{'='*70}", flush=True)


def _decode_shard(args):
    """Decode a single shard — runs in a worker process."""
    shard_file, sp_model_path, min_len = args
    import sentencepiece as spm
    import numpy as np
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    bos = sp.bos_id()
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(shard_file, dtype="<i4", count=256)
    n = int(header[2])
    tokens = np.fromfile(shard_file, dtype="<u2", count=n, offset=header_bytes)
    docs = []
    current = []
    for t in tokens:
        if t == bos:
            if current:
                text = sp.decode(current)
                if len(text.strip()) > min_len:
                    docs.append(text)
            current = []
        else:
            current.append(int(t))
    if current:
        text = sp.decode(current)
        if len(text.strip()) > min_len:
            docs.append(text)
    return str(shard_file.name), n, docs


def decode_all_shards(max_docs=None, num_workers=None):
    """Decode text from ALL SP binary shards in parallel."""
    import multiprocessing as mp
    step("STEP 1: Decoding documents from SP shards (parallel)")

    shard_files = sorted(SHARD_DIR.glob("fineweb_train_*.bin"))
    val_files = sorted(SHARD_DIR.glob("fineweb_val_*.bin"))
    sp_model_path = str(SP_MODEL)

    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(shard_files))
    print(f"  Found {len(shard_files)} training shards, using {num_workers} workers")

    train_args = [(f, sp_model_path, 50) for f in shard_files]
    all_docs = []
    with mp.Pool(num_workers) as pool:
        for name, n, docs in pool.imap(_decode_shard, train_args):
            all_docs.extend(docs)
            print(f"  {name}: {n:,} tokens -> {len(docs):,} docs (total: {len(all_docs):,})", flush=True)
            if max_docs and len(all_docs) >= max_docs:
                pool.terminate()
                all_docs = all_docs[:max_docs]
                print(f"  Reached max_docs={max_docs}, stopping")
                break

    val_args = [(f, sp_model_path, 50) for f in val_files]
    val_docs = []
    with mp.Pool(min(num_workers, len(val_files))) as pool:
        for name, n, docs in pool.imap(_decode_shard, val_args):
            val_docs.extend(docs)
            print(f"  {name}: {n:,} tokens -> {len(docs):,} val docs", flush=True)

    print(f"\n  Total: {len(all_docs):,} train docs, {len(val_docs):,} val docs")
    return all_docs, val_docs


def train_bpe(texts, num_merges=4000):
    """Train BESE BPE using fast indexed approach."""
    step(f"STEP 2: Training BESE BPE ({num_merges} merges on {len(texts)} docs)")
    from bese_fast_bpe import train_bpe_merges_fast, FastBESEBPETokenizer

    t0 = time.time()
    merges = train_bpe_merges_fast(texts, num_merges=num_merges, verbose=True)
    tok = FastBESEBPETokenizer(merges=merges)
    elapsed = time.time() - t0
    print(f"\n  BPE training took {elapsed:.1f}s")
    print(f"  Vocab size: {tok.vocab_size}")

    # Byte check
    bpt = tok.get_bytes_per_token_lut()
    ok = fail = 0
    for text in texts[:200]:
        enc = tok.encode(text)
        tb = int(sum(bpt[t] for t in enc))
        ub = len(text.encode("utf-8"))
        if tb == ub:
            ok += 1
        else:
            fail += 1
    print(f"  Byte check: {ok} OK, {fail} FAIL (out of {ok+fail})")
    if fail > 0:
        print("  WARNING: Byte check failures detected!")

    # Compression ratio
    total_tokens = sum(len(tok.encode(t)) for t in texts[:1000])
    total_bytes = sum(len(t.encode("utf-8")) for t in texts[:1000])
    print(f"  Compression: {total_tokens/total_bytes:.3f} tokens/byte (on 1K docs)")

    # Save tokenizer
    TOK_DIR.mkdir(parents=True, exist_ok=True)
    tok_path = TOK_DIR / f"bese_bpe_{num_merges}.json"
    tok.save(tok_path)
    print(f"  Saved {tok_path}")

    return tok, tok_path


def export_bese_shards(tok, train_texts, val_texts):
    """Export BESE+BPE binary shards matching upstream format."""
    step(f"STEP 3: Exporting BESE shards ({len(train_texts)} train, {len(val_texts)} val docs)")

    BESE_SHARD_DIR.mkdir(parents=True, exist_ok=True)
    HEADER_INTS = 256
    SHARD_SIZE = 100_000_000  # ~100M tokens per shard

    def write_shard(path, tokens):
        header = np.zeros(HEADER_INTS, dtype="<i4")
        header[0] = 20240520
        header[1] = 1
        header[2] = int(tokens.shape[0])
        with open(path, "wb") as f:
            f.write(header.tobytes())
            f.write(tokens.astype("<u2").tobytes())
        return int(tokens.shape[0])

    # Encode and write validation shard
    print("  Encoding validation docs...")
    t0 = time.time()
    val_chunks = []
    for i, text in enumerate(val_texts):
        enc = tok.encode(text)
        val_chunks.append(enc.astype(np.uint16))
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{len(val_texts)} val docs encoded...", flush=True)
    val_tokens = np.concatenate(val_chunks) if val_chunks else np.array([], dtype=np.uint16)
    val_path = BESE_SHARD_DIR / "fineweb_val_000000.bin"
    n = write_shard(val_path, val_tokens)
    print(f"  Val shard: {n:,} tokens ({time.time()-t0:.1f}s)")

    # Encode and write training shards
    print("  Encoding training docs...")
    t0 = time.time()
    train_chunks = []
    total_train_tokens = 0
    shard_idx = 0

    for i, text in enumerate(train_texts):
        enc = tok.encode(text)
        train_chunks.append(enc.astype(np.uint16))
        total_train_tokens += len(enc)

        if total_train_tokens >= SHARD_SIZE:
            shard_tokens = np.concatenate(train_chunks)
            shard_path = BESE_SHARD_DIR / f"fineweb_train_{shard_idx:06d}.bin"
            n = write_shard(shard_path, shard_tokens)
            print(f"  Train shard {shard_idx}: {n:,} tokens")
            train_chunks = []
            total_train_tokens = 0
            shard_idx += 1

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(train_texts) - i - 1) / rate
            print(f"    {i+1}/{len(train_texts)} train docs ({rate:.0f} docs/s, ~{remaining:.0f}s remaining)", flush=True)

    if train_chunks:
        shard_tokens = np.concatenate(train_chunks)
        shard_path = BESE_SHARD_DIR / f"fineweb_train_{shard_idx:06d}.bin"
        n = write_shard(shard_path, shard_tokens)
        print(f"  Train shard {shard_idx}: {n:,} tokens")

    elapsed = time.time() - t0
    total_shards = shard_idx + (1 if train_chunks else 0)
    print(f"\n  Export complete: {total_shards} train shards + 1 val shard ({elapsed:.1f}s)")
    print(f"  Shards written to {BESE_SHARD_DIR}")


def run_training(name, data_path, tokenizer_path, vocab_size, train_script,
                 num_layers=11, model_dim=512, mlp_mult=3, num_gpus=8,
                 extra_env=None):
    """Run a training job and return the output."""
    step(f"TRAINING: {name} ({num_layers}L/{model_dim}d/{mlp_mult}x MLP, {num_gpus} GPUs)")

    env = os.environ.copy()
    env.update({
        "RUN_ID": name,
        "DATA_PATH": str(data_path),
        "TOKENIZER_PATH": str(tokenizer_path),
        "VOCAB_SIZE": str(vocab_size),
        "NUM_LAYERS": str(num_layers),
        "MODEL_DIM": str(model_dim),
        "MLP_MULT": str(mlp_mult),
        "VAL_LOSS_EVERY": "500",
        "TRAIN_LOG_EVERY": "100",
        "MAX_WALLCLOCK_SECONDS": "600",
    })
    if extra_env:
        env.update(extra_env)

    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={num_gpus}",
        str(train_script),
    ]
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Key env: VOCAB_SIZE={vocab_size} NUM_LAYERS={num_layers} "
          f"MODEL_DIM={model_dim} MLP_MULT={mlp_mult}")

    t0 = time.time()
    output_lines = []
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)
    proc.wait()
    elapsed = time.time() - t0

    output = "".join(output_lines)
    if proc.returncode != 0:
        print(f"\n  ERROR: Process exited with code {proc.returncode}")
        raise RuntimeError(f"Training '{name}' failed with exit code {proc.returncode}")
    print(f"\n  Wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return output


def extract_metrics(output):
    """Extract metrics from training output. Prefers sliding window / TTT BPB."""
    metrics = {}
    for line in output.strip().split("\n"):
        # Sliding window (most accurate non-TTT score)
        if "final_int8_zlib_roundtrip_exact" in line and "val_bpb:" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["sliding_bpb"] = float(part.split(":")[1])
                if part.startswith("val_loss:"):
                    metrics["sliding_loss"] = float(part.split(":")[1])
        # INT6 roundtrip (quantized model quality)
        if "final_int6_roundtrip_exact" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["int6_bpb"] = float(part.split(":")[1])
        # TTT (best possible score)
        if "legal_ttt_exact" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["ttt_bpb"] = float(part.split(":")[1])
        # SLOT
        if "slot_lbfgs_exact" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["slot_bpb"] = float(part.split(":")[1])
        # Submission size
        if "Total submission size" in line and "bytes" in line:
            m = re.search(r'(\d+)\s*bytes', line)
            if m:
                metrics["size_bytes"] = int(m.group(1))
        # Serialized model size
        if "Serialized model int6+lzma:" in line:
            m = re.search(r'(\d+)\s*bytes', line)
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


def main():
    parser = argparse.ArgumentParser(description="BESE v3: End-to-end Parameter Golf submission")
    parser.add_argument("--bese-only", action="store_true", help="Skip baseline, run BESE only")
    parser.add_argument("--baseline-only", action="store_true", help="Skip BESE, run baseline only")
    parser.add_argument("--num-merges", type=int, default=4000, help="Number of BPE merges")
    parser.add_argument("--max-docs", type=int, default=None, help="Max docs to decode (None=all)")
    parser.add_argument("--num-layers", type=int, default=11, help="Transformer layers")
    parser.add_argument("--model-dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--mlp-mult", type=int, default=3, help="MLP multiplier")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--ttt", action="store_true", help="Enable TTT eval")
    parser.add_argument("--slot", action="store_true", help="Enable SLOT eval")
    parser.add_argument("--skip-decode", action="store_true", help="Skip decode if shards exist")
    args = parser.parse_args()

    t_start = time.time()

    # Detect number of GPUs
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        detected_gpus = len([l for l in result.stdout.strip().split("\n") if "GPU" in l])
        if detected_gpus > 0 and args.num_gpus != detected_gpus:
            print(f"  Detected {detected_gpus} GPUs, using all of them")
            args.num_gpus = detected_gpus
    except Exception:
        pass

    # Step 1: Decode all shards
    tok_path = None
    if not args.baseline_only:
        if args.skip_decode and BESE_SHARD_DIR.exists() and list(BESE_SHARD_DIR.glob("*.bin")):
            print("  Skipping decode, using existing BESE shards")
        else:
            train_texts, val_texts = decode_all_shards(max_docs=args.max_docs)

            # Step 2: Train BPE (use subset for BPE training)
            bpe_train_texts = train_texts[:50000]
            tok, tok_path = train_bpe(bpe_train_texts, num_merges=args.num_merges)

            # Step 3: Export shards
            export_bese_shards(tok, train_texts, val_texts)
            del train_texts, val_texts  # free memory

    # Step 4: Run training
    results = {}

    if not args.bese_only:
        # Baseline: upstream SP1024 with winning architecture
        baseline_out = run_training(
            name="baseline_sp1024",
            data_path=SHARD_DIR,
            tokenizer_path=SP_MODEL,
            vocab_size=1024,
            train_script=PG_DIR / "train_gpt.py",
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            mlp_mult=args.mlp_mult,
            num_gpus=args.num_gpus,
            extra_env={
                "TTT_ENABLED": "1" if args.ttt else "0",
                "EVAL_STRIDE": "64",
            },
        )
        results["baseline"] = extract_metrics(baseline_out)

    if not args.baseline_only:
        # Determine BESE tokenizer path
        if tok_path is None:
            candidates = sorted(TOK_DIR.glob("bese_bpe_*.json"))
            if not candidates:
                raise FileNotFoundError(f"No BESE tokenizer found in {TOK_DIR}. Run without --skip-decode first.")
            tok_path = candidates[-1]
        from bese_fast_bpe import FastBESEBPETokenizer
        tok = FastBESEBPETokenizer.load(str(tok_path))

        # BESE: winning architecture + BESE tokenizer
        bese_extra_env = {
            "BESE_TOKENIZER_ROOT": str(BESE_DIR / "tokenizer"),
            "EVAL_STRIDE": "64",
            "TTT_ENABLED": "1" if args.ttt else "0",
            "SLOT_ENABLED": "1" if args.slot else "0",
        }
        bese_out = run_training(
            name="bese_v3",
            data_path=BESE_SHARD_DIR,
            tokenizer_path=tok_path,
            vocab_size=tok.vocab_size,
            train_script=BESE_DIR / "integration" / "train_gpt_bese.py",
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            mlp_mult=args.mlp_mult,
            num_gpus=args.num_gpus,
            extra_env=bese_extra_env,
        )
        results["bese"] = extract_metrics(bese_out)

    # Step 5: Report
    step("RESULTS SUMMARY")
    for name, metrics in results.items():
        best = metrics.get("best_bpb", "N/A")
        sliding = metrics.get("sliding_bpb", "N/A")
        ttt = metrics.get("ttt_bpb", "N/A")
        slot = metrics.get("slot_bpb", "N/A")
        int6 = metrics.get("int6_bpb", "N/A")
        size = metrics.get("size_bytes")
        size_str = f"{size/1e6:.2f} MB" if size else "N/A"
        model_size = metrics.get("model_bytes")
        model_str = f"{model_size/1e6:.2f} MB" if model_size else "N/A"
        print(f"  {name}:")
        print(f"    best_bpb={best}  sliding={sliding}  ttt={ttt}  slot={slot}")
        print(f"    int6_bpb={int6}  model={model_str}  total={size_str}")

    if "baseline" in results and "bese" in results:
        b_bpb = results["baseline"].get("best_bpb")
        e_bpb = results["bese"].get("best_bpb")
        if b_bpb and e_bpb:
            diff = e_bpb - b_bpb
            print(f"\n  BESE vs Baseline: {diff:+.4f} BPB ({'BESE better' if diff < 0 else 'Baseline better'})")

    total = time.time() - t_start
    print(f"\n  Total wall time: {total/60:.1f} min")


if __name__ == "__main__":
    main()
