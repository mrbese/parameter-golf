#!/usr/bin/env python3
"""
Standalone artifact reassembler for BESE v7.

Rebuilds the submission artifact with a smaller n-gram table (max-n=2 instead of 3):
  1. Rebuilds n-gram table at /workspace/artifacts/ngram_table_v7.bin with --max-n 2
  2. Loads /workspace/bese/final_model.pt (123 MB unquantized checkpoint)
  3. Runs INT6 quantization + LZMA9 compression (all CPU, no GPU needed)
  4. Bundles the new smaller n-gram table into the artifact
  5. Saves final_model.int6.ptz + copies to /workspace/checkpoints/v7/

Usage (on the RunPod pod):
  cd /workspace/bese && python scripts/reassemble_v7.py

Target: artifact < 16,000,000 bytes
"""

from __future__ import annotations

import io
import lzma
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BESE_DIR = Path(os.environ.get("BESE_DIR", "/workspace/bese"))
NET_VOL = Path("/workspace")

NGRAM_TABLE = NET_VOL / "artifacts" / "ngram_table_v7.bin"
MODEL_PT = BESE_DIR / "final_model.pt"
ARTIFACT_OUT = BESE_DIR / "final_model.int6.ptz"
ARTIFACT_BACKUP = NET_VOL / "checkpoints" / "v7" / "final_model.int6.ptz"

NUM_LAYERS = 12
SIZE_LIMIT = 16_000_000  # 16 MB hard limit


# ---------------------------------------------------------------------------
# Step 1: Rebuild n-gram table with max-n=2
# ---------------------------------------------------------------------------
def rebuild_ngram(max_n: int = 2) -> int:
    shard_dir = NET_VOL / "bese_shards_v7"
    shards = sorted(shard_dir.glob("fineweb_train_*.bin"))
    if not shards:
        raise FileNotFoundError(f"No training shards found in {shard_dir}")
    first_shard = shards[0]
    print(f"\nRebuilding n-gram table (max-n={max_n}) from {first_shard.name}...")
    if NGRAM_TABLE.exists():
        NGRAM_TABLE.unlink()
        print("  Removed old n-gram table")
    t0 = time.time()
    subprocess.check_call(
        [
            sys.executable,
            str(BESE_DIR / "scripts" / "build_ngram_table.py"),
            "--shard", str(first_shard),
            "--output", str(NGRAM_TABLE),
            "--max-n", str(max_n),
            "--top-k", "1",
        ],
        cwd=str(BESE_DIR),
    )
    elapsed = time.time() - t0
    size = NGRAM_TABLE.stat().st_size
    print(f"  Done in {elapsed:.1f}s: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")
    return size


# ---------------------------------------------------------------------------
# Quantization helpers (copied from integration/train_gpt_bese.py)
# ---------------------------------------------------------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
        "q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,"
        "ve_shared.scale,attn_gate,vr_lambda",
    ).split(",") if p
)

_INT8_CLIP_Q = 99.99984 / 100.0
_INT8_SCALE_DTYPE = torch.float16


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), _INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=_INT8_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), _INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


def quantize_int6_per_row(
    t: Tensor,
    clip_range: int = 31,
    clip_percentiles: list[float] | None = None,
) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if clip_percentiles is None:
        clip_percentiles = [0.9990, 0.9995, 0.9999, 0.99999, 1.0]
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float("inf")
        for pct in clip_percentiles:
            row_clip = (
                torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
            )
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(
                torch.round(t32 / s.float()[:, None]), -clip_range, clip_range
            ).to(torch.int8)
            err = (t32 - q.float() * s.float()[:, None]).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale


def _unbank_state_dict(sd: dict, num_layers: int) -> dict:
    """Convert 3D bank tensors into individual 2D tensors with standard names."""
    out: dict = {}
    n = num_layers
    for name, tensor in sd.items():
        if name == "qo_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_q.weight"] = tensor[i]
                out[f"blocks.{i}.attn.proj.weight"] = tensor[n + i]
        elif name == "kv_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_k.weight"] = tensor[i]
                out[f"blocks.{i}.attn.c_v.weight"] = tensor[n + i]
        elif name == "mlp_up_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.fc.weight"] = tensor[i]
        elif name == "mlp_down_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.proj.weight"] = tensor[i]
        else:
            out[name] = tensor
    return out


def mixed_quantize_int6(state_dict: dict, int6_cats: set[str]) -> tuple[dict, dict]:
    result: dict = {}
    meta: dict = {}
    total = len(state_dict)
    for idx, (name, tensor) in enumerate(state_dict.items(), 1):
        if idx % 20 == 0 or idx == total:
            print(f"  Quantizing param {idx}/{total}: {name[:60]}", flush=True)
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            if cat == "mlp":
                q, s = quantize_int6_per_row(
                    t, clip_range=31, clip_percentiles=[0.9995, 0.9999, 1.0]
                )
            else:  # attn
                q, s = quantize_int6_per_row(
                    t, clip_range=31, clip_percentiles=[0.999, 0.9995, 0.9999, 0.99999, 1.0]
                )
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta


# ---------------------------------------------------------------------------
# Step 2+3: Load model, quantize, bundle, compress, save
# ---------------------------------------------------------------------------
def reassemble_artifact() -> int:
    if not MODEL_PT.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PT}")

    print(f"\nLoading {MODEL_PT} ...")
    t0 = time.time()
    sd_full = torch.load(str(MODEL_PT), map_location="cpu", weights_only=False)
    print(f"  Loaded in {time.time() - t0:.1f}s — {len(sd_full)} tensors")

    print("\nUnbanking state dict ...")
    unbanked_sd = _unbank_state_dict(sd_full, NUM_LAYERS)
    print(f"  {len(unbanked_sd)} unbanked tensors")

    print("\nQuantizing INT6 (CPU) ...")
    t0 = time.time()
    quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, {"mlp", "attn"})
    print(f"  Quantized in {time.time() - t0:.1f}s")

    save_dict: dict = {"w": quant_result, "m": quant_meta}

    # Bundle n-gram table
    if NGRAM_TABLE.exists():
        with open(NGRAM_TABLE, "rb") as nf:
            save_dict["ngram"] = nf.read()
        print(f"  Bundled n-gram: {len(save_dict['ngram']):,} bytes")
    else:
        print("  WARNING: No n-gram table found — not bundling!")

    print("\nSerializing + LZMA9 compressing ...")
    t0 = time.time()
    buf = io.BytesIO()
    torch.save(save_dict, buf)
    raw_bytes = buf.getvalue()
    print(f"  Raw serialized: {len(raw_bytes):,} bytes ({len(raw_bytes) / 1_000_000:.2f} MB)")
    compressed = lzma.compress(raw_bytes, preset=9)
    elapsed = time.time() - t0
    print(
        f"  Compressed in {elapsed:.1f}s: {len(compressed):,} bytes "
        f"({len(compressed) / 1_000_000:.2f} MB)  ratio={len(raw_bytes)/len(compressed):.1f}x"
    )

    with open(ARTIFACT_OUT, "wb") as f:
        f.write(compressed)
    print(f"\n  Saved to {ARTIFACT_OUT}")

    # Copy to network volume backup
    import shutil
    ARTIFACT_BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ARTIFACT_OUT, ARTIFACT_BACKUP)
    print(f"  Backed up to {ARTIFACT_BACKUP}")

    return len(compressed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 68)
    print("  BESE v7 Artifact Reassembler (max-n=2 n-gram, no retraining)")
    print("=" * 68)

    t_start = time.time()

    # Step 1: Rebuild n-gram table
    ngram_size = rebuild_ngram(max_n=2)

    # Steps 2+3: Quantize + compress + bundle
    artifact_size = reassemble_artifact()

    total_elapsed = time.time() - t_start

    print("\n" + "=" * 68)
    print("  REASSEMBLY SUMMARY")
    print("=" * 68)
    print(f"  N-gram table (max-n=2): {ngram_size:,} bytes ({ngram_size / 1024:.1f} KB)")
    print(f"  Artifact size:          {artifact_size:,} bytes ({artifact_size / 1_000_000:.3f} MB)")
    print(f"  Size limit:             {SIZE_LIMIT:,} bytes ({SIZE_LIMIT / 1_000_000:.0f} MB)")

    if artifact_size < SIZE_LIMIT:
        headroom = SIZE_LIMIT - artifact_size
        print(f"\n  ✓ SIZE CHECK: PASS")
        print(f"    Headroom: {headroom:,} bytes ({headroom / 1024:.1f} KB)")
    else:
        over = artifact_size - SIZE_LIMIT
        print(f"\n  ✗ SIZE CHECK: FAIL — {over:,} bytes OVER limit!")
        print("    Consider: --max-n 1 or reducing top-k")

    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print("=" * 68)


if __name__ == "__main__":
    main()
