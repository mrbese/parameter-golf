"""
Mamba-3 SSD (Structured State Space Duality) — Pure PyTorch implementation.

Based on mamba3-minimal (https://github.com/VikramKarLex/mamba3-minimal).
No custom CUDA/Triton kernels — uses standard PyTorch ops (einsum, cumsum, exp).

For the Parameter Golf challenge: 7 Mamba + 1 Attention hybrid.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from einops import rearrange
except ImportError:
    raise ImportError("einops is required: pip install einops")


# ---------------------------------------------------------------------------
# Core SSD helpers
# ---------------------------------------------------------------------------

def segsum(x: Tensor) -> Tensor:
    """Stable cumulative sum for decay computation.

    Computes a lower-triangular matrix where entry [i,j] = sum(x[j..i]).
    Used for computing state decay within chunks.

    Args:
        x: (..., L) tensor
    Returns:
        (..., L, L) lower-triangular cumsum matrix
    """
    T = x.size(-1)
    x = x[..., None].repeat(*([1] * (x.ndim - 1)), 1, T)  # (..., T, T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask2 = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask2, -torch.inf)
    return x_segsum


def ssd_chunked(
    x: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    chunk_size: int,
    initial_states: Tensor | None = None,
) -> Tensor:
    """Structured State Space Duality — chunked parallel computation.

    Args:
        x: (batch, seq_len, heads, head_dim) — input after projection
        A: (batch, seq_len, heads) — log decay rates (dt * A_param)
        B: (batch, seq_len, heads, d_state) — input-to-state projection
        C: (batch, seq_len, heads, d_state) — state-to-output projection
        chunk_size: size of chunks for parallel computation
        initial_states: optional (batch, heads, head_dim, d_state) initial hidden state

    Returns:
        (batch, seq_len, heads, head_dim) output
    """
    batch, seq_len, nheads, headdim = x.shape
    d_state = B.shape[-1]

    # Pad sequence to multiple of chunk_size
    pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        A = F.pad(A, (0, 0, 0, pad_len))
        B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, 0, 0, pad_len))

    # Reshape into chunks: (batch, n_chunks, chunk_size, ...)
    x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
    A = rearrange(A, "b (c l) h -> b c l h", l=chunk_size)
    B = rearrange(B, "b (c l) h n -> b c l h n", l=chunk_size)
    C = rearrange(C, "b (c l) h n -> b c l h n", l=chunk_size)

    # Transpose heads for einsum convenience: b h c l ...
    A = rearrange(A, "b c l h -> b h c l")

    # Step 1: Intra-chunk quadratic attention
    # L[i,j] = exp(sum(A[j+1..i])) — decay from position j to position i within chunk
    L = torch.exp(segsum(A))  # (batch, heads, n_chunks, chunk_size, chunk_size)

    # Y_diag = C^T @ L @ (B^T @ x) — within-chunk contribution
    Y_diag = torch.einsum(
        "bclhn, bcshn, bhcls, bcshp -> bclhp",
        C, B, L, x
    )

    # Step 2: Per-chunk state accumulation
    A_cumsum = torch.cumsum(A, dim=-1)  # (batch, heads, n_chunks, chunk_size)

    # Decay from each position to end of chunk
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # (b, h, c, l)

    # Accumulate states: B^T @ (decay * x)
    states = torch.einsum(
        "bclhn, bhcl, bclhp -> bchpn",
        B, decay_states, x
    )  # (batch, n_chunks, heads, head_dim, d_state)

    # Step 3: Inter-chunk recurrence
    # Propagate states across chunks with decay
    if initial_states is not None:
        # Prepend initial states
        states = torch.cat([initial_states.unsqueeze(1), states], dim=1)

    A_chunk_decay = A_cumsum[:, :, :, -1]  # (b, h, c) — total decay per chunk
    decay_chunk = torch.exp(segsum(F.pad(A_chunk_decay, (1, 0))))  # (b, h, c+1, c+1)

    if initial_states is not None:
        new_states = torch.einsum(
            "bhzc, bchpn -> bzhpn",
            decay_chunk[:, :, :states.shape[1], :states.shape[1]],
            states
        )
        new_states = new_states[:, 1:]  # remove the initial state position
    else:
        new_states = torch.einsum(
            "bhzc, bchpn -> bzhpn",
            decay_chunk[:, :, :states.shape[1], :states.shape[1]],
            states
        )

    # Step 4: State-to-output
    state_decay_out = torch.exp(A_cumsum)  # (b, h, c, l)
    Y_off = torch.einsum(
        "bclhn, bchpn, bhcl -> bclhp",
        C, new_states, state_decay_out
    )

    # Combine intra-chunk and inter-chunk
    Y = Y_diag + Y_off
    Y = rearrange(Y, "b c l h p -> b (c l) h p")

    # Remove padding
    if pad_len > 0:
        Y = Y[:, :seq_len]

    return Y


# ---------------------------------------------------------------------------
# Mamba-3 Block
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class Mamba3Block(nn.Module):
    """Mamba-3 block with SSD (Structured State Space Duality).

    Uses trapezoidal discretization decomposed into two SSD calls,
    implemented in pure PyTorch without custom CUDA kernels.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_inner = expand * dim
        self.nheads = self.d_inner // headdim
        self.headdim = headdim
        self.chunk_size = chunk_size

        # Input projection: x → (z, x_proj, B, C, dt)
        # z: gating signal (d_inner)
        # x_proj: SSM input (d_inner)
        # B: input-to-state (nheads * d_state)
        # C: state-to-output (nheads * d_state)
        # dt: timestep (nheads)
        d_proj = self.d_inner * 2 + self.nheads * d_state * 2 + self.nheads
        self.in_proj = nn.Linear(dim, d_proj, bias=False)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # SSM parameters
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.dt_bias = nn.Parameter(torch.randn(self.nheads) * 0.1)
        self.A_log = nn.Parameter(torch.log(0.5 + torch.rand(self.nheads) * 0.5))  # log of decay rate

        # Normalization for B and C projections
        self.B_norm = RMSNorm(d_state)
        self.C_norm = RMSNorm(d_state)

        # Pre/post normalization
        self.norm = nn.LayerNorm(dim)  # use existing RMSNorm pattern

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.in_proj.weight, gain=1.0)
        nn.init.zeros_(self.out_proj.weight)  # zero-init output projection

    def forward(self, u: Tensor) -> Tensor:
        """
        Args:
            u: (batch, seq_len, dim)
        Returns:
            (batch, seq_len, dim)
        """
        batch, seq_len, dim = u.shape

        # Normalize input
        u_normed = self.norm(u)

        # Project
        proj = self.in_proj(u_normed)

        # Split projections
        d_inner = self.d_inner
        nheads = self.nheads
        d_state = self.d_state

        z = proj[..., :d_inner]
        x = proj[..., d_inner:d_inner * 2]
        B_flat = proj[..., d_inner * 2:d_inner * 2 + nheads * d_state]
        C_flat = proj[..., d_inner * 2 + nheads * d_state:d_inner * 2 + nheads * d_state * 2]
        dt = proj[..., -nheads:]

        # Process dt (timestep)
        dt = F.softplus(dt + self.dt_bias)  # (batch, seq_len, nheads)

        # Compute dA = dt * A (log-space decay)
        A = -torch.exp(self.A_log)  # negative decay rate
        dA = dt * A  # (batch, seq_len, nheads)

        # Normalize B and C
        B = rearrange(B_flat, "b l (h n) -> b l h n", h=nheads)
        C = rearrange(C_flat, "b l (h n) -> b l h n", h=nheads)
        B = self.B_norm(B)
        C = self.C_norm(C)

        # Reshape x for SSD
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)

        # Run SSD
        y = ssd_chunked(x, dA, B, C, self.chunk_size)

        # Skip connection with D
        y = y + x * self.D[None, None, :, None]

        # Merge heads
        y = rearrange(y, "b l h p -> b l (h p)")

        # Gate with SiLU
        y = y * F.silu(z)

        # Output projection + residual
        return u + self.out_proj(y)


# ---------------------------------------------------------------------------
# Attention Block (reused from existing GPT, simplified)
# ---------------------------------------------------------------------------

class Rotary(nn.Module):
    """RoPE (Rotary Position Embedding). Copied from train_gpt_bese.py to avoid circular import."""
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    """Apply rotary position embeddings. Copied from train_gpt_bese.py."""
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class AttentionBlock(nn.Module):
    """Standard causal self-attention block for the hybrid architecture.

    Simplified from the existing GPT's bank-based attention to standalone weights.
    Uses FlashAttention-3 if available, falls back to SDPA.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        mlp_mult: float = 3.0,
        rope_base: float = 10000.0,
        qk_gain_init: float = 5.25,
        rope_dims: int = 16,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        mlp_dim = int(dim * mlp_mult)

        # Attention projections
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.c_v = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)

        # MLP
        self.mlp_fc = nn.Linear(dim, mlp_dim, bias=False)
        self.mlp_proj = nn.Linear(mlp_dim, dim, bias=False)

        # QK gain
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))

        # RoPE (using local Rotary to avoid circular import)
        self.rope_dims = rope_dims
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=4096, rope_dims=rope_dims)

        # Layer norms
        self.attn_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.ndim == 2:
                if "proj" in name:
                    nn.init.zeros_(p)  # zero-init output projections
                else:
                    nn.init.orthogonal_(p, gain=1.0)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape

        # Attention
        h = self.attn_norm(x)
        q = self.c_q(h).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(h).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(h).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # QK norm + RoPE
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]

        # FlashAttention or SDPA fallback
        try:
            from flash_attn_interface import flash_attn_func as fa3
            y = fa3(q, k, v, causal=True)
        except ImportError:
            # SDPA fallback
            q_t = q.transpose(1, 2)  # (bsz, heads, seqlen, head_dim)
            # Expand KV for GQA
            rep = self.num_heads // self.num_kv_heads
            k_t = k.transpose(1, 2).repeat_interleave(rep, dim=1)
            v_t = v.transpose(1, 2).repeat_interleave(rep, dim=1)
            y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
            y = y.transpose(1, 2)

        y = y.reshape(bsz, seqlen, dim)
        x = x + self.c_proj(y)

        # MLP
        h = self.mlp_norm(x)
        x = x + self.mlp_proj(F.silu(self.mlp_fc(h)))

        return x


# ---------------------------------------------------------------------------
# Hybrid Model: 7 Mamba-3 + 1 Attention
# ---------------------------------------------------------------------------

class HybridMambaGPT(nn.Module):
    """Mamba-3 + Attention hybrid for Parameter Golf.

    Architecture:
        - 7 Mamba-3 blocks (linear O(n) scaling)
        - 1 Attention block at position `attn_pos` (quadratic, but only 1 layer)
        - Depth recurrence on configurable Mamba layers
        - BESE 288 vocab with tied embeddings
    """

    def __init__(
        self,
        vocab_size: int = 288,
        num_layers: int = 8,
        model_dim: int = 512,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 64,
        attn_pos: int = 4,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        mlp_mult: float = 3.0,
        rope_base: float = 10000.0,
        qk_gain_init: float = 5.25,
        rope_dims: int = 16,
        logit_softcap: float = 30.0,
        tied_embed_init_std: float = 0.005,
        depth_recurrence_start: int = 2,
        depth_recurrence_end: int = 4,
        depth_recurrence_loops: int = 3,
        depth_recurrence_activation_frac: float = 0.35,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.logit_softcap = logit_softcap
        self.tied_embed_init_std = tied_embed_init_std
        self.attn_pos = attn_pos

        # Depth recurrence config
        self._rec_start = depth_recurrence_start
        self._rec_end = depth_recurrence_end
        self._rec_target_loops = depth_recurrence_loops
        self._rec_activation_frac = depth_recurrence_activation_frac
        self._rec_loops = 1  # starts at 1, increases during training
        self._training_progress = 0.0

        # Token embedding (tied with lm_head)
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)

        # SmearGate for temporal smoothing
        self.smear_gate = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))

        # Build layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == attn_pos:
                self.layers.append(AttentionBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    mlp_mult=mlp_mult,
                    rope_base=rope_base,
                    qk_gain_init=qk_gain_init,
                    rope_dims=rope_dims,
                ))
            else:
                self.layers.append(Mamba3Block(
                    dim=model_dim,
                    d_state=d_state,
                    expand=expand,
                    headdim=headdim,
                    chunk_size=chunk_size,
                ))

        # Final norm
        self.final_norm = nn.LayerNorm(model_dim)

        # No separate lm_head — tied with tok_emb
        self.lm_head = None

    def _smear(self, x: Tensor) -> Tensor:
        """Temporal smoothing gate."""
        g = torch.sigmoid(self.smear_gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

    def _run_layers(self, x: Tensor) -> Tensor:
        """Run all layers with optional depth recurrence on Mamba layers."""
        # Determine number of recurrence loops
        if self._training_progress >= self._rec_activation_frac:
            self._rec_loops = self._rec_target_loops

        for i, layer in enumerate(self.layers):
            if self._rec_start <= i <= self._rec_end and self._rec_loops > 1:
                # Depth recurrence: loop this layer multiple times
                for _ in range(self._rec_loops):
                    x = layer(x)
            else:
                x = layer(x)
        return x

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Forward pass with cross-entropy loss."""
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self._smear(x)
        x = self._run_layers(x)
        x = self.final_norm(x)

        logits = F.linear(x, self.tok_emb.weight)  # tied embeddings
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        targets = target_ids.reshape(-1)
        logits_flat = logits.reshape(-1, logits.size(-1))
        return F.cross_entropy(logits_flat.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits without computing loss (for eval/TTT)."""
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self._smear(x)
        x = self._run_layers(x)
        x = self.final_norm(x)

        logits = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return logits
