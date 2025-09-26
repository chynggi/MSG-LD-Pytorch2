from functools import reduce
from typing import Callable, Literal, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from packaging import version
from torch import einsum, nn

try:
    from flash_attn import flash_attn_func, flash_attn_kvpacked_func
except ImportError:  # pragma: no cover - fallback path
    flash_attn_func = None
    flash_attn_kvpacked_func = None

try:
    import natten
except ImportError:  # pragma: no cover - optional dependency
    natten = None


class FourierFeatures(nn.Module):
    def __init__(self, in_features: int, out_features: int, std: float = 1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn(out_features // 2, in_features) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = 2 * math.pi * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def normalize(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    dim = list(range(1, x.ndim))
    n = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    alpha = np.sqrt(n.numel() / x.numel())
    return x / torch.add(eps, n, alpha=alpha)


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


def create_causal_mask(i: int, j: int, device: torch.device) -> torch.Tensor:
    return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)


def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        seq_start_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len, device = x.shape[1], x.device
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max positional embedding {self.max_seq_len}"
            )
        if pos is None:
            pos = torch.arange(seq_len, device=device)
        if seq_start_pos is not None:
            pos = (pos - seq_start_pos[..., None]).clamp(min=0)
        pos_emb = self.emb(pos) * self.scale
        return pos_emb


class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dimension must be divisible by 2")
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)
        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        seq_start_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len, device = x.shape[1], x.device
        if pos is None:
            pos = torch.arange(seq_len, device=device)
        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]
        emb = einsum("i, j -> i j", pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        use_xpos: bool = False,
        scale_base: float = 512,
        interpolation_factor: float = 1.0,
        base: float = 10000,
        base_rescale_factor: float = 1.0,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        if use_xpos:
            scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
            self.scale_base = scale_base
            self.register_buffer("scale", scale)
        else:
            self.register_buffer("scale", None)
        self.interpolation_factor = interpolation_factor

    def forward_from_seq_len(self, seq_len: int) -> Tuple[torch.Tensor, float]:
        device = self.inv_freq.device
        t = torch.arange(seq_len, device=device)
        return self.forward(t)

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, float]:
        device = self.inv_freq.device
        t = t.to(torch.float32) / self.interpolation_factor
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        if self.scale is None:
            return freqs, 1.0
        seq_len = t.shape[0]
        power = (torch.arange(seq_len, device=device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = torch.cat((scale, scale), dim=-1)
        return freqs, scale


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    t: torch.Tensor, freqs: torch.Tensor, scale: float = 1
) -> torch.Tensor:
    out_dtype = t.dtype
    dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    seq_len = t.shape[-2]
    freqs, t = freqs.to(dtype), t.to(dtype)
    freqs = freqs[-seq_len:, :]
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, "b n d -> b 1 n d")
    t, t_unrotated = t[..., : freqs.shape[-1]], t[..., freqs.shape[-1] :]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)
    return torch.cat((t, t_unrotated), dim=-1)


class LayerNorm(nn.Module):
    def __init__(self, dim: int, bias: bool = False, fix_scale: bool = False):
        super().__init__()
        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))
        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta)


class GLU(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: Callable,
        use_conv: bool = False,
        conv_kernel_size: int = 3,
    ):
        super().__init__()
        self.act = activation
        if use_conv:
            self.proj = nn.Conv1d(dim_in, dim_out * 2, conv_kernel_size, padding=conv_kernel_size // 2)
        else:
            self.proj = nn.Linear(dim_in, dim_out * 2)
        self.use_conv = use_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            x = rearrange(x, "b n d -> b d n")
            x = self.proj(x)
            x = rearrange(x, "b d n -> b n d")
        else:
            x = self.proj(x)
        x, gate = x.chunk(2, dim=-1)
        return x * self.act(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: float = 4,
        no_bias: bool = False,
        glu: bool = True,
        use_conv: bool = False,
        conv_kernel_size: int = 3,
        zero_init_output: bool = True,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        activation = nn.SiLU()
        dim_out = dim if dim_out is None else dim_out
        if glu:
            linear_in = GLU(dim, inner_dim, activation, use_conv=use_conv, conv_kernel_size=conv_kernel_size)
        else:
            layers = [nn.Linear(dim, inner_dim, bias=not no_bias), activation]
            if use_conv:
                layers = [
                    Rearrange("b n d -> b d n"),
                    nn.Conv1d(dim, inner_dim, conv_kernel_size, padding=conv_kernel_size // 2, bias=not no_bias),
                    Rearrange("b d n -> b n d"),
                    activation,
                ]
            linear_in = nn.Sequential(*layers)
        if use_conv:
            linear_out = nn.Conv1d(inner_dim, dim_out, conv_kernel_size, padding=conv_kernel_size // 2, bias=not no_bias)
            self.ff = nn.Sequential(
                linear_in,
                Rearrange("b n d -> b d n"),
                linear_out,
                Rearrange("b d n -> b n d"),
            )
        else:
            linear_out = nn.Linear(inner_dim, dim_out, bias=not no_bias)
            self.ff = nn.Sequential(linear_in, linear_out)
        if zero_init_output:
            nn.init.zeros_(linear_out.weight)
            if not no_bias:
                nn.init.zeros_(linear_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_heads: int = 64,
        dim_context: Optional[int] = None,
        causal: bool = False,
        zero_init_output: bool = True,
        qk_norm: Literal["l2", "ln", "none"] = "none",
        natten_kernel_size: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.causal = causal
        dim_kv = dim_context if dim_context is not None else dim
        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads
        if dim_context is not None:
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)
        self.qk_norm = qk_norm
        if qk_norm == "ln":
            self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1e-6)
            self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1e-6)
        self.natten_kernel_size = natten_kernel_size
        if natten_kernel_size is None:
            self.use_pt_flash = torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse("2.0.0")
            self.use_fa_flash = torch.cuda.is_available() and flash_attn_func is not None
            self.sdp_kwargs = dict(enable_flash=True, enable_math=True, enable_mem_efficient=True)

    def flash_attn(self, q, k, v, mask=None, causal=None):
        batch, heads, q_len = q.shape[0], q.shape[1], q.shape[-2]
        kv_heads = k.shape[1]
        if heads != kv_heads:
            heads_per_kv_head = heads // kv_heads
            k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim=1), (k, v))
        causal = self.causal if causal is None else causal
        if q_len == 1 and causal:
            causal = False
        if mask is not None:
            mask = mask.expand(batch, heads, q_len, k.shape[-2])
        if k.shape[-2] > q_len and causal:
            causal_mask = create_causal_mask(q_len, k.shape[-2], device=q.device)
            if mask is None:
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False
        row_is_entirely_masked = None
        if mask is not None and causal:
            causal_mask = create_causal_mask(q_len, k.shape[-2], device=q.device)
            mask = mask & ~causal_mask
            row_is_entirely_masked = ~mask.any(dim=-1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked
            causal = False
        with torch.backends.cuda.sdp_kernel(**self.sdp_kwargs):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)
        if row_is_entirely_masked is not None:
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.0)
        return out

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor, float]] = None,
        causal: Optional[bool] = None,
    ) -> torch.Tensor:
        has_context = context is not None
        if hasattr(self, "to_q"):
            q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.num_heads)
            k, v = self.to_kv(context if has_context else x).chunk(2, dim=-1)
            k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.kv_heads), (k, v))
        else:
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v))
        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        elif self.qk_norm == "ln":
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rotary_pos_emb is not None and not has_context:
            freqs, scale = rotary_pos_emb
            q_dtype, k_dtype = q.dtype, k.dtype
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            freqs = freqs.to(torch.float32)
            q = apply_rotary_pos_emb(q, freqs, scale)
            k = apply_rotary_pos_emb(k, freqs, scale)
            q = q.to(q_dtype)
            k = k.to(k_dtype)
        input_mask = context_mask if context_mask is not None else mask
        masks = []
        final_attn_mask = None
        if input_mask is not None:
            masks.append(~rearrange(input_mask, "b n -> b 1 1 n"))
        if masks:
            final_attn_mask = ~or_reduce(masks)
        causal = self.causal if causal is None else causal
        if self.natten_kernel_size is not None:
            if natten is None:
                raise ImportError("natten is required for neighborhood attention")
            dtype_in = q.dtype
            q, k, v = map(lambda t: t.to(torch.float32), (q, k, v))
            attn = natten.functional.natten1dqk(q, k, kernel_size=self.natten_kernel_size, dilation=1)
            if final_attn_mask is not None:
                attn = attn.masked_fill(final_attn_mask, -torch.finfo(attn.dtype).max)
            attn = F.softmax(attn, dim=-1, dtype=torch.float32)
            out = natten.functional.natten1dav(attn, v, kernel_size=self.natten_kernel_size, dilation=1).to(dtype_in)
        elif getattr(self, "use_fa_flash", False):
            if final_attn_mask is not None:
                raise NotImplementedError("masking not supported with FlashAttention2")
            q, k, v = map(lambda t: rearrange(t, "b h n d -> b n h d").to(torch.float16), (q, k, v))
            out = flash_attn_func(q, k, v, causal=causal)
            out = rearrange(out, "b n h d -> b h n d").to(torch.float32)
        elif getattr(self, "use_pt_flash", False):
            out = self.flash_attn(q, k, v, mask=final_attn_mask, causal=causal)
        else:
            if self.num_heads != self.kv_heads:
                heads_per_kv_head = self.num_heads // self.kv_heads
                k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim=1), (k, v))
            scale = 1.0 / math.sqrt(q.shape[-1])
            dots = einsum("b h i d, b h j d -> b h i j", q, k) * scale
            mask_value = -torch.finfo(dots.dtype).max
            if final_attn_mask is not None:
                dots = dots.masked_fill(~final_attn_mask, mask_value)
            if causal:
                dots = dots.masked_fill(create_causal_mask(dots.shape[-2], dots.shape[-1], device=dots.device), mask_value)
            attn = F.softmax(dots, dim=-1, dtype=torch.float32).type(dots.dtype)
            out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        if mask is not None:
            out = out.masked_fill(~rearrange(mask, "b n -> b n 1"), 0.0)
        return out


class ConformerModule(nn.Module):
    def __init__(self, dim: int, norm_kwargs=None):
        super().__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        self.in_norm = LayerNorm(dim, **norm_kwargs)
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.glu = GLU(dim, dim, nn.SiLU())
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
        self.mid_norm = LayerNorm(dim, **norm_kwargs)
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(x)
        x = rearrange(x, "b n d -> b d n")
        x = self.pointwise_conv(x)
        x = rearrange(x, "b d n -> b n d")
        x = self.glu(x)
        x = rearrange(x, "b n d -> b d n")
        x = self.depthwise_conv(x)
        x = rearrange(x, "b d n -> b n d")
        x = self.mid_norm(x)
        x = self.swish(x)
        x = rearrange(x, "b n d -> b d n")
        x = self.pointwise_conv_2(x)
        x = rearrange(x, "b d n -> b n d")
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_heads: int = 64,
        cross_attend: bool = False,
        dim_context: Optional[int] = None,
        global_cond_dim: Optional[int] = None,
        causal: bool = False,
        zero_init_branch_outputs: bool = True,
        conformer: bool = False,
        layer_ix: int = -1,
        remove_norms: bool = False,
        attn_kwargs=None,
        ff_kwargs=None,
        norm_kwargs=None,
    ):
        super().__init__()
        if attn_kwargs is None:
            attn_kwargs = {}
        if ff_kwargs is None:
            ff_kwargs = {}
        if norm_kwargs is None:
            norm_kwargs = {}
        self.pre_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
        self.self_attn = Attention(
            dim,
            dim_heads=dim_heads,
            causal=causal,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs,
        )
        self.cross_attend = cross_attend
        if cross_attend:
            self.cross_attend_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
            self.cross_attn = Attention(
                dim,
                dim_heads=dim_heads,
                dim_context=dim_context,
                causal=causal,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs,
            )
        self.ff_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)
        self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs) if conformer else None
        self.global_cond_dim = global_cond_dim
        if global_cond_dim:
            self.to_scale_shift_gate = nn.Sequential(
                nn.SiLU(),
                nn.Linear(global_cond_dim, dim * 6, bias=False),
            )
            nn.init.zeros_(self.to_scale_shift_gate[1].weight)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor, float]] = None,
    ) -> torch.Tensor:
        if self.global_cond_dim and global_cond is not None:
            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = self.to_scale_shift_gate(global_cond).unsqueeze(1).chunk(6, dim=-1)
            residual = x
            x = self.pre_norm(x)
            x = x * (1 + scale_self) + shift_self
            x = self.self_attn(x, mask=mask, rotary_pos_emb=rotary_pos_emb)
            x = x * torch.sigmoid(1 - gate_self)
            x = x + residual
            if context is not None:
                x = x + self.cross_attn(self.cross_attend_norm(x), context=context, context_mask=context_mask)
            if self.conformer is not None:
                x = x + self.conformer(x)
            residual = x
            x = self.ff_norm(x)
            x = x * (1 + scale_ff) + shift_ff
            x = self.ff(x)
            x = x * torch.sigmoid(1 - gate_ff)
            x = x + residual
        else:
            x = x + self.self_attn(self.pre_norm(x), mask=mask, rotary_pos_emb=rotary_pos_emb)
            if context is not None:
                x = x + self.cross_attn(self.cross_attend_norm(x), context=context, context_mask=context_mask)
            if self.conformer is not None:
                x = x + self.conformer(x)
            x = x + self.ff(self.ff_norm(x))
        return x


class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        dim_in: Optional[int] = None,
        dim_out: Optional[int] = None,
        dim_heads: int = 64,
        cross_attend: bool = False,
        cond_token_dim: Optional[int] = None,
        global_cond_dim: Optional[int] = None,
        causal: bool = False,
        rotary_pos_emb: bool = True,
        zero_init_branch_outputs: bool = True,
        conformer: bool = False,
        use_sinusoidal_emb: bool = False,
        use_abs_pos_emb: bool = False,
        abs_pos_emb_max_length: int = 10000,
        **kwargs,
    ):
        super().__init__()
        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()
        self.layers = nn.ModuleList([])
        self.causal = causal
        self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32)) if rotary_pos_emb else None
        self.use_sinusoidal_emb = use_sinusoidal_emb
        self.use_abs_pos_emb = use_abs_pos_emb
        if use_sinusoidal_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)
        elif use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length)
        for i in range(depth):
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads=dim_heads,
                    cross_attend=cross_attend,
                    dim_context=cond_token_dim,
                    global_cond_dim=global_cond_dim,
                    causal=causal,
                    zero_init_branch_outputs=zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    **kwargs,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        prepend_embeds: Optional[torch.Tensor] = None,
        prepend_mask: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        return_info: bool = False,
        **kwargs,
    ):
        batch, seq = x.shape[:2]
        x = self.project_in(x)
        if prepend_embeds is not None:
            if prepend_embeds.shape[-1] != x.shape[-1]:
                raise ValueError("prepend embedding dim must match model dim")
            x = torch.cat((prepend_embeds, x), dim=-2)
            if prepend_mask is not None or mask is not None:
                mask = mask if mask is not None else torch.ones((batch, seq), device=x.device, dtype=torch.bool)
                prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_embeds.shape[-2]), device=x.device, dtype=torch.bool)
                mask = torch.cat((prepend_mask, mask), dim=-1)
        if self.rotary_pos_emb is not None:
            rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])
        else:
            rotary_pos_emb = None
        if self.use_sinusoidal_emb or self.use_abs_pos_emb:
            x = x + self.pos_emb(x)
        info = {"hidden_states": []}
        for layer in self.layers:
            x = checkpoint(layer, x, rotary_pos_emb=rotary_pos_emb, global_cond=global_cond, mask=mask, **kwargs)
            if return_info:
                info["hidden_states"].append(x)
        x = self.project_out(x)
        if return_info:
            return x, info
        return x
