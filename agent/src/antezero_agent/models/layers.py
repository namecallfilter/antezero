"""Building blocks of the policy (models/rl_policy.py): a pre-norm Transformer encoder with the parameter names, init and
maths of nn.TransformerEncoder, on scaled_dot_product_attention (faster), plus small helpers (a zero-initialised
bias-free linear, zero-padding a linear's inputs, symlog)."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_inputs(lin, n_in):
    """Grow nn.Linear lin to n_in inputs: the new input columns are zero (no random draws)."""
    w = lin.weight
    if n_in > w.shape[1]:
        lin.weight = nn.Parameter(torch.cat([w.data, w.data.new_zeros(w.shape[0], n_in - w.shape[1])], 1))
        lin.in_features = n_in


class ZeroLinear(nn.Module):
    """nn.Linear without bias that starts at zero, without drawing random numbers (a submodule, so its weight comes
    after the parameters registered before it)."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_out, n_in))

    def forward(self, x):
        return F.linear(x, self.weight)


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


class FastSelfAttention(nn.Module):
    """nn.MultiheadAttention's self-attention (batch_first, no dropout) with the same parameters (in_proj_weight,
    in_proj_bias, out_proj), the same init (and random draws, in the same order) and the same maths."""

    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d, d))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d))
        self.out_proj = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.in_proj_weight)   # (nn.MultiheadAttention._reset_parameters)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, pad):
        Bn, L, d = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).view(Bn, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=~pad[:, None, None, :])
        return self.out_proj(o.transpose(1, 2).reshape(Bn, L, d))


class FastEncoderLayer(nn.Module):
    """nn.TransformerEncoderLayer(d, heads, 4 d, dropout 0, batch_first, norm_first, relu) with the same parameter
    names, registration order and init: x + attn(norm1(x)), then x + linear2(relu(linear1(norm2(x))))."""

    def __init__(self, d, heads):
        super().__init__()
        self.self_attn = FastSelfAttention(d, heads)
        self.linear1 = nn.Linear(d, 4 * d)
        self.linear2 = nn.Linear(4 * d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x, pad):
        x = x + self.self_attn(self.norm1(x), pad)
        return x + self.linear2(F.relu(self.linear1(self.norm2(x))))


class FastEncoder(nn.Module):
    """nn.TransformerEncoder of FastEncoderLayers: copies of one layer (as nn.TransformerEncoder deep-copies its
    layer), the same state-dict keys (layers.i.*), forward(src, src_key_padding_mask) with True = padding."""

    def __init__(self, layer, n):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])

    def forward(self, x, src_key_padding_mask):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return x
