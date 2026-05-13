import copy

import torch
import torch.nn as nn
from einops import rearrange


class MotionTemporalBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, hs):
        t, b, q, d = hs.shape
        hs_btq = rearrange(hs, 't b q d -> t (b q) d')
        attn_out = self.self_attn(hs_btq, hs_btq, hs_btq)[0]
        hs_btq = self.norm1(hs_btq + attn_out)
        hs_btq = self.norm2(hs_btq + self.ffn(hs_btq))
        return rearrange(hs_btq, 't (b q) d -> t b q d', b=b, q=q)


class TemporalDecoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layer, dropout):
        super().__init__()
        self.n_layer = n_layer
        self.layer = nn.ModuleList([copy.deepcopy(MotionTemporalBlock(d_model, n_heads, dropout)) for _ in range(n_layer)])

    def forward(self, hs, tgt=None):
        for i in range(self.n_layer):
            hs = self.layer[i](hs)
        return hs
