import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from diffusers import DDPMScheduler
except ImportError as e:
    raise ImportError(
        "Please install diffusers first: pip install diffusers[torch] transformers accelerate"
    ) from e


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def pairwise_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    b1 = box_cxcywh_to_xyxy(boxes1)
    b2 = box_cxcywh_to_xyxy(boxes2)
    lt = torch.maximum(b1[..., :, None, :2], b2[..., None, :, :2])
    rb = torch.minimum(b1[..., :, None, 2:], b2[..., None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((b1[..., 2] - b1[..., 0]).clamp(min=0) * (b1[..., 3] - b1[..., 1]).clamp(min=0))
    area2 = ((b2[..., 2] - b2[..., 0]).clamp(min=0) * (b2[..., 3] - b2[..., 1]).clamp(min=0))
    union = area1[..., :, None] + area2[..., None, :] - inter
    return inter / (union + 1e-6)


class MotionAwareTemporalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, b, q, d = x.shape
        x_ = rearrange(x, 't b q d -> t (b q) d')
        attn_out = self.attn(x_, x_, x_)[0]
        x_ = self.norm1(x_ + attn_out)
        x_ = self.norm2(x_ + self.ffn(x_))
        return rearrange(x_, 't (b q) d -> t b q d', b=b, q=q)


class ConstrainedSlotAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        topk: int = 8,
        diag_bias: float = 2.0,
        iou_bias: float = 1.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.topk = topk
        self.diag_bias = diag_bias
        self.iou_bias = iou_bias
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, ref_boxes: torch.Tensor) -> torch.Tensor:
        t, b, q, d = x.shape
        residual = x
        qx = self.q_proj(x).view(t, b, q, self.n_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        kx = self.k_proj(x).view(t, b, q, self.n_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        vx = self.v_proj(x).view(t, b, q, self.n_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        attn_logits = torch.matmul(qx, kx.transpose(-1, -2)) * self.scale
        iou_bias = pairwise_box_iou(ref_boxes, ref_boxes).unsqueeze(2)
        eye = torch.eye(q, device=x.device, dtype=x.dtype).view(1, 1, 1, q, q)
        attn_logits = attn_logits + self.iou_bias * iou_bias + self.diag_bias * eye
        if self.topk is not None and self.topk < q:
            topk_idx = attn_logits.topk(self.topk, dim=-1).indices
            mask = torch.zeros_like(attn_logits, dtype=torch.bool)
            mask.scatter_(-1, topk_idx, True)
            attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
        attn = attn_logits.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, vx)
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(t, b, q, d)
        out = self.out_proj(out)
        return self.norm(residual + out)


class TextConditioner(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(self, traj: torch.Tensor, text_tok: torch.Tensor, text_mask: torch.Tensor = None) -> torch.Tensor:
        t, b, q, d = traj.shape
        q_tokens = rearrange(traj, 't b q d -> (b t) q d')
        kv_tokens = repeat(text_tok, 'b l d -> (b t) l d', t=t)
        key_padding_mask = None if text_mask is None else repeat(~text_mask, 'b l -> (b t) l', t=t)
        cross, _ = self.cross_attn(
            query=q_tokens,
            key=kv_tokens,
            value=kv_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        fused = self.norm1(q_tokens + cross)
        fused = self.norm2(fused + self.ffn(fused))
        gate = self.gate(torch.cat([q_tokens, fused], dim=-1))
        fused = gate * fused + (1.0 - gate) * q_tokens
        return rearrange(fused, '(b t) q d -> t b q d', b=b, t=t)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long)
        if timesteps.dim() == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.float()
        device = timesteps.device
        half_dim = self.dim // 2
        if half_dim == 0:
            return timesteps[:, None]
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * exponent)
        args = timesteps[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualTemporalDenoiseBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, dropout: float = 0.0, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.time_proj = nn.Linear(time_dim, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm1(x)
        h = F.silu(h + cond)
        h = self.conv1(h)
        h = h + self.time_proj(time_emb).unsqueeze(-1)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return residual + h


class LengthPreservingTemporalDenoiser(nn.Module):
    def __init__(self, latent_dim: int, time_embed_dim: int = None, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed_dim = time_embed_dim or latent_dim * 4
        self.time_embed = SinusoidalTimeEmbedding(self.time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.in_proj = nn.Conv1d(latent_dim, latent_dim, kernel_size=1)
        self.cond_proj = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
        )
        self.blocks = nn.ModuleList(
            [ResidualTemporalDenoiseBlock(latent_dim, self.time_embed_dim, dropout=dropout) for _ in range(depth)]
        )
        self.out_norm = nn.GroupNorm(8, latent_dim)
        self.out_proj = nn.Conv1d(latent_dim, latent_dim, kernel_size=1)

    def _expand_timestep(self, timestep, batch_size: int, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(timestep):
            ts = timestep.to(device)
            if ts.dim() == 0:
                ts = ts.repeat(batch_size)
            elif ts.shape[0] == 1 and batch_size > 1:
                ts = ts.repeat(batch_size)
            elif ts.shape[0] != batch_size:
                ts = ts.reshape(-1)
                if ts.shape[0] != batch_size:
                    raise ValueError(f"Unexpected timestep shape {tuple(ts.shape)} for batch size {batch_size}")
            return ts.long()
        return torch.full((batch_size,), int(timestep), device=device, dtype=torch.long)

    def forward(self, sample: torch.Tensor, cond: torch.Tensor, timestep) -> torch.Tensor:
        batch_size = sample.shape[0]
        timesteps = self._expand_timestep(timestep, batch_size, sample.device)
        time_emb = self.time_mlp(self.time_embed(timesteps))
        cond_feat = self.cond_proj(cond)
        h = self.in_proj(sample)
        for block in self.blocks:
            h = block(h, time_emb, cond_feat)
        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_proj(h)


class TemporalContextEnhancer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        slot_topk: int = 8,
        slot_diag_bias: float = 2.0,
        slot_iou_bias: float = 1.0,
    ):
        super().__init__()
        self.motion = MotionAwareTemporalAttention(d_model, n_heads, dropout)
        self.slot = ConstrainedSlotAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            topk=slot_topk,
            diag_bias=slot_diag_bias,
            iou_bias=slot_iou_bias,
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, traj: torch.Tensor, ref_boxes: torch.Tensor):
        traj = self.motion(traj)
        traj = self.slot(traj, ref_boxes)
        return self.out_norm(traj)


class BoxTrajectoryDiffusionHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        latent_dim: int = 64,
        num_inference_scheduler_steps: int = 50,
        num_infer_steps: int = 4,
        infer_noise_scale: float = 0.15,
        refine_strength: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.box_in = nn.Sequential(
            nn.Linear(4, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.text_conditioner = TextConditioner(d_model, n_heads, dropout)
        self.cond_in = nn.Linear(d_model, latent_dim)
        self.box_out = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 4),
        )
        self.box_gate = nn.Sequential(
            nn.Linear(8, 32),
            nn.GELU(),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )
        self.latent_to_query = nn.Linear(latent_dim, d_model)
        self.query_delta = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.query_gate = nn.Sequential(
            nn.Linear(d_model * 2 + 8, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.query_norm = nn.LayerNorm(d_model)

        self.denoiser = LengthPreservingTemporalDenoiser(
            latent_dim=latent_dim,
            time_embed_dim=latent_dim * 4,
            depth=4,
            dropout=dropout,
        )

        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_inference_scheduler_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=False,
        )
        self.num_infer_steps = num_infer_steps
        self.infer_noise_scale = infer_noise_scale
        self.refine_strength = max(0.0, min(1.0, float(refine_strength)))

    def _build_cond_feat(
        self,
        query_feat: torch.Tensor,
        text_tokens: torch.Tensor,
        text_token_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.text_conditioner(query_feat, text_tokens, text_token_mask)

    def _encode_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        return rearrange(self.box_in(boxes), 'b t q c -> (b q) c t')

    def _encode_cond(self, cond_feat: torch.Tensor) -> torch.Tensor:
        cond_btqd = rearrange(cond_feat, 't b q d -> b t q d')
        return rearrange(self.cond_in(cond_btqd), 'b t q c -> (b q) c t')

    def _fuse_queries(
        self,
        pred_boxes: torch.Tensor,
        refined_boxes: torch.Tensor,
        query_feat: torch.Tensor,
        cond_feat: torch.Tensor,
        latent_sample: torch.Tensor,
    ) -> torch.Tensor:
        B, T, Q, _ = pred_boxes.shape
        query_btqd = rearrange(query_feat, 't b q d -> b t q d')
        cond_btqd = rearrange(cond_feat, 't b q d -> b t q d')
        latent_btqd = rearrange(latent_sample, '(b q) c t -> b t q c', b=B, q=Q)
        latent_query = self.latent_to_query(latent_btqd)
        delta = self.query_delta(torch.cat([cond_btqd, latent_query], dim=-1))
        gate = self.query_gate(torch.cat([query_btqd, delta, pred_boxes, refined_boxes], dim=-1))
        refined_query = self.query_norm(query_btqd + gate * delta)
        return rearrange(refined_query, 'b t q d -> t b q d')

    def refine_trajectory(
        self,
        pred_boxes: torch.Tensor,
        query_feat: torch.Tensor,
        text_tokens: torch.Tensor,
        text_token_mask: torch.Tensor,
    ):
        # pred_boxes: [B, T, Q, 4], query_feat: [T, B, Q, D]
        cond_feat = self._build_cond_feat(query_feat, text_tokens, text_token_mask)
        if self.num_infer_steps <= 0:
            return pred_boxes, query_feat, cond_feat

        B, T, Q, _ = pred_boxes.shape
        sample = self._encode_boxes(pred_boxes)
        cond = self._encode_cond(cond_feat)

        self.scheduler.set_timesteps(self.num_infer_steps, device=pred_boxes.device)
        timesteps = self.scheduler.timesteps
        if len(timesteps) == 0:
            return pred_boxes, query_feat, cond_feat

        refine_steps = max(1, int(round(len(timesteps) * float(self.refine_strength))))
        timesteps = timesteps[-refine_steps:]
        start_ts = torch.full(
            (sample.shape[0],),
            int(timesteps[0]),
            device=sample.device,
            dtype=torch.long,
        )

        noise = torch.randn_like(sample) * self.infer_noise_scale
        sample = self.scheduler.add_noise(sample, noise, start_ts)

        for timestep in timesteps:
            model_sample = self.scheduler.scale_model_input(sample, timestep)
            pred_noise = self.denoiser(model_sample, cond, timestep)
            sample = self.scheduler.step(pred_noise, timestep, sample).prev_sample

        refined = self.box_out(rearrange(sample, '(b q) c t -> b t q c', b=B, q=Q)).sigmoid()
        gate = self.box_gate(torch.cat([pred_boxes, refined], dim=-1))
        refined = ((1.0 - gate) * pred_boxes + gate * refined).clamp(0.0, 1.0)
        refined_query = self._fuse_queries(pred_boxes, refined, query_feat, cond_feat, sample)
        return refined, refined_query, cond_feat
