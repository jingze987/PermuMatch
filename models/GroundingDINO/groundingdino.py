# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model components.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import os
import torch
import torch.nn.functional as F
from torch import nn
import torchvision

from models.dino_util import get_tokenlizer
from models.dino_util.misc import (
    NestedTensor,
    inverse_sigmoid,
    nested_tensor_from_tensor_list,
)

from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer import build_transformer, DeformableTransformerDecoderLayer
from .utils import MLP, ContrastiveEmbed, _set_lora_linear
from .segmentation import FPNSpatialDecoder
from ..dino_util.misc import clean_state_dict
from .utils import gen_sineembed_for_position, _get_clones
from einops import rearrange, repeat
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from .temporal_modules import TemporalDecoder

class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
            self,
            backbone,
            transformer,
            num_queries,
            aux_loss=False,
            iter_update=False,
            query_dim=2,
            num_feature_levels=1,
            nheads=8,
            # two stage
            two_stage_type="no",  # ['no', 'standard']
            dec_pred_bbox_embed_share=True,
            two_stage_class_embed_share=True,
            two_stage_bbox_embed_share=True,
            num_patterns=0,
            dn_number=100,
            dn_box_noise_scale=0.4,
            dn_label_noise_ratio=0.5,
            dn_labelbook_size=100,
            text_encoder_type="bert-base-uncased",
            sub_sentence_present=True,
            max_text_len=256,
            num_classes=1,
            dropout=0.0,
            # lora
            dec_lora=False,
            enc_lora=False,
            lora_rank=16,
            # segmentation
            pixel_decoder=None,
            mask_decoder=None,
            temporal_layer=3,
            tracking_alpha=0.1,
            tracking_beta=0.5,
            full_tune=False,
            trainable_key_list=[],
            use_diffusion_temporal=False,
            diffusion_layer_idx=-1,
            diffusion_latent_dim=128,
            diffusion_total_steps=50,
            diffusion_infer_steps=4,
            diffusion_noise_scale=0.15,
            diffusion_refine_strength=0.5,
            slot_topk=8,
            slot_diag_bias=2.0,
            slot_iou_bias=1.0,
            use_lightweight_emd=False,
            emd_weight=0.25,
            emd_candidate_topk=5,
            emd_roi_size=3,
            emd_sinkhorn_iters=5,
            emd_eps=0.07,
            use_memory_relation_distill=False,
            memory_relation_tau=1.0,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer__.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = 256
        self.num_classes = num_classes
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # denoising-query configuration retained for checkpoint compatibility
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        # freeze

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            # num_backbone_outs = len(backbone.num_channels)
            num_backbone_outs = len(backbone.num_channels[-3:])  # Not use the first-layer feature
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[-3:][_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-3:][0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self.ref_point_head = self.transformer.decoder.ref_point_head
        self._reset_parameters()

        if enc_lora or dec_lora:
            if enc_lora:
                _set_lora_linear(self.transformer.encoder, lora_rank)
            if dec_lora:
                _set_lora_linear(self.transformer.decoder, lora_rank)

        if pixel_decoder and mask_decoder:
            self.mask = True
            self.pixel_decoder = pixel_decoder
            self.mask_decoder = mask_decoder
        else:
            self.mask = False

        # temporal aggregation
        _temporal_decoder = TemporalDecoder(hidden_dim, nheads, temporal_layer, dropout)
        temporal_decoder_list = [_temporal_decoder for i in range(transformer.num_decoder_layers)]
        self.temporal_decoder = nn.ModuleList(temporal_decoder_list)
        self.tracking_alpha = tracking_alpha
        self.tracking_beta = float(tracking_beta)
        self.use_lightweight_emd = use_lightweight_emd
        self.emd_weight = emd_weight
        self.emd_candidate_topk = emd_candidate_topk
        self.emd_roi_size = emd_roi_size
        self.emd_sinkhorn_iters = emd_sinkhorn_iters
        self.emd_eps = emd_eps

        self.use_diffusion_temporal = use_diffusion_temporal
        self.diffusion_layer_idx = transformer.num_decoder_layers - 1 if diffusion_layer_idx < 0 else diffusion_layer_idx
        self.use_memory_relation_distill = use_memory_relation_distill
        self.memory_relation_tau = float(memory_relation_tau)
        self.mask_out_stride = 4
        self.temporal_context = None
        self.box_diffusion = None
        if use_diffusion_temporal:
            from .diffusion_temporal_modules import TemporalContextEnhancer, BoxTrajectoryDiffusionHead
            self.temporal_context = TemporalContextEnhancer(
                d_model=hidden_dim,
                n_heads=nheads,
                dropout=dropout,
                slot_topk=slot_topk,
                slot_diag_bias=slot_diag_bias,
                slot_iou_bias=slot_iou_bias,
            )
            self.box_diffusion = BoxTrajectoryDiffusionHead(
                d_model=hidden_dim,
                n_heads=nheads,
                latent_dim=diffusion_latent_dim,
                num_inference_scheduler_steps=diffusion_total_steps,
                num_infer_steps=diffusion_infer_steps,
                infer_noise_scale=diffusion_noise_scale,
                refine_strength=diffusion_refine_strength,
                dropout=dropout,
            )

        # freeze parameters that are not used by the inference-time fine-tuned heads
        if not full_tune:
            for n, p in self.named_parameters():
                if any(keyword in n for keyword in trainable_key_list):
                    continue
                p.requires_grad_(False)

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def infer(self, all_samples: NestedTensor, captions, targets, crop_len=36, **kw):
        all_samples = copy.deepcopy(all_samples)

        if isinstance(all_samples, (list, torch.Tensor)):
            all_samples = nested_tensor_from_tensor_list(all_samples)

        B, T, H, W = all_samples.mask.shape
        assert B == 1, "Inference only supports for batch size = 1"
        device = all_samples.device
        nl = self.transformer.num_decoder_layers - 1

        captions = [t if t.endswith(".") else t + "." for t in captions]

        # encoder texts

        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(
            device
        )

        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                                        :, : self.max_text_len, : self.max_text_len
                                        ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                                        :, : self.max_text_len, : self.max_text_len
                                        ]

        all_features, all_hs, all_reference, all_memory_enc = [], [], [], []
        all_text_dict = {
            "encoded_text": [],
            "text_token_mask": [],
            "position_ids": [],
            "text_self_attention_masks": []
        }
        for clip_id in range(0, T, crop_len):
            samples = NestedTensor(
                all_samples.tensors[:, clip_id: clip_id + crop_len],
                all_samples.mask[:, clip_id: clip_id + crop_len]
            )

            samples.tensors = rearrange(samples.tensors, 'b t c h w -> (b t) c h w')
            samples.mask = rearrange(samples.mask, 'b t h w -> (b t) h w')

            t = samples.tensors.shape[0]
            text_dict = {
                "encoded_text": repeat(encoded_text, 'b ... -> (b t) ...', t=t),
                "text_token_mask": repeat(text_token_mask, 'b ... -> (b t) ...', t=t),
                "position_ids": repeat(position_ids, 'b ... -> (b t) ...', t=t),
                "text_self_attention_masks": repeat(text_self_attention_masks, 'b ... -> (b t) ...', t=t),
            }

            features, poss = self.backbone(samples)
            poss = poss[-3:]


            srcs = []
            masks = []
            for l, feat in enumerate(features[-3:]):
                src, mask = feat.decompose()
                srcs.append(self.input_proj[l](src))
                masks.append(mask)
                assert mask is not None
            if self.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1].tensors)
                    else:
                        src = self.input_proj[l](srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    poss.append(pos_l)

            input_query_bbox = input_query_label = attn_mask = dn_meta = None
            output = self.transformer(
                srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
            )
            hs, reference, hs_enc, ref_enc, init_box_proposal, memory_enc, selected_query_list = output
            all_hs.append(hs[nl])
            all_reference.append(reference[nl])
            all_memory_enc.append(memory_enc)
            all_features.append(features[0])
            for k, v in text_dict.items():
                all_text_dict[k].append(v)

        # merge into video features
        hs = torch.cat(all_hs, dim=0)   # (video_len, q, d)
        reference = torch.cat(all_reference, dim=0)   # (video_len, q, d)
        memory_enc = [torch.cat(mem, dim=0) for mem in list(zip(*all_memory_enc))]
        features = NestedTensor(*[torch.cat(x, dim=0) for x in list(zip(*[feat.decompose() for feat in all_features]))])
        text_dict = {k: torch.cat(v, dim=0) for k, v in all_text_dict.items()}

        # prepare high-resolution visual tokens for tracking
        memory_enc.insert(0, features.tensors)
        mask_features = self.pixel_decoder(memory_enc[-1], memory_enc[:-1][::-1])

        # sort the queries via tracking
        text_feat = rearrange(text_dict['encoded_text'], '(b t) l d -> t b l d', t=T)
        alpha = self.tracking_alpha
        hs = rearrange(hs, '(b t) q d -> t b q d', t=T)
        reference = rearrange(reference, '(b t) q d -> t b q d', t=T)
        roi_tokens = None
        if self.mask and self.use_lightweight_emd:
            roi_tokens = self._extract_roi_tokens(mask_features, rearrange(reference, 't b q d -> (b t) q d'))
            roi_tokens = rearrange(roi_tokens, '(b t) q k d -> t b q k d', t=T)
        hs_tracked = [hs[0]]
        ref_tracked = [reference[0]]
        hs_memory = hs[0]
        token_memory = roi_tokens[0] if roi_tokens is not None else None
        for t in range(1, T):
            cur_tokens = roi_tokens[t] if roi_tokens is not None else None
            ind = self.match_from_embds(hs_memory, hs[t], token_memory, cur_tokens)
            ind = torch.tensor(ind, device=device)
            hs_sorted = torch.gather(hs[t], 1, ind[:, :, None].expand(-1, -1, self.hidden_dim))
            ref_sorted = torch.gather(reference[t], 1, ind[:, :, None].expand(-1, -1, 4))
            conf = self._compute_tracking_conf(hs_sorted, text_feat[t])
            hs_memory = (1 - alpha * conf) * hs_memory + alpha * conf * hs_sorted
            hs_memory = F.normalize(hs_memory, dim=-1)
            if token_memory is not None:
                token_sorted = torch.gather(
                    cur_tokens,
                    1,
                    ind[:, :, None, None].expand(-1, -1, cur_tokens.shape[2], cur_tokens.shape[3])
                )
                token_memory = (1 - alpha * conf.unsqueeze(-1)) * token_memory + alpha * conf.unsqueeze(-1) * token_sorted
                token_memory = F.normalize(token_memory, dim=-1)
            hs_tracked.append(hs_sorted)
            ref_tracked.append(ref_sorted)
        hs = rearrange(torch.stack(hs_tracked, 0), 't b q d -> t (b q) d')
        reference = rearrange(torch.stack(ref_tracked, 0), 't b q d -> (b t) q d')

        # temporal fusion / diffusion enhancement
        hs_t_student = rearrange(hs, 't (b q) d -> t b q d', b=B)
        hs_t_post = None
        if self.use_diffusion_temporal and self.temporal_context is not None and nl == self.diffusion_layer_idx:
            ref_t = rearrange(reference, '(b t) q d -> t b q d', b=B)
            hs_t_post = self.temporal_context(
                traj=hs_t_student,
                ref_boxes=ref_t,
            )
            hs = rearrange(hs_t_post, 't b q d -> (b t) q d', b=B)
        else:
            hs_t_post = self.temporal_decoder[nl](hs_t_student)
            hs = rearrange(hs_t_post, 't b q d -> (b t) q d', b=B)

        # bbox
        delta_unsig = self.bbox_embed[nl](hs)
        coord_unsig = delta_unsig + inverse_sigmoid(reference)
        coord = coord_unsig.sigmoid()
        logits_pre = torch.max(self.class_embed[nl](hs, text_dict), dim=-1, keepdim=True)[0]
        if self.use_diffusion_temporal and self.box_diffusion is not None and nl == self.diffusion_layer_idx and hs_t_post is not None:
            coord_t = rearrange(coord, '(b t) q c -> b t q c', b=B)
            coord_t, hs_t_post, _ = self.box_diffusion.refine_trajectory(
                pred_boxes=coord_t,
                query_feat=hs_t_post,
                text_tokens=encoded_text,
                text_token_mask=text_token_mask,
            )
            coord = rearrange(coord_t, 'b t q c -> (b t) q c')
            hs = rearrange(hs_t_post, 't b q d -> (b t) q d', b=B)

        # logits
        logits = torch.max(self.class_embed[nl](hs, text_dict), dim=-1, keepdim=True)[0]

        # segmentation
        mask_features_flat = mask_features.flatten(2).permute(2, 0, 1)
        mask_features_padding = features.mask.flatten(1)
        start_index = torch.tensor(0, device=device)
        spatial_shapes = torch.as_tensor(mask_features.shape[-2:], device=device)[None, :]
        valid_ratios = self.transformer.get_valid_ratio(features.mask)
        valid_ratios = torch.cat([valid_ratios, valid_ratios], -1)[:, None, :]
        query_ref_input = coord * valid_ratios  # use diffusion-refined trajectory
        query_sine_embed = gen_sineembed_for_position(query_ref_input)
        query_pos = self.ref_point_head(query_sine_embed).transpose(0, 1)  # nq, bs, 256
        query_ref_input = query_ref_input.transpose(0, 1)[:, :, None, :]
        mask_embed = hs.transpose(0, 1)

        # we perform with clips since deformable attention only support max length of 36
        mask_embed_list = []
        for clip_id in range(0, T, crop_len):
            clip_mask_embed = mask_embed[:, clip_id: clip_id+crop_len]
            for mask_layer in self.mask_decoder:
                clip_mask_embed = mask_layer(
                    tgt=clip_mask_embed,
                    tgt_query_pos=query_pos[:, clip_id: clip_id+crop_len],
                    tgt_reference_points=query_ref_input[:, clip_id: clip_id+crop_len],
                    memory_text=text_dict["encoded_text"][clip_id: clip_id+crop_len],
                    text_attention_mask=~text_dict["text_token_mask"][clip_id: clip_id+crop_len],
                    memory=mask_features_flat[:, clip_id: clip_id+crop_len],
                    memory_key_padding_mask=mask_features_padding[clip_id: clip_id+crop_len],
                    memory_level_start_index=start_index,
                    memory_spatial_shapes=spatial_shapes,
                )
            mask_embed_list.append(clip_mask_embed.transpose(0, 1))
        mask_embed = torch.cat(mask_embed_list, dim=0)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        out = {"pred_logits": rearrange(logits, '(b t) q k -> b t q k', b=B),
               "pred_boxes": rearrange(coord, '(b t) q k -> b t q k', b=B),
               "pred_masks": rearrange(outputs_mask, '(b t) q h w -> b t q h w', b=B)
               }

        return out


    def _cxcywh_to_xyxy(self, boxes):
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)

    def _extract_roi_tokens(self, feat_map, ref_boxes):
        if (not self.use_lightweight_emd) or feat_map is None or ref_boxes is None:
            return None

        bt, c, h, w = feat_map.shape
        q = ref_boxes.shape[1]
        boxes = ref_boxes.detach().clamp(0.0, 1.0)
        boxes = self._cxcywh_to_xyxy(boxes)
        boxes[..., 0::2] = boxes[..., 0::2].clamp(0.0, 1.0)
        boxes[..., 1::2] = boxes[..., 1::2].clamp(0.0, 1.0)

        min_w = 1.0 / max(w, 1)
        min_h = 1.0 / max(h, 1)
        cxcy = 0.5 * (boxes[..., :2] + boxes[..., 2:])
        wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=torch.tensor([min_w, min_h], device=boxes.device))
        boxes = torch.cat((cxcy - 0.5 * wh, cxcy + 0.5 * wh), dim=-1)
        boxes[..., 0::2] = boxes[..., 0::2] * max(w - 1, 1)
        boxes[..., 1::2] = boxes[..., 1::2] * max(h - 1, 1)

        batch_idx = torch.arange(bt, device=feat_map.device, dtype=boxes.dtype).view(bt, 1, 1).expand(-1, q, 1)
        rois = torch.cat((batch_idx, boxes), dim=-1).reshape(-1, 5)
        pooled = torchvision.ops.roi_align(
            feat_map,
            rois,
            output_size=(self.emd_roi_size, self.emd_roi_size),
            spatial_scale=1.0,
            aligned=True,
        )
        pooled = pooled.flatten(2).transpose(1, 2)
        pooled = F.normalize(pooled, dim=-1)
        return pooled.view(bt, q, self.emd_roi_size * self.emd_roi_size, c)

    def _sinkhorn_emd(self, mem_tokens, cur_tokens):
        local_cost = 1.0 - torch.bmm(mem_tokens, cur_tokens.transpose(1, 2))
        kernel = torch.exp(-local_cost / max(self.emd_eps, 1e-4)).clamp_min(1e-8)

        num_pairs, n_mem, n_cur = local_cost.shape
        a = local_cost.new_full((num_pairs, n_mem), 1.0 / max(n_mem, 1))
        b = local_cost.new_full((num_pairs, n_cur), 1.0 / max(n_cur, 1))
        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(self.emd_sinkhorn_iters):
            kv = torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
            u = a / kv
            ktu = torch.bmm(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
            v = b / ktu

        transport = u.unsqueeze(-1) * kernel * v.unsqueeze(-2)
        return (transport * local_cost).sum(dim=(1, 2))

    def _compute_tracking_conf(self, feats, text_feat_t):
        feats_norm = F.normalize(feats, dim=-1)
        text_norm = F.normalize(text_feat_t, dim=-1)

        sent_feat = text_norm[:, 0]
        sent_score = F.cosine_similarity(feats_norm, sent_feat.unsqueeze(1), dim=-1)
        token_score = torch.einsum("bqd,bld->bql", feats_norm, text_norm).max(dim=-1)[0]

        sent_score = 0.5 * (sent_score + 1.0)
        token_score = 0.5 * (token_score + 1.0)
        beta = max(0.0, min(1.0, float(self.tracking_beta)))
        conf = beta * sent_score + (1.0 - beta) * token_score
        return conf.clamp_(0.0, 1.0).unsqueeze(-1)

    @torch.no_grad()
    def match_from_embds(self, mem, cur, mem_tokens=None, cur_tokens=None):
        # mem, cur: b, q, d
        mem = F.normalize(mem, dim=-1)
        cur = F.normalize(cur, dim=-1)
        cos_sim = torch.matmul(mem, cur.transpose(1, 2))
        cost_embd = 1 - cos_sim

        cost = cost_embd.clone()
        use_visual_tokens = (
            self.use_lightweight_emd
            and mem_tokens is not None
            and cur_tokens is not None
            and self.emd_weight > 0
        )

        if use_visual_tokens:
            bsz, num_queries, _, token_dim = mem_tokens.shape
            refine_topk = min(self.emd_candidate_topk, cur.shape[1])
            for b in range(bsz):
                cand_idx = torch.topk(cost_embd[b], k=refine_topk, dim=-1, largest=False).indices
                mem_idx = torch.arange(num_queries, device=mem.device).unsqueeze(1).expand(-1, refine_topk).reshape(-1)
                cur_idx = cand_idx.reshape(-1)

                mem_tok = mem_tokens[b, mem_idx].reshape(-1, mem_tokens.shape[2], token_dim)
                cur_tok = cur_tokens[b, cur_idx].reshape(-1, cur_tokens.shape[2], token_dim)
                emd_cost = self._sinkhorn_emd(mem_tok, cur_tok)

                mixed_cost = (1.0 - self.emd_weight) * cost_embd[b, mem_idx, cur_idx] + self.emd_weight * emd_cost
                cost[b, mem_idx, cur_idx] = mixed_cost

        C = cost.cpu()  # memory x current

        # permutation that makes current aligns to memory
        indices = np.stack([linear_sum_assignment(c)[1] for c in C], axis=0)

        return indices


def build_groundingdino(args, num_classes):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    hidden_dim = transformer.d_model

    pixel_decoder = FPNSpatialDecoder(hidden_dim, 2 * [hidden_dim] + [backbone.num_channels[0]], args.SegHead.mask_dim)

    mask_dec_layer = DeformableTransformerDecoderLayer(
        hidden_dim,
        n_levels=1,
        n_heads=args.SegHead.n_heads,
        n_points=args.SegHead.n_points,
        dropout=args.SegHead.dropout,
        use_text_cross_attention=args.SegHead.use_text_cross_attention,
    )
    mask_decoder = _get_clones(mask_dec_layer, args.SegHead.n_layers)

    trainable_key_list = ["feat_map", "LayerNorm", "backbone.0.norm0", ".bbox_embed", "lora",
                          "pixel_decoder", "mask_decoder", "classifier"]
    if not args.single_frame:
        trainable_key_list.append("temporal")
        if getattr(args, "use_diffusion_temporal", False):
            trainable_key_list.extend(["temporal_context", "box_diffusion"])

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
        num_classes=num_classes,
        dec_lora=args.dec_lora,
        enc_lora=args.enc_lora,
        lora_rank=args.lora_rank,
        pixel_decoder=pixel_decoder,
        mask_decoder=mask_decoder,
        trainable_key_list=trainable_key_list,
        full_tune=args.full_tune,
        temporal_layer=args.temporal_layer,
        tracking_alpha=args.tracking_alpha,
        tracking_beta=getattr(args, "tracking_beta", 0.5),
        dropout=args.dropout,
        use_diffusion_temporal=getattr(args, "use_diffusion_temporal", False),
        diffusion_layer_idx=getattr(args, "diffusion_layer_idx", -1),
        diffusion_latent_dim=getattr(args, "diffusion_latent_dim", 128),
        diffusion_total_steps=getattr(args, "diffusion_total_steps", getattr(args, "diffusion_train_steps", 50)),
        diffusion_infer_steps=getattr(args, "diffusion_infer_steps", 4),
        diffusion_noise_scale=getattr(args, "diffusion_noise_scale", 0.15),
        diffusion_refine_strength=getattr(args, "diffusion_refine_strength", 0.5),
        slot_topk=getattr(args, "slot_topk", 8),
        slot_diag_bias=getattr(args, "slot_diag_bias", 2.0),
        slot_iou_bias=getattr(args, "slot_iou_bias", 1.0),
        use_lightweight_emd=getattr(args, "use_lightweight_emd", False),
        emd_weight=getattr(args, "emd_weight", 0.25),
        emd_candidate_topk=getattr(args, "emd_candidate_topk", 5),
        emd_roi_size=getattr(args, "emd_roi_size", 3),
        emd_sinkhorn_iters=getattr(args, "emd_sinkhorn_iters", 5),
        emd_eps=getattr(args, "emd_eps", 0.07),
        use_memory_relation_distill=getattr(args, "use_memory_relation_distill", False),
        memory_relation_tau=getattr(args, "memory_relation_tau", 1.0),
    )
    pretrained_path = getattr(args, "pretrained_path", None)
    if pretrained_path and os.path.isfile(pretrained_path):
        print("load pretrained GroundingDINO from {} ...".format(pretrained_path))
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    elif pretrained_path:
        print(
            "Warning: pretrained GroundingDINO checkpoint not found at {}. "
            "Continuing; the inference checkpoint loaded by eval/inference_mevis.py must contain the required weights."
            .format(pretrained_path)
        )
    return model
