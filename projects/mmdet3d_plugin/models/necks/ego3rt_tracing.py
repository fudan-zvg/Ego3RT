# ------------------------------------------------------------------------------------------------
# Ego3RT
# Copyright (c) 2022 ZhangVision Group. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/fundamentalvision/Deformable-DETR
# ------------------------------------------------------------------------------------------------

import copy
from typing import Optional, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_norm_layer
from projects.mmdet3d_plugin.ops.msda.modules import MSDeformAttn, MVMSAdaptiveAttn

from mmdet.models import NECKS
from ..utils.bev_aug import BEVRandomRotateScale, BEVRandomFlip
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from .utils import PositionEmbeddingSine, TransformerEncoderLayer, TransformerEncoder, BackTracingDecoderLayer, BackTracingDecoder, CustomBottleneck



@NECKS.register_module()
class Ego3rtTracing(nn.Module):
    def __init__(self, input_dims=[512, 1024, 2048], d_model=512, nhead=4, bev_shape=64, polar_size=(80, 256),
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", 
                 num_feature_levels=3, dec_n_points=3, enc_n_points=4,
                 topdown_cfg=None, topdown_layers=1, norm_cfg=None,
                 pc_range=None,
                 ):
        super().__init__()

        hidden_dim = d_model
        # Input Projection
        input_proj_list = []
        for i in range(num_feature_levels):
            for in_channels in input_dims:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
        self.input_proj = nn.ModuleList(input_proj_list)
        # decoder
        self.d_model = d_model
        self.nhead = nhead
        self.polar_size = polar_size

        num_queries = polar_size[0] * polar_size[1]
        self.num_queries = num_queries
        self.pc_range = pc_range
        self.bev_shape = bev_shape
        self.num_feature_levels = num_feature_levels
        self.dec_n_points = dec_n_points
        self.query_embeds = nn.Embedding(num_queries, hidden_dim*2)
        self.pos = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2)

        self.bev_aug = nn.ModuleList([BEVRandomRotateScale(pc_range=pc_range), BEVRandomFlip()])

        encoder_layer = TransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = BackTracingDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels * 6, nhead, dec_n_points, polar_size, norm_cfg)

        self.decoder = BackTracingDecoder(decoder_layer, num_decoder_layers, polar_size)

        self.bev_encoder = nn.Sequential(
                *[CustomBottleneck(**topdown_cfg.block_cfg) for _ in range(topdown_layers)])                                            

        # x forward, y right
        self.im_eyes_polar = self.polar2cart(polar_size)
        self.im_eyes_cart = self.cart2polar(polar_size, bev_shape)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MVMSAdaptiveAttn):
                m._reset_parameters()
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)


    def polar2cart(self, polar_bev_shape):
        # From polar coordinate to cartesian coordinate
        R, C = polar_bev_shape
        delta = 2 * math.pi / C
        base_point_r = torch.tile(torch.arange(1, R+1).unsqueeze(1) / R, (1, C))
        base_point_c = torch.tile(torch.arange(0, C) * delta, (R, 1))
        base_point_sin = base_point_c.sin()
        base_point_cos = base_point_c.cos()

        base_point_x = (base_point_r * base_point_cos + 1) / 2
        base_point_y = (base_point_r * base_point_sin + 1) / 2
        im_eyes_cart = torch.stack(
            [base_point_x,
             base_point_y,
             torch.ones(R, C) / 2,
             ], -1).view(-1, R * C, 3)

        return im_eyes_cart

    def cart2polar(self, polar_bev_shape, rec_bev_shape):
        # From cartesian coordinate to polar coordinate
        R, C = polar_bev_shape
        delta = 2 * math.pi / C
        base_point_x = torch.tile(torch.arange(0, rec_bev_shape) / (rec_bev_shape - 1), (rec_bev_shape, 1))
        base_point_x = (base_point_x - 0.5) * 2**0.5
        base_point_y = base_point_x.T.flip(0)
        base_point_r = torch.sqrt(torch.square(base_point_x) + torch.square(base_point_y))
        base_point_r = (base_point_r * (R + 1) - 1) / R
        tan = base_point_y / base_point_x
        ind_neg = base_point_x < 0.
        ind_pn = torch.logical_and(~ind_neg, base_point_y < 0.)
        ind_pp = torch.logical_and(~ind_neg, ~ind_pn)
        base_point_c = torch.atan(tan) * ind_pp + \
                       (torch.atan(tan) + 2 * math.pi) * ind_pn + \
                       (torch.atan(tan) + math.pi) * ind_neg
        base_point_c = base_point_c / delta / C
        base_point = torch.stack([base_point_c, base_point_r], dim=-1) * 2 - 1
        return base_point

    def back_tracing(self, im_eyes_polar, pc_range, img_metas):
        lidar2img = []
        B = im_eyes_polar.shape[0]
        for img_meta in img_metas:
            lidar2cam_rts = img_meta['lidar2cam']
            intrinsics = img_meta['cam_intrinsic']
            for i in range(len(lidar2cam_rts)):
                lidar2cam_rt = lidar2cam_rts[i]
                intrinsic = intrinsics[i]
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                lidar2img.append(lidar2img_rt)
        lidar2img = np.asarray(lidar2img)
        lidar2img = im_eyes_polar.new_tensor(lidar2img).unsqueeze(0).repeat(B,1,1,1)  # (B, N, 4, 4)
        im_eyes = im_eyes_polar.clone()  # (B, N, 3)
        new_pc_range = [prange * 2**0.5 for prange in pc_range]
        im_eyes[..., 0:1] = im_eyes[..., 0:1] * (new_pc_range[3] - new_pc_range[0]) + new_pc_range[0]
        im_eyes[..., 1:2] = im_eyes[..., 1:2] * (new_pc_range[4] - new_pc_range[1]) + new_pc_range[1]
        im_eyes[..., 2:3] = im_eyes[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        # im_eyes (B, num_queries, 4)
        im_eyes = torch.cat((im_eyes, torch.ones_like(im_eyes[..., :1])), -1)
        B, num_query = im_eyes.size()[:2]
        num_cam = lidar2img.size(1)
        im_eyes = im_eyes.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
        lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
        focus_cam = torch.matmul(lidar2img, im_eyes).squeeze(-1)
        eps = 1e-5
        mask = (focus_cam[..., 2:3] > eps)
        focus_cam = focus_cam[..., 0:2] / torch.maximum(
            focus_cam[..., 2:3], torch.ones_like(focus_cam[..., 2:3]) * eps)
        focus_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        focus_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
        focus_cam = (focus_cam - 0.5) * 2
        mask = (mask & (focus_cam[..., 0:1] > -1.0)
                & (focus_cam[..., 0:1] < 1.0)
                & (focus_cam[..., 1:2] > -1.0)
                & (focus_cam[..., 1:2] < 1.0))
        mask = torch.nan_to_num(mask)

        return focus_cam, mask.squeeze(-1)

    def generate_attn_mask(self, eye_masks):
        # Decide which view to look for each imaginary eye.
        eye_masks = eye_masks.transpose(1, 2)
        eye_masks[eye_masks.int().sum(-1) == 0] = True
        attn_mask = eye_masks.repeat_interleave(self.num_feature_levels, dim=2)
        attn_mask = attn_mask.repeat_interleave(self.dec_n_points, dim=2)
        attn_mask = ~attn_mask
        return attn_mask

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def transformer(self, srcs, masks, query_embeds, pos_embeds, focus_points, attn_mask):
        n_cam = len(srcs)
        memories = []
        spatial_shapes_dec = None
        level_start_index_dec = None
        mask_flattens_dec = []
        valid_ratios_dec = []
        for cam in range(n_cam):
            src_flatten = []
            mask_flatten = []
            lvl_pos_embed_flatten = []
            spatial_shapes = []
            for lvl, (src, mask, pos_embed) in enumerate(zip(srcs[cam], masks[cam], pos_embeds[cam])):
                bs, c, h, w = src.shape
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)
                src = src.flatten(2).transpose(1, 2)
                mask = mask.flatten(1)
                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)
                src_flatten.append(src)
                mask_flatten.append(mask)
            src_flatten = torch.cat(src_flatten, 1)
            mask_flatten = torch.cat(mask_flatten, 1)
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks[cam]], 1)

            spatial_shapes_dec = spatial_shapes
            level_start_index_dec = level_start_index
            mask_flattens_dec.append(mask_flatten)
            valid_ratios_dec.append(valid_ratios)
            # encoder
            memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                                  mask_flatten)
            memories.append(memory)

        memories = torch.cat(memories, dim=1)
        bs, _, c = memories.shape
        assert spatial_shapes_dec is not None and level_start_index_dec is not None
        level_start_index_dec = torch.cat((spatial_shapes_dec.new_zeros((1,)),
                                           spatial_shapes_dec.prod(1).repeat(n_cam).cumsum(0)[:-1]))
        spatial_shapes_dec = spatial_shapes_dec.repeat(n_cam, 1)
        mask_flattens_dec = torch.cat(mask_flattens_dec, dim=1)
        valid_ratios_dec = torch.cat(valid_ratios_dec, dim=1)
        query_embed, tgt = torch.split(query_embeds, c, dim=-1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        # decoder
        feat_3d = self.decoder(tgt, focus_points, memories,
                                      spatial_shapes_dec, level_start_index_dec,
                                      valid_ratios_dec, query_embed, mask_flattens_dec, attn_mask=attn_mask)

        return feat_3d

    def forward(self, features, img_metas, gt_bboxes_3d=None, gt_labels_3d=None):
        srcs = []
        masks = []
        pos = []
        batch_size, Ncams, input_img_channel, input_img_h, input_img_w = features[0].shape

        img_masks = features[0].new_zeros(
            (batch_size, input_img_h, input_img_w))

        for lvl, feature in enumerate(features):
            feature = feature.permute(1, 0, 2, 3, 4)  # [N, B, C, H, W]
            for feat in feature:
                srcs.append(self.input_proj[lvl](feat))
                pos.append(self.pos(feat).to(features[0].device))
                masks.append(F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0))

        srcs_shuffle = []
        masks_shuffle = []
        pos_shuffle = []

        for i in range(Ncams):
            srcs_shuffle.append(srcs[i::Ncams])
            masks_shuffle.append(masks[i::Ncams])
            pos_shuffle.append(pos[i::Ncams])

        focus_cam, eye_masks = self.back_tracing(im_eyes_polar=self.im_eyes_polar,
                                                                    pc_range=self.pc_range,
                                                                    img_metas=img_metas)
        focus_cam = (focus_cam + 1) / 2.
        focus_cam = focus_cam.transpose(1, 2).repeat_interleave(3, dim=2).to(features[0].device)
        attn_mask = self.generate_attn_mask(eye_masks).to(features[0].device)

        feat_3d = self.transformer(srcs=srcs_shuffle, masks=masks_shuffle,
                                   query_embeds=self.query_embeds.weight,
                                   pos_embeds=pos_shuffle, focus_points=focus_cam, attn_mask=attn_mask)

        bs, num_que, dim = feat_3d.shape
        feat_3d = feat_3d.permute(0, 2, 1).reshape(bs, dim, self.polar_size[0], self.polar_size[1])
        sampling_grid = self.im_eyes_cart.to(feat_3d.device).unsqueeze(0).repeat(bs, 1, 1, 1)
        feat_3d = F.grid_sample(feat_3d, sampling_grid, align_corners=False)
        feat_3d = feat_3d.flip(2)
        if self.training:
            for aug in self.bev_aug:
                feat_3d = aug(feat_3d, gt_bboxes_3d, gt_labels_3d)
        feat_3d = self.bev_encoder(feat_3d)
        return feat_3d


