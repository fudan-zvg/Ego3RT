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
from mmdet.models.backbones.resnet import Bottleneck
from projects.mmdet3d_plugin.ops.msda.modules import MSDeformAttn, MVMSAdaptiveAttn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DWFFN(nn.Module):
    def __init__(self, d_model=320, d_ffn=1024,norm_cfg=None):
        super(DWFFN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(d_model, d_ffn, kernel_size=1),
            build_norm_layer(norm_cfg, d_ffn)[1],
            nn.GELU()
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(d_ffn, d_ffn, kernel_size=3, stride=1, padding=1, groups=d_ffn, bias=True),
            build_norm_layer(norm_cfg, d_ffn)[1],
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(d_ffn, d_model, kernel_size=1),
            build_norm_layer(norm_cfg, d_model)[1],
        )

    def forward(self, x, polar_bev_shape):
        b, n, c = x.shape
        h, w = polar_bev_shape
        x = x.transpose(1,2).reshape(b, c, h, w)
        x = self.conv2(self.dwconv(self.conv1(x)))
        x = x.reshape(b, c, n).transpose(1,2)
        return x


class PolarAttention(nn.Module):
    # from https://github.com/lucidrains/axial-attention
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(PolarAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = nn.Conv1d(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.qkv_transform(x)
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = stacked_similarity.view(N * W, 3, self.groups, H, H).sum(dim=1)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = stacked_output.view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class BackTracingDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, polar_bev_shape=(80, 256), norm_cfg=None):
        super().__init__()

        self.d_model = d_model
        # MVAA
        self.cross_attn = MVMSAdaptiveAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Deformable attention
        self.dec_self_attn = MSDeformAttn(d_model, 1, 4, 4)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # polar attention
        self.dec_axial_attn = PolarAttention(d_model, d_model, n_heads, polar_bev_shape[0])
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # ffn
        self.polar_bev_shape = polar_bev_shape
        self.dwffn = DWFFN(d_model, d_ffn, norm_cfg)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, query_pos, focus_points, src, src_spatial_shapes, level_start_index,
                bev_ref, bev_spatial_shapes, bev_start_index,
                src_padding_mask=None, attn_mask=None):
        # Deformable attention
        tgt2 = self.norm1(tgt)
        tgt2 = self.dec_self_attn(self.with_pos_embed(tgt2, query_pos), bev_ref, tgt2, bev_spatial_shapes, bev_start_index)
        tgt = tgt + self.dropout2(tgt2)

        # polar attention
        tgt2 = self.norm3(tgt)
        b, n, c = tgt2.shape
        h, w = self.polar_bev_shape
        tgt2 = tgt2.transpose(1,2).reshape(b, c, h, w)
        tgt2 = self.dec_axial_attn(tgt2)
        tgt2 = tgt2.reshape(b, c, n).transpose(1, 2)
        tgt = tgt + self.dropout3(tgt2)

        # MVAA
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(self.with_pos_embed(tgt2, query_pos),
                               focus_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask, attn_mask)

        tgt = tgt + self.dropout1(tgt2)

        tgt = tgt + self.dwffn(tgt, self.polar_bev_shape)
        return tgt


class BackTracingDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, polar_bev_shape):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        d_model = decoder_layer.d_model
        bbox_embed = Mlp(d_model, d_model, 2)
        self.bbox_embed = _get_clones(bbox_embed, num_layers)
        self.class_embed = None
        self.polar_bev_shape = polar_bev_shape

    @staticmethod
    def get_bev_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, tgt, focus_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios, 
                query_pos=None, src_padding_mask=None, attn_mask=None):
        output = tgt
        bs, n, c = tgt.shape
        R, Ci = self.polar_bev_shape
        spatial_shape = (R, Ci)
        bev_spatial_shapes = torch.as_tensor(spatial_shape, dtype=torch.long, device=tgt.device).unsqueeze(0)
        bev_start_index = bev_spatial_shapes.new_zeros((1,))
        bev_valid_ratios = torch.ones(1, 1, 2, device=tgt.device).float()
        bev_ref = self.get_bev_reference_points(bev_spatial_shapes, bev_valid_ratios, device=tgt.device)

        for layer in self.layers:
            assert focus_points.shape[-1] == 2
            focus_points_input = focus_points * src_valid_ratios[:, None]
            output = layer(output, query_pos, focus_points_input, src, src_spatial_shapes, src_level_start_index,
                           bev_ref, bev_spatial_shapes, bev_start_index,
                           src_padding_mask, attn_mask)

        return output


class CustomBottleneck(Bottleneck):
    expansion = 1