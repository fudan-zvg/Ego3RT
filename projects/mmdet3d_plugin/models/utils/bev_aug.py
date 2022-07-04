# ------------------------------------------------------------------------------------------------
# Ego3RT
# Copyright (c) 2022 ZhangVision Group. All Rights Reserved.
# ------------------------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import random


class BEVRandomRotateScale(nn.Module):
    def __init__(self,
                 prob=0.5,
                 max_rotate_degree=22.5,
                 scaling_ratio_range=(0.95, 1.05),
                 pc_range=[-50., -50., -5., 50., 50., 3.]
                 ):
        super(BEVRandomRotateScale, self).__init__()
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.prob = prob
        self.max_rotate_degree = max_rotate_degree
        self.scaling_ratio_range = scaling_ratio_range
        self.pc_range = pc_range

    def forward(self, feat, gt_bboxes_3d, gt_labels_3d):
        B, C, H, W = feat.shape
        prob = random.uniform(0, 1)
        if prob > self.prob or not self.training:
            return feat
        else:
            rotation_degree = random.uniform(-self.max_rotate_degree,
                                             self.max_rotate_degree)
            rotation_matrix = self._get_rotation_matrix(rotation_degree)
            scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                           self.scaling_ratio_range[1])
            scaling_matrix = self._get_scaling_matrix(scaling_ratio)

            # change gt_bboxes_3d
            for batch in range(len(gt_bboxes_3d)):
                gt_bboxes_3d[batch].scale(scaling_ratio)
                gt_bboxes_3d[batch].rotate(rotation_matrix)
                gt_bboxes_3d[batch], gt_labels_3d[batch] = self.filter_gt_bboxes(gt_bboxes_3d[batch],
                                                                                 gt_labels_3d[batch])
            rotation_matrix = torch.from_numpy(rotation_matrix).to(feat.device)
            scaling_matrix = torch.from_numpy(scaling_matrix).to(feat.device)
            warp_matrix = rotation_matrix @ scaling_matrix
            warp_matrix = warp_matrix[:2, :2]
            x = torch.arange(0, W, 1.)
            y = torch.arange(0, H, 1.)
            yy, xx = torch.meshgrid(y, x)
            grids = torch.stack([xx, yy], dim=-1)
            norm_grids = (2 * (grids + 0.5) / torch.tensor([W, H]) - 1)
            norm_grids = warp_matrix @ norm_grids.to(warp_matrix.device).unsqueeze(-1)
            norm_grids = norm_grids.squeeze(-1)
            norm_grids = norm_grids.unsqueeze(0).repeat(B, 1, 1, 1)
            rotated_feat = F.grid_sample(feat, norm_grids, padding_mode='zeros', align_corners=False)
            return rotated_feat

    def filter_gt_bboxes(self, gt_bboxes_3d, gt_labels_3d):
        bev_range = [self.pc_range[0], self.pc_range[1], self.pc_range[3], self.pc_range[4]]
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]
        return gt_bboxes_3d, gt_labels_3d

    @staticmethod
    def _get_rotation_matrix(rotate_degrees):
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.],
             [0., 0., 1.]],
            dtype=np.float32)
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio):
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.],
             [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix



class BEVRandomFlip(nn.Module):
    def __init__(self, prob=0.5, ):
        super(BEVRandomFlip, self).__init__()
        self.prob = prob

    def forward(self, feat, gt_bboxes_3d, gt_labels_3d=None):
        B, C, H, W = feat.shape
        prob = random.uniform(0, 1)
        if prob > self.prob or not self.training:
            return feat
        else:
            h_or_v = random.uniform(0, 1)
            if h_or_v > 0.5:
                flip_type = 'horizontal'
                flipped_feat = feat.flip(-2)
            else:
                flip_type = 'vertical'
                flipped_feat = feat.flip(-1)
            for batch in range(len(gt_bboxes_3d)):
                gt_bboxes_3d[batch].flip(flip_type)

        return flipped_feat

