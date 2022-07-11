# ------------------------------------------------------------------------------------------------
# Ego3RT
# Copyright (c) 2022 ZhangVision Group. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/WangYueFt/detr3d
# ------------------------------------------------------------------------------------------------

import copy
import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class ScalePadMultiViewImage(object):
    """Scale and pad the multi-view image.
    Args:
        virtual_img_size (tuple, compulsory): input image size
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, virtual_img_size, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.virtual_img_size = virtual_img_size
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _scale_img(self, results):
        """Pad images according to ``self.size``."""
        for i in range(len(results['img'])):
            img_size = results['img'][i].shape
            cam_intrinsic = results['cam_intrinsic'][i]

            scale_x = self.virtual_img_size[0] / img_size[1]
            scale_y = self.virtual_img_size[1] / img_size[0]

            cam_intrinsic[0, 2] = cam_intrinsic[0, 2] * scale_x
            cam_intrinsic[1, 2] = cam_intrinsic[1, 2] * scale_y

            cam_intrinsic[0, 0] = cam_intrinsic[0, 0] * scale_x
            cam_intrinsic[1, 1] = cam_intrinsic[1, 1] * scale_y
            results['cam_intrinsic'][i] = cam_intrinsic
        resize_img = [np.round(mmcv.imresize(
                img, self.virtual_img_size)) for img in results['img']]

        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in resize_img]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in resize_img]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._scale_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str

