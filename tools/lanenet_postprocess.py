#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-30 上午10:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_postprocess.py
# @IDE: PyCharm Community Edition
"""
LaneNet model post process
"""
import os.path as ops
import math

import cv2
import glog as log
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


class _LaneFeat(object):
    """
    """
    def __init__(self, feat, coord, class_id=-1):
        """
        lane feat object
        :param feat: lane embeddng feats [feature_1, feature_2, ...]
        :param coord: lane coordinates [x, y]
        :param class_id: lane class id
        """
        self._feat = feat
        self._coord = coord
        self._class_id = class_id

    @property
    def feat(self):
        """
        :return:
        """
        return self._feat

    @feat.setter
    def feat(self, value):
        """
        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        if value.dtype != np.float32:
            value = np.array(value, dtype=np.float64)

        self._feat = value

    @property
    def coord(self):
        """
        :return:
        """
        return self._coord

    @coord.setter
    def coord(self, value):
        """
        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        if value.dtype != np.int32:
            value = np.array(value, dtype=np.int32)

        self._coord = value

    @property
    def class_id(self):
        """
        :return:
        """
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        """
        :param value:
        :return:
        """
        if not isinstance(value, np.int64):
            raise ValueError('Class id must be integer')

        self._class_id = value


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self):
        """
        """
        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([175, 0, 0]),
                           np.array([0, 175, 0]),
                           np.array([0, 0, 175]),
                           np.array([125, 0, 0]),
                           np.array([0, 125, 0]),
                           np.array([0, 0, 125]),
                           np.array([75, 0, 0]),
                           np.array([0, 75, 0]),
                           np.array([0, 0, 75]),
                           np.array([250, 250, 0]),
                           np.array([0, 250, 250]),
                           np.array([250, 0, 250]),
                           np.array([175, 175, 0]),
                           np.array([0, 175, 175]),
                           np.array([175, 0, 175]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([75, 75, 0]),
                           np.array([0, 75, 75]),
                           np.array([75, 0, 75]),
                           np.array([50, 200, 50]),
                           np.array([200, 50, 200]),
                           np.array([200, 200, 50]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100]),
                           np.array([100, 100, 50]),
                           np.array([200, 100, 100]),
                           np.array([100, 200, 100]),
                           np.array([100, 100, 200]),
                           ]

    def _embedding_feats_dbscan_cluster(self, embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        # db = DBSCAN(eps=0.35, min_samples=1000)
        db = DBSCAN(eps=0.35, min_samples=10)
        # print("embedding_image_feats",embedding_image_feats.shape)
        features = embedding_image_feats
        features = StandardScaler().fit_transform(embedding_image_feats)
        # print("features",features.shape)
        db.fit(features)
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)
        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 255) # 前景点坐标
        # print("idx",idx)
        lane_embedding_feats = instance_seg_ret[idx] # 前景点类别
        # print("lane_embedding_feats",lane_embedding_feats.shape)
        # print(np.expand_dims(lane_embedding_feats,axis=1))
        # print(lane_embedding_feats.transpose())
        idx_scale = np.vstack((idx[0] / 100.0, idx[1] / 100.0)).transpose()
        # print("idx_scale",idx_scale.shape)
        # print(idx_scale)
        lane_embedding_feats = np.hstack((np.expand_dims(lane_embedding_feats,axis=1), idx_scale))
        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result, seg_result):
        """
        :param binary_seg_result:
        :param instance_seg_result:
        :param seg_result:
        :return:
        """
        # print("seg_result",seg_result.shape)
        # get embedding feats and coords
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )
        # print("get_lane_embedding_feats_result",get_lane_embedding_feats_result['lane_embedding_feats'].shape) # 3165
        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
        )
        # print("dbscan_cluster_result",dbscan_cluster_result)
        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']
        coord = get_lane_embedding_feats_result['lane_coordinates']
        line_clouds = np.zeros(shape=[len(unique_labels)-1, binary_seg_result.shape[0], binary_seg_result.shape[1]], dtype=np.uint8)

        if db_labels is None:
            return None, None

        lane_coords = []

        for index, label in enumerate(unique_labels.tolist()):
            if label == -1:
                continue
            idx = np.where(db_labels == label)
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))
            mask[pix_coord_idx] = self._color_map[index]
            # line_clouds[label][pix_coord_idx] = np.sum(seg_result[1:, coord[idx][:, 1], coord[idx][:, 0]])
            line_clouds[label][pix_coord_idx] = 1
            lane_coords.append(coord[idx])

        return mask, lane_coords, line_clouds


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """
    def __init__(self):
        """
        :param ipm_remap_file_path: ipm generate file path
        """

        self._cluster = _LaneNetCluster()

        # remap_file_load_ret = self._load_remap_matrix()
        # self._remap_to_ipm_x = remap_file_load_ret['remap_to_ipm_x']
        # self._remap_to_ipm_y = remap_file_load_ret['remap_to_ipm_y']

        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

    def postprocess(self, binary_seg_result, instance_seg_result, seg_result,
                    min_area_threshold=100):
        """
        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :return:
        """
        # convert binary_seg_result
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8).squeeze()
        # apply image morphology operation to fill in the hold and reduce the small area
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=5) # 预处理
        # print("morphological_ret", np.nonzero(morphological_ret[0]))
        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret) # 预处理
        # print("connect_components_analysis_ret", np.nonzero(connect_components_analysis_ret))
        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]
        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        # apply embedding features cluster
        # print("morphological_ret", morphological_ret.shape,  "instance_seg_result", instance_seg_result.shape)
        mask_image, lane_coords, line_clouds = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result.squeeze(),
            seg_result=seg_result
        )
        # print("mask_image",mask_image, "lane_coords",lane_coords)
        return mask_image, lane_coords, line_clouds
        # if mask_image is None:
        #     return {
        #         'mask_image': None,
        #         'fit_params': None,
        #         'source_image': None,
        #     }

        # lane line fit
        # fit_params = []
        # src_lane_pts = []  # lane pts every single lane
        # for lane_index, coords in enumerate(lane_coords):
            # if data_source == 'tusimple':
            #     tmp_mask = np.zeros(shape=(720, 1280), dtype=np.uint8)
            #     tmp_mask[tuple((np.int_(coords[:, 1] * 720 / 256), np.int_(coords[:, 0] * 1280 / 512)))] = 255
            # else:
            #     raise ValueError('Wrong data source now only support tusimple')
            # tmp_ipm_mask = cv2.remap(
            #     tmp_mask,
            #     self._remap_to_ipm_x,
            #     self._remap_to_ipm_y,
            #     interpolation=cv2.INTER_NEAREST
            # )
            # nonzero_y = np.array(tmp_ipm_mask.nonzero()[0])
            # nonzero_x = np.array(tmp_ipm_mask.nonzero()[1])

            # fit_param = np.polyfit(nonzero_y, nonzero_x, 2)
            # fit_params.append(fit_param)

            # [ipm_image_height, ipm_image_width] = tmp_ipm_mask.shape
            # plot_y = np.linspace(10, ipm_image_height, ipm_image_height - 10)
            # fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
            # fit_x = fit_param[0] * plot_y ** 3 + fit_param[1] * plot_y ** 2 + fit_param[2] * plot_y + fit_param[3]

            # lane_pts = []
            # for index in range(0, plot_y.shape[0], 5):
            #     src_x = self._remap_to_ipm_x[
            #         int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
            #     if src_x <= 0:
            #         continue
            #     src_y = self._remap_to_ipm_y[
            #         int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
            #     src_y = src_y if src_y > 0 else 0
            #
            #     lane_pts.append([src_x, src_y])
            #
            # src_lane_pts.append(lane_pts)

        # tusimple test data sample point along y axis every 10 pixels
        # source_image_width = source_image.shape[1]
        # for index, single_lane_pts in enumerate(src_lane_pts):
        #     single_lane_pt_x = np.array(single_lane_pts, dtype=np.float32)[:, 0]
        #     single_lane_pt_y = np.array(single_lane_pts, dtype=np.float32)[:, 1]
        #     # if data_source == 'tusimple':
        #     #     start_plot_y = 240
        #     #     end_plot_y = 720
        #     # else:
        #     #     raise ValueError('Wrong data source now only support tusimple')
        #     step = int(math.floor((end_plot_y - start_plot_y) / 10))
        #     for plot_y in np.linspace(start_plot_y, end_plot_y, step):
        #         diff = single_lane_pt_y - plot_y
        #         fake_diff_bigger_than_zero = diff.copy()
        #         fake_diff_smaller_than_zero = diff.copy()
        #         fake_diff_bigger_than_zero[np.where(diff <= 0)] = float('inf')
        #         fake_diff_smaller_than_zero[np.where(diff > 0)] = float('-inf')
        #         idx_low = np.argmax(fake_diff_smaller_than_zero)
        #         idx_high = np.argmin(fake_diff_bigger_than_zero)
        #
        #         previous_src_pt_x = single_lane_pt_x[idx_low]
        #         previous_src_pt_y = single_lane_pt_y[idx_low]
        #         last_src_pt_x = single_lane_pt_x[idx_high]
        #         last_src_pt_y = single_lane_pt_y[idx_high]
        #
        #         if previous_src_pt_y < start_plot_y or last_src_pt_y < start_plot_y or \
        #                 fake_diff_smaller_than_zero[idx_low] == float('-inf') or \
        #                 fake_diff_bigger_than_zero[idx_high] == float('inf'):
        #             continue
        #
        #         interpolation_src_pt_x = (abs(previous_src_pt_y - plot_y) * previous_src_pt_x +
        #                                   abs(last_src_pt_y - plot_y) * last_src_pt_x) / \
        #                                  (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
        #         interpolation_src_pt_y = (abs(previous_src_pt_y - plot_y) * previous_src_pt_y +
        #                                   abs(last_src_pt_y - plot_y) * last_src_pt_y) / \
        #                                  (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
        #
        #         if interpolation_src_pt_x > source_image_width or interpolation_src_pt_x < 10:
        #             continue
        #
        #         lane_color = self._color_map[index].tolist()
        #         cv2.circle(source_image, (int(interpolation_src_pt_x),
        #                                   int(interpolation_src_pt_y)), 5, lane_color, -1)
        # ret = {
        #     'mask_image': mask_image,
        #     'fit_params': fit_params,
        #     'source_image': source_image,
        # }
        #
        # return ret