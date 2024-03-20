#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines import LoadAnnotations3D


@PIPELINES.register_module()
class CustomLoadMultiViewImageFromFiles(object):

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = [mmcv.imread(name, self.color_type) for name in filename]
        if self.to_float32:
            img = [_.astype(np.float32) for _ in img]
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = [img_.shape for img_ in img]
        results['ori_shape'] = [img_.shape for img_ in img]
        # Set initial values for default meta_keys
        results['pad_shape'] = [img_.shape for img_ in img]
        results['crop_shape'] = [np.zeros(2) for img_ in img]
        results['scale_factor'] = [1.0 for img_ in img]
        num_channels = 1 if len(img[0].shape) < 3 else img[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations3DLaneSegment(LoadAnnotations3D):

    def __init__(self,
                 with_lane_3d=True,
                 with_lane_label_3d=True,
                 with_lane_adj=True,
                 with_lane_lste_adj=False,
                 with_lane_type=False,
                 with_bbox_3d=False,
                 with_label_3d=False,
                 with_area=False,
                 **kwargs):
        super().__init__(with_bbox_3d, with_label_3d, **kwargs)
        self.with_lane_3d = with_lane_3d
        self.with_lane_label_3d = with_lane_label_3d
        self.with_lane_type = with_lane_type
        self.with_lane_adj = with_lane_adj
        self.with_lane_lste_adj = with_lane_lste_adj
        self.with_area = with_area

    def _load_lanes_3d(self, results):
        results['gt_lanes_3d'] = results['ann_info']['gt_lanes_3d']
        if self.with_lane_label_3d:
            results['gt_lane_labels_3d'] = results['ann_info']['gt_lane_labels_3d']
        if self.with_lane_adj:
            results['gt_lane_adj'] = results['ann_info']['gt_lane_adj']
        if self.with_lane_lste_adj:
            results['gt_lane_lste_adj'] = results['ann_info']['gt_lane_lste_adj']
        if self.with_lane_type:
            results['gt_lane_left_type'] = results['ann_info']['gt_lane_left_type']
            results['gt_lane_right_type'] = results['ann_info']['gt_lane_right_type']
        if self.with_area:
            results['gt_areas_3d'] = results['ann_info']['gt_areas_3d']
            results['gt_area_labels_3d'] = results['ann_info']['gt_area_labels_3d']
        return results

    def __call__(self, results):
        results = super().__call__(results)
        if self.with_lane_3d:
            results = self._load_lanes_3d(results)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = super().__repr__()
        repr_str += f'{indent_str}with_lane_3d={self.with_lane_3d}, '
        repr_str += f'{indent_str}with_lane_lable_3d={self.with_lane_label_3d}, '
        repr_str += f'{indent_str}with_lane_adj={self.with_lane_adj}, '
        repr_str += f'{indent_str}with_lane_lste_adj={self.with_lane_lste_adj}, '
        repr_str += f'{indent_str}with_lane_type={self.with_lane_type}, '
        repr_str += f'{indent_str}with_area={self.with_area}, '
        return repr_str
