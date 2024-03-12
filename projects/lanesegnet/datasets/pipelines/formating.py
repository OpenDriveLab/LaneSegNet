#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.pipelines import DefaultFormatBundle3D


@PIPELINES.register_module()
class CustomFormatBundle3DLane(DefaultFormatBundle3D):
    """Custom formatting bundle for 3D Lane.
    """

    def __init__(self, class_names, **kwargs):
        super(CustomFormatBundle3DLane, self).__init__(class_names, **kwargs)

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'gt_lanes_3d' in results:
            results['gt_lanes_3d'] = DC(
                to_tensor(results['gt_lanes_3d']))
        if 'gt_lane_labels_3d' in results:
            results['gt_lane_labels_3d'] = DC(
                to_tensor(results['gt_lane_labels_3d']))
        if 'gt_lane_adj' in results:
            results['gt_lane_adj'] = DC(
                to_tensor(results['gt_lane_adj']))
        if 'gt_lane_lste_adj' in results:
            results['gt_lane_lste_adj'] = DC(
                to_tensor(results['gt_lane_lste_adj']))
        if 'gt_lane_left_type' in results:
            results['gt_lane_left_type'] = DC(
                to_tensor(results['gt_lane_left_type']))  
        if 'gt_lane_right_type' in results:
            results['gt_lane_right_type'] = DC(
                to_tensor(results['gt_lane_right_type']))  
        if 'gt_instance_masks' in results:
            results['gt_instance_masks'] = DC(
                to_tensor(results['gt_instance_masks']))
        if 'gt_areas_3d' in results:
            results['gt_areas_3d'] = DC(
                to_tensor(results['gt_areas_3d']))
        if 'gt_area_labels_3d' in results:
            results['gt_area_labels_3d'] = DC(
                to_tensor(results['gt_area_labels_3d']))

        results = super(CustomFormatBundle3DLane, self).__call__(results)
        return results
