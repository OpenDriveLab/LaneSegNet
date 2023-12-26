#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST


@MATCH_COST.register_module()
class LaneL1Cost(object):

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, lane_pred, gt_lanes):
        lane_cost = torch.cdist(lane_pred, gt_lanes, p=1)
        return lane_cost * self.weight
