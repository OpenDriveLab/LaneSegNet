#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import torch
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.models.utils.transformer import inverse_sigmoid
from .util import normalize_3dlane

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class LaneSegmentHungarianAssigner3D(BaseAssigner):

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 mask_cost=dict(type='CrossEntropyLossCost', weight=1.0),
                 dice_cost=dict(type='DiceCost', weight=1.0),
                 normalize_gt=False,
                 pc_range=None):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.mask_cost = build_match_cost(mask_cost)
        self.dice_cost = build_match_cost(dice_cost)
        self.normalize_gt = normalize_gt
        self.pc_range = pc_range

    def assign(self,
               lanes_pred,
               mask_pred,
               cls_pred,
               gt_lanes,
               gt_mask,
               gt_labels):

        num_gts, num_bboxes = gt_lanes.size(0), lanes_pred.size(0)
        # 1. assign -1 by default
        assigned_gt_inds = lanes_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = lanes_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)

        if self.normalize_gt:
            normalized_gt_lanes = normalize_3dlane(gt_lanes, self.pc_range)
        else:
            normalized_gt_lanes = gt_lanes

        # regression L1 cost    
        reg_cost = self.reg_cost(lanes_pred, normalized_gt_lanes)

        # mask cost
        mask_cost = self.mask_cost(mask_pred, gt_mask)

        # dice cost
        dice_cost = self.dice_cost(mask_pred, gt_mask)

        # weighted sum of above two costs
        cost = cls_cost + reg_cost + mask_cost + dice_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            lanes_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            lanes_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0

        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
