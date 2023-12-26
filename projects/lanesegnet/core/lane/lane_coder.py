#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
import torch
from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from .util import denormalize_3dlane


@BBOX_CODERS.register_module()
class LaneSegmentPseudoCoder(BaseBBoxCoder):

    def __init__(self, denormalize=False):
        self.denormalize = denormalize

    def encode(self):
        pass

    def decode_single(self, cls_scores, lane_preds, lane_left_type_scores, lane_right_type_scores):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            lane_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """

        cls_scores = cls_scores.sigmoid()
        scores, labels = cls_scores.max(-1)
        left_type_scores, left_type_labels = lane_left_type_scores.sigmoid().max(-1)
        right_type_scores, right_type_labels = lane_right_type_scores.sigmoid().max(-1)
        if self.denormalize:
            final_lane_preds = denormalize_3dlane(lane_preds, self.pc_range)
        else:
            final_lane_preds = lane_preds

        predictions_dict = {
            'lane3d': final_lane_preds.detach().cpu().numpy(),
            'scores': scores.detach().cpu().numpy(),
            'labels': labels.detach().cpu().numpy(),
            'left_type_scores': left_type_scores.detach().cpu().numpy(),
            'left_type_labels': left_type_labels.detach().cpu().numpy(),
            'right_type_scores': right_type_scores.detach().cpu().numpy(),
            'right_type_labels': right_type_labels.detach().cpu().numpy()
        }

        return predictions_dict

    def decode(self, preds_dicts):

        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_lanes_preds = preds_dicts['all_lanes_preds'][-1]
        all_lanes_left_types = preds_dicts['all_lanes_left_type'][-1]
        all_lanes_right_types = preds_dicts['all_lanes_right_type'][-1]
        
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(
                all_cls_scores[i], all_lanes_preds[i], all_lanes_left_types[i], all_lanes_right_types[i]))
        return predictions_list
