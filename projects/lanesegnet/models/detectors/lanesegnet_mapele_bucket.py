#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import copy
import numpy as np
import torch

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.models.builder import build_head
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from ...utils.builder import build_bev_constructor


@DETECTORS.register_module()
class LaneSegNetMapElementBucket(MVXTwoStageDetector):

    def __init__(self,
                 bev_constructor=None,
                 lane_head=None,
                 lsls_head=None,
                 bbox_head=None,
                 lste_head=None,
                 area_head=None,
                 video_test_mode=False,
                 **kwargs):

        super(LaneSegNetMapElementBucket, self).__init__(**kwargs)

        if bev_constructor is not None:
            self.bev_constructor = build_bev_constructor(bev_constructor)

        if lane_head is not None:
            lane_head.update(train_cfg=self.train_cfg.lane)
            self.pts_bbox_head = build_head(lane_head)
        else:
            self.pts_bbox_head = None
        
        if lsls_head is not None:
            self.lsls_head = build_head(lsls_head)
        else:
            self.lsls_head = None

        if bbox_head is not None:
            bbox_head.update(train_cfg=self.train_cfg.bbox)
            self.bbox_head = build_head(bbox_head)
        else:
            self.bbox_head = None

        if lste_head is not None:
            self.lste_head = build_head(lste_head)
        else:
            self.lste_head = None
 
        if area_head is not None:
            area_head.update(train_cfg=self.train_cfg.area)
            self.area_head = build_head(area_head)
        else:
            self.area_head = None

        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img',))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.bev_constructor(img_feats, img_metas, prev_bev)
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img',))
    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_lanes_3d=None,
                      gt_lane_labels_3d=None,
                      gt_lane_adj=None,
                      gt_lane_left_type=None,
                      gt_lane_right_type=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_lane_lste_adj=None,
                      gt_areas_3d=None,
                      gt_area_labels_3d=None,
                      gt_instance_masks=None,
                      gt_bboxes_ignore=None,
                      ):

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        if self.video_test_mode:
            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        else:
            prev_bev = None

        img_metas = [each[len_queue-1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bev_feats = self.bev_constructor(img_feats, img_metas, prev_bev)

        losses = dict()
        outs = self.pts_bbox_head(img_feats, bev_feats, img_metas)
        loss_inputs = [outs, gt_lanes_3d, gt_lane_labels_3d, gt_instance_masks, gt_lane_left_type, gt_lane_right_type]
        lane_losses, lane_assign_result = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        for loss in lane_losses:
            losses['lane_head.' + loss] = lane_losses[loss]
        lane_feats = outs['history_states']

        if self.lsls_head is not None:
            lsls_losses = self.lsls_head.forward_train(lane_feats, lane_assign_result, lane_feats, lane_assign_result, gt_lane_adj)
            for loss in lsls_losses:
                losses['lsls_head.' + loss] = lsls_losses[loss]

        if self.bbox_head is not None:
            front_view_img_feats = [lvl[:, 0] for lvl in img_feats]
            batch_input_shape = tuple(img[0, 0].size()[-2:])
            bbox_img_metas = []
            for img_meta in img_metas:
                bbox_img_metas.append(
                    dict(
                        batch_input_shape=batch_input_shape,
                        img_shape=img_meta['img_shape'][0],
                        scale_factor=img_meta['scale_factor'][0],
                        crop_shape=img_meta['crop_shape'][0]))
                img_meta['batch_input_shape'] = batch_input_shape

            te_losses = {}
            bbox_outs = self.bbox_head(front_view_img_feats, bbox_img_metas)
            bbox_losses, te_assign_result = self.bbox_head.loss(bbox_outs, gt_bboxes, gt_labels, bbox_img_metas, gt_bboxes_ignore)
            for loss in bbox_losses:
                te_losses['bbox_head.' + loss] = bbox_losses[loss]

            if self.lste_head is not None:
                te_feats = bbox_outs['history_states']
                lste_losses = self.lste_head.forward_train(lane_feats, lane_assign_result, te_feats, te_assign_result, gt_lane_lste_adj)
                for loss in lste_losses:
                    te_losses['lste_head.' + loss] = lste_losses[loss]
        
            num_gt_bboxes = sum([len(gt) for gt in gt_labels])
            if num_gt_bboxes == 0:
                for loss in te_losses:
                    te_losses[loss] *= 0

            losses.update(te_losses)
        
        if self.area_head is not None:
            outs = self.area_head(img_feats, bev_feats, img_metas)
            loss_inputs = [outs, gt_areas_3d, gt_area_labels_3d]
            area_losses, _ = self.area_head.loss(*loss_inputs, img_metas=img_metas)
            for loss in area_losses:
                losses['area_head.' + loss] = area_losses[loss]

        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0]['can_bus'][-1] = 0
            img_metas[0]['can_bus'][:3] = 0

        new_prev_bev, results_list = self.simple_test(
            img_metas, img, prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return results_list

    def simple_test_pts(self, x, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function"""
        batchsize = len(img_metas)

        bev_feats = self.bev_constructor(x, img_metas, prev_bev)
        outs = self.pts_bbox_head(x, bev_feats, img_metas)

        lane_results = self.pts_bbox_head.get_lanes(
            outs, img_metas, rescale=rescale)

        if self.lsls_head is not None:
            lane_feats = outs['history_states']
            lsls_results = self.lsls_head.get_relationship(lane_feats, lane_feats)
            lsls_results = [result.detach().cpu().numpy() for result in lsls_results]
        else:
            lsls_results = [None for _ in range(batchsize)]

        if self.bbox_head is not None:
            front_view_img_feats = [lvl[:, 0] for lvl in x]
            batch_input_shape = tuple(img[0, 0].size()[-2:])
            bbox_img_metas = []
            for img_meta in img_metas:
                bbox_img_metas.append(
                    dict(
                        batch_input_shape=batch_input_shape,
                        img_shape=img_meta['img_shape'][0],
                        scale_factor=img_meta['scale_factor'][0],
                        crop_shape=img_meta['crop_shape'][0]))
                img_meta['batch_input_shape'] = batch_input_shape
            bbox_outs = self.bbox_head(front_view_img_feats, bbox_img_metas)
            bbox_results = self.bbox_head.get_bboxes(bbox_outs, bbox_img_metas, rescale=rescale)
        else:
            bbox_results = [None for _ in range(batchsize)]
        
        if self.bbox_head is not None and self.lste_head is not None:
            te_feats = bbox_outs['history_states']
            lste_results = self.lste_head.get_relationship(lane_feats, te_feats)
            lste_results = [result.detach().cpu().numpy() for result in lste_results]
        else:
            lste_results = [None for _ in range(batchsize)]
        
        if self.area_head is not None:
            outs = self.area_head(x, bev_feats, img_metas)
            area_results = self.area_head.get_areas(
                outs, img_metas, rescale=rescale)

        return bev_feats, lane_results, lsls_results, bbox_results, lste_results, area_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        results_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, lane_results, lsls_results, bbox_results, lste_results, area_results = self.simple_test_pts(
            img_feats, img_metas, img, prev_bev, rescale=rescale)
        for result_dict, lane, lsls, bbox, lste, area in zip(
            results_list, lane_results, lsls_results, bbox_results, lste_results, area_results):
            result_dict['lane_results'] = lane
            result_dict['lsls_results'] = lsls
            result_dict['bbox_results'] = bbox
            result_dict['lste_results'] = lste
            result_dict['area_results'] = area
        return new_prev_bev, results_list
