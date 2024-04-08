#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import os
import random
import copy
import tqdm

import numpy as np
import torch
import mmcv
import cv2

import shapely
from shapely.geometry import LineString
from mmdet.datasets import DATASETS

from openlanev2.lanesegment.evaluation import evaluate as openlanev2_evaluate
from .openlanev2_subset_A_lanesegnet_dataset import OpenLaneV2_subset_A_LaneSegNet_Dataset
from ..core.lane.util import fix_pts_interpolate
from ..core.visualizer.lane_segment import draw_annotation_bev


@DATASETS.register_module()
class OpenLaneV2_subset_A_MapElementBucket_Dataset(OpenLaneV2_subset_A_LaneSegNet_Dataset):

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information
        """
        info = self.data_infos[index]
        ann_info = info['annotation']

        gt_lanes = []
        gt_lane_labels_3d = []
        gt_lane_left_type = []
        gt_lane_right_type = []

        for idx, lane in enumerate(ann_info['lane_segment']):
            centerline = lane['centerline']
            LineString_lane = LineString(centerline)
            left_boundary = lane['left_laneline']
            LineString_left_boundary = LineString(left_boundary)
            right_boundary = lane['right_laneline']
            LineString_right_boundary = LineString(right_boundary)
            gt_lanes.append([LineString_lane, LineString_left_boundary, LineString_right_boundary])
            gt_lane_labels_3d.append(0)
            gt_lane_left_type.append(lane['left_laneline_type'])
            gt_lane_right_type.append(lane['right_laneline_type'])

        topology_lsls = np.array(ann_info['topology_lsls'], dtype=np.float32)

        te_bboxes = np.array([np.array(sign['points'], dtype=np.float32).flatten() for sign in ann_info['traffic_element']])
        te_labels = np.array([sign['attribute'] for sign in ann_info['traffic_element']], dtype=np.int64)
        if len(te_bboxes) == 0:
            te_bboxes = np.zeros((0, 4), dtype=np.float32)
            te_labels = np.zeros((0, ), dtype=np.int64)

        topology_lste = np.array(ann_info['topology_lste'], dtype=np.float32)

        gt_areas_3d = []
        gt_area_labels_3d = []
        for area in ann_info['area']:
            gt_areas_3d.append(fix_pts_interpolate(area['points'], 20))
            gt_area_labels_3d.append(area['category'] - 1)

        annos = dict(
            gt_lanes_3d = gt_lanes,
            gt_lane_labels_3d = gt_lane_labels_3d,
            gt_lane_adj = topology_lsls,
            bboxes = te_bboxes,
            labels = te_labels,
            gt_lane_lste_adj = topology_lste,
            gt_lane_left_type = gt_lane_left_type,
            gt_lane_right_type = gt_lane_right_type,
            gt_areas_3d = gt_areas_3d,
            gt_area_labels_3d = gt_area_labels_3d
        )
        return annos

    def format_openlanev2_gt(self):
        gt_dict = {}
        for idx in range(len(self.data_infos)):
            info = copy.deepcopy(self.data_infos[idx])
            key = (self.split, info['segment_id'], str(info['timestamp']))
            areas = []
            for area in info['annotation']['area']:
                    points = area['points']
                    if len(points) != 20:
                        points = fix_pts_interpolate(points, 20)
                    area['points'] = points
                    areas.append(area)
            info['annotation']['area'] = areas
            gt_dict[key] = info
        return gt_dict

    def format_results(self, results, out_dir=None, logger=None, **kwargs):
        if out_dir is not None:
            logger.info(f'Starting format results...')
            data_type = np.float16
        else:
            data_type = np.float32

        pred_dict = {}
        pred_dict['method'] = 'LaneSegNet'
        pred_dict['team'] = 'dummy'
        pred_dict['authors'] = []
        pred_dict['e-mail'] = 'dummy'
        pred_dict['institution / company'] = 'OpenDriveLab'
        pred_dict['country / region'] = 'CN'
        pred_dict['results'] = {}
        for idx, result in enumerate(tqdm.tqdm(results, ncols=80, desc='Formatting results')):
            info = self.data_infos[idx]
            key = (self.split, info['segment_id'], str(info['timestamp']))

            pred_info = dict(
                lane_segment = [],
                area = [],
                traffic_element = [],
                topology_lsls = None,
                topology_lste = None
            )

            if result['lane_results'] is not None:
                lane_results = result['lane_results']
                scores = lane_results[1]
                valid_indices = np.argsort(-scores)
                lanes = lane_results[0][valid_indices]
                labels = lane_results[2][valid_indices]
                scores = scores[valid_indices]
                lanes = lanes.reshape(-1, lanes.shape[-1] // 3, 3)

                # left_type_scores = lane_results[3][valid_indices]
                # right_type_scores = lane_results[5][valid_indices]
                left_type_labels = lane_results[4][valid_indices]
                right_type_labels = lane_results[6][valid_indices]

                for pred_idx, (lane, score, label) in enumerate(zip(lanes, scores, labels)):
                    pred_lane_segment = {}
                    pred_lane_segment['id'] = 20000 + pred_idx
                    pred_lane_segment['centerline'] = fix_pts_interpolate(lane[:self.points_num], 10).astype(data_type)
                    pred_lane_segment['left_laneline'] = fix_pts_interpolate(lane[self.points_num:self.points_num * 2], 10).astype(data_type)
                    pred_lane_segment['right_laneline'] = fix_pts_interpolate(lane[self.points_num * 2:], 10).astype(data_type)
                    pred_lane_segment['left_laneline_type'] = left_type_labels[pred_idx]
                    pred_lane_segment['right_laneline_type'] = right_type_labels[pred_idx]
                    pred_lane_segment['confidence'] = score.item()
                    pred_info['lane_segment'].append(pred_lane_segment)

            if result['area_results'] is not None:
                area_results = result['area_results']
                scores = area_results[1]
                area_valid_indices = np.argsort(-scores)
                areas = area_results[0][area_valid_indices]
                labels = area_results[2][area_valid_indices]
                scores = scores[area_valid_indices]
                for pred_idx, (area, score, label) in enumerate(zip(areas, scores, labels)):
                    pred_area = {}
                    pred_area['id'] = 30000 + pred_idx
                    pred_area['points'] = fix_pts_interpolate(area, 20).astype(data_type)
                    pred_area['category'] = label + 1
                    pred_area['confidence'] = score.item()
                    pred_info['area'].append(pred_area)

            if result['bbox_results'] is not None:
                te_results = result['bbox_results']
                scores = te_results[1]
                te_valid_indices = np.argsort(-scores)
                tes = te_results[0][te_valid_indices]
                scores = scores[te_valid_indices]
                class_idxs = te_results[2][te_valid_indices]
                for pred_idx, (te, score, class_idx) in enumerate(zip(tes, scores, class_idxs)):
                    te_info = dict(
                        id = 10000 + pred_idx,
                        category = 1 if class_idx < 4 else 2,
                        attribute = class_idx,
                        points = te.reshape(2, 2).astype(data_type),
                        confidence = score
                    )
                    pred_info['traffic_element'].append(te_info)

            if result['lsls_results'] is not None:
                pred_info['topology_lsls'] = result['lsls_results'].astype(np.float32)[valid_indices][:, valid_indices]
            else:
                pred_info['topology_lsls'] = np.zeros((len(pred_info['lane_segment']), len(pred_info['lane_segment'])), dtype=np.float32)

            if result['lste_results'] is not None:
                topology_lste = result['lste_results'].astype(np.float32)[valid_indices]
                pred_info['topology_lste'] = topology_lste
            else:
                pred_info['topology_lste'] = np.zeros((len(pred_info['lane_segment']), len(pred_info['traffic_element'])), dtype=np.float32)

            pred_dict['results'][key] = dict(predictions=pred_info)

        if out_dir is not None:
            logger.info(f'Saving results to {out_dir}...')
            mmcv.dump(pred_dict, os.path.join(out_dir, 'submission.pkl'))

        return pred_dict

    def evaluate(self, results, logger=None, show=False, out_dir=None, **kwargs):
        """Evaluation in Openlane-V2 subset_A dataset.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            show (bool): Whether to visualize the results.
            out_dir (str): Path of directory to save the results.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        if show:
            assert out_dir, 'Expect out_dir when show is set.'
            logger.info(f'Visualizing results at {out_dir}...')
            self.show(results, out_dir)
            logger.info(f'Visualize done.')

        logger.info(f'Starting format results...')
        gt_dict = self.format_openlanev2_gt()
        pred_dict = self.format_results(results, logger=logger)

        logger.info(f'Starting openlanev2 evaluate...')
        metric_results = openlanev2_evaluate(gt_dict, pred_dict)['OpenLane-V2 UniScore']
        return metric_results


    def show(self, results, out_dir, score_thr=0.3, show_num=20, **kwargs):
        """Show the results.

        Args:
            results (list[dict]): Testing results of the dataset.
            out_dir (str): Path of directory to save the results.
            score_thr (float): The threshold of score.
            show_num (int): The number of images to be shown.
        """
        for idx, result in enumerate(results):
            if idx % 5 != 0:
                continue
            if idx // 5 > show_num:
                break

            info = self.data_infos[idx]

            pred_result = self.format_results([result])
            pred_result = list(pred_result['results'].values())[0]['predictions']
            pred_result = self._filter_by_confidence(pred_result, score_thr)

            pv_imgs = []
            for cam_name, cam_info in info['sensor'].items():
                image_path = os.path.join(self.data_root, cam_info['image_path'])
                image_pv = mmcv.imread(image_path)
                pv_imgs.append(image_pv)

            surround_img = self._render_surround_img(pv_imgs)
            output_path = os.path.join(out_dir, f'{info["segment_id"]}/{info["timestamp"]}/surround.jpg')
            mmcv.imwrite(surround_img, output_path)

            conn_img_gt = draw_annotation_bev(info['annotation'])
            conn_img_pred = draw_annotation_bev(pred_result)
            divider = np.ones((conn_img_gt.shape[0], 7, 3), dtype=np.uint8) * 128
            conn_img = np.concatenate([conn_img_gt, divider, conn_img_pred], axis=1)[..., ::-1]

            output_path = os.path.join(out_dir, f'{info["segment_id"]}/{info["timestamp"]}/bev.jpg')
            mmcv.imwrite(conn_img, output_path)
