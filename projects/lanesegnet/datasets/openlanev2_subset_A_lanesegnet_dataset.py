#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import os
import random
import copy

import numpy as np
import torch
import mmcv
import cv2

import shapely
from shapely.geometry import LineString
from pyquaternion import Quaternion
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset

from .openlanev2_evaluate_custom import lanesegnet_evaluate
from ..core.lane.util import fix_pts_interpolate
from ..core.visualizer.lane_segment import draw_annotation_bev

@DATASETS.register_module()
class OpenLaneV2_subset_A_LaneSegNet_Dataset(Custom3DDataset):
    CAMS = ('ring_front_center', 'ring_front_left', 'ring_front_right',
            'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right')
    LANE_CLASSES = ('lane_segment', 'ped_crossing')
    TE_CLASSES = ('traffic_light', 'road_sign')
    TE_ATTR_CLASSES = ('unknown', 'red', 'green', 'yellow',
                       'go_straight', 'turn_left', 'turn_right',
                       'no_left_turn', 'no_right_turn', 'u_turn', 'no_u_turn',
                       'slight_left', 'slight_right')
    MAP_CHANGE_LOGS = [
        '75e8adad-50a6-3245-8726-5e612db3d165',
        '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
        'af170aac-8465-3d7b-82c5-64147e94af7d',
        '6e106cf8-f6dd-38f6-89c8-9be7a71e7275',
    ]

    def __init__(self,
                 data_root,
                 ann_file,
                 queue_length=1,
                 filter_empty_te=False,
                 filter_map_change=False,
                 points_num=10,
                 split='train',
                 **kwargs):
        self.filter_map_change = filter_map_change
        self.split = split
        super().__init__(data_root, ann_file, **kwargs)
        self.queue_length = queue_length
        self.filter_empty_te = filter_empty_te
        self.points_num = points_num
        self.LANE_CLASSES = self.CLASSES

    def load_annotations(self, ann_file):
        """Load annotation from a olv2 pkl file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: Annotation info from the json file.
        """
        data_infos = mmcv.load(ann_file, file_format='pkl')
        if isinstance(data_infos, dict):
            if self.filter_map_change and self.split == 'train':
                data_infos = [info for info in data_infos.values() if info['meta_data']['source_id'] not in self.MAP_CHANGE_LOGS]
            else:
                data_infos = list(data_infos.values())
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines.
        """
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['timestamp'],
            scene_token=info['segment_id']
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_name, cam_info in info['sensor'].items():
                image_path = cam_info['image_path']
                image_paths.append(os.path.join(self.data_root, image_path))

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['extrinsic']['rotation'])
                lidar2cam_t = cam_info['extrinsic']['translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t

                intrinsic = np.array(cam_info['intrinsic']['K'])
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)

                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and len(annos['gt_lane_labels_3d']) == 0:
                return None
            if self.filter_empty_te and len(annos['labels']) == 0:
                return None

        can_bus = np.zeros(18)
        rotation = Quaternion._from_matrix(np.array(info['pose']['rotation']))
        can_bus[:3] = info['pose']['translation']
        can_bus[3:7] = rotation
        patch_angle = rotation.yaw_pitch_roll[0] / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        input_dict['can_bus'] = can_bus
        input_dict['lidar2global_rotation'] = np.array(info['pose']['rotation'])

        return input_dict

    def ped2lane_segment(self, points):
        assert points.shape[0] == 5
        dir_vector = points[1] - points[0]
        dir = np.rad2deg(np.arctan2(dir_vector[1], dir_vector[0]))

        if dir < -45 or dir > 135:
            left_boundary = points[[2, 3]]
            right_boundary = points[[1, 0]]
        else:
            left_boundary = points[[0, 1]]
            right_boundary = points[[3, 2]]
        
        centerline = LineString((left_boundary + right_boundary) / 2)
        left_boundary = LineString(left_boundary)
        right_boundary = LineString(right_boundary)

        return centerline, left_boundary, right_boundary

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

        for area in ann_info['area']:
            if area['category'] == 1 and 'ped_crossing' in self.LANE_CLASSES:
                centerline, left_boundary, right_boundary = self.ped2lane_segment(area['points'])
                gt_lanes.append([centerline, left_boundary, right_boundary])
                gt_lane_labels_3d.append(1)
                gt_lane_left_type.append(0)
                gt_lane_right_type.append(0)

            elif area['category'] == 2 and 'road_boundary' in self.LANE_CLASSES:
                raise NotImplementedError

        topology_lsls = np.array(ann_info['topology_lsls'], dtype=np.float32)

        te_bboxes = np.array([np.array(sign['points'], dtype=np.float32).flatten() for sign in ann_info['traffic_element']])
        te_labels = np.array([sign['attribute'] for sign in ann_info['traffic_element']], dtype=np.int64)
        if len(te_bboxes) == 0:
            te_bboxes = np.zeros((0, 4), dtype=np.float32)
            te_labels = np.zeros((0, ), dtype=np.int64)

        topology_lste = np.array(ann_info['topology_lste'], dtype=np.float32)

        annos = dict(
            gt_lanes_3d = gt_lanes,
            gt_lane_labels_3d = gt_lane_labels_3d,
            gt_lane_adj = topology_lsls,
            bboxes = te_bboxes,
            labels = te_labels,
            gt_lane_lste_adj = topology_lste,
            gt_lane_left_type = gt_lane_left_type,
            gt_lane_right_type = gt_lane_right_type,
        )
        return annos

    def prepare_train_data(self, index):
        data_queue = []

        # temporal aug
        prev_indexs_list = list(range(index-self.queue_length, index))
        random.shuffle(prev_indexs_list)
        prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)


        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        sample_idx = input_dict['sample_idx']
        scene_token = input_dict['scene_token']
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or len(example['gt_lane_labels_3d']._data) == 0):
            return None
        if self.filter_empty_te and \
                (example is None or len(example['gt_labels']._data) == 0):
            return None

        data_queue.insert(0, example)
        for i in prev_indexs_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if input_dict['sample_idx'] < sample_idx and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                if self.filter_empty_gt and \
                    (example is None or len(example['gt_lane_labels_3d']._data) == 0):
                    return None
                sample_idx = input_dict['sample_idx']
            data_queue.insert(0, copy.deepcopy(example))
        return self.union2one(data_queue)

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if i == 0:
                metas_map[i]['prev_bev'] = False
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def format_openlanev2_gt(self):
        gt_dict = {}
        for idx in range(len(self.data_infos)):
            info = copy.deepcopy(self.data_infos[idx])
            key = (self.split, info['segment_id'], str(info['timestamp']))
            areas = []
            for area in info['annotation']['area']:
                if area['category'] == 1:
                    points = area['points']
                    left_boundary = fix_pts_interpolate(points[[0, 1]], 10)
                    right_boundary = fix_pts_interpolate(points[[2, 3]], 10)
                    area['points'] = np.concatenate([left_boundary, right_boundary], axis=0)
                    areas.append(area)
            info['annotation']['area'] = areas
            gt_dict[key] = info
        return gt_dict

    def format_results(self, results, jsonfile_prefix=None):
        pred_dict = {}
        pred_dict['method'] = 'LaneSegNet'
        pred_dict['authors'] = []
        pred_dict['e-mail'] = 'dummy'
        pred_dict['institution / company'] = 'OpenDriveLab'
        pred_dict['country / region'] = 'CN'
        pred_dict['results'] = {}
        for idx, result in enumerate(results):
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

                left_type_scores = lane_results[3][valid_indices]
                left_type_labels = lane_results[4][valid_indices]
                right_type_scores = lane_results[5][valid_indices]
                right_type_labels = lane_results[6][valid_indices]

                pred_area_index = []
                for pred_idx, (lane, score, label) in enumerate(zip(lanes, scores, labels)):
                    if label == 0:
                        points = lane.astype(np.float32)
                        pred_lane_segment = {}
                        pred_lane_segment['id'] = 20000 + pred_idx
                        pred_lane_segment['centerline'] = fix_pts_interpolate(points[:self.points_num], 10)
                        pred_lane_segment['left_laneline'] = fix_pts_interpolate(points[self.points_num:self.points_num * 2], 10)
                        pred_lane_segment['right_laneline'] = fix_pts_interpolate(points[self.points_num * 2:], 10)
                        pred_lane_segment['left_laneline_type'] = left_type_labels[pred_idx]
                        pred_lane_segment['right_laneline_type'] = right_type_labels[pred_idx]
                        pred_lane_segment['confidence'] = score.item()
                        pred_info['lane_segment'].append(pred_lane_segment)

                    elif label == 1:
                        points = lane.astype(np.float32)
                        pred_ped = {}
                        pred_ped['id'] = 20000 + pred_idx
                        pred_points = np.concatenate((fix_pts_interpolate(points[self.points_num:self.points_num * 2], 10),
                                                      fix_pts_interpolate(points[self.points_num * 2:][::-1], 10)), axis=0)
                        pred_ped['points'] = pred_points
                        pred_ped['category'] = label
                        pred_ped['confidence'] = score.item()
                        pred_info['area'].append(pred_ped)
                        pred_area_index.append(pred_idx)

                    elif label == 2:
                        raise NotImplementedError

            if result['bbox_results'] is not None:
                te_results = result['bbox_results']
                scores = te_results[1]
                te_valid_indices = np.argsort(-scores)
                tes = te_results[0][te_valid_indices]
                scores = scores[te_valid_indices]
                class_idxs = te_results[2][te_valid_indices]
                for pred_idx, (te, score, class_idx) in enumerate(zip(tes, scores, class_idxs)):
                    te_info = dict(
                        id = 20000 + pred_idx,
                        category = 1 if class_idx < 4 else 2,
                        attribute = class_idx,
                        points = te.reshape(2, 2).astype(np.float32),
                        confidence = score
                    )
                    pred_info['traffic_element'].append(te_info)

            if result['lsls_results'] is not None:
                topology_lsls_area = result['lsls_results'].astype(np.float32)[valid_indices][:, valid_indices]
                topology_lsls_area = np.delete(topology_lsls_area, pred_area_index, axis=0)
                topology_lsls = np.delete(topology_lsls_area, pred_area_index, axis=1)
                pred_info['topology_lsls'] = topology_lsls
            else:
                pred_info['topology_lsls'] = np.zeros((len(pred_info['lane_segment']), len(pred_info['lane_segment'])), dtype=np.float32)

            if result['lste_results'] is not None:
                topology_lste_area = result['lste_results'].astype(np.float32)[valid_indices]
                topology_lste = np.delete(topology_lste_area, pred_area_index, axis=0)
                pred_info['topology_lste'] = topology_lste
            else:
                pred_info['topology_lste'] = np.zeros((len(pred_info['lane_segment']), len(pred_info['traffic_element'])), dtype=np.float32)

            pred_dict['results'][key] = dict(predictions=pred_info)

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
        pred_dict = self.format_results(results)

        logger.info(f'Starting openlanev2 evaluate...')
        metric_results = lanesegnet_evaluate(gt_dict, pred_dict)
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


    @staticmethod
    def _render_surround_img(images):
        all_image = []
        img_height = images[1].shape[0]

        for idx in [1, 0, 2, 5, 3, 4, 6]:
            if idx  == 0:
                all_image.append(images[idx][356:1906, :])
                all_image.append(np.full((img_height, 20, 3), (255, 255, 255), dtype=np.uint8))
            elif idx == 6 or idx == 2:
                all_image.append(images[idx])
            else:
                all_image.append(images[idx])
                all_image.append(np.full((img_height, 20, 3), (255, 255, 255), dtype=np.uint8))

        surround_img_upper = None
        surround_img_upper = np.concatenate(all_image[:5], 1)

        surround_img_down = None
        surround_img_down = np.concatenate(all_image[5:], 1)
        scale = surround_img_upper.shape[1] / surround_img_down.shape[1]
        surround_img_down = cv2.resize(surround_img_down, None, fx=scale, fy=scale)

        divider = np.full((25, surround_img_down.shape[1], 3), (255, 255, 255), dtype=np.uint8)

        surround_img = np.concatenate((surround_img_upper, divider, surround_img_down), 0)
        surround_img = cv2.resize(surround_img, None, fx=0.5, fy=0.5)

        return surround_img

    @staticmethod
    def _filter_by_confidence(annotations, threshold=0.3):
        annotations = annotations.copy()
        areas = annotations['area']
        ls_mask = []
        lane_segments = []
        for ls in annotations['lane_segment']:
            if ls['confidence'] > threshold:
                ls_mask.append(True)
                lane_segments.append(ls)
            else:
                ls_mask.append(False)
        ls_mask = np.array(ls_mask, dtype=bool)
        areas = [area for area in annotations['area'] if area['confidence'] > threshold]

        traffic_elements = annotations['traffic_element']
        te_mask = []
        tes = []
        for te in traffic_elements:
            if te['confidence'] > threshold:
                te_mask.append(True)
                tes.append(te)
            else:
                te_mask.append(False)
        te_mask = np.array(te_mask, dtype=bool)

        annotations['lane_segment'] = lane_segments
        annotations['area'] = areas
        annotations['traffic_element'] = tes
        annotations['topology_lsls'] = annotations['topology_lsls'][ls_mask][:, ls_mask] > 0.5
        annotations['topology_lste'] = annotations['topology_lste'][ls_mask][:, te_mask] > 0.5
        return annotations
