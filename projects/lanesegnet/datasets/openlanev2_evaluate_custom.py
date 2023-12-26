#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
from tqdm import tqdm
from openlanev2.lanesegment.io import io
from openlanev2.lanesegment.evaluation.distance import (pairwise, area_distance,
                                                        lane_segment_distance, lane_segment_distance_c)
from openlanev2.lanesegment.evaluation.evaluate import (THRESHOLDS_LANESEG, THRESHOLDS_AREA,
                                                        _mAP_over_threshold, _mAP_topology_lsls)


def lanesegnet_evaluate(ground_truth, predictions, verbose=True):

    if isinstance(ground_truth, str):
        ground_truth = io.pickle_load(ground_truth)

    if predictions is None:
        preds = {}
        print('\nDummy evaluation on ground truth.\n')
    else:
        if isinstance(predictions, str):
            predictions = io.pickle_load(predictions)
        predictions = predictions['results']

    gts = {}
    preds = {}
    for token in ground_truth.keys():
        gts[token] = ground_truth[token]['annotation']
        if predictions is None:
            preds[token] = gts[token]
            for i, _ in enumerate(preds[token]['lane_segment']):
                preds[token]['lane_segment'][i]['confidence'] = np.float32(1)
            for i, _ in enumerate(preds[token]['area']):
                preds[token]['area'][i]['confidence'] = np.float32(1)
            for i, _ in enumerate(preds[token]['traffic_element']):
                preds[token]['traffic_element'][i]['confidence'] = np.float32(1)
        else:
            preds[token] = predictions[token]['predictions']

    assert set(gts.keys()) == set(preds.keys()), '#frame differs'

    """
        calculate distances between gts and preds    
    """

    distance_matrices = {
        'laneseg': {},
        'area': {},
    }

    for token in tqdm(gts.keys(), desc='calculating distances:', ncols=80, disable=not verbose):

        mask = pairwise(
            [gt for gt in gts[token]['lane_segment']],
            [pred for pred in preds[token]['lane_segment']],
            lane_segment_distance_c,
            relax=True,
        ) < THRESHOLDS_LANESEG[-1]

        distance_matrices['laneseg'][token] = pairwise(
            [gt for gt in gts[token]['lane_segment']],
            [pred for pred in preds[token]['lane_segment']],
            lane_segment_distance,
            mask=mask,
            relax=True,
        )

        distance_matrices['area'][token] = pairwise(
            [gt for gt in gts[token]['area']],
            [pred for pred in preds[token]['area']],
            area_distance,
        )

    """
        evaluate
    """

    metrics = {
        'mAP': 0
    }

    metrics['AP_ls'] = _mAP_over_threshold(
        gts=gts, 
        preds=preds, 
        distance_matrices=distance_matrices['laneseg'], 
        distance_thresholds=THRESHOLDS_LANESEG,
        object_type='lane_segment',
        filter=lambda _: True,
        inject=True, # save tp for eval on graph
    ).mean()

    metrics['AP_ped'] = _mAP_over_threshold(
        gts=gts, 
        preds=preds, 
        distance_matrices=distance_matrices['area'], 
        distance_thresholds=THRESHOLDS_AREA, 
        object_type='area',
        filter=lambda x: x['category'] == 1,
        inject=False,
    ).mean()

    metrics['TOP_lsls'] = _mAP_topology_lsls(gts, preds, THRESHOLDS_LANESEG)

    metrics['mAP'] = (metrics['AP_ls'] + metrics['AP_ped']) / 2

    return metrics
