#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
import torch
from shapely.geometry import LineString

def normalize_3dlane(lanes, pc_range):
    normalized_lanes = lanes.clone()
    normalized_lanes[..., 0::3] = (lanes[..., 0::3] - pc_range[0]) / (pc_range[3] - pc_range[0])
    normalized_lanes[..., 1::3] = (lanes[..., 1::3] - pc_range[1]) / (pc_range[4] - pc_range[1])
    normalized_lanes[..., 2::3] = (lanes[..., 2::3] - pc_range[2]) / (pc_range[5] - pc_range[2])
    normalized_lanes = torch.clamp(normalized_lanes, 0, 1)

    return normalized_lanes

def denormalize_3dlane(normalized_lanes, pc_range):
    lanes = normalized_lanes.clone()
    lanes[..., 0::3] = (normalized_lanes[..., 0::3] * (pc_range[3] - pc_range[0]) + pc_range[0])
    lanes[..., 1::3] = (normalized_lanes[..., 1::3] * (pc_range[4] - pc_range[1]) + pc_range[1])
    lanes[..., 2::3] = (normalized_lanes[..., 2::3] * (pc_range[5] - pc_range[2]) + pc_range[2])
    return lanes

def fix_pts_interpolate(lane, n_points):
    ls = LineString(lane)
    distances = np.linspace(0, ls.length, n_points)
    lane = np.array([ls.interpolate(distance).coords[0] for distance in distances])
    return lane
