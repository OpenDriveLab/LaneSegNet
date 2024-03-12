#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt

COLOR_DICT = {  # RGB [0, 1]
    'centerline': np.array([243, 90, 2]) / 255,
    'laneline': np.array([0, 32, 127]) / 255,
    'ped_crossing': np.array([255, 192, 0]) / 255,
    'road_boundary': np.array([220, 30, 0]) / 255,
}
LINE_PARAM = {
    0: {'color': COLOR_DICT['laneline'], 'alpha': 0.3, 'linestyle': ':'},       # none
    1: {'color': COLOR_DICT['laneline'], 'alpha': 0.75, 'linestyle': 'solid'},  # solid
    2: {'color': COLOR_DICT['laneline'], 'alpha': 0.75, 'linestyle': '--'},     # dashed
    'ped_crossing': {'color': COLOR_DICT['ped_crossing'], 'alpha': 1, 'linestyle': 'solid'},
    'road_boundary': {'color': COLOR_DICT['road_boundary'], 'alpha': 1, 'linestyle': 'solid'}
}
BEV_RANGE = [-50, 50, -25, 25]

def _draw_centerline(ax, lane_centerline):
    points = np.asarray(lane_centerline['points'])
    color = COLOR_DICT['centerline']
    # draw line
    ax.plot(points[:, 1], points[:, 0], color=color, alpha=1.0, linewidth=0.6)
    # draw start and end vertex
    ax.scatter(points[[0, -1], 1], points[[0, -1], 0], color=color, s=1)
    # draw arrow
    ax.annotate('', xy=(points[-1, 1], points[-1, 0]),
                xytext=(points[-2, 1], points[-2, 0]),
                arrowprops=dict(arrowstyle='->', lw=0.6, color=color))

def _draw_line(ax, line):
    points = np.asarray(line['points'])
    config = LINE_PARAM[line['linetype']]
    ax.plot(points[:, 1], points[:, 0], linewidth=0.6, **config)

def _draw_lane_segment(ax, lane_segment, with_centerline, with_laneline):
    if with_centerline:
        _draw_centerline(ax, {'points': lane_segment['centerline']})
    if with_laneline:
        _draw_line(ax, {'points': lane_segment['left_laneline'], 'linetype': lane_segment['left_laneline_type']})
        _draw_line(ax, {'points': lane_segment['right_laneline'], 'linetype': lane_segment['right_laneline_type']})

def _draw_area(ax, area):
    if area['category'] == 1:  # ped crossing with lane segment style.
        _draw_line(ax, {'points': area['points'], 'linetype': 'ped_crossing'})
    elif area['category'] == 2:  # road boundary
        _draw_line(ax, {'points': area['points'], 'linetype': 'road_boundary'})

def draw_annotation_bev(annotation, with_centerline=True, with_laneline=True, with_area=True):

    fig, ax = plt.figure(figsize=(2, 4), dpi=200), plt.gca()
    ax.set_aspect('equal')
    ax.set_ylim([BEV_RANGE[0], BEV_RANGE[1]])
    ax.set_xlim([BEV_RANGE[2], BEV_RANGE[3]])
    ax.invert_xaxis()
    ax.grid(False)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.tight_layout(pad=0.2)

    for lane_segment in annotation['lane_segment']:
        _draw_lane_segment(ax, lane_segment, with_centerline, with_laneline)
    if with_area:
        for area in annotation['area']:
            _draw_area(ax, area)

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data
