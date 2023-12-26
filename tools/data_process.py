import numpy as np
from tqdm import tqdm
from shapely.geometry import LineString
from openlanev2.lanesegment.io import io

"""
This script is used to collect the data from the original OpenLane-V2 dataset.
The results will be saved in OpenLane-V2 folder.
The main difference between this script and the original one is that we don't interpolate the points for ped crossing and road bouadary.
"""

def _fix_pts_interpolate(curve, n_points):
    ls = LineString(curve)
    distances = np.linspace(0, ls.length, n_points)
    curve = np.array([ls.interpolate(distance).coords[0] for distance in distances], dtype=np.float32)
    return curve

def collect(root_path : str, data_dict : dict, collection : str, n_points : dict) -> None:

    data_list = [(split, segment_id, timestamp.split('.')[0]) \
        for split, segment_ids in data_dict.items() \
            for segment_id, timestamps in segment_ids.items() \
                for timestamp in timestamps
    ]
    meta = {}
    for split, segment_id, timestamp in tqdm(data_list, desc=f'collecting {collection}', ncols=100):
        identifier = (split, segment_id, timestamp)
        frame = io.json_load(f'{root_path}/{split}/{segment_id}/info/{timestamp}-ls.json')

        for k, v in frame['pose'].items():
            frame['pose'][k] = np.array(v, dtype=np.float64)
        for camera in frame['sensor'].keys():
            for para in ['intrinsic', 'extrinsic']:
                for k, v in frame['sensor'][camera][para].items():
                    frame['sensor'][camera][para][k] = np.array(v, dtype=np.float64)

        if 'annotation' not in frame:
            meta[identifier] = frame
            continue

        # NOTE: We don't interpolate the points for ped crossing and road bouadary.
        for i, area in enumerate(frame['annotation']['area']):
            frame['annotation']['area'][i]['points'] = np.array(area['points'], dtype=np.float32)
        for i, lane_segment in enumerate(frame['annotation']['lane_segment']):
            frame['annotation']['lane_segment'][i]['centerline'] = _fix_pts_interpolate(np.array(lane_segment['centerline']), n_points['centerline'])
            frame['annotation']['lane_segment'][i]['left_laneline'] = _fix_pts_interpolate(np.array(lane_segment['left_laneline']), n_points['left_laneline'])
            frame['annotation']['lane_segment'][i]['right_laneline'] = _fix_pts_interpolate(np.array(lane_segment['right_laneline']), n_points['right_laneline'])
        for i, traffic_element in enumerate(frame['annotation']['traffic_element']):
            frame['annotation']['traffic_element'][i]['points'] = np.array(traffic_element['points'], dtype=np.float32)
        frame['annotation']['topology_lsls'] = np.array(frame['annotation']['topology_lsls'], dtype=np.int8)
        frame['annotation']['topology_lste'] = np.array(frame['annotation']['topology_lste'], dtype=np.int8)
        meta[identifier] = frame

    io.pickle_dump(f'{root_path}/{collection}.pkl', meta)

if __name__ == '__main__':
    root_path = 'data/OpenLane-V2'
    file = f'{root_path}/data_dict_subset_A.json'
    subset = 'data_dict_subset_A'
    for split, segments in io.json_load(file).items():
        collect(
            root_path,
            {split: segments},
            f'{subset}_{split}_lanesegnet',
            n_points={
                'centerline': 10,
                'left_laneline': 10,
                'right_laneline': 10
            },
        )
