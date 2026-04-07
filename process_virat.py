"""
Process VIRAT dataset into MID format.

VIRAT dataset structure:
- 8 scenes (after merging clips by camera):
    Scene 1: 000001 + 000002 + 000007 + 000008 (same camera, merged with frame offsets)
    Scene 2: 000200_00 + 000200_03 + 000200_05 (same camera)
    Scene 3: 000201_03 + 000201_04 + 000201_05 + 000201_06 + 000201_07 (same camera)
    Scene 4: 000204_00 + 000204_04 (same camera)
    Scene 5: 000205_01 (single clip)
    Scene 6: 040000_09 (single clip)
    Scene 7: 040001_01 (single clip)
    Scene 8: 040003_02 (single clip)

Processing steps:
1. Load geom.yml + types.yml for each clip
2. Filter Person tracks only
3. Merge clips within same camera (frame offset + global person ID)
4. Downsample 30fps -> 10fps (every 3rd frame)
5. Compute center_x, center_y from bbox
6. Standardize per scene (mean=0, std=1 per scene)
7. Compute velocity, acceleration
8. Output as .pkl (dill, same format as ETH-UCY)
"""

import os
import re
import numpy as np
import pandas as pd
import dill

from environment import Environment, Scene, Node, derivative_of

# VIRAT fps is 30, we downsample to 10 (every 3rd frame)
VIRAT_FPS = 30
TARGET_FPS = 10
FRAME_STEP = VIRAT_FPS // TARGET_FPS  # 3
DT = 1.0 / TARGET_FPS  # 0.1 seconds between frames at 10fps

# Standardization for pixel coordinates (per scene, mean=0, std=1)
STANDARDIZATION = {
    'PEDESTRIAN': {
        'position': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}},
        'velocity': {'x': {'mean': 0, 'std': 2}, 'y': {'mean': 0, 'std': 2}},
        'acceleration': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}}
    }
}

# 3 independent VIRAT scenes (clips from same camera merged)
# S1: 000001/000002/000007/000008 -> same camera
# S2: 040000/040001/040003 -> same camera
# S3: 000200/000201/000204/000205 -> same camera (all 0002xx clips)
VIRAT_SCENES = {
    'virat_s1': {
        'name': 'virat_s1',
        'clips': [
            ('VIRAT_S_000001', 'train'),
            ('VIRAT_S_000002', 'train'),
            ('VIRAT_S_000007', 'validate'),
            ('VIRAT_S_000008', 'validate'),
        ],
        'split_type': 'time',  # 80% train / 20% test by time
    },
    'virat_s2': {
        'name': 'virat_s2',
        'clips': [
            ('VIRAT_S_040003_02_000197_000552', 'validate'),
            ('VIRAT_S_040001_01_000448_001101', 'train'),  # file is in train/ dir
            ('VIRAT_S_040000_09_001194_001574', 'validate'),
        ],
        'split_type': 'time',  # 80% train / 20% test by time (ignore clip labels)
    },
    'virat_s3': {
        'name': 'virat_s3',
        'clips': [
            ('VIRAT_S_000200_03_000657_000899', 'train'),
            ('VIRAT_S_000200_05_001525_001575', 'train'),
            ('VIRAT_S_000200_00_000100_000171', 'validate'),
            ('VIRAT_S_000201_03_000640_000672', 'train'),
            ('VIRAT_S_000201_05_001081_001215', 'train'),
            ('VIRAT_S_000201_06_001354_001397', 'train'),
            ('VIRAT_S_000201_07_001485_001581', 'train'),
            ('VIRAT_S_000201_04_000682_000822', 'validate'),
            ('VIRAT_S_000204_00_000000_000109', 'train'),
            ('VIRAT_S_000204_04_000738_000977', 'train'),
            ('VIRAT_S_000205_01_000197_000342', 'validate'),
        ],
        'split_type': 'time',
    },
}


def maybe_makedirs(path_to_create):
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise


def load_geom_and_types(clip_name, split):
    """Load geom.yml and types.yml for a clip, return person track DataFrame."""
    ann_dir = 'VIRAT-dataset/annotations/' + split
    geom_file = os.path.join(ann_dir, clip_name + '.geom.yml')
    types_file = os.path.join(ann_dir, clip_name + '.types.yml')

    # Get person track IDs from types.yml
    person_ids = set()
    with open(types_file, 'r') as f:
        for line in f:
            if 'Person' in line and 'id1:' in line:
                m = re.search(r'id1:\s*(\d+)', line)
                if m:
                    person_ids.add(int(m.group(1)))

    # Load trajectory data from geom.yml
    rows = []
    with open(geom_file, 'r') as f:
        for line in f:
            if 'geom:' not in line:
                continue
            m_id = re.search(r'id1:\s*(\d+)', line)
            m_ts = re.search(r'ts0:\s*(\d+)', line)
            m_g0 = re.search(r'g0:\s*([\d\s]+)', line)
            if not (m_id and m_ts and m_g0):
                continue
            track_id = int(m_id.group(1))
            if track_id not in person_ids:
                continue
            frame_id = int(m_ts.group(1))
            parts = m_g0.group(1).strip().split()
            if len(parts) < 4:
                continue
            left, top, right, bottom = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            rows.append({
                'frame_id': frame_id,
                'track_id': track_id,
                'center_x': center_x,
                'center_y': center_y,
            })

    return pd.DataFrame(rows)


def merge_scene_clips(scene_info):
    """
    Load and merge all clips for a scene at 30fps, then downsample to 10fps.
    Returns a DataFrame with global frame_id, global unique track_id, and raw center_x/y.
    """
    all_rows = []
    global_person_id = 0
    global_frame_offset = 0  # cumulative at 30fps

    for clip_name, clip_split in scene_info['clips']:
        msg = '    Loading ' + clip_name + ' (' + clip_split + ')...'
        print(msg)
        df = load_geom_and_types(clip_name, clip_split)

        if df.empty:
            print('    Warning: ' + clip_name + ' has no person data')
            continue

        # Remap person IDs to globally unique within this scene
        old_to_new = {}
        for old_id in sorted(df['track_id'].unique()):
            old_to_new[old_id] = global_person_id
            global_person_id += 1
        df['track_id'] = df['track_id'].map(old_to_new)

        # Apply cumulative frame offset at full (30fps) resolution
        df['frame_id'] = df['frame_id'] + global_frame_offset

        # Update cumulative offset for next clip
        clip_max_frame = int(df['frame_id'].max())
        global_frame_offset = clip_max_frame + FRAME_STEP

        # Track which clip each row belongs to for train/test splitting
        df['clip_split'] = clip_split

        all_rows.append(df[['frame_id', 'track_id', 'center_x', 'center_y', 'clip_split']])

    if not all_rows:
        return pd.DataFrame()

    # Merge all clips
    merged = pd.concat(all_rows, ignore_index=True)
    merged.sort_values('frame_id', inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Downsample: 30fps -> 10fps (keep every 3rd frame)
    merged = merged[merged['frame_id'] % FRAME_STEP == 0].copy()
    merged.reset_index(drop=True, inplace=True)

    # Renumber frames to be continuous from 0
    merged['frame_id'] = merged['frame_id'] // FRAME_STEP

    return merged


def create_scene_from_df(df, scene_name, data_class, node_type_enum):
    """Create a MID Scene from a merged DataFrame."""
    if df.empty or len(df) == 0:
        return None

    max_timesteps = int(df['frame_id'].max()) + 1

    scene = Scene(
        timesteps=max_timesteps,
        dt=DT,
        name=scene_name + '_' + data_class,
        aug_func=None
    )

    data_columns = pd.MultiIndex.from_product(
        [['position', 'velocity', 'acceleration'], ['x', 'y']]
    )

    for node_id in df['track_id'].unique():
        node_df = df[df['track_id'] == node_id].sort_values('frame_id')
        if len(node_df) < 2:
            continue

        # Skip nodes with large frame gaps
        frame_diff = np.diff(node_df['frame_id'].values)
        if not np.all(frame_diff <= FRAME_STEP * 2):
            continue

        x = node_df['center_x'].values.astype(float)
        y = node_df['center_y'].values.astype(float)

        vx = derivative_of(x, DT)
        vy = derivative_of(y, DT)
        ax = derivative_of(vx, DT)
        ay = derivative_of(vy, DT)

        data_dict = {
            ('position', 'x'): x,
            ('position', 'y'): y,
            ('velocity', 'x'): vx,
            ('velocity', 'y'): vy,
            ('acceleration', 'x'): ax,
            ('acceleration', 'y'): ay,
        }

        node_data = pd.DataFrame(data_dict, columns=data_columns)
        node = Node(
            node_type=node_type_enum.PEDESTRIAN,
            node_id=str(node_id),
            data=node_data
        )
        node.first_timestep = int(node_df['frame_id'].iloc[0])
        scene.nodes.append(node)

    print('    Scene: ' + scene.name + ', nodes: ' + str(len(scene.nodes)) + ', timesteps: ' + str(max_timesteps))
    return scene


def process_scene(scene_key, scene_info):
    """Process a single VIRAT scene. Outputs separate train/test .pkl files per scene."""
    scene_name = scene_info['name']
    data_folder_name = 'processed_data_virat'
    attention_radius = {('PEDESTRIAN', 'PEDESTRIAN'): 5.0}

    print('\nProcessing ' + scene_key + ' (' + scene_name + ')...')

    merged_df = merge_scene_clips(scene_info)

    if merged_df.empty:
        print('    No data for ' + scene_key)
        return

    n_persons = merged_df['track_id'].nunique()
    n_rows = len(merged_df)
    max_frame = int(merged_df['frame_id'].max())
    print('    Merged: ' + str(n_rows) + ' rows, ' + str(n_persons) + ' persons, frames 0-' + str(max_frame))

    # Per-scene standardization (fit on ALL data)
    for col in ['center_x', 'center_y']:
        mean = merged_df[col].mean()
        std = merged_df[col].std()
        if std < 1e-6:
            std = 1.0
        merged_df[col] = (merged_df[col] - mean) / std
        print('    Standardized ' + col + ': mean=' + str(round(mean, 2)) + ', std=' + str(round(std, 2)) + ' -> mean=0, std=1')

    # Split train/test: 80/20 time split
    split_frame = max(1, int(max_frame * 0.8))
    train_df = merged_df[merged_df['frame_id'] <= split_frame].copy()
    test_df = merged_df[merged_df['frame_id'] > split_frame].copy()
    print('    Time split at frame ' + str(split_frame) + '/'+ str(max_frame) + ': train=' + str(len(train_df)) + ' rows, test=' + str(len(test_df)) + ' rows')

    # Drop the clip_split column before creating scenes
    if 'clip_split' in train_df.columns:
        train_df.drop(columns=['clip_split'], inplace=True)
    if 'clip_split' in test_df.columns:
        test_df.drop(columns=['clip_split'], inplace=True)

    # Create Environment (needed for NodeType enum)
    env_template = Environment(
        node_type_list=['PEDESTRIAN'],
        standardization=STANDARDIZATION
    )
    env_template.attention_radius = attention_radius

    # Create train/test scenes using the Environment's NodeType enum
    train_scene = create_scene_from_df(train_df, scene_name, 'train', env_template.NodeType)
    test_scene = create_scene_from_df(test_df, scene_name, 'test', env_template.NodeType)

    # Save per-scene pkl files
    if train_scene and len(train_scene.nodes) > 0:
        env_train = Environment(
            node_type_list=['PEDESTRIAN'],
            standardization=STANDARDIZATION
        )
        env_train.attention_radius = attention_radius
        env_train.scenes = [train_scene]
        out_path = os.path.join(data_folder_name, scene_name + '_train.pkl')
        with open(out_path, 'wb') as f:
            dill.dump(env_train, f, protocol=dill.HIGHEST_PROTOCOL)
        print('    Saved train: ' + out_path + ' (nodes=' + str(len(train_scene.nodes)) + ')')
    else:
        print('    Warning: no train nodes for this scene!')

    if test_scene and len(test_scene.nodes) > 0:
        env_test = Environment(
            node_type_list=['PEDESTRIAN'],
            standardization=STANDARDIZATION
        )
        env_test.attention_radius = attention_radius
        env_test.scenes = [test_scene]
        out_path = os.path.join(data_folder_name, scene_name + '_test.pkl')
        with open(out_path, 'wb') as f:
            dill.dump(env_test, f, protocol=dill.HIGHEST_PROTOCOL)
        print('    Saved test:  ' + out_path + ' (nodes=' + str(len(test_scene.nodes)) + ')')
    else:
        print('    Warning: no test nodes for this scene!')


def main():
    data_folder_name = 'processed_data_virat'
    maybe_makedirs(data_folder_name)

    print('Processing ' + str(len(VIRAT_SCENES)) + ' VIRAT scenes...')

    for scene_key, scene_info in VIRAT_SCENES.items():
        process_scene(scene_key, scene_info)

    # Cleanup old mixed pkl files (legacy virat_train/test.pkl without scene prefix)
    for old_file in ['virat_train.pkl', 'virat_test.pkl']:
        old_path = os.path.join(data_folder_name, old_file)
        if os.path.exists(old_path):
            print('    Removing old file: ' + old_path)
            os.remove(old_path)


if __name__ == '__main__':
    main()
