import mmcv
import os
import glob
import numpy as np
import open3d as o3d

from tools.data_converter.uma3d_data_utils import UMA3DData


def get_max_pc_size(data_path):
    """Get maximum size of point cloud in dataset"""
    pc_files = glob.glob(os.path.join(data_path, "pointclouds", "*.pcd"))
    max_size = 0
    for file in pc_files:
        pcd = o3d.io.read_point_cloud(file)
        points = np.asarray(pcd.points, dtype=np.float32)
        max_size = max(points.shape[0], max_size)

    return max_size


def create_uma3d_infos(data_path,
                       pkl_prefix='uma3d',
                       save_path=None,
                       workers=4):
    """Create info file of uma3d dataset.
    
    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data.
        pkl_prefix (str): Prefix of the pkl to be saved. Default: 'sunrgbd'.
        save_path (str): Path of the pkl to be saved. Default: None.
        workers (int): Number of threads to be used. Default: 4.
    """
    assert os.path.exists(data_path)
    save_path = data_path if save_path is None else save_path
    assert os.path.exists(save_path)

    # generate infos for detection task
    train_filename = os.path.join(save_path,
                                  f'{pkl_prefix}_infos_train.pkl')
    val_filename = os.path.join(save_path, f'{pkl_prefix}_infos_val.pkl')

    pc_max_size = get_max_pc_size(data_path)
    train_dataset = UMA3DData(root_path=data_path,
                              split='train', pc_max_size=pc_max_size)
    val_dataset = UMA3DData(root_path=data_path,
                            split='val', pc_max_size=pc_max_size)

    infos_train = train_dataset.get_infos(
        num_workers=workers, has_label=True)
    mmcv.dump(infos_train, train_filename, 'pkl')
    print(f'{pkl_prefix} info train file is saved to {train_filename}')

    infos_val = val_dataset.get_infos(
        num_workers=workers, has_label=True)
    mmcv.dump(infos_val, val_filename, 'pkl')
    print(f'{pkl_prefix} info val file is saved to {val_filename}')
