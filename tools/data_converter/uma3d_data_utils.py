import mmcv
import numpy as np
import open3d as o3d
import json
from os import path as osp
from pathlib import Path
from concurrent import futures as futures

CLASSES = ['small_buoy', 'tall_buoy', 'dock']


class UMA3DInstance(object):

    def __init__(self, ann_data):
        self.translation = np.array(ann_data['translation'])
        self.size = np.array(ann_data['size'])
        self.rotation = np.array(ann_data['rotation'])
        self.classname = ann_data['class_label']
        self.label = CLASSES.index(self.classname)

        # TODO: figure out how rotating along all three
        #   dimensions works in MMDetection3D
        # For now, single rotation attribute is set as 0
        self.rotation_y = 0

        # TODO: is this assumption correct?
        # (That ann_data['translation'] is supposed to be the center
        #   of the bounding box as opposed to one of the corners)
        # Supervisely's documentation is outdated so the answer
        #   is not on there. But we need some way to verify
        self.box3d = np.concatenate([
            self.translation,
            self.size,
            [self.rotation_y]
        ])


class UMA3DData(object):
    """UMA3D data.
    
    Generate uma3d infos for uma3d_converter.
    
    Args:
        root_path (str): Root path of the raw data.
        split (str): Set split type of the data. Default: 'train'.
    """

    def __init__(self, root_path, split='train', pc_max_size=None):
        assert split in ['train', 'val']
        self.root_dir = root_path
        self.split = split
        self.pc_max_size = pc_max_size
        split_file = osp.join(self.root_dir, f'{self.split}_indices.pickle')
        mmcv.check_file_exist(split_file)
        self.sample_idx_list = mmcv.load(split_file)

    def __len__(self):
        return len(self.sample_idx_list)

    def get_point_cloud(self, index):
        """Get pointcloud data from pointclouds<index>.pcd"""
        pts_filename = osp.join(self.root_dir, 'pointclouds',
                                '{}.pcd'.format(index))
        return o3d.io.read_point_cloud(pts_filename)

    def get_annotation(self, index):
        """Get annotation data from annotations/<index>.pcd.json."""
        anns = json.load(osp.join(
            self.root_dir, 'annotations', '{}.pcd.json'.format(index)))
        return anns
    
    #def get_image(self, index):
    #    pass

    def get_infos(self, num_workers=4, has_label=True, sample_idx_list=None):
        """Get data infos.
        
        This method gets information from the raw data.

        Args:
            num_workers (int): Number of threads to be used. Default: 4.
            has_label (bool): Whether the data has label. Default: True.
        
        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            
            info = dict()
            pc_info = {'num_features': 3, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            info['pts_path_orig'] = osp.join('pointclouds',
                                             sample_idx + '.pcd')
            info['pts_path'] = osp.join('pointclouds_padded',
                                        sample_idx + '.pcd')

            # Pad, rewrite pcd data to shape (self.pc_max_size, 3)
            # (all input point clouds to the model must have the same dims)
            pcd = self.get_point_cloud(sample_idx)
            pcd_pts = np.asarray(pcd.points, dtype=np.float32)
            padding = np.full((self.pc_max_size - pcd_pts.shape[0], 3), 1000)
            pcd_pts = np.vstack((pcd_pts, padding))
            pcd.points = o3d.utility.Vector3dVector(pcd_pts)
            o3d.io.write_point_cloud(
                osp.join(self.root_dir, info['pts_path']), pcd)

            # Process annotation data
            if has_label:
                ann_path = osp.join(
                    self.root_dir, 'annotations', sample_idx + '.pcd.json')
                with open(ann_path, 'r') as ann_file:
                    ann_contents = json.load(ann_file)

                obj_list = [UMA3DInstance(ann) for ann in ann_contents
                                if ann['class_label'] in CLASSES]
                
                annotations = {}
                annotations['gt_num'] = len(obj_list)
                annotations['index'] = np.arange(
                    len(obj_list), dtype=np.int32)
                
                annotations['name'] = np.array([
                    obj.classname for obj in obj_list
                ])
                annotations['class'] = np.array([
                    obj.label for obj in obj_list
                ])

                annotations['location'] = np.concatenate([
                    obj.translation.reshape(1, 3) for obj in obj_list
                ], axis=0)
                annotations['dimensions'] = np.array([
                    obj.size for obj in obj_list
                ])
                annotations['rotation_y'] = np.array([
                    obj.rotation_y for obj in obj_list
                ])

                annotations['gt_boxes_upright_depth'] = np.stack([
                    obj.box3d for obj in obj_list
                ], axis=0)
                info['annos'] = annotations

            return info
        
        pc_padded_dir = osp.join(self.root_dir, 'pointclouds_padded')
        Path(pc_padded_dir).mkdir(parents=True, exist_ok=True)

        sample_idx_list = sample_idx_list if \
            sample_idx_list is not None else self.sample_idx_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_idx_list)
        return list(infos)