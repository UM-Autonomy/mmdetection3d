import numpy as np 
import open3d as o3d

from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadPointsFromPCD(LoadPointsFromFile):
    """Loads Points from PCD.
    
    Load points in .pcd format from file.

    Args: same as args of superclass.
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        super().__init__(coord_type,
                         load_dim,
                         use_dim,
                         shift_height,
                         use_color,
                         file_client_args)
    
    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        pcd = o3d.io.read_point_cloud(pts_filename)
        points = np.asarray(pcd.points, dtype=np.float32)

        return points
