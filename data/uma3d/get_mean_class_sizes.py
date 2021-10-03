"""
Compute mean bounding box dimensions for each class in the data.
This is a necessary input into VoteNet.

Usage:
python get_mean_class_sizes.py
"""

import glob
import json
import numpy as np

CLASSES = ["small_buoy", "tall_buoy", "dock"]


def get_mean_class_sizes():
    files = glob.glob("data/annotations/*.pcd.json")

    # Initialize 3D list
    # Index 0: classes
    # Index 1: dimensions (x, y, z)
    sizes = []
    for _ in range(len(CLASSES)): # class
        sizes_class = []
        for _ in range(3):  # xyz
            sizes_class.append([])
        sizes.append(sizes_class)

    for file in files:
        with open(file, "r") as f:
            ann_data = json.load(f)

        for ann in ann_data:
            class_index = CLASSES.index(ann["class_label"])
            for dim_index, dim_size in enumerate(ann["size"]):
                sizes[class_index][dim_index].append(dim_size)

    mean_sizes = np.zeros((len(CLASSES), 3))
    for i in range(len(CLASSES)):
        for j in range(3):
            mean_sizes[i][j] = np.array(sizes[i][j]).mean()

    print(mean_sizes)


if __name__ == '__main__':
    get_mean_class_sizes()
