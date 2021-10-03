"""
Script to append https://supervise.ly/ data to uma3d dataset.

Args:
    -a: path to directory containing annotations in universal format,
        generated from https://github.com/EricWiener/universal-devkit
        (REQUIRED argument)
    -p: path to directory containing pointclouds in .pcd format
        (REQUIRED argument)
    -n: name of dataset to store in meta.json (REQUIRED argument)
    -i: path to directory containing images, if any (OPTIONAL argument)

Usage:
python append_dataset.py -a <ann_dir> -p <pcl_dir> -n <name>
/OR/
python append_dataset.py -a <ann_dir> -p <pcl_dir> -n <name> -i <img_dir>
"""

import argparse
import json
import os  # path.join, getcwd, rename
import shutil  # copy, copytree
from ntpath import basename
from glob import glob
from pathlib import Path

from utils import get_meta_file, save_meta_file

OUTPUT_PATHS = (os.path.join(os.getcwd(), "data", "annotations"),
    os.path.join(os.getcwd(), "data", "pointclouds"),
    os.path.join(os.getcwd(), "data", "related_images"))
OUTPUT_EXT = (".pcd.json", ".pcd")


def append_dataset(ann_dir, pcl_dir, img_dir, name):
    """Append a new Supervisely dataset to the uma3d dataset."""

    for path in OUTPUT_PATHS:
        Path(path).mkdir(parents=True, exist_ok=True)

    meta_data = get_meta_file()

    # Retrieve paths to annotation, pointcloud, and (if exists) images
    #   for all timestamps with at least one annotation
    nonempty_datapaths = []
    for ann_path in glob("{}/*.pcd.json".format(ann_dir)):
        # Get the filename with extensions (ex. 1212129.pcd.json)

        with open(ann_path, "r") as ann_file:
            annotation = json.load(ann_file)

        if len(annotation) == 0:
            continue
        timestamp = basename(ann_path).split(".")[-3]

        datapath_set = []
        datapath_set.append(timestamp)
        datapath_set.append(ann_path)
        datapath_set.append("{}/{}.pcd".format(pcl_dir, timestamp))

        if img_dir:
            datapath_set.append("{}/{}_pcd".format(img_dir, timestamp))

        nonempty_datapaths.append(datapath_set)

    from_index = meta_data["next_index"]
    from_index = f"{from_index:05}"
    
    # Currently, indices are hardcoded to be 5-digit numerical strings.
    # This leaves a range of "00000" to "99999", or 100000 different timestamps.
    # TODO: this can be changed if desired
    to_index = meta_data["next_index"] + len(nonempty_datapaths) - 1
    if to_index >= 100000:
        print("ABORTED: integer value of ending index would surpass five digits.")
        exit(1)
    to_index = f"{to_index:05}"

    # Perform copy operations for specified files
    indices_added = 0
    for i, datapath_set in enumerate(nonempty_datapaths):
        # Check if timestamp already exists in dataset. If so, skip.
        # (this is a safeguard against accidentally adding duplicate
        #  copies of the same dataset)
        if datapath_set[0] in meta_data["index_to_timestamp"].values():
            print("timestamp {} already exists in dataset. Skipping...".format(datapath_set[0]))
            continue

        index = i + meta_data["next_index"]
        index = f"{index:05}"
        indices_added += 1

        meta_data["index_to_timestamp"][index] = datapath_set[0]

        for j, path in enumerate(datapath_set[1:3]):  # ann + pcl

            shutil.copy(path, OUTPUT_PATHS[j])  # copy file

            filename = basename(path)
            copy_filename = os.path.join(OUTPUT_PATHS[j], filename)
            copy_filename_new = os.path.join(
                OUTPUT_PATHS[j], f"{index}{OUTPUT_EXT[j]}"
            )
            os.rename(copy_filename, copy_filename_new)  # rename file

        if len(datapath_set) == 4:  # img

            dirname = datapath_set[3].split("/")[-1]  # e.g. 1573334491937654504_pcd
            copy_dirname = os.path.join(OUTPUT_PATHS[2], dirname)
            shutil.copytree(datapath_set[3], copy_dirname)  # copy dir

            copy_dirname_new = os.path.join(OUTPUT_PATHS[2], index)
            os.rename(copy_dirname, copy_dirname_new)  # rename dir

    # Finish updating metadata variables, then save to file.
    if indices_added > 0:
        meta_data["dataset_to_range"][name] = {
            "from": from_index, "to": to_index
        }
        meta_data["num_datasets"] = len(meta_data["dataset_to_range"])
        meta_data["next_index"] += len(nonempty_datapaths)

        save_meta_file(meta_data)


if __name__ == "__main__":
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-a", "--ann_dir", type=str, default="",
        help="Directory path to the universal annotations"
    )
    ap.add_argument(
        "-p", "--pcl_dir", type=str, default="",
        help="Directory path to the pointclouds"
    )
    ap.add_argument(
        "-n", "--name", type=str, default="",
        help="Name of dataset to store in data/meta.json"
    )
    ap.add_argument(
        "-i", "--img_dir", type=str, default="",
        help="Directory path to the images"
    )
    args = vars(ap.parse_args())
    append_dataset(args["ann_dir"], args["pcl_dir"],
        args["img_dir"], args["name"]
    )
