# UMA3D dataset

This file describes the structure and setup of the `mmdetection3d/data/uma3d` directory which contains the pointcloud and annotation data needed (and additional, currently unused image data) for UM::Autonomy's 3D Deep Learning Project.

## About

### Dataset Structure

The `data/` folder contains the following (after running all necessary scripts).

`data`  
├── `annotations`  
│  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── `XXXXX.pcd.json`  
├── `pointclouds`  
│  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  └── `XXXXX.pcd`  
├── `related_images`  
│  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── `XXXXX`  
│  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     ├── `default_webcam`  
│    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   └── `default_webcam.json`  
├── &nbsp;`meta.json`  
├── &nbsp;`train_indices.pickle`  
├── &nbsp;`val_indices.pickle`  
├── &nbsp;`class_summary.csv`  
├── &nbsp;`dataset_summary.csv`  
├── &nbsp;`uma3d_infos_train.pkl`  
└── &nbsp;`uma3d_infos_val.pkl`  

### Dataset Contents

* `annotations/` contains annotation data after being run through `supervisely_3d_to_universal.py` from [this repository](https://github.com/EricWiener/universal-devkit).
* `pointclouds/` contains raw pointcloud data.
* `related_images/` contains image data, each image paired with a meta file.
* `meta.json` contains metadata for the dataset.
* `train_indices.pickle` and `val_indices.pickle` contain lists of indices for the training and validation sets, respectively.
* `class_summary.csv` and `dataset_summary.csv` contains a table breaking down the class and dataset distribution, respectively, for each of the training and validation sets.
* `uma3d_infos_train.pkl` and `uma3d_info_val.pkl` contain lists of re-formatted annotation data, and serve as the final input into the model pipeline.

### Scripts

Directly under the `uma3d` folder exist the following scripts (check the docstring of each file for usage instructions):

* `append_dataset.py`: append Supervisely dataset to uma3d dataset.
* `gen_trainval.py`: generate training and validation index pickle files, along with class and dataset summaries.
* `get_mean_class_sizes.py`: compute and print mean bounding box sizes for each class across each xyz dimension. Necessary information to run VoteNet.

In addition, under the `tools` directory contains a `create_data.py` script, which relies on `uma3d_converter.py` and `uma3d_data_utils.py`. This will preprocess both the training and validation data into two `pkl` files, which will serve as the ultimate input for the training script.

## Download/Setup Data

Data can be found on Supervisely (https://supervise.ly/), the data annotation site that UMA's deep learning projects currently use. After logging in, perform the following steps:
1. Select a dataset.
2. Underneath "Download as", select ".json + pointclouds".
3. Run `.json` annotations through `supervisely_3d_to_universal.py` from [this repository](https://github.com/EricWiener/universal-devkit).
4. For each Supervisely dataset, run `append_dataset.py`, specifying dataset name and directories containing annotations, pointclouds, and (optionally) image data.
5. Run `gen_trainval.py`. (NOTE: this needs to be rerun after any part of the uma3d dataset is added, removed, or modified, including after running Step 4)
6. Navigate back to the root folder of the repo. From there, run the command `python tools/create_data.py uma3d --root-path ./data/uma3d/data --out-dir ./data/uma3d/data --extra-tag uma3d`. (NOTE: this needs to be rerun anytime after Step 5 is rerun)
7. The data should now be ready for training and evaluation.

## Future Implementations
(as of September 5, 2021)
* Script to cleanly remove Supervisely datasets from uma3d dataset.
    * All files from that dataset should be removed from `annotations/`, `pointclouds/`, and `related_images/`, and any trace of them should be removed from `meta.json`.
* Script to compress indices in uma3d dataset, in case datasets are removed.  
    * Example: If the dataset with indices "00100" to "00199" is removed, `00200.pcd` should be renamed to `00100.pcd`, `00200.pcd.json` should be renamed to `00100.pcd.json`, and the `00200/` image directory (if exists) should be renamed to `00100/`. For indices "00201", "00202", etc., the same operations should be applied to all relevant files. The `meta.json` file should also be modified.