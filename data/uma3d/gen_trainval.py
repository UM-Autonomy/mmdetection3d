"""
Script to generate training and validation data from uma3d,
    along with dataset + class summaries of timestamps + annotations,
    respectively.

Uses rule that all indices divisible by 4 are placed in the
    validation set; remainder are placed in the training set.
TODO: other rules can be implemented in the future.

NOTE: in the context of uma3d, "testing" could be interpreted by
    doing so in the water. As a result, there is no testing set.

Usage:
python gen_trainval.py
"""

import pickle
import json
import pandas as pd

from utils import get_meta_file


def gen_trainval():
    """Split into training and validation; write to files."""

    # Assign to train or val
    meta_data = get_meta_file()
    indices = [[], []]  # indices[0]: train; indices[1]: val.
    for index in meta_data["index_to_timestamp"]:
        i = 1 if int(index) % 4 == 0 else 0
        indices[i].append(index)

    index_to_dataset = {}
    for name, range_ in meta_data["dataset_to_range"].items():
        for index_int in range(int(range_["from"]), int(range_["to"]) + 1):
            index_to_dataset[f"{index_int:05}"] = name

    # Compute summaries
    counts = [[{}, {}], [{}, {}]]  # counts[0]: dataset. counts[1]: class.
    for i in range(2):
        for index in indices[i]:
            # Retrieve dataset value
            dataset = index_to_dataset[index]
            counts[0][i][dataset] = counts[0][i].get(dataset, 0) + 1

            # Retrieve annotation value
            ann_filename = "data/annotations/{}.pcd.json".format(index)
            with open(ann_filename, "r") as f:
                ann_contents = json.load(f)
            for ann in ann_contents:
                label = ann["class_label"]
                counts[1][i][label] = counts[1][i].get(label, 0) + 1

    # Create dataframes of summaries
    df = [None, None]
    for i in range(2):
        dataset_names = sorted(list(
            set(counts[i][0].keys()).union(set(counts[i][1].keys()))))
        data = {"train": [], "val": [], "TOTAL": []}
        for name in dataset_names:
            data["train"].append(counts[i][0].get(name, 0))
            data["val"].append(counts[i][1].get(name, 0))
            data["TOTAL"].append(data["train"][-1] + data["val"][-1])
        dataset_names.append("TOTAL")
        for col in data.keys():
            data[col].append(sum(data[col]))
        df[i] = pd.DataFrame(data, index=dataset_names)

    # Write to files
    with open("data/train_indices.pickle", "wb") as train_file:
        pickle.dump(indices[0], train_file)
    with open("data/val_indices.pickle", "wb") as val_file:
        pickle.dump(indices[1], val_file)
    df[0].to_csv("data/dataset_summary.csv", index=True)
    df[1].to_csv("data/class_summary.csv", index=True)


if __name__ == '__main__':
    gen_trainval()