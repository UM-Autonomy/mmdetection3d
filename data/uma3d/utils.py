"""
Files containing utility functions for dataset.
"""

import json


def get_meta_file():
    """Return data structure containing dataset metadata.
    
    If file exists, load file and return file contents.
    Otherwise, return data structure with attribute keys
        and empty values.
    """
    try:
        with open("data/meta.json", "r") as meta_file:
            meta_data = json.load(meta_file)
    except OSError:
        meta_data = {
            "next_index": 0,
            "num_datasets": 0,
            "dataset_to_range": {},
            "index_to_timestamp": {}
        }
    return meta_data


def save_meta_file(meta_data):
    """Save metadata as input parameter."""
    with open("data/meta.json", "w") as meta_file:
        json.dump(meta_data, meta_file, indent=4)