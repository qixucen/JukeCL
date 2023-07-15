import os
from .gtzan import GTZAN


def get_dataset(dataset_dir, subset, download=True):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    d = GTZAN(root=dataset_dir, download=download, subset=subset)
    return d
