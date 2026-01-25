import h5py

from .base import BaseDataset

class H5Dataset(BaseDataset):
    """
    Dataset to open .h5 files
    """
    def __init__(self, path):
        self.file_path = path

        with h5py.File(self.file_path, "r") as f:
            print("Top-level keys:", list(f.keys()))
            print(f["labels"])
            print(f["depths"])
        

class NYUDepthDataset(H5Dataset):
    """
    https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
    """