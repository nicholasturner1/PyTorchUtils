import os
import itertools

import numpy as np
import h5py
import torch

from augmentor import Augment
from dataprovider3 import DataProvider, Dataset


class Sampler(torch.utils.data.IterableDataset):

    def __init__(self, datadir, spec, vols=[], mode="train", aug=None, seed=None):
        assert mode in ["train","val"], f"invalid mode: {mode}"

        super(Sampler, self).__init__()
        self.seed = seed
        self.build(datadir, vols, spec, aug)

    def __iter__(self):
        """
        Sets RNG seed, and feeds an iterator to the DataLoader.
        Shouldn't need to modify this
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single process
            np.random.seed(self.seed)
        else:
            if self.seed is not None:
                np.random.seed(self.seed * (worker_info.id + 1))
            else:
                np.random.seed(worker_info.id + 1)
        
        return (self.sample() for _ in itertools.count())

    def build(self, datadir, vols, spec, aug):
        """Builds an internal instance of the DataProvider class"""
        print("Spec")
        print(spec)
        dp = DataProvider(spec)
        for vol in vols:
            print("Vol: {}".format(vol))
            dp.add_dataset(self.build_dataset(datadir, vol))

        dp.set_augment(aug)
        dp.set_imgs(["input"])
        dp.set_segs(["cleft_label"])
        self.dataprovider = dp

    def build_dataset(self, datadir, vol):
        img = read_h5(os.path.join(datadir, vol + "_img.h5"))
        clf = read_h5(os.path.join(datadir, vol + "_syn.h5")).astype("float32")

        #Preprocessing
        img = (img / 255.).astype("float32")
        clf[clf != 0] = 1

        # Create Dataset.
        dset = Dataset()
        dset.add_data(key='input', data=img)
        dset.add_data(key='cleft_label', data=clf)
        return dset

    def sample(self):
        """
        Pulls a sample from the DataProvider class.
        One can add extra functionality here for post-processing, etc.
        """
        return self.dataprovider()


def read_h5(fname, dset_name="/main"):
    assert os.path.isfile(fname)
    with h5py.File(fname) as f:
        return f[dset_name][()]
