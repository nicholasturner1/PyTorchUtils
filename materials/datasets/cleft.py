import os
import itertools

import numpy as np
import h5py
import torch

import dataprovider3


class TrainingDataset(torch.utils.data.IterableDataset):

    def __init__(self, datadir, spec, vols=[], aug=None,
                 seed=None, verbose=False, rank=None):

        super(TrainingDataset, self).__init__()
        self.seed = seed
        self.rank = rank
        self.build(datadir, vols, spec, aug)
        self.verbose = verbose

    def __iter__(self):
        """
        Sets RNG seed, and feeds an iterator to the DataLoader.
        Shouldn't need to modify this
        """
        if self.seed is None:
            seed_ = None
        elif self.rank is None:
            seed_ = self.seed
        else:
            seed_ = self.seed + self.rank

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and seed_ is not None:
            seed = seed_ * (worker_info.id + 1)
        else:
            seed = seed_

        np.random.seed(seed)

        return (self.sample() for _ in itertools.count())

    def build(self, datadir, vols, spec, aug):
        """Builds an internal instance of the DataProvider class"""
        print("Spec")
        print(spec)
        dp = dataprovider3.DataProvider(spec)
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

        # Preprocessing
        img = (img / 255.).astype("float32")
        clf[clf != 0] = 1

        # Create Dataset.
        dset = dataprovider3.Dataset()
        dset.add_data(key='input', data=img)
        dset.add_data(key='cleft_label', data=clf)
        return dset

    def sample(self):
        """
        Pulls a sample from the DataProvider class.
        One can add extra functionality here for post-processing, etc.
        """
        if self.verbose:
            print("starting sampling")
            sample = self.dataprovider()
            print("ending sampling")
            return sample
        else:
            return self.dataprovider()


def read_h5(fname, dset_name="/main"):
    assert os.path.isfile(fname)
    while True:
        try:
            with h5py.File(fname, 'r') as f:
                return f[dset_name][()]
        except OSError:
            pass
