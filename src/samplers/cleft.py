import os

import h5py

from augmentor import Augment
from dataprovider3 import DataProvider, Dataset


class Sampler(object):

    def __init__(self, datadir, spec, vols=[], mode="train", aug=None):
        assert mode in ["train","val"], "invalid mode:{}".format(mode)
        datadir = os.path.expanduser(datadir)
        self.build(datadir, vols, spec, aug)

    def __call__(self):
        return self.dataprovider()

    def build(self, datadir, vols, spec, aug):
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


def read_h5(fname, dset_name="/main"):
    assert os.path.isfile(fname)
    with h5py.File(fname) as f:
        return f[dset_name].value

