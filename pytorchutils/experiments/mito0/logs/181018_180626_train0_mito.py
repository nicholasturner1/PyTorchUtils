import os

import numpy as np
import h5py

from augmentor import Augment
from dataprovider3 import DataProvider, Dataset


class Sampler(object):

    def __init__(self, datadir, patchsz, vols=[], aug=None):
        datadir = os.path.expanduser(datadir)
        self.build(datadir, vols, patchsz, aug)

    def __call__(self):
        return self.dataprovider()

    def build(self, datadir, vols, patchsz, aug):
        spec = self.make_spec(patchsz)
        print("Spec")
        print(spec)
        dp = DataProvider(spec)
        for vol in vols:
            print("Vol: {}".format(vol))
            dp.add_dataset(self.build_dataset(datadir, vol))
        dp.set_augment(aug)
        dp.set_imgs(["input"])
        dp.set_segs(["seg"])
        self.dataprovider = dp

    def make_spec(self, patchsz):
        return dict(input = patchsz, 
                    mito_label = patchsz,
                    mito_mask = patchsz)

    def build_dataset(self, datadir, vol):
        img = read_h5(os.path.join(datadir, f"pinky_sept/{vol}_img.h5"))
        mit = read_h5(os.path.join(datadir, f"pinky_oct18/{vol}_pruned.h5")).astype("float32")

        #Preprocessing
        img = (img / 255.).astype("float32")
        mit[mit != 0] = 1
        msk = np.ones(mit.shape, dtype=np.uint8).astype("float32")

        # Create Dataset.
        dset = Dataset()
        dset.add_data(key='input', data=img)
        dset.add_data(key='mito_label', data=mit)
        dset.add_mask(key='mito_mask', data=msk, loc=False)
        return dset


def read_h5(fname, dset_name="/main"):
    assert os.path.isfile(fname)
    with h5py.File(fname) as f:
        return f[dset_name].value

