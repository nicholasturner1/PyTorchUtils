import os
import itertools
import random

import numpy as np
import h5py
import torch

from dataprovider3 import DataProvider, Dataset


class Sampler(torch.utils.data.IterableDataset):

    def __init__(self, datadir, spec, vols=[],
                 mode="train", aug=None, seed=None):
        assert mode in ["train", "val"], f"invalid mode: {mode}"

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

    def make_spec(self, patchsz):
        return dict(input=patchsz, seg=patchsz, 
                    seg_mask=patchsz, clefts=patchsz)

    def build(self, datadir, vols, patchsz, aug):
        """Builds an internal instance of the DataProvider class"""
        spec = self.make_spec(patchsz)
        print("Spec")
        print(spec)
        self.dsets = dict()
        self.edges = dict()
        dp = DataProvider(spec)
        for vol in vols:
            print("Vol: {}".format(vol))
            dset, edges = self.build_dataset(datadir, vol)
            # Storing to associate these volumes to edges during sampling
            self.dsets[vol] = dset
            self.edges[vol] = edges
            dp.add_dataset(dset)

        self.dset_names = list(self.dsets.keys())
        dp.set_augment(aug)
        dp.set_imgs(['input'])
        dp.set_segs(['seg','clefts'])
        self.dataprovider = dp

    def build_dataset(self, datadir, vol):
        img = read_h5(os.path.join(datadir, f"{vol}_img.h5"))
        seg = read_h5(os.path.join(datadir, f"{vol}_seg.h5"))
        clf = read_h5(os.path.join(datadir, f"{vol}_clf.h5"))

        msk_fname = os.path.join(datadir, f"{vol}_assign_mask.h5")
        if os.path.exists(msk_fname):
            msk = read_h5(msk_fname).astype(np.float32)
        else:
            msk = np.ones(clf.shape, dtype=np.float32)

        edg = read_edge_csv(os.path.join(datadir, f"{vol}_edges.csv"))

        remove_missing_clefts(clf, edg)

        # Preprocessing
        img = (img / 255.).astype("float32")

        # Create Dataset.
        dset = Dataset()
        dset.add_data(key='input', data=img)
        dset.add_data(key='clefts', data=clf)
        dset.add_data(key='seg', data=seg)
        dset.add_data(key='seg_mask', data=msk)
        dset.add_mask(key='clefts_mask', data=msk, loc=True, shifts=True)
        return dset, edg

    def sample(self):
        """
        Pulls a sample from the DataProvider class.
        One can add extra functionality here for post-processing, etc.
        """
        dset_name = random.choice(self.dset_names)
        while True:
            sample = self.dataprovider.random_sample(
                         dset=self.dsets[dset_name])
            if not self.sample_empty(sample):
                break

        edges = self.edges[dset_name]
        return self.postprocess(sample, edges)

    def sample_empty(self, sample, thresh=10):
        "Ensures that more than thresh voxels are labeled"
        return (sample["clefts"] != 0).sum() < thresh

    def postprocess(self, sample, edges):
        final_sample = dict()
        final_sample["input"], i = self.make_input(sample)
        final_sample["avan_label"] = self.make_output(sample, edges[i])
        final_sample["avan_mask"] = self.make_mask(sample)
        return final_sample

    def make_input(self, sample):
        """Binarize a cleft, zero others, stack w/ image"""
        #First id is always 0
        cleft_id = np.random.choice(np.unique(sample["clefts"])[1:])
        cleft_mask = (sample["clefts"] == cleft_id).astype("float32")
        return np.concatenate((sample["input"], cleft_mask), axis=0), cleft_id

    def make_output(self, sample, edge):
        """Label presynaptic and postsynaptic segment voxels"""
        seg = sample["seg"]
        seg_label = np.zeros((2,) + seg.shape[-3:], dtype=np.float32)

        seg_label[0,...] = seg == edge[0]
        seg_label[1,...] = seg == edge[1]

        return seg_label

    def make_mask(self, sample):
        return np.concatenate((sample["seg_mask"],
                               sample["seg_mask"]), axis=0)
 


def read_h5(fname, dset_name="/main"):
    assert os.path.isfile(fname), f"{fname} does not exist"
    with h5py.File(fname) as f:
        return f[dset_name][()]


def read_edge_csv(fname, delim=","):
    edges = dict()
    with open(fname) as f:
        for l in f.readlines():
            fields = l.strip().split(delim)
            edges[int(fields[0])] = (int(fields[1]),int(fields[2]))
    return edges


def remove_missing_clefts(clf, edg):
    all_keys = set(edg.keys())

    for i in np.unique(clf)[1:]:
        if i not in all_keys:
            clf[clf == i] = 0
