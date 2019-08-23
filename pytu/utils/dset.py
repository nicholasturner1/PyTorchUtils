#!/usr/bin/env python
__doc__ = """

Dataset Wrappers for using PyTorch parallelism

Nicholas Turner <nturner@cs.princeton.edu>, 2018
Kisuk Lee <kisuklee@mit.edu>, 2018
"""

import numpy as np
from torch.utils import data


class Dataset(data.Dataset):
    "Wrapper class for asynchronous sampling functions"

    def __init__(self, sampler, size=10000000):
        super(Dataset, self).__init__()
        self.sampler = sampler
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        "Pulls a sample"
        return self.sampler()


class DataLoader(object):

    def __init__(self, sampler, batch_size,
                 num_workers=1, seed=0, size=10000000):

        sample_spec = SampleSpec(sampler().keys())
        dataset = Dataset(sampler, size)

        def worker_init_fn(worker_id):
            """DataProvider seed depends on numpy's"""
            seed = seed+worker_id
            np.random.seed(seed)

        dataloader = data.DataLoader(dataset, batch_size=batch_size,
                                     num_workers=num_workers, pin_memory=True,
                                     worker_init_fn=worker_init_fn)

        self.dataiter = iter(dataloader)
        self.inputs = spec.get_inputs()

    def __call__(self):
        sample = next(self.dataiter)
        for k in sample:
            is_input = k in self.inputs
            sample[k] = torch.from_numpy(sample[k]).cuda(non_blocking=is_input)
        return sample

