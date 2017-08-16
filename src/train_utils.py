#!/usr/bin/env python
__doc__ = """

Training Utilities

Nicholas Turner <nturner@cs.princeton.edu>, 2017
Jingpeng Wu <jingpeng.wu@gmail.com>,
Kisuk Lee <kisuklee@mit.edu>, 2015-2017
"""

import torch
from torch.autograd import Variable

import os
import h5py
# Async sampling.
from Queue import Queue
import threading

#===========================================


def sampler_daemon(sampler, q):
    " Function run by the thread "
    while True:
        if not q.full():
            q.put(sampler(imgs=["input"]))
        else:
            q.join()

class AsyncSampler(object):
    " Wrapper class for asynchronous sampling functions "

    def __init__(self, sampler, queue_size=10):

        self.q = Queue(queue_size)
        self.t = threading.Thread(target=sampler_daemon, args=(sampler, self.q))
        self.t.daemon = True
        self.t.start()

    def get(self):
        " Pulls a sample from the queue "
        res = self.q.get()
        self.q.task_done()
        return res

#===========================================

class SampleSpec(object):
    """
    Class specifying the purpose of each volume within a dataset sample

    The three possible classes are: input, label, mask
    Links each label to a mask if it exists
    """

    def __init__(self, sample_keys):

        #Assigning keys to purposes
        (self._inputs,
        self._labels,
        self._masks ) = self._parse_sample_keys(sample_keys)

        self._mask_lookup = self._create_mask_lookup()

    def get_inputs(self):
        return self._inputs

    def get_labels(self):
        return self._labels

    def get_masks(self):
        return self._masks

    def has_mask(self, label_name):
        " Returns whether a label has a matched mask "
        assert self._mask_lookup.has_key(label_name), "{} not in lookup".format(label_name)
        return self._mask_lookup[label_name] is not None

    def get_mask_name(self, label_name):
        assert self._mask_lookup.has_key(label_name)
        return self._mask_lookup[label_name]

    def get_mask_index(self, label_name):
        assert self._mask_lookup.has_key(label_name)
        return self._masks.index( self._mask_lookup[label_name] )

    #================================================
    # Non-interface functions
    #================================================

    def _parse_sample_keys(self, keys):
        """
        Assigns keys to purposes within (inputs, labels, masks) by
        inspecting their names

        All names containing _mask  -> masks
        All names containing _label -> labels
        Else                        -> inputs
        """
        inputs = []; labels = []; masks = []

        for k in keys:
            if   "_mask"  in k:
                masks.append(k)
            elif "_label" in k:
                labels.append(k)
            else:
                inputs.append(k)

        return sorted(inputs), sorted(labels), sorted(masks)

    def _create_mask_lookup(self):
        """
        Creates a lookup dictionary between labels and their respective masks

        Assumes labels and masks are already defined (keys already parsed)
        """

        lookup = {} #init
        for l in self._labels:

            mask_name_candidate1 = l.replace("_label","_mask")
            mask_name_candidate2 = l + "_mask"

            if   mask_name_candidate1 in self._masks:
                lookup[l] = mask_name_candidate1
            elif mask_name_candidate2 in self._masks:
                lookup[l] = mask_name_candidate2
            else:
                lookup[l] = None

        return lookup


#===========================================

class LearningMonitor:
    """
    LearningMonitor - a record keeping class for training
    neural networks, including functionality for maintaining
    running averages
    """

    def __init__(self, fname=None):
        """Initialize LearningMonitor."""
        if fname is None:
            #Each dict holds nums & denoms for each running average it records
            self.train = dict(numerators=dict(), denominators=dict())  # Train stats.
            self.test  = dict(numerators=dict(), denominators=dict())  # Test stats.
        else:
            self.load(fname)

    def append_train(self, iter, data):
        """Add train stats."""
        self._append(iter, data, 'train')

    def append_test(self, iter, data):
        """Add test stats."""
        self._append(iter, data, 'test')

    def add_to_num(self, data, phase):
        """Accumulate data to numerators"""
        self._add_to_avg(data, True, phase)

    def add_to_denom(self, data, phase):
        """Accumulate data to denominators"""
        self._add_to_avg(data, False, phase)

    def get_last_iter(self):
        """Return the last iteration number."""
        ret = 0
        if 'iter' in self.train and 'iter' in self.test:
            ret = max(self.train['iter'][-1],self.test['iter'][-1])
        return ret

    def get_last_value(self, key, phase):
        " Extract the last value from one of the records "
        assert phase=="train" or phase=="test", "invalid phase {}".format(phase)
        d = getattr(self, phase)
        return d[key][-1]

    def load(self, fname):
        """Initialize by loading from a h5 file."""
        assert(os.path.exists(fname))
        f = h5py.File(fname, 'r', driver='core')
        # Train stats.
        train = f['/train']
        for key, data in train.iteritems():
            self.train[key] = list(data.value)
        # Test stats.
        test = f['/test']
        for key, data in test.iteritems():
            self.test[key] = list(data.value)
        f.close()

    def save(self, fname, elapsed, base_lr=0):
        """Save stats."""
        if os.path.exists(fname):
            os.remove(fname)
        # Crate h5 file to save.
        f = h5py.File(fname)
        # Train stats.
        for key, data in self.train.iteritems():
            if key == "numerators" or key == "denominators":
              continue
            f.create_dataset('/train/{}'.format(key), data=data)
        # Test stats.
        for key, data in self.test.iteritems():
            if key == "numerators" or key == "denominators":
              continue
            f.create_dataset('/test/{}'.format(key), data=data)
        # Iteration speed in (s/iteration).
        f.create_dataset('/elapsed', data=elapsed)
        f.create_dataset('/base_lr', data=base_lr)
        f.close()

    def compute_avgs(self, iter, phase):
        """
        Finalizes the running averages, and appends them onto the train & test records
        """

        d = getattr(self, phase)
        nums = d["numerators"]
        denoms = d["denominators"]

        avgs = { k : nums[k] / denoms[k] for k in nums.keys() }
        self._append(iter, avgs, phase)

        #Resetting averages
        for k in nums.keys():
          nums[k] = 0.0; denoms[k] = 0.0

    ####################################################################
    ## Non-interface functions
    ####################################################################

    def _add_to_avg(self, data, numerators, phase):
        assert phase=="train" or phase=="test", "invalid phase {}".format(phase)

        term = "numerators" if numerators else "denominators"
        d = getattr(self, phase)[term]
        for key, val in data.items():
            if key not in d:
              d[key] = 0.0
            d[key] += val


    def _append(self, iter, data, phase):
        assert phase=='train' or phase=='test', "invalid phase {}".format(phase)

        d = getattr(self, phase)
        # Iteration.
        if 'iter' not in d:
            d['iter'] = list()
        d['iter'].append(iter)
        # Stats.
        for key, val in data.items():
            if key not in d:
                d[key] = list()
            d[key].append(val)

#===========================================
# Other Util Functions
#===========================================

def masks_not_empty(sample, mask_names):
    counts = [ sample[name].sum() for name in mask_names ]
    return 0 in counts

def make_variable(np_arr, requires_grad=True, volatile=False):
    if not volatile:
      return Variable(torch.from_numpy(np_arr.copy()), requires_grad=requires_grad).cuda()
    else:
      return Variable(torch.from_numpy(np_arr.copy()), volatile=True).cuda()
