#!/usr/bin/env python
__doc__ = """

Miscellaneous Utils

Nicholas Turner <nturner@cs.princeton.edu>, 2017
"""

import torch
from torch.autograd import Variable


import os, re, shutil
import datetime
import h5py


def log_tagged_modules(module_fnames, log_dir, phase, iter_num=0):
    
    timestamp = datetime.datetime.now().strftime("%d%m%y_%H%M%S")

    for fname in module_fnames:
        basename = os.path.basename(fname)
        output_basename = "{}_{}{}_{}".format(timestamp, phase, iter_num, basename)

        shutil.copyfile(fname, os.path.join(log_dir, output_basename))


def save_chkpt(model, learning_monitor, iter_num, model_dir, log_dir):

    # Save model
    chkpt_fname = os.path.join(model_dir, "model{}.chkpt".format(iter_num))
    torch.save(model.state_dict(), chkpt_fname)

    # Save learning monitor 
    lm_fname = os.path.join(log_dir, "stats{}.h5".format(iter_num))
    learning_monitor.save(lm_fname, iter_num)


def load_chkpt(model, learning_monitor, iter_num, model_dir, log_dir):

    # Load model params
    chkpt_fname = os.path.join(model_dir, "model{}.chkpt".format(iter_num))
    model.load_state_dict(torch.load(chkpt_fname))
    
    # Load learning monitor
    lm_fname = os.path.join(log_dir, "stats{}.h5".format(iter_num))
    learning_monitor.load(lm_fname)


def iter_from_chkpt_fname(chkpt_fname):
    """ Extracts the iteration number from a network checkpoint """
    basename = os.path.basename(chkpt_fname)
    return int(re.findall(r"\d+", basename)[0])


def masks_not_empty(sample, mask_names):
    """ Tests whether a sample has any non-masked values """
    counts = [ sample[name].sum() for name in mask_names ]
    return 0 in counts


def make_variable(np_arr, requires_grad=True, volatile=False):
    """ Creates a torch.autograd.Variable from a np array """
    if not volatile:
      return Variable(torch.from_numpy(np_arr.copy()), requires_grad=requires_grad).cuda()
    else:
      return Variable(torch.from_numpy(np_arr.copy()), volatile=True).cuda()

def read_h5(fname):
    
    with h5py.File(fname) as f:
        d = f["/main"].value

    return d


def write_h5(data, fname):

    if os.path.exists(fname):
      os.remove(fname)

    with h5py.File(fname) as f:
        f.create_dataset("/main",data=data)

