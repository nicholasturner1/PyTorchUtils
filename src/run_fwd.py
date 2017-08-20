#!/usr/bin/env python

import os, sys, imp
import collections

import torch
from torch.nn import functional as F
import dataprovider as dp

import forward
import utils


EXPT_NAME     = sys.argv[1]
GPU           = sys.argv[2]
MODEL_FNAME   = sys.argv[3]
CHKPT_FNAME   = sys.argv[4]
DSET_NAMES    = sys.argv[5:]

DATA_DIR = "/usr/people/nturner/research/datasets/SNEMI3D/train_val_test/"

os.environ["CUDA_VISIBLE_DEVICES"] = GPU


#============================================================
params = {}

#Model params
params["in_dim"]      = 1
params["output_spec"] = collections.OrderedDict(psd_label=1)
params["depth"]       = 4
params["batch_norm"]  = False
params["activation"]  = F.sigmoid
params["chkpt_fname"] = CHKPT_FNAME

#IO/Record params
params["expt_dir"]    = "experiments/{}".format(EXPT_NAME)
params["model_dir"]   = os.path.join(params["expt_dir"], "models")
params["log_dir"]     = os.path.join(params["expt_dir"], "logs")
params["fwd_dir"]     = os.path.join(params["expt_dir"], "forward")

#Dataset params
params["dsets"]       = DSET_NAMES
params["input_spec"]  = collections.OrderedDict(input=(18,160,160)) #dp dataset spec
params["scan_spec"]   = collections.OrderedDict(psd=(1,18,160,160)) 
params["scan_params"] = dict(stride=(0.5,0.5,0.5), blend="bump")


#============================================================

#Modules used for record-keeping
params["modules_used"] = [MODEL_FNAME, "layers.py"]


#Use-specific Module imports
model_module = imp.load_source("Model", MODEL_FNAME)
Model = model_module.Model


#============================================================

def create_network(in_dim, depth, output_spec, 
                   batch_norm, chkpt_fname, **params):

    net = Model(in_dim, output_spec, depth, bn=batch_norm).cuda()

    net.load_state_dict(torch.load(chkpt_fname))

    return net


def make_forward_scanner(dset_name, input_spec, scan_spec, scan_params, **params):
    """ Creates a DataProvider ForwardScanner from a dset name """

    # Reading EM image
    img = utils.read_h5(os.path.join(DATA_DIR, dset_name + "_img.h5"))
    img = (img / 255.).astype("float32")

    # Creating DataProvider Dataset
    vd = dp.VolumeDataset()

    vd.add_raw_data(key="input", data=img)
    vd.set_spec(input_spec)

    # Returning DataProvider ForwardScanner
    return dp.ForwardScanner(vd, scan_spec, params=scan_params)
    

def save_output(output, dset_name, iter_num, fwd_dir, **params):
    """ Saves the volumes within a DataProvider ForwardScanner """

    for k in output.outputs.data.iterkeys():

        output_data = output.outputs.get_data(k)

        basename = "{}_{}_{}.h5".format(dset_name, k, iter_num)
        full_fname = os.path.join( fwd_dir, basename )

        utils.write_h5(output_data[0,:,:,:], full_fname)


#============================================================

def main(**params):

    net = create_network(**params)

    iter_num = utils.iter_from_chkpt_fname(params["chkpt_fname"])
    utils.log_tagged_modules(params["modules_used"], params["log_dir"], 
                             "fwd", iter_num)

    for dset in params["dsets"]:
        print(dset)

        fs = make_forward_scanner(dset, **params)

        output = forward.forward(net, fs, params["scan_spec"], 
                                 activation=params["activation"])

        save_output(output, dset, iter_num, **params)


if __name__ == "__main__":

    main(**params)
