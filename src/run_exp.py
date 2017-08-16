#!/usr/bin/env python

import os, re
import shutil
import sys
import imp

import torch

import train
import train_utils as utils


EXPT_NAME     = sys.argv[1] 
GPU           = sys.argv[2]
SAMPLER_FNAME = sys.argv[3]
MODEL_FNAME   = sys.argv[4]
CHKPT_FNAME   = sys.argv[5] if len(sys.argv) > 5 else None


os.environ["CUDA_VISIBLE_DEVICES"] = GPU


params = dict()

#Model params
params["in_dimension"] = 1
params["output_spec"]  = [("psd_label",1)]
params["depth"]        = 4
params["batch_norm"]   = False

#Training procedure params
params["lr"]          = 0.00001
params["max_iter"]    = 1000000
params["test_intv"]   = 1000
params["test_iter"]   = 100
params["avgs_intv"]   = 50
params["chkpt_intv"]  = 10000
params["chkpt_fname"] = CHKPT_FNAME

#IO/Record params #SOME HAVE HOOKS TO TRAIN.PY
params["expt_dir"]   = "experiments/{}".format(EXPT_NAME)
params["model_dir"]  = os.path.join(params["expt_dir"], "models")
params["logs_dir"]   = os.path.join(params["expt_dir"], "logs")
params["fwd_dir"]    = os.path.join(params["expt_dir"], "forward")

#Sampling params
params["data_dir"]     = "~/research/datasets/SNEMI3D/"
params["train_sets"]   = ["train"]
params["val_sets"]     = ["val"]


#modules used for record-keeping
modules_used = [MODEL_FNAME,"layers.py",SAMPLER_FNAME, "loss.py"]


#Use-specific Module imports
import loss
sampler_module = imp.load_source("Sampler",SAMPLER_FNAME)
model_module = imp.load_source("Model",MODEL_FNAME)
Model = model_module.Model


def last_iter_from_chkpt(chkpt_fname):
    basename = os.path.basename(chkpt_fname)
    return int(re.findall(r"\d+", basename)[0])


def start_training():

    net = Model(params["in_dimension"], params["output_spec"], 
               params["depth"], bn=params["batch_norm"]).cuda()

    if params["chkpt_fname"] is not None:
        net.load_state_dict(torch.load(params["chkpt_fname"]))
        last_iter = last_iter_from_chkpt(params["chkpt_fname"])
    else:
        last_iter = 0

    loss_fn = loss.BinomialCrossEntropyWithLogits()

    optimizer = torch.optim.Adam( net.parameters(), lr=params["lr"] )

    train_sampler = utils.AsyncSampler(sampler_module.Sampler(params["data_dir"],
                                                      dsets=params["train_sets"], 
                                                      mode="train"))

    val_sampler   = utils.AsyncSampler(sampler_module.Sampler(params["data_dir"], 
                                                      dsets=params["val_sets"],   
                                                      mode="test"))

    train.train(net, loss_fn, optimizer, train_sampler, val_sampler, 
                last_iter=last_iter, **params)


def make_reqd_dirs():

    for d in [params["model_dir"], params["logs_dir"], params["fwd_dir"]]:
      if not os.path.isdir(d):
        os.makedirs(d)


def copy_record_files():
    """ Keep a copy of the used modules as records """

    target = os.path.join(params["logs_dir"], __file__)
    shutil.copyfile(__file__, target)
    for f in modules_used:
        target = os.path.join(params["logs_dir"], os.path.basename(f))
        shutil.copyfile(f, target)


def main():

    make_reqd_dirs()
    copy_record_files()
    start_training()

if __name__ == "__main__":
    main()
