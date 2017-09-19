#!/usr/bin/env python

import os, re, imp, sys
import shutil
import collections

import torch

import train
import utils


EXPT_NAME     = sys.argv[1] 
GPU           = sys.argv[2]
SAMPLER_FNAME = sys.argv[3]
MODEL_FNAME   = sys.argv[4]
CHKPT_NUM     = int(sys.argv[5]) if len(sys.argv) > 5 else 0


os.environ["CUDA_VISIBLE_DEVICES"] = GPU


params = dict()

#Model params
params["in_dim"]       = 1
params["output_spec"]  = collections.OrderedDict(psd_label=1)
params["depth"]        = 4
params["batch_norm"]   = False

#Training procedure params
params["lr"]          = 0.00001
params["max_iter"]    = 1000000
params["test_intv"]   = 1000
params["test_iter"]   = 100
params["avgs_intv"]   = 50
params["chkpt_intv"]  = 10000
params["chkpt_num"]   = CHKPT_NUM

#IO/Record params 
params["expt_dir"]   = "experiments/{}".format(EXPT_NAME)
params["model_dir"]  = os.path.join(params["expt_dir"], "models")
params["log_dir"]    = os.path.join(params["expt_dir"], "logs")
params["fwd_dir"]    = os.path.join(params["expt_dir"], "forward")

#Sampling params
params["data_dir"]     = "/usr/people/nturner/research/datasets/SNEMI3D/"
params["train_sets"]   = ["train"]
params["val_sets"]     = ["roncal_val"]


#modules used for record-keeping
params["modules_used"] = [__file__, MODEL_FNAME, "layers.py", 
                          SAMPLER_FNAME, "loss.py"]

#Use-specific Module imports
import loss
sampler_module = imp.load_source("Sampler",SAMPLER_FNAME)
model_module = imp.load_source("Model",MODEL_FNAME)
Model = model_module.Model



def start_training(in_dim, output_spec, depth, batch_norm, chkpt_num,
                   lr, train_sets, val_sets, data_dir, **params):

    net = Model(in_dim, output_spec, depth, bn=batch_norm).cuda()
    monitor = utils.LearningMonitor()

    if chkpt_num != 0:
        utils.load_chkpt(net, monitor, chkpt_num, 
                         params["model_dir"], 
                         params["log_dir"])

    loss_fn = loss.BinomialCrossEntropyWithLogits()

    optimizer = torch.optim.Adam( net.parameters(), lr=lr )

    train_sampler = utils.AsyncSampler(sampler_module.Sampler(data_dir,
                                                      dsets=train_sets, 
                                                      mode="train"))

    val_sampler   = utils.AsyncSampler(sampler_module.Sampler(data_dir, 
                                                      dsets=val_sets,   
                                                      mode="test"))

    train.train(net, loss_fn, optimizer, train_sampler, val_sampler, 
                last_iter=chkpt_num, monitor=monitor, **params)


def make_reqd_dirs(model_dir, log_dir, fwd_dir, **params):

    for d in [model_dir, log_dir, fwd_dir]:
      if not os.path.isdir(d):
        os.makedirs(d)
      

def main(**params):

    make_reqd_dirs(**params)

    utils.log_tagged_modules(params["modules_used"], 
                             params["log_dir"], "train", 
                             iter_num=params["chkpt_num"])

    start_training(**params)

if __name__ == "__main__":
    main(**params)
