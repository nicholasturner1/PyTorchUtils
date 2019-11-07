"""
Training Script

Put all the ugly things that change with every experiment here

Nicholas Turner, 2017-9
"""
import os.path as osp
import collections

import torch
import tensorboardX

from pytu import utils
from pytu import train
from pytu import loss


def main(**args):

    # args should be the info you need to specify the params
    # for a given experiment, but only params should be used below
    params = fill_params(**args)

    utils.set_gpus(params["gpus"])

    utils.make_required_dirs(**params)

    tstamp = utils.timestamp()
    utils.log_params(params, tstamp=tstamp)
    utils.log_tagged_modules(params["modules_used"],
                             params["log_dir"], "train",
                             chkpt_num=params["chkpt_num"],
                             tstamp=tstamp)

    start_training(**params)


def fill_params(expt_name, chkpt_num, batch_sz, gpus,
                sampler_fname, model_fname, augmentor_fname, **args):

    params = {}

    # Model params
    params["in_spec"] = dict(input=(1, 18, 160, 160))
    params["output_spec"] = collections.OrderedDict(
                                cleft=(1, 18, 160, 160))
    params["width"] = [16, 32, 64]

    # Training procedure params
    params["max_iter"] = 1000000
    params["lr"] = 0.00001
    params["test_intv"] = 1000
    params["test_iter"] = 10
    params["avgs_intv"] = 50
    params["chkpt_intv"] = 100
    params["warm_up"] = 20
    params["chkpt_num"] = chkpt_num
    params["batch_size"] = batch_sz

    # Sampling params
    params["data_dir"] = osp.expanduser(
                             "~/seungmount/research/Nick/datasets/SNEMI3D/")
    assert osp.isdir(params["data_dir"]), "non-existent data directory"
    params["train_sets"] = ["K_val"]
    params["val_sets"] = ["K_val"]
    params["patchsz"] = (18, 160, 160)
    params["sampler_spec"] = dict(input=params["patchsz"],
                                  cleft_label=params["patchsz"])
    params["num_workers"] = 4*batch_sz

    # GPUS
    params["gpus"] = gpus

    # IO/Record params
    params["expt_name"] = expt_name
    params["expt_dir"] = "experiments/{}".format(expt_name)
    params["model_dir"] = osp.join(params["expt_dir"], "models")
    params["log_dir"] = osp.join(params["expt_dir"], "logs")
    params["fwd_dir"] = osp.join(params["expt_dir"], "forward")
    params["tb_train"] = osp.join(params["expt_dir"], "tb/train")
    params["tb_val"] = osp.join(params["expt_dir"], "tb/val")

    # Use-specific Module imports
    params["model_class"] = utils.load_source(model_fname).Model
    params["sampler_class"] = utils.load_source(sampler_fname).Sampler
    params["augmentor_constr"] = utils.load_source(augmentor_fname
                                                   ).get_augmentation

    # "Schema" for turning the parameters above into arguments
    #  for the model class
    params["model_args"] = [params["in_spec"], params["output_spec"],
                            params["width"]]
    params["model_kwargs"] = {}

    # Modules used for record-keeping
    params["modules_used"] = [__file__, model_fname,
                              sampler_fname, augmentor_fname]

    return params


def start_training(tb_train, tb_val, lr, chkpt_num, **params):

    # PyTorch Model
    net = utils.create_network(**params)
    train_writer = tensorboardX.SummaryWriter(tb_train)
    val_writer = tensorboardX.SummaryWriter(tb_val)
    monitor = utils.LearningMonitor()

    # Loading model checkpoint (if applicable)
    if chkpt_num != 0:
        net, monitor = utils.load_chkpt(net, monitor, chkpt_num, **params)

    # Dataset Sampling
    train_loader = utils.load_data(**params, train=True)
    val_loader = utils.load_data(**params, train=False)

    loss_fn = loss.BinomialCrossEntropyWithLogits()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train.train(net, loss_fn, optimizer, train_loader, val_loader,
                train_writer=train_writer, val_writer=val_writer,
                last_iter=chkpt_num, monitor=monitor,
                **params)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("expt_name",
                        help="Experiment name")
    parser.add_argument("model_fname",
                        help="Model architecture filename")
    parser.add_argument("sampler_fname",
                        help="Sampler filename")
    parser.add_argument("augmentor_fname",
                        help="Data augmentor module filename")
    parser.add_argument("--batch_sz",  type=int, default=1,
                        help="Batch size for each sample")
    parser.add_argument("--chkpt_num", type=int, default=0,
                        help="Checkpoint Number")
    parser.add_argument("--gpus", default=["0"], nargs="+")

    args = parser.parse_args()

    main(**vars(args))
