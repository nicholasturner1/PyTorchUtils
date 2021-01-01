"""
Miscellaneous Utils

Nicholas Turner <nturner@cs.princeton.edu>, 2017
"""

import importlib
import datetime
import shutil
import types
import os
import re

import torch
from torch.utils.data import DataLoader
import tensorboardX
import numpy as np
import h5py


__all__ = ["timestamp",
           "make_required_dirs", "log_tagged_modules", "logparams",
           "create_network", "load_network", "load_learning_monitor",
           "save_chkpt", "loadchkpt", "iter_from_chkpt_fname",
           "initmodel", "initloss", "initopt", "initloaders", "initwriters",
           "load_data", "load_source",
           "to_torch", "masks_empty",
           "read_h5", "write_h5",
           "set_gpus", "init_seed", "logfile"]


def make_required_dirs(args):
    required_dirs = ["modeldir", "logdir", "fwddir", "tb_train", "tb_val"]

    for d in required_dirs:
        path = getattr(args, d)
        if not os.path.isdir(path):
            os.makedirs(path)


def logfile(filename, tag, logdir=None, tstamp=None):

    tstamp = tstamp if tstamp is not None else timestamp()

    basename = os.path.basename(filename)
    output_basename = f"{tstamp}_{tag}"

    shutil.copyfile(filename, os.path.join(logdir, output_basename))


def logparams(args, tstamp=None, logdir=None):
    param_dict = vars(args)

    if logdir is None:
        assert hasattr(args, "logdir"), "log dir not specified"
        logdir = args.logdir

    tstamp = tstamp if tstamp is not None else timestamp()

    output_basename = f"{tstamp}_params.csv"

    with open(os.path.join(logdir, output_basename), "w+") as f:
        for (k, v) in param_dict.items():
            f.write(f"{k};{v}\n")


def log_tagged_modules(module_fnames, log_dir,
                       phase, chkpt_num=0, tstamp=None):

    tstamp = tstamp if tstamp is not None else timestamp()

    for fname in module_fnames:
        basename = os.path.basename(fname)
        output_basename = f"{tstamp}_{phase}{chkpt_num}_{basename}"

        shutil.copyfile(fname, os.path.join(log_dir, output_basename))


def save_chkpt(model, learning_monitor, opt, chkpt_num, model_dir, log_dir):
    # Save model
    chkpt_fname = os.path.join(model_dir, f"model{chkpt_num}.chkpt")
    torch.save(model.module.state_dict(), chkpt_fname)

    # Save optimizer state
    opt_fname = os.path.join(log_dir, f"opt{chkpt_num}.chkpt")
    torch.save(opt.state_dict(), opt_fname)

    # Save learning monitor
    lm_fname = os.path.join(log_dir, f"stats{chkpt_num}.h5")
    learning_monitor.save(lm_fname, chkpt_num)


def initmodel(args, device):
    modelconstr = load_source(args.modelfilename,
                              "model", args.logdir,
                              args.timestamp).Model
    basemodel = modelconstr(*args.modelargs, **args.modelkwargs).to(device)

    return torch.nn.parallel.DistributedDataParallel(
               basemodel, device_ids=[device])


def create_network(model_class, model_args, model_kwargs,
                   chkpt_num=0, model_dir=None, **params):
    net = torch.nn.DataParallel(
              model_class(*model_args, **model_kwargs)).cuda()

    if chkpt_num > 0:
        load_network(net, chkpt_num, model_dir)

    return net


def load_network(model, chkpt_num, model_dir):
    chkpt_fname = os.path.join(model_dir, f"model{chkpt_num}.chkpt")
    model.module.load_state_dict(torch.load(chkpt_fname))

    return model


def initloss(args, device):
    constructor = load_source(args.lossfilename,
                              "loss", args.logdir, args.timestamp).Loss
    return constructor(*args.lossargs, **args.losskwargs)


def initopt(args, model, device):
    constructor = load_source(args.lossfilename,
                              "opt", args.logdir, args.timestamp).Optimizer

    return constructor(model.parameters(), *args.optargs, **args.optkwargs)


def load_optimizer(opt, chkpt_num, log_dir):
    opt_fname = os.path.join(log_dir, f"opt{chkpt_num}.chkpt")
    opt.load_state_dict(torch.load(opt_fname))

    return opt


def load_learning_monitor(learning_monitor, chkpt_num, log_dir):

    lm_fname = os.path.join(log_dir, f"stats{chkpt_num}.h5")
    learning_monitor.load(lm_fname)

    return learning_monitor


def loadchkpt(model, learning_monitor, opt, args):

    m = load_network(model, args.chkptnum, args.modeldir)
    opt = load_optimizer(opt, args.chkptnum, args.logdir)
    lm = load_learning_monitor(learning_monitor, args.chkptnum, args.log_dir)

    return m, lm, opt


def initloaders(args, rank):
    augconstr = load_source(args.augfilename,
                            "aug", args.logdir,
                            args.timestamp).Augmentor
    aug = augconstr(*args.augargs, **args.augkwargs)

    dsetconstr = load_source(args.datasetfilename,
                             "dataset", args.logdir,
                             args.timestamp).Dataset
    traindset = dsetconstr(*args.trainsamplerargs,
                           **args.trainsamplerkwargs,
                           aug=aug)
    valdset = dsetconstr(*args.valsamplerargs,
                         **args.valsamplerkwargs,
                         aug=aug)

    trainloader = wrapdataset(traindset, rank, args)
    valloader = wrapdataset(valdset, rank, args)

    return iter(trainloader), iter(valloader)


def load_data(sampler_class, augmentor_constr, data_dir, patchsz,
              train_sets, val_sets, batch_size, num_workers, train=True,
              **params):
    aug = augmentor_constr(train)
    if train:
        sampler = sampler_class(data_dir, patchsz, train_sets,
                                mode="train", aug=aug)
    else:
        sampler = sampler_class(data_dir, patchsz, val_sets,
                                mode="val", aug=aug)

    loader = DataLoader(sampler, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True)

    return iter(loader)


def wrapdataset(dset, rank, args):

    if isinstance(dset, torch.utils.data.IterableDataset):
        # Assume that the iterable dataset already randomizes things
        sampler = None
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(
                      dset, num_replicas=len(args.gpus), rank=rank)

    return torch.utils.data.DataLoader(
        dataset=dset, batch_size=args.batchsize, shuffle=False,
        num_workers=0, pin_memory=True, sampler=sampler)


def initwriters(args):
    trainwriter = tensorboardX.SummaryWriter(args.tb_train)
    valwriter = tensorboardX.SummaryWriter(args.tb_val)

    return trainwriter, valwriter


def load_source(fname, module_name="module", log_dir=None, tstamp=None):
    """Updated version of imp.load_source(fname)"""
    loader = importlib.machinery.SourceFileLoader(module_name, fname)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)

    if log_dir is not None:
        logfile(fname, module_name, log_dir, tstamp=tstamp)

    return mod


def iter_from_chkpt_fname(chkpt_fname):
    """ Extracts the iteration number from a network checkpoint """
    basename = os.path.basename(chkpt_fname)
    return int(re.findall(r"\d+", basename)[0])


def to_torch(np_arr, device, block=True):
    tensor = torch.from_numpy(np.ascontiguousarray(np_arr))
    return tensor.to(device, non_blocking=(not block))


def masks_empty(sample, mask_names):
    """ Tests whether a sample has any non-masked values """
    return any(not torch.any(sample[name] != 0) for name in mask_names)


def init_seed(worker_id, random=False):
    "Setting the random seed for each sampler thread within a loader"
    if random:
        seed = torch.IntTensor(1).random_().item()
    else:
        seed = worker_id

    torch.manual_seed(seed)
    np.random.seed(seed)


def set_gpus(gpu_list):
    """ Sets the gpus visible to this process """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)


def read_h5(fname):

    with h5py.File(fname, 'r') as f:
        d = f["/main"][()]

    return d


def write_h5(data, fname):

    if os.path.exists(fname):
        os.remove(fname)

    with h5py.File(fname, 'w') as f:
        f.create_dataset("/main", data=data)


def timestamp():
    return datetime.datetime.now().strftime("%d%m%y_%H%M%S")
