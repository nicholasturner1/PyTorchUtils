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


def make_required_dirs(args):
    """Sets up a directory structure for the current experiment"""
    required_dirs = ["modeldir", "logdir", "fwddir", "tb_train", "tb_val"]

    for d in required_dirs:
        path = getattr(args, d)
        if not os.path.isdir(path):
            os.makedirs(path)


def logfile(filename, tag, logdir=None, tstamp=None):
    "Logs a file within the log directory with a timestamp and name tag"
    tstamp = tstamp if tstamp is not None else timestamp()

    basename = os.path.basename(filename)
    output_basename = f"{tstamp}_{tag}"

    shutil.copyfile(filename, os.path.join(logdir, output_basename))


def logparams(args, tstamp=None, logdir=None):
    "Logs the parameters of the args object in the log directory"
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
    "Logs a set of modules in the log directory"

    tstamp = tstamp if tstamp is not None else timestamp()

    for fname in module_fnames:
        basename = os.path.basename(fname)
        output_basename = f"{tstamp}_{phase}{chkpt_num}_{basename}"

        shutil.copyfile(fname, os.path.join(log_dir, output_basename))


def save_chkpt(model, learning_monitor, opt, chkpt_num, model_dir, log_dir):
    "Saves model parameters, optimizer state, and loss history"
    # Save model
    chkpt_fname = os.path.join(model_dir, f"model{chkpt_num}.pt")
    torch.save(model.module.state_dict(), chkpt_fname)

    # Save optimizer state
    opt_fname = os.path.join(log_dir, f"opt{chkpt_num}.pt")
    torch.save(opt.state_dict(), opt_fname)

    # Save learning monitor
    lm_fname = os.path.join(log_dir, f"stats{chkpt_num}.h5")
    learning_monitor.save(lm_fname, chkpt_num)


def initmodel(args, device, distrib=True):
    "Initializes an instance of a PyTorch model from a source file"
    modelconstr = load_source(args.modelfilename,
                              "model", args.logdir,
                              args.timestamp).Model
    basemodel = modelconstr(*args.modelargs, **args.modelkwargs).to(device)

    if distrib:
        return torch.nn.parallel.DistributedDataParallel(
                   basemodel, device_ids=[device])
    else:
        return basemodel


def create_network(model_class, model_args, model_kwargs,
                   chkpt_num=0, model_dir=None, **params):
    net = torch.nn.DataParallel(
              model_class(*model_args, **model_kwargs)).cuda()

    if chkpt_num > 0:
        load_network(net, chkpt_num, model_dir)

    return net


def load_network(model, chkpt_num, model_dir, module=True):
    "Loads model parameters into an intialized model"
    chkpt_fname = os.path.join(model_dir, f"model{chkpt_num}.pt")
    if module:
        model.module.load_state_dict(torch.load(chkpt_fname))
    else:
        model.load_state_dict(torch.load(chkpt_fname))

    return model


def initloss(args, device):
    "Initializes an instance of a PyTorch loss module from a source file"
    constructor = load_source(args.lossfilename,
                              "loss", args.logdir, args.timestamp).Loss
    return constructor(*args.lossargs, **args.losskwargs)


def initopt(args, model, device):
    "Initializes an instance of a PyTorch optimizer from a source file"
    constructor = load_source(args.lossfilename,
                              "opt", args.logdir, args.timestamp).Optimizer

    return constructor(model.parameters(), *args.optargs, **args.optkwargs)


def load_optimizer(opt, chkpt_num, log_dir):
    "Loads an optimizer state into an initialized optimizer"
    opt_fname = os.path.join(log_dir, f"opt{chkpt_num}.pt")
    opt.load_state_dict(torch.load(opt_fname))

    return opt


def load_learning_monitor(learning_monitor, chkpt_num, log_dir):
    "Loads loss history"
    lm_fname = os.path.join(log_dir, f"stats{chkpt_num}.h5")
    learning_monitor.load(lm_fname)

    return learning_monitor


def loadchkpt(model, learning_monitor, opt, args):
    """
    Loads network parameters, optimizer state, and loss history
    from a saved checkpoint
    """
    m = load_network(model, args.chkptnum, args.modeldir)
    opt = load_optimizer(opt, args.chkptnum, args.logdir)
    lm = load_learning_monitor(learning_monitor, args.chkptnum, args.logdir)

    return m, lm, opt


def initloaders(args, rank):
    "Initialize data loaders for training and validation"
    augconstr = load_source(args.augfilename,
                            "aug", args.logdir,
                            args.timestamp).Augmentor
    trainaug = augconstr(True, *args.augargs, **args.augkwargs)
    valaug = augconstr(False, *args.augargs, **args.augkwargs)

    dsetconstr = load_source(args.datasetfilename,
                             "dataset", args.logdir,
                             args.timestamp).Dataset
    traindset = dsetconstr(*args.trainsamplerargs,
                           **args.trainsamplerkwargs,
                           aug=trainaug)
    valdset = dsetconstr(*args.valsamplerargs,
                         **args.valsamplerkwargs,
                         aug=valaug)

    trainloader = wrapdataset(traindset, rank, args)
    valloader = wrapdataset(valdset, rank, args)

    return iter(trainloader), iter(valloader)


def initinferencedataset(args, wrap=True):
    "Initialize a dataset for sample-wise validation"
    dsetconstr = load_source(args.datasetfilename,
                             "inferencedataset", args.logdir,
                             args.timestamp).InferenceDataset

    return dsetconstr(*args.datasetargs, **args.datasetkwargs)


def wrapdataset(dset, rank, args):
    """Wraps a training dataset for distributed data loading"""
    if isinstance(dset, torch.utils.data.IterableDataset):
        # Assume that the iterable dataset already randomizes things
        sampler = None
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(
                      dset, num_replicas=len(args.gpus), rank=rank)

    return torch.utils.data.DataLoader(
        dataset=dset, batch_size=args.batchsize, shuffle=False,
        num_workers=0, pin_memory=True, sampler=sampler)


def load_data(sampler_class, augmentor_constr, data_dir, patchsz,
              train_sets, val_sets, batch_size, num_workers, train=True,
              **params):
    "Initialize data loaders for training and validation - old"
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


def initwriters(args):
    "Initializes tensorboard writers"
    rankstr = str(args.rank)
    traindir = os.path.join(args.tb_train, rankstr)
    valdir = os.path.join(args.tb_val, rankstr)
    if not os.path.isdir(traindir):
        os.makedirs(traindir)
    if not os.path.isdir(valdir):
        os.makedirs(valdir)

    trainwriter = tensorboardX.SummaryWriter(traindir)
    valwriter = tensorboardX.SummaryWriter(valdir)

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
    """Extracts the iteration number from a network checkpoint"""
    basename = os.path.basename(chkpt_fname)
    return int(re.findall(r"\d+", basename)[0])


def group_sample(sample, sample_spec, gpu=0, phase="train"):
    """ Creates the Torch tensors for a sample """

    inputs = sample_spec.get_inputs()
    labels = sample_spec.get_labels()
    masks = sample_spec.get_masks()

    input_vars = [to_torch(sample[k], gpu, block=False) for k in inputs]
    label_vars = [to_torch(sample[k], gpu, block=False) for k in labels]
    mask_vars = [to_torch(sample[k], gpu, block=False) for k in masks]

    return input_vars, label_vars, mask_vars


def to_torch(np_arr, device, block=True):
    """Converts a numpy array to a torch tensor on the specified device"""
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
        return f["/main"][()]


def write_h5(data, fname):
    if os.path.exists(fname):
        os.remove(fname)

    opts = dict(compression="gzip", compression_opts=4)
    with h5py.File(fname, 'w') as f:
        f.create_dataset("/main", data=data, **opts)


def timestamp():
    return datetime.datetime.now().strftime("%d%m%y_%H%M%S")
