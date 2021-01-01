import os

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import tensorboardX

from . import train
from . import utils


def run_training(args):

    utils.set_gpus(args.gpus)

    os.environ["MASTER_ADDR"] = "0.0.0.0"
    os.environ["MASTER_PORT"] = str(args.port)

    mp.spawn(trainingprocess, nprocs=len(args.gpus), args=(args,))


def trainingprocess(rank, args, torch_seed=12345):

    dist.init_process_group(
        backend="nccl", init_method="env://",
        world_size=len(args.gpus), rank=rank)

    args.rank = rank
    args.device = f"cuda:{args.gpus[rank]}"
    torch.manual_seed(torch_seed)
    #torch.cuda.device(gpu)

    model = initmodel(args, args.device)
    lossfn, opt = initopt(args, model, args.device)
    trainloader, valloader = initloaders(args, rank)
    trainwriter, valwriter = initwriters(args)
    monitor = utils.LearningMonitor()

    if args.chkptnum > 0:
        model, monitor, opt = utils.loadchkpt(model, monitor, opt, args)

    train.train(model, lossfn, opt, trainloader, valloader,
                train_writer=trainwriter, val_writer=valwriter,
                last_iter=args.chkptnum, monitor=monitor, args=args) #, rank=rank)


def initmodel(args, device):
    modelconstr = utils.load_source(args.modelfilename,
                                    "model", args.logdir,
                                    args.timestamp).Model
    basemodel = modelconstr(*args.modelargs, **args.modelkwargs).to(device)

    return torch.nn.parallel.DistributedDataParallel(
               basemodel, device_ids=[device])


def initopt(args, model, device):
    lossconstr = utils.load_source(args.lossfilename,
                                   "loss", args.logdir,
                                   args.timestamp).Loss
    loss = lossconstr(*args.lossargs, **args.losskwargs)

    optconstr = utils.load_source(args.lossfilename,
                                  "opt", args.logdir,
                                  args.timestamp).Optimizer
    opt = optconstr(model.parameters(), *args.optargs, **args.optkwargs)

    return loss, opt


def initloaders(args, rank):
    augconstr = utils.load_source(args.augfilename,
                                  "aug", args.logdir,
                                  args.timestamp).Augmentor
    aug = augconstr(*args.augargs, **args.augkwargs)

    dsetconstr = utils.load_source(args.datasetfilename,
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
