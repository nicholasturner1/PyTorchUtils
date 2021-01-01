import os
import signal

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from . import train
from . import utils


def run_training(args):

    utils.set_gpus(args.gpus)

    os.environ["MASTER_ADDR"] = "0.0.0.0"
    os.environ["MASTER_PORT"] = str(args.port)

    # signal code allows for cleaner Ctrl+C handling
    # https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python  # noqa
    orig_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    context = mp.spawn(trainingprocess, join=False,
                       nprocs=len(args.gpus), args=(args,))
    signal.signal(signal.SIGINT, orig_sigint_handler)

    waitforinterrupt(context)


def trainingprocess(rank, args, torch_seed=12345):

    dist.init_process_group(
        backend="nccl", init_method="env://",
        world_size=len(args.gpus), rank=rank)

    args.rank = rank
    args.device = f"cuda:{args.gpus[rank]}"
    torch.manual_seed(torch_seed)

    model = utils.initmodel(args, args.device)
    lossfn = utils.initloss(args, args.device)
    opt = utils.initopt(args, model, args.device)
    trainloader, valloader = utils.initloaders(args, rank)
    trainwriter, valwriter = utils.initwriters(args)
    monitor = utils.LearningMonitor()

    if args.chkptnum > 0:
        model, monitor, opt = utils.loadchkpt(model, monitor, opt, args)

    train.train(model, lossfn, opt, trainloader, valloader,
                train_writer=trainwriter, val_writer=valwriter,
                last_iter=args.chkptnum, monitor=monitor, args=args, rank=rank)


def waitforinterrupt(context):
    try:
        while not context.join():
            pass
        return
    except KeyboardInterrupt:
        # A warning that mentions leaked semaphores is currently
        # unavoidable here
        # https://www.gitmemory.com/issue/pytorch/pytorch/23117/513478582
        for process in context.processes:
            if process.is_alive():
                process.terminate()
