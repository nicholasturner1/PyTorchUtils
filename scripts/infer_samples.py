"""
Inference script for small cutout samples

Nicholas Turner, 2020
"""
import os
import time
import argparse
import os.path as osp

import torch
from pytu import run
from pytu import utils


def addparams(args):

    # Sampling params
    patchsize = (20, 160, 160)
    args.datasetargs = [patchsize]
    args.datasetkwargs = dict(samples=readlines(args.samplefilename))

    return args


def infersamples(args):
    assert len(args.gpus) == 1
    assert args.batchsize == 1
    utils.set_gpus(args.gpus)
    args.device = "cuda:0"

    model = utils.initmodel(args, args.device, distrib=False)
    model = utils.load_network(model, args.chkptnum, args.modeldir, module=False)
    dset = utils.initinferencedataset(args)
    loader = utils.wrapdataset(dset, 0, args)

    samplespec = None
    for (i, (samplename, sample)) in enumerate(loader):
        samplename = samplename[0]  # batchsize == 1 above
        print(f"#{i+1}/{len(dset)}:")
        print(samplename)
        start = time.time()
        if samplespec is None:
            samplespec = utils.SampleSpec(sample.keys())

        preds = infersample(model, sample, samplespec, args.device)
        savepreds(args, samplename, preds)
        print(f"{time.time() - start:.3f}s")


def infersample(model, sample, samplespec, device):
    inputs, labels, masks = utils.group_sample(sample, samplespec, device)
    with torch.no_grad():
        return (torch.sigmoid(p) for p in model(*inputs))


def savepreds(args, name, preds):
    for (i, vol) in enumerate(preds):
        preppedname = prepname(name)
        outputfilename = (f"{args.fwddir}/{preppedname}"
                          f"_{args.chkptnum}_{i}_{args.tag}.h5")
        utils.write_h5(vol.cpu().numpy(), outputfilename)


def prepname(name):
    basename = os.path.basename(name)
    prefix = basename.replace("img.h5", "")
    return f"n{prefix}" if "negatives" in name else prefix


def main(args):
    """
    Adds user specified parameters to the args object, 
    sets up a few other things, and starts training
    """
    args = addparams(args)
    args = utils.filldefaults(args)
    utils.sanitycheck(args)
    args.timestamp = utils.timestamp()

    utils.make_required_dirs(args)
    utils.logparams(args, tstamp=args.timestamp)
    utils.logfile(__file__, "run_samples.py", args.logdir, args.timestamp)

    infersamples(args)


def readlines(filename):
    with open(filename) as f:
        return [line.strip() for line in f.readlines()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "experimentname", help="Experiment name")
    parser.add_argument(
        "modelfilename", help="Model architecture filename")
    parser.add_argument(
        "datasetfilename", help="Dataset filename")
    parser.add_argument(
        "--chkptnum", help="Checkpoint Number",
        type=int, default=0)
    parser.add_argument(
        "--gpus", help="GPUs indices to use",
        default=["0"], nargs="+")
    parser.add_argument(
        "samplefilename", help="Inference sample filename",
        default="validationset")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--indices_to_save", nargs='+', type=int, default=None)

    args = parser.parse_args()

    main(args)
