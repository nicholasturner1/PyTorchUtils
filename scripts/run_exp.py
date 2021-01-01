"""
Training Script

Nicholas Turner, 2017-20
"""
import argparse
import os.path as osp

from pytu import run
from pytu import utils


def addparams(args):

    # Training procedure params
    args.max_iter = 1000000
    args.lr = 0.00001
    args.test_intv = 1000
    args.test_iter = 10
    args.avgs_intv = 50
    args.chkpt_intv = 10
    args.warm_up = 20

    # Sampling params
    datadir = osp.expanduser("~/seungmount/research/Nick/datasets/SNEMI3D/")
    samplerpatchsz = (18, 160, 160)
    samplerspec = dict(input=samplerpatchsz, cleft_label=samplerpatchsz)
    args.samplerargs = [datadir, samplerspec]
    args.trainsets = ["K_val"]
    args.valsets = ["K_val"]

    # Model params
    inspec = dict(input=(1, 18, 160, 160))
    outspec = dict(cleft=(1, 18, 160, 160))
    #width = [16, 32, 64]
    width = [16, 32]
    args.modelargs = [inspec, outspec, width]

    return args


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
    utils.logfile(__file__, "run_exp.py", args.logdir, args.timestamp)

    run.run_training(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "experimentname", help="Experiment name")
    parser.add_argument(
        "modelfilename", help="Model architecture filename")
    parser.add_argument(
        "datasetfilename", help="Dataset filename")
    parser.add_argument(
        "augfilename", help="Data augmentor module filename")
    parser.add_argument(
        "lossfilename", help="Loss and optimizer filename")
    parser.add_argument(
        "--batchsize", help="Batch size for each sample",
        type=int, default=1)
    parser.add_argument(
        "--chkptnum", help="Checkpoint Number",
        type=int, default=0)
    parser.add_argument(
        "--gpus", help="GPUs indices to use",
        default=["0"], nargs="+")

    args = parser.parse_args()

    main(args)
