"""
Utilities to make creating the args object easier
"""
import os.path as osp
import copy


def filldefaults(args):

    # Should already be defined by the training script, but just in case
    fill(args, "gpus", ["0"])
    fill(args, "batchsize", 1)
    fill(args, "chkptnum", 0)

    # Default IO directory structure
    assert hasattr(args, "experimentname")
    fill(args, "exptdir", f"experiments/{args.experimentname}")
    fill(args, "modeldir", osp.join(args.exptdir, "models"))
    fill(args, "logdir", osp.join(args.exptdir, "logs"))
    fill(args, "fwddir", osp.join(args.exptdir, "forward"))
    fill(args, "tb_train", osp.join(args.exptdir, "tb/train"))
    fill(args, "tb_val", osp.join(args.exptdir, "tb/val"))

    # Default (empty) arguments to interpreted components
    fillargs(args, "model")
    fillargs(args, "loss")
    fillargs(args, "opt")
    fillargs(args, "aug")
    fillargs(args, "sampler")

    # Splitting sampler args into two (if not already defined)
    fill(args, "trainsamplerargs", args.samplerargs)
    trainsamplerkwargs = copy.copy(args.samplerkwargs)
    trainsamplerkwargs["mode"] = "train"
    trainsamplerkwargs["vols"] = args.trainsets
    fill(args, "trainsamplerkwargs", trainsamplerkwargs)
    fill(args, "valsamplerargs", args.samplerargs)
    valsamplerkwargs = copy.copy(args.samplerkwargs)
    valsamplerkwargs["mode"] = "val"
    valsamplerkwargs["vols"] = args.valsets
    fill(args, "valsamplerkwargs", valsamplerkwargs)

    # For running multiple processes
    fill(args, "port", 54321)

    return args


def fill(args, attrname, value):
    if not hasattr(args, attrname):
        setattr(args, attrname, value)


def fillargs(args, rootname):
    fill(args, f"{rootname}args", [])
    fill(args, f"{rootname}kwargs", {})


def sanitycheck(args):

    assert isinstance(args.gpus, list)
    assert len(args.gpus) > 0
    assert isinstance(args.gpus[0], str)
    assert int(args.gpus[0]) >= 0  # testing converstion to int

    assert isinstance(args.batchsize, int)
    assert args.batchsize > 0

    assert isinstance(args.chkptnum, int)
    assert args.chkptnum >= 0

    # __args should be lists, __kwargs should be dicts
    assertargtypes(args, "model")
    assertargtypes(args, "loss")
    assertargtypes(args, "opt")
    assertargtypes(args, "aug")
    assertargtypes(args, "sampler")
    assertargtypes(args, "trainsampler")
    assertargtypes(args, "valsampler")

    assert isinstance(args.port, int)
    assert args.port > 0


def assertargtypes(args, rootname):
    assert isinstance(getattr(args, f"{rootname}args"), list), (
        f"args.{rootname}args is not a list")
    assert isinstance(getattr(args, f"{rootname}kwargs"), dict), (
        f"args.{rootname}kwargs is not a dict")
