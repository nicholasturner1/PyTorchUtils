"""
Training

Nicholas Turner <nturner@cs.princeton.edu>, 2017-9
"""
import time

import torch

from . import utils


REQUIRED_PARAMS = ["max_iter", "test_intv", "test_iter",
                   "avgs_intv", "chkpt_intv", "exptdir",
                   "modeldir", "logdir", "batchsize", "warm_up"]


def train(model, loss_fn, optimizer, sampler, val_sampler=None,
          last_iter=0, train_writer=None, val_writer=None,
          monitor=None, args=None, rank=None):
    """ Generalized training function """

    assert params_defined(args), "Params under-specified"

    if monitor is None:
        monitor = utils.LearningMonitor()

    # Determine the names of inputs, labels, masks
    sample_spec = utils.SampleSpec(next(sampler).keys())
    mask_names = sample_spec.get_masks()

    model_w_loss = utils.wrapmodel(model, loss_fn, sample_spec)

    printR0(rank, "======= BEGIN TRAINING LOOP ========")
    for i in range(last_iter, args.max_iter):
        start = time.time()

        # Make sure no mask is empty (data for all tasks)
        sample = fetch_nonempty_sample(sampler, mask_names)

        inputs, labels, masks = utils.group_sample(sample, sample_spec,
                                                   args.device, "train")

        # Running forward pass, evaluating loss fn
        losses, nmsks = model_w_loss(inputs, labels, masks)

        losses, nmsks = sum_to_scalar(losses, nmsks)

        # Need all processes to have completed their forward pass
        # before computing the backward pass
        torch.distributed.barrier()
        update_model(optimizer, losses)
        log_errors(monitor, losses, nmsks, i)

        # Elapsed time.
        elapsed = time.time() - start
        log_elapsed_time(monitor, elapsed, i, "train")

        if val_sampler is not None and i % args.test_intv == 0:
            run_validation(model_w_loss, val_sampler, args.test_iter,
                           loss_fn, sample_spec, monitor, val_writer, i,
                           args, rank)

        if i % args.avgs_intv == 0 or i < last_iter + args.warm_up:
            monitor.compute_avgs(i, "train")

            # Displaying stats (both to console and TensorBoard)
            avg_losses = {k: monitor.get_last_value(k, "train")
                          for k in losses.keys()}
            avg_time = monitor.get_last_value("iter_time", "train")

            write_averages(train_writer, avg_losses, avg_time, i)

            # rounding losses for display
            print_log_output(i, avg_losses, avg_time, rank)

        if i % args.chkpt_intv == 0 and i != last_iter and args.rank == 0:
            printR0(rank, "SAVE CHECKPOINT: {} iters.".format(i))
            utils.save_chkpt(model, monitor, optimizer,
                             i, args.modeldir, args.logdir)
        torch.distributed.barrier()


def write_averages(writer, losses, time, i):
    """ Writes the average losses and iter time to a TensorBoard writer """
    if writer is not None:
        writer.add_scalar("Time Avg", time, i)
        for (k, v) in losses.items():
            writer.add_scalar("Loss {} Avg".format(k), v, i)
    writer.flush()


def log_elapsed_time(monitor, elapsed_time, i, phase="train"):
    """ Stores the iteration time within the LearningMonitor """
    monitor.add_to_num({"iter_time": elapsed_time}, phase)
    monitor.add_to_denom({"iter_time": 1}, phase)


def log_errors(monitor, losses, nmsks, i, phase="train"):
    """ Adds the losses to the running averages within the LearningMonitor """

    assert losses.keys() == nmsks.keys(), "Mismatched losses and nmsks"

    # Extracting values from Tensors
    losses = {k: v.item() for (k, v) in losses.items()}
    nmsks = {k: v.item() for (k, v) in nmsks.items()}

    monitor.add_to_num(losses, phase)
    monitor.add_to_denom(nmsks, phase)


def update_model(optimizer, losses):
    """ Runs the backward pass and updates model parameters """
    optimizer.zero_grad()
    total_loss = sum(losses.values())
    total_loss.backward()
    optimizer.step()


def params_defined(args):
    """ Checks whether all required parameters have been defined """
    for param in REQUIRED_PARAMS:
        if not hasattr(args, param):
            print(param)
            return False

    return True


def fetch_nonempty_sample(sampler, masks, num=1):
    """
    Pulls samples from the sampler with SOME unmasked
    voxels for each task
    """
    sample = next(sampler)

    while utils.masks_empty(sample, masks):
        sample = next(sampler)

    return sample


def run_validation(model_w_loss, sampler, num_iters, loss_fn,
                   sample_spec, monitor, writer, i, args, rank):

    mask_names = sample_spec.get_masks()
    printR0(rank, "------ BEGIN VALIDATION LOOP -------")
    with torch.no_grad():
        start = time.time()
        for j in range(num_iters):
            # Make sure no mask is empty (data for all tasks)
            sample = fetch_nonempty_sample(sampler, mask_names)

            inputs, labels, masks = utils.group_sample(sample, sample_spec,
                                                       args.device, "test")

            # Running forward pass, evaluating loss fn
            losses, nmsks, preds = model_w_loss(
                                   inputs, labels, masks, return_preds=True)

            losses, nmsks = sum_to_scalar(losses, nmsks)
            log_errors(monitor, losses, nmsks, i, "test")

            # Elapsed time.
            elapsed = time.time() - start
            log_elapsed_time(monitor, elapsed, i, "test")
            start = time.time()

    monitor.compute_avgs(i, "test")
    avg_losses = {k: monitor.get_last_value(k, "test")
                  for k in losses.keys()}
    avg_time = monitor.get_last_value("iter_time", "test")
    write_averages(writer, avg_losses, avg_time, i)

    write_images(writer, inputs, preds, labels, i)

    printR0(rank, "TEST ", end="")
    print_log_output(i, avg_losses, avg_time, rank)
    printR0(rank, "------- END VALIDATION LOOP --------")


def write_images(writer, inputs, preds, labels, i, sigmoid_preds=True):
    write_images_(writer, inputs, "input", i)
    write_images_(writer, preds, "pred", i, sigmoid=sigmoid_preds)
    write_images_(writer, labels, "label", i)
    writer.flush()


def write_images_(writer, vols, vol_type, i, sigmoid=False):
    for (j, vol) in enumerate(vols):
        if sigmoid:
            vol = torch.sigmoid(vol)

        if len(vol.size()) != 5:
            continue

        for k in range(vol.size(1)):
            tag = f"{vol_type}_{j}_{k}"
            writer.add_images(tag, vol[0:1, k, ...],
                              global_step=i, dataformats="CNHW")


def print_log_output(i, avg_losses, avg_time, rank):
    """Printing log output to the terminal screen"""
    printR0(rank, f"iter: {i}; ", end="")
    printR0(rank, "{ ", end="")
    for (k, avg) in avg_losses.items():
        printR0(rank, f"{k}: {avg:.2e}, ", end="")
    printR0(rank, "} ", end="")
    printR0(rank, f" (iter_time = {avg_time:.5f}s on avg)")


def sum_to_scalar(*args):
    """Adding losses/nmsks together that were evaluated in parallel"""
    new_args = list()
    for arg in args:
        new_args.append({k: v.sum() for (k, v) in arg.items()})

    return new_args


def printR0(rank, msg, *args, **kwargs):
    if rank == 0 or rank is None:
        print(msg, *args, **kwargs)
