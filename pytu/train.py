"""
Training

Nicholas Turner <nturner@cs.princeton.edu>, 2017-9
"""
import time

import torch

from . import utils


REQUIRED_PARAMS = ["max_iter", "test_intv", "test_iter",
                   "avgs_intv", "chkpt_intv", "expt_dir",
                   "model_dir", "log_dir", "batch_size",
                   "warm_up"]


def train(model, loss_fn, optimizer, sampler, val_sampler=None, last_iter=0,
          train_writer=None, val_writer=None, monitor=None, **params):
    """ Generalized training function """

    assert params_defined(params), "Params under-specified"

    if monitor is None:
        monitor = utils.LearningMonitor()

    # Determine the names of inputs, labels, masks
    sample_spec = utils.SampleSpec(next(sampler).keys())
    mask_names = sample_spec.get_masks()

    model_w_loss = utils.wrapmodel(model, loss_fn, sample_spec)

    print("======= BEGIN TRAINING LOOP ========")
    for i in range(last_iter, params['max_iter']):
        start = time.time()

        # Make sure no mask is empty (data for all tasks)
        sample = fetch_nonempty_sample(sampler, mask_names)

        inputs, labels, masks = group_sample(sample, sample_spec, "train")

        # Running forward pass, evaluating loss fn
        losses, nmsks = model_w_loss(inputs, labels, masks)

        losses, nmsks = sum_to_scalar(losses, nmsks)
        update_model(optimizer, losses)
        log_errors(monitor, losses, nmsks, i)

        # Elapsed time.
        elapsed = time.time() - start
        log_elapsed_time(monitor, elapsed, i, "train")

        if val_sampler is not None and i % params["test_intv"] == 0:
            run_validation(model_w_loss, val_sampler, params["test_iter"],
                           loss_fn, sample_spec, monitor, val_writer, i)

        if i % params["avgs_intv"] == 0 or i < last_iter + params["warm_up"]:
            monitor.compute_avgs(i, "train")

            # Displaying stats (both to console and TensorBoard)
            avg_losses = {k: monitor.get_last_value(k, "train")
                          for k in losses.keys()}
            avg_time = monitor.get_last_value("iter_time", "train")

            write_averages(train_writer, avg_losses, avg_time, i)

            # rounding losses for display
            print_log_output(i, avg_losses, avg_time)

        if i % params["chkpt_intv"] == 0 and i != last_iter:
            print("SAVE CHECKPOINT: {} iters.".format(i))
            utils.save_chkpt(model, monitor, optimizer, 
                             i, params["model_dir"],
                             params["log_dir"])


def write_averages(writer, losses, time, i):
    """ Writes the average losses and iter time to a TensorBoard writer """
    if writer is not None:
        writer.add_scalar("Time Avg", time, i)
        for (k, v) in losses.items():
            writer.add_scalar("Loss {} Avg".format(k), v, i)


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


def params_defined(params):
    """ Checks whether all required parameters have been defined """

    defined_keys = set(params.keys())
    for param in REQUIRED_PARAMS:
        if param not in defined_keys:
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


def group_sample(sample, sample_spec, phase="train"):
    """ Creates the Torch tensors for a sample """

    inputs = sample_spec.get_inputs()
    labels = sample_spec.get_labels()
    masks = sample_spec.get_masks()

    input_vars = [utils.to_torch(sample[k], block=True) for k in inputs]
    label_vars = [utils.to_torch(sample[k], block=False) for k in labels]
    mask_vars = [utils.to_torch(sample[k], block=False) for k in masks]

    return input_vars, label_vars, mask_vars


def run_validation(model_w_loss, sampler, num_iters, loss_fn,
                   sample_spec, monitor, writer, i):

    mask_names = sample_spec.get_masks()
    print("------- BEGIN VALIDATION LOOP --------")
    with torch.no_grad():
        start = time.time()
        for j in range(num_iters):

            # Make sure no mask is empty (data for all tasks)
            sample = fetch_nonempty_sample(sampler, mask_names)

            inputs, labels, masks = group_sample(sample, sample_spec, "test")

            # Running forward pass, evaluating loss fn
            losses, nmsks = model_w_loss(inputs, labels, masks)

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

    print("TEST ", end="")
    print_log_output(i, avg_losses, avg_time)
    print("------- END VALIDATION LOOP --------")


def print_log_output(i, avg_losses, avg_time):
    """Printing log output to the terminal screen"""
    print(f"iter: {i}; ", end="")
    print("{ ", end="")
    for (k, avg) in avg_losses.items():
        print(f"{k}: {avg:.2e}, ", end="")
    print("} ", end="")
    print(f" (iter_time = {avg_time:.5f}s on avg)")


def sum_to_scalar(*args):
    """Adding losses/nmsks together that were evaluated in parallel"""
    new_args = list()
    for arg in args:
        new_args.append({k: v.sum() for (k, v) in arg.items()})

    return new_args
