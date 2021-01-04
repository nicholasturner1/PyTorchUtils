import torch
from torch import nn


def wrapmodel(model, loss_fn, sample_spec):

    if isinstance(model, nn.DataParallel):
        return nn.DataParallel(WrappedModel(
                                   model.module, loss_fn, sample_spec)).cuda()
    else:
        return WrappedModel(model, loss_fn, sample_spec)


class WrappedModel(nn.Module):

    def __init__(self, model, loss_fn, sample_spec):
        """
        Wrapping the loss function evaluation with the forward pass to
        minimize cross-gpu talk
        """
        super(WrappedModel, self).__init__()

        self.model = model
        self.loss_fn = loss_fn
        self.sample_spec = sample_spec

    def forward(self, inputs, labels, masks, return_preds=False):
        preds = self.model(*inputs)
        losses, nmsks = self.eval_error(preds, labels, masks)
        if not return_preds:
            return losses, nmsks
        else:
            return losses, nmsks, preds

    def eval_error(self, preds, labels, masks):
        """
        Evaluates the error of the predictions according to the available
        labels and masks

        Assumes labels are ordered according to the sample_spec
        """
        label_names = self.sample_spec.get_labels()

        assert len(label_names) == len(labels), \
            "Mismatched labels and label names"
        assert len(preds) == len(labels), \
            "Mismatched preds and labels"

        losses, nmsks = dict(), dict()

        for (pred, label, label_name) in zip(preds, labels, label_names):

            if self.sample_spec.has_mask(label_name):
                mask = masks[self.sample_spec.get_mask_index(label_name)]

                losses[label_name] = self.loss_fn(pred, label, mask)
                nmsks[label_name] = mask.sum()

            else:
                losses[label_name] = self.loss_fn(pred, label)
                # Wrapping the value in a torch Tensor to give a
                #  uniform interface
                # (particularly for log_errors and DataParallel's gather)
                nmsks[label_name] = torch.tensor((label.nelement(),),
                                                 device=label.device)

        # DataParallel doesn't like concatenating scalars
        losses, nmsks = self.format_losses(losses, nmsks)
        return losses, nmsks

    def format_losses(self, *args):
        """
        DataParallel doesn't like concatenating scalars (gives a warning),
        so I'll add an extra dimension to those values.
        """
        new_args = list()
        for arg in args:
            new_arg = dict()
            for (k, v) in arg.items():
                new_arg[k] = v.unsqueeze(0) if v.ndim == 0 else v
            new_args.append(new_arg)

        return new_args
