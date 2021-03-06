"""
Loss functions.

Nicholas Turner <nturner.cs@princeton.edu>, 2017
"""

import torch
from torch import nn
import numpy as np

from .balance import gunpowder_balance


class BinomialCrossEntropyWithLogits(nn.Module):
    """ 
    A version of BCE w/ logits with the ability to mask
    out regions of output
    """

    def __init__(self):
      nn.Module.__init__(self)
      self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, label, mask=None):

      #Need masking for this application
      # copied from PyTorch's github repo
      err = self.bce(pred, label)

      if mask is None:
        cost = err.sum() #/ np.prod(err.size())
      else:
        cost = (err * mask).sum() #/ mask.sum()

      return cost


class MSELoss(nn.Module):
    """
    A version of BCE w/ logits with the ability to mask
    out regions of output
    """

    def __init__(self, size_average=True):
        nn.Module.__init__(self)
        self.mse = nn.MSELoss(reduction='none')
        self.reduce = torch.mean if size_average else torch.sum

    def forward(self, pred, label, mask=None):

        loss = self.mse(pred,label)
        if mask is not None:
            loss *= mask
        return self.reduce(loss)
      

class MSELossRebal(nn.Module):
    """ Mean Squared Error with rebalancing"""

    def __init__(self, balance_thresh=0):
        
        nn.Module.__init__(self)

        self.mse = nn.functional.mse_loss
        self.balance_thresh = balance_thresh

    def forward(self, pred, label, mask):
        mask *= gunpowder_balance(label, mask, self.balance_thresh)
        return (self.mse(pred, label, reduce=False) * mask).sum()
