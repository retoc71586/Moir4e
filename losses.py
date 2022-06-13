"""
Some loss functions.

Copyright ETH Zurich, Manuel Kaufmann
"""
import torch

def mse(predictions, targets):
    """
    Compute the MSE.
    :param predictions: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :param targets: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :return: The MSE between predictions and targets.
    """
    diff = predictions - targets
    loss_per_sample_and_seq = (diff * diff).sum(dim=-1)  # (N, F)
    return loss_per_sample_and_seq.mean()


def avg_l1(predictions, targets):
    diff = predictions - targets
    loss_per_sample_and_seq = torch.abs(diff).sum(dim=-1) 
    return loss_per_sample_and_seq.mean()
