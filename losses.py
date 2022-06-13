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


def rmse(predictions, targets):
    """
    Compute the RMSE.
    :param predictions: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :param targets: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :return: The RMSE between predictions and targets.
    """
    diff = predictions - targets
    loss_per_sample_and_seq = (diff * diff).sum(dim=-1)  # (N, F)
    loss_per_sample_and_seq = loss_per_sample_and_seq.sqrt()
    return loss_per_sample_and_seq.mean()


def loss_pose_all_mean(predictions, targets):
    """
    Loss computed as described https://github.com/eth-ait/spl/blob/6b37cc0a61c69b6e43187800d6589eb9cfaa9799/spl/model/base_model.py
    """
    diff = predictions - targets
    return (diff * diff).mean()


def loss_pose_joint_sum(predictions, targets, n_frames = 144):
    """
    Loss computed as described https://github.com/eth-ait/spl/blob/6b37cc0a61c69b6e43187800d6589eb9cfaa9799/spl/model/base_model.py
    """

    n_baches, n_frames, joints_x_dof = predictions.shape

    joints = 15
    dof = joints_x_dof // joints

    diff = predictions - targets
    loss_per_joint = (diff * diff).view(-1, n_frames, joints, dof)
    loss_per_joint = loss_per_joint.sum(dim=-1)
    loss_per_joint = loss_per_joint.sqrt() # (N, F, J)
    loss_per_sample_and_seq = loss_per_joint.sum(dim=-1) # (N, F)

    return loss_per_sample_and_seq.mean()

def loss_pose_joint_sum_squared(predictions, targets):
    """
    Loss computed as described https://github.com/eth-ait/spl/blob/6b37cc0a61c69b6e43187800d6589eb9cfaa9799/spl/model/base_model.py
    """
    diff = predictions - targets
    per_joint_loss = (diff * diff).view(-1, 143, 15, 9) #TODO: adjust such that it doesn't use hardcoded
    per_joint_loss = per_joint_loss.sum(dim=-1)
    per_joint_loss = per_joint_loss.sum(dim=-1)

    return per_joint_loss.mean()


def mse_joint(predictions, targets):
    """
    Compute the MSE.
    :param predictions: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :param targets: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :return: The MSE between predictions and targets.
    """
    diff = predictions - targets
    loss_per_sample_and_seq = (diff * diff).sum(dim=-1)  # (N, F)
    return loss_per_sample_and_seq.sum()

def avg_l1(predictions, targets):
    """ 
    The average l1 loss from:
    https://github.com/wei-mao-2019/LearnTrajDep/blob/master/utils/loss_funcs.py#L7
    """
    diff = predictions - targets
    loss_per_sample_and_seq = torch.abs(diff).sum(dim=-1) 
    return loss_per_sample_and_seq.mean()
