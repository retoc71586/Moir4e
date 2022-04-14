"""
Helper functions to compute evaluation metrics.

Copyright ETH Zurich, Manuel Kaufmann
"""
import cv2
import copy
import numpy as np
import torch

from fk import SMPL_MAJOR_JOINTS
from fk import SMPL_NR_JOINTS
from fk import SMPL_PARENTS
from fk import sparse_to_full
from fk import local_rot_to_global


def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden


def is_valid_rotmat(rotmats, thresh=1e-6):
    """
    Checks that the rotation matrices are valid, i.e. R*R' == I and det(R) == 1
    Args:
        rotmats: A np array of shape (..., 3, 3).
        thresh: Numerical threshold.

    Returns:
        True if all rotation matrices are valid, False if at least one is not valid.
    """
    # check we have a valid rotation matrix
    rotmats_t = np.transpose(rotmats, tuple(range(len(rotmats.shape[:-2]))) + (-1, -2))
    is_orthogonal = np.all(np.abs(np.matmul(rotmats, rotmats_t) - eye(3, rotmats.shape[:-2])) < thresh)
    det_is_one = np.all(np.abs(np.linalg.det(rotmats) - 1.0) < thresh)
    return is_orthogonal and det_is_one


def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).

    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest


def angle_diff(predictions, targets):
    """
    Computes the angular distance between the target and predicted rotations. We define this as the angle that is
    required to rotate one rotation into the other. This essentially computes || log(R_diff) || where R_diff is the
    difference rotation between prediction and target.

    Args:
        predictions: np array of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3)
        targets: np array of same shape as `predictions`

    Returns:
        The geodesic distance for each joint as an np array of shape (..., n_joints)
    """
    assert predictions.shape[-1] == predictions.shape[-2] == 3
    assert targets.shape[-1] == targets.shape[-2] == 3

    ori_shape = predictions.shape[:-2]
    preds = np.reshape(predictions, [-1, 3, 3])
    targs = np.reshape(targets, [-1, 3, 3])

    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = np.matmul(preds, np.transpose(targs, [0, 2, 1]))

    # convert `r` to angle-axis representation and extract the angle, which is our measure of difference between
    # the predicted and target orientations
    angles = []
    for i in range(r.shape[0]):
        aa, _ = cv2.Rodrigues(r[i])
        angles.append(np.linalg.norm(aa))
    angles = np.array(angles)

    return np.reshape(angles, ori_shape)


class MetricsEngine(object):
    """
    Compute and aggregate various motion metrics. It keeps track of the metric values per frame, so that we can
    evaluate them for different sequence lengths. It assumes that inputs are in rotation matrix format.
    """
    def __init__(self, target_lengths):
        """
        Initializer.
        Args:
            target_lengths: List of target sequence lengths that should be evaluated.
        """
        self.target_lengths = target_lengths
        self.all_summaries_op = None
        self.n_samples = 0
        self._should_call_reset = False  # a guard to avoid stupid mistakes
        self.metrics_agg = {"joint_angle": None}
        self.summaries = {k: {t: None for t in target_lengths} for k in self.metrics_agg}

    def reset(self):
        """
        Reset all metrics.
        """
        self.metrics_agg = {"joint_angle": None}
        self.n_samples = 0
        self._should_call_reset = False  # now it's again safe to compute new values

    def compute(self, predictions, targets, reduce_fn="mean"):
        """
        Compute the joint angle metric. Predictions and targets are assumed to be in rotation matrix format.
        Args:
            predictions: An np array of shape (n, seq_length, n_joints*9)
            targets: An np array of the same shape as `predictions`
            reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].

        Returns:
            A dictionary {"joint_angle" -> values} where the values are given per batch entry and frame as an np array
            of shape (n, seq_length).
        """
        assert predictions.shape[-1] % 9 == 0, "predictions are not rotation matrices"
        assert targets.shape[-1] % 9 == 0, "targets are not rotation matrices"
        assert reduce_fn in ["mean", "sum"]
        assert not self._should_call_reset, "you should reset the state of this class after calling `finalize`"
        dof = 9
        n_joints = len(SMPL_MAJOR_JOINTS)
        batch_size = predictions.shape[0]
        seq_length = predictions.shape[1]
        assert n_joints*dof == predictions.shape[-1], "unexpected number of joints"

        # first reshape everything to (-1, n_joints * 9)
        pred = np.reshape(predictions, [-1, n_joints*dof]).copy()
        targ = np.reshape(targets, [-1, n_joints*dof]).copy()

        # enforce valid rotations
        pred_val = np.reshape(pred, [-1, n_joints, 3, 3])
        pred = get_closest_rotmat(pred_val)
        pred = np.reshape(pred, [-1, n_joints*dof])

        # check that the rotations are valid
        pred_are_valid = is_valid_rotmat(np.reshape(pred, [-1, n_joints, 3, 3]))
        assert pred_are_valid, 'predicted rotation matrices are not valid'
        targ_are_valid = is_valid_rotmat(np.reshape(targ, [-1, n_joints, 3, 3]))
        assert targ_are_valid, 'target rotation matrices are not valid'

        # add missing joints
        pred = sparse_to_full(pred, SMPL_MAJOR_JOINTS, SMPL_NR_JOINTS, rep="rot_mat")
        targ = sparse_to_full(targ, SMPL_MAJOR_JOINTS, SMPL_NR_JOINTS, rep="rot_mat")

        # make sure we don't consider the root orientation
        assert pred.shape[-1] == SMPL_NR_JOINTS*dof
        assert targ.shape[-1] == SMPL_NR_JOINTS*dof
        pred[:, 0:9] = np.eye(3, 3).flatten()
        targ[:, 0:9] = np.eye(3, 3).flatten()

        metrics = dict()
        select_joints = SMPL_MAJOR_JOINTS
        reduce_fn_np = np.mean if reduce_fn == "mean" else np.sum

        # compute the joint angle diff on the global rotations
        pred_global = local_rot_to_global(pred, SMPL_PARENTS, left_mult=False,
                                          rep="rot_mat")  # (-1, full_n_joints, 3, 3)
        targ_global = local_rot_to_global(targ, SMPL_PARENTS, left_mult=False,
                                          rep="rot_mat")  # (-1, full_n_joints, 3, 3)
        v = angle_diff(pred_global[:, select_joints], targ_global[:, select_joints])  # (-1, n_joints)
        v = np.reshape(v, [batch_size, seq_length, n_joints])
        metrics["joint_angle"] = reduce_fn_np(v, axis=-1)

        return metrics

    def aggregate(self, new_metrics):
        """
        Aggregate the metrics.
        Args:
            new_metrics: Dictionary of new metric values to aggregate. Each entry is expected to be a numpy array
            of shape (batch_size, seq_length).
        """
        assert isinstance(new_metrics, dict)
        assert list(new_metrics.keys()) == list(self.metrics_agg.keys())

        # sum over the batch dimension
        for m in new_metrics:
            if self.metrics_agg[m] is None:
                self.metrics_agg[m] = np.sum(new_metrics[m], axis=0)
            else:
                self.metrics_agg[m] += np.sum(new_metrics[m], axis=0)

        # keep track of the total number of samples processed
        batch_size = new_metrics[list(new_metrics.keys())[0]].shape[0]
        self.n_samples += batch_size

    def compute_and_aggregate(self, predictions, targets, reduce_fn="mean"):
        """
        Computes the joint angle metric values and aggregates them directly.
        Args:
            predictions: An np array of shape (n, seq_length, n_joints*dof)
            targets: An np array of the same shape as `predictions`
            reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].
        """
        if isinstance(predictions, torch.Tensor):
            ps = predictions.detach().cpu().numpy()
            ts = targets.detach().cpu().numpy()
        else:
            ps = predictions
            ts = targets
        new_metrics = self.compute(ps, ts, reduce_fn)
        self.aggregate(new_metrics)

    def get_final_metrics(self):
        """
        Finalize and return the metrics - this should only be called once all the data has been processed.
        :return: A dictionary of the final aggregated metrics per time step.
        """
        self._should_call_reset = True  # make sure to call `reset` before new values are computed
        assert self.n_samples > 0

        for m in self.metrics_agg:
            self.metrics_agg[m] = self.metrics_agg[m] / self.n_samples

        # return a copy of the metrics so that the class can be re-used again immediately
        return copy.deepcopy(self.metrics_agg)

    @staticmethod
    def get_summary_string(final_metrics):
        """Create a summary string from the given metrics, e.g. for printing to the console."""
        seq_length = final_metrics[list(final_metrics.keys())[0]].shape[0]
        s = "metrics until {}:".format(seq_length)
        for m in sorted(final_metrics):
            val = np.sum(final_metrics[m])
            s += "   {}: {:.3f}".format(m, val)
        return s

    def to_tensorboard_log(self, metrics, writer, global_step, prefix=''):
        """Write metrics to tensorboard."""
        for m in metrics:
            for t in self.target_lengths:
                metric_value = np.sum(metrics[m][:t])
                writer.add_scalar('{}/{}/until {}'.format(m, prefix, t), metric_value, global_step)
