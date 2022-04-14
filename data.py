"""
Loading data stored in LMDB format.

Copyright ETH Zurich, Manuel Kaufmann
"""
import lmdb
import numpy as np
import os
import torch

from configuration import CONSTANTS as C
from torch.utils.data import Dataset


class AMASSSample(object):
    """Represents a single sequence of poses."""

    def __init__(self, seq_id, poses):
        """
        Initializer.
        :param seq_id: The unique identifier of this sequence.
        :param poses: A numpy array of shape (N_FRAMES, N_JOINTS * DOF)
        """
        self.seq_id = seq_id
        self.poses = poses  # (N_FRAMES, N_JOINTS * DOF)

    @property
    def n_frames(self):
        return self.poses.shape[0]

    def to_tensor(self):
        """Create PyTorch tensors."""
        self.poses = torch.from_numpy(self.poses).to(dtype=C.DTYPE)
        return self

    def extract_window(self, start_frame, end_frame):
        """Extract a subwindow."""
        return AMASSSample(self.seq_id, self.poses[start_frame:end_frame])


class AMASSBatch(object):
    """Represents a mini-batch of `AMASSSample`s."""

    def __init__(self, seq_ids, poses):
        """
        Initializer.
        :param seq_ids: List of IDs per sequence.
        :param poses: A tensor of shape (N, MAX_SEQ_LEN, N_JOINTS*DOF)
        """
        self.seq_ids = seq_ids
        self.poses = poses

    @staticmethod
    def from_sample_list(samples):
        """Collect a set of AMASSSamples into a batch with padding if necessary."""
        ids = []
        poses = []
        for sample in samples:
            ids.append(sample.seq_id)
            poses.append(sample.poses)
        # Here we would typically need to pad the batch, but we assume at thi point windows of fixed sizes are
        # extracted from an AMASSSample, so no need for any padding.
        poses = torch.stack(poses)
        return AMASSBatch(ids, poses)

    @property
    def batch_size(self):
        return self.poses.shape[0]

    @property
    def seq_length(self):
        """The maximum sequence length of this batch (i.e. including potential padding)."""
        return self.poses.shape[1]

    def to_gpu(self):
        """Move data to GPU."""
        self.poses = self.poses.to(device=C.DEVICE)
        return self


class LMDBDataset(Dataset):
    """Access LMDB data."""

    def __init__(self, lmdb_path, transform, filter_seq_len=None):
        """
        Initializer.
        :param lmdb_path: Path to the LMDB dataset.
        :param transform: Pytorch transforms to be applied to each sample, can be None.
        :param filter_seq_len: Minimum sequence length for samples. Any sample below (if given) will be rejected.
        """
        super(LMDBDataset).__init__()
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env = None
        self.dof = 135  # 15 joints, 3x3 rotation matrices for each joint

        self.open_lmdb()
        with self.env.begin(write=False) as txn:
            seq_lens_bytes = txn.get('seq_lengths'.encode())
            self.seq_lengths = np.frombuffer(seq_lens_bytes, dtype=np.int64).copy()
        self.env.close()
        self.env = None

        if filter_seq_len is not None:
            self.valid_idxs = np.where(self.seq_lengths >= filter_seq_len)[0]
        else:
            self.valid_idxs = list(range(len(self.seq_lengths)))

    def open_lmdb(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, subdir=os.path.isdir(self.lmdb_path),
                                 readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return len(self.valid_idxs)

    def __getitem__(self, idx):
        self.open_lmdb()
        index = self.valid_idxs[idx]
        poses_key = "poses{}".format(index).encode()
        id_key = "id{}".format(index).encode()
        with self.env.begin(write=False) as txn:
            sample = AMASSSample(seq_id=txn.get(id_key).decode(),
                                 poses=np.frombuffer(txn.get(poses_key), dtype=np.float32).copy().reshape(-1, self.dof))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

