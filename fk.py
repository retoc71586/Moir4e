"""
Helper functions to perform forward kinematics with the SMPL body model.

Copyright ETH Zurich, Manuel Kaufmann
"""
import numpy as np
import quaternion
import cv2

SMPL_MAJOR_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
SMPL_NR_JOINTS = 24
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
SMPL_JOINTS = ['pelvis', 'l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee', 'spine2', 'l_ankle', 'r_ankle', 'spine3',
               'l_foot', 'r_foot', 'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder',
               'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand']
SMPL_JOINT_MAPPING = {i: x for i, x in enumerate(SMPL_JOINTS)}


def sparse_to_full(joint_angles_sparse, sparse_joints_idxs, tot_nr_joints, rep="rot_mat"):
    """
    Pad the given sparse joint angles with identity elements to retrieve a full skeleton with `tot_nr_joints`
    many joints.
    Args:
        joint_angles_sparse: An np array of shape (N, len(sparse_joints_idxs) * dof)
          or (N, len(sparse_joints_idxs), dof)
        sparse_joints_idxs: A list of joint indices pointing into the full skeleton given by range(0, tot_nr_joints)
        tot_nr_jonts: Total number of joints in the full skeleton.
        rep: Which representation is used, rot_mat or quat

    Returns:
        The padded joint angles as an array of shape (N, tot_nr_joints*dof)
    """
    joint_idxs = sparse_joints_idxs
    assert rep in ["rot_mat", "quat", "aa"]
    dof = 9 if rep == "rot_mat" else 4 if rep == "quat" else 3
    n_sparse_joints = len(sparse_joints_idxs)
    angles_sparse = np.reshape(joint_angles_sparse, [-1, n_sparse_joints, dof])

    # fill in the missing indices with the identity element
    smpl_full = np.zeros(shape=[angles_sparse.shape[0], tot_nr_joints, dof])  # (N, tot_nr_joints, dof)
    if rep == "quat":
        smpl_full[..., 0] = 1.0
    elif rep == "rot_mat":
        smpl_full[..., 0] = 1.0
        smpl_full[..., 4] = 1.0
        smpl_full[..., 8] = 1.0
    else:
        pass  # nothing to do for angle-axis

    smpl_full[:, joint_idxs] = angles_sparse
    smpl_full = np.reshape(smpl_full, [-1, tot_nr_joints * dof])
    return smpl_full


def local_rot_to_global(joint_angles, parents, rep="rot_mat", left_mult=False):
    """
    Converts local rotations into global rotations by "unrolling" the kinematic chain.
    Args:
        joint_angles: An np array of rotation matrices of shape (N, nr_joints*dof)
        parents: A np array specifying the parent for each joint
        rep: Which representation is used for `joint_angles`
        left_mult: If True the local matrix is multiplied from the left, rather than the right

    Returns:
        The global rotations as an np array of rotation matrices in format (N, nr_joints, 3, 3)
    """
    assert rep in ["rot_mat", "quat", "aa"]
    n_joints = len(parents)
    if rep == "rot_mat":
        rots = np.reshape(joint_angles, [-1, n_joints, 3, 3])
    elif rep == "quat":
        rots = quaternion.as_rotation_matrix(quaternion.from_float_array(
            np.reshape(joint_angles, [-1, n_joints, 4])))
    else:
        rots = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(
            np.reshape(joint_angles, [-1, n_joints, 3])))

    out = np.zeros_like(rots)
    dof = rots.shape[-3]
    for j in range(dof):
        if parents[j] < 0:
            # root rotation
            out[..., j, :, :] = rots[..., j, :, :]
        else:
            parent_rot = out[..., parents[j], :, :]
            local_rot = rots[..., j, :, :]
            lm = local_rot if left_mult else parent_rot
            rm = parent_rot if left_mult else local_rot
            out[..., j, :, :] = np.matmul(lm, rm)

    return out


class ForwardKinematics(object):
    """
    FK Engine
    """
    def __init__(self, offsets, parents, left_mult=False, major_joints=None, norm_idx=None, no_root=True):
        self.offsets = offsets
        if norm_idx is not None:
            self.offsets = self.offsets / np.linalg.norm(self.offsets[norm_idx])
        self.parents = parents
        self.n_joints = len(parents)
        self.major_joints = major_joints
        self.left_mult = left_mult
        self.no_root = no_root
        assert self.offsets.shape[0] == self.n_joints

    def fk(self, joint_angles):
        """
        Perform forward kinematics. This requires joint angles to be in rotation matrix format.
        Args:
            joint_angles: np array of shape (N, n_joints*3*3)

        Returns:
            The 3D joint positions as a an array of shape (N, n_joints, 3)
        """
        assert joint_angles.shape[-1] == self.n_joints * 9
        angles = np.reshape(joint_angles, [-1, self.n_joints, 3, 3])
        n_frames = angles.shape[0]
        positions = np.zeros([n_frames, self.n_joints, 3])
        rotations = np.zeros([n_frames, self.n_joints, 3, 3])  # intermediate storage of global rotation matrices
        if self.left_mult:
            offsets = self.offsets[np.newaxis, np.newaxis, ...]  # (1, 1, n_joints, 3)
        else:
            offsets = self.offsets[np.newaxis, ..., np.newaxis]  # (1, n_joints, 3, 1)

        if self.no_root:
            angles[:, 0] = np.eye(3)

        for j in range(self.n_joints):
            if self.parents[j] == -1:
                # this is the root, we don't consider any root translation
                positions[:, j] = 0.0
                rotations[:, j] = angles[:, j]
            else:
                # this is a regular joint
                if self.left_mult:
                    positions[:, j] = np.squeeze(np.matmul(offsets[:, :, j], rotations[:, self.parents[j]])) + \
                                      positions[:, self.parents[j]]
                    rotations[:, j] = np.matmul(angles[:, j], rotations[:, self.parents[j]])
                else:
                    positions[:, j] = np.squeeze(np.matmul(rotations[:, self.parents[j]], offsets[:, j])) + \
                                      positions[:, self.parents[j]]
                    rotations[:, j] = np.matmul(rotations[:, self.parents[j]], angles[:, j])

        return positions

    def from_aa(self, joint_angles):
        """
        Get joint positions from angle axis representations in shape (N, n_joints*3).
        """
        angles = np.reshape(joint_angles, [-1, self.n_joints, 3])
        angles_rot = np.zeros(angles.shape + (3,))
        for i in range(angles.shape[0]):
            for j in range(self.n_joints):
                angles_rot[i, j] = cv2.Rodrigues(angles[i, j])[0]
        return self.fk(np.reshape(angles_rot, [-1, self.n_joints * 9]))

    def from_rotmat(self, joint_angles):
        """
        Get joint positions from rotation matrix representations in shape (N, N_JOINTS*3*3).
        """
        return self.fk(joint_angles)

    def from_sparse(self, joint_angles_sparse, rep="rot_mat", return_sparse=True):
        """
        Get joint positions from reduced set of H36M joints.
        Args:
            joint_angles_sparse: np array of shape (N, len(sparse_joint_idxs) * dof))
            sparse_joints_idxs: List of indices into `H36M_JOINTS` pointing out which SMPL joints are used in
              `pose_sparse`. If None defaults to `H36M_MAJOR_JOINTS`.
            rep: "rot_mat" or "quat", which representation is used for the angles in `joint_angles_sparse`
            return_sparse: If True it will return only the positions of the joints given in `sparse_joint_idxs`.

        Returns:
            The joint positions as an array of shape (N, len(sparse_joint_idxs), 3) if `return_sparse` is True
            otherwise (N, H36M_NR_JOINTS, 3).
        """
        assert self.major_joints is not None
        assert rep == "rot_mat"
        smpl_full = sparse_to_full(joint_angles_sparse, self.major_joints, self.n_joints, rep)
        positions = self.from_rotmat(smpl_full)
        if return_sparse:
            positions = positions[:, self.major_joints]
        return positions


class SMPLForwardKinematics(ForwardKinematics):
    """
    Forward Kinematics for the skeleton defined by SMPL.
    """
    def __init__(self):
        # this are the offsets stored under `J` in the SMPL model pickle file
        offsets = np.array([[-8.76308970e-04, -2.11418723e-01, 2.78211200e-02],
                            [7.04848876e-02, -3.01002533e-01, 1.97749280e-02],
                            [-6.98883278e-02, -3.00379160e-01, 2.30254335e-02],
                            [-3.38451650e-03, -1.08161861e-01, 5.63597909e-03],
                            [1.01153808e-01, -6.65211904e-01, 1.30860155e-02],
                            [-1.06040718e-01, -6.71029623e-01, 1.38401121e-02],
                            [1.96440985e-04, 1.94957852e-02, 3.92296547e-03],
                            [8.95999143e-02, -1.04856032e+00, -3.04155922e-02],
                            [-9.20120818e-02, -1.05466743e+00, -2.80514913e-02],
                            [2.22362284e-03, 6.85680141e-02, 3.17901760e-02],
                            [1.12937580e-01, -1.10320516e+00, 8.39545265e-02],
                            [-1.14055299e-01, -1.10107698e+00, 8.98482216e-02],
                            [2.60992373e-04, 2.76811197e-01, -1.79753042e-02],
                            [7.75218998e-02, 1.86348444e-01, -5.08464100e-03],
                            [-7.48091986e-02, 1.84174211e-01, -1.00204779e-02],
                            [3.77815350e-03, 3.39133394e-01, 3.22299558e-02],
                            [1.62839013e-01, 2.18087461e-01, -1.23774789e-02],
                            [-1.64012068e-01, 2.16959041e-01, -1.98226746e-02],
                            [4.14086325e-01, 2.06120683e-01, -3.98959248e-02],
                            [-4.10001734e-01, 2.03806676e-01, -3.99843890e-02],
                            [6.52105424e-01, 2.15127546e-01, -3.98521818e-02],
                            [-6.55178550e-01, 2.12428626e-01, -4.35159074e-02],
                            [7.31773168e-01, 2.05445019e-01, -5.30577698e-02],
                            [-7.35578759e-01, 2.05180646e-01, -5.39352281e-02]])

        # need to convert them to compatible offsets
        smpl_offsets = np.zeros([24, 3])
        smpl_offsets[0] = offsets[0]
        for idx, pid in enumerate(SMPL_PARENTS[1:]):
            smpl_offsets[idx+1] = offsets[idx + 1] - offsets[pid]

        # normalize so that right thigh has length 1
        super(SMPLForwardKinematics, self).__init__(smpl_offsets, SMPL_PARENTS, norm_idx=4,
                                                    left_mult=False, major_joints=SMPL_MAJOR_JOINTS)
