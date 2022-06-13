"""
Helper functions to visualize motion. Run this script to visualize 10 random validation samples.

Copyright ETH Zurich, Manuel Kaufmann
"""
import numpy as np
from matplotlib import pyplot as plt, animation as animation
from matplotlib.animation import writers
from mpl_toolkits.mplot3d import Axes3D

from motion_metrics import get_closest_rotmat
from motion_metrics import is_valid_rotmat


_prop_cycle = plt.rcParams['axes.prop_cycle']
_colors = _prop_cycle.by_key()['color']


class Visualizer(object):
    """
    Helper class to visualize SMPL joint angles parameterized as rotation matrices.
    """
    def __init__(self, fk_engine):
        self.fk_engine = fk_engine
        self.is_sparse = True
        self.expected_n_input_joints = len(self.fk_engine.major_joints) if self.is_sparse else self.fk_engine.n_joints

    def visualize_with_gt(self, seed, prediction, target, title):
        """
        Visualize prediction and ground truth side by side.
        Args:
            seed: A np array of shape (seed_seq_length, n_joints*dof)
            prediction: A np array of shape (target_seq_length, n_joints*dof)
            target: A np array of shape (target_seq_length, n_joints*dof)
            title: Title of the plot
        """
        self.visualize_rotmat(seed, prediction, target, title)

    def visualize(self, seed, prediction, title):
        """
        Visualize prediction only.
        Args:
            seed: A np array of shape (seed_seq_length, n_joints*dof)
            prediction: A np array of shape (target_seq_length, n_joints*dof)
            title: Title of the plot
        """
        self.visualize_rotmat(seed, prediction, title=title)

    def visualize_rotmat(self, seed, prediction, target=None, title=''):
        def _to_positions(angles_):
            full_seq = np.concatenate([seed, angles_], axis=0)

            # Make sure the rotations are valid.
            full_seq_val = np.reshape(full_seq, [-1, n_joints, 3, 3])
            full_seq = get_closest_rotmat(full_seq_val)
            full_seq = np.reshape(full_seq, [-1, n_joints * dof])

            # Check that rotation matrices are valid.
            full_are_valid = is_valid_rotmat(np.reshape(full_seq, [-1, n_joints, 3, 3]))
            assert full_are_valid, 'rotation matrices are not valid rotations'

            # Compute positions.
            if self.is_sparse:
                full_seq_pos = self.fk_engine.from_sparse(full_seq, return_sparse=False)  # (N, full_n_joints, 3)
            else:
                full_seq_pos = self.fk_engine.from_rotmat(full_seq)

            # Swap y and z because SMPL defines y to be up.
            full_seq_pos = full_seq_pos[..., [0, 2, 1]]
            return full_seq_pos

        assert seed.shape[-1] == prediction.shape[-1] == self.expected_n_input_joints * 9
        n_joints = self.expected_n_input_joints
        dof = 9

        pred_pos = _to_positions(prediction)
        positions = [pred_pos]
        colors = [_colors[0]]
        titles = ['prediction']

        if target is not None:
            assert prediction.shape[-1] == target.shape[-1]
            assert prediction.shape[0] == target.shape[0]
            targ_pos = _to_positions(target)
            positions.append(targ_pos)
            colors.append(_colors[1])
            titles.append('target')

        visualize_positions(positions=positions,
                            colors=colors,
                            titles=titles,
                            fig_title=title,
                            parents=self.fk_engine.parents,
                            change_color_after_frame=(seed.shape[0], None))


def visualize_positions(positions, colors, titles, fig_title, parents, change_color_after_frame=None, overlay=False,
                        fps=60):
    """
    Visualize motion given 3D positions. Can visualize several motions side by side. If the sequence lengths don't
    match, all animations are displayed until the shortest sequence length.
    Args:
        positions: a list of np arrays in shape (seq_length, n_joints, 3) giving the 3D positions per joint and frame
        colors: list of color for each entry in `positions`
        titles: list of titles for each entry in `positions`
        fig_title: title for the entire figure
        parents: skeleton structure
        fps: frames per second
        change_color_after_frame: after this frame id, the color of the plot is changed (for each entry in `positions`)
        overlay: if true, all entries in `positions` are plotted into the same subplot
    """
    seq_length = np.amin([pos.shape[0] for pos in positions])
    n_joints = positions[0].shape[1]
    pos = positions

    # create figure with as many subplots as we have skeletons
    fig = plt.figure(figsize=(16, 9))
    plt.clf()
    n_axes = 1 if overlay else len(pos)
    axes = [fig.add_subplot(1, n_axes, i + 1, projection='3d') for i in range(n_axes)]
    fig.suptitle(fig_title)

    # create point object for every bone in every skeleton
    all_lines = []
    for i, joints in enumerate(pos):
        idx = 0 if overlay else i
        ax = axes[idx]
        lines_j = [
            ax.plot(joints[0:1, n,  0], joints[0:1, n, 1], joints[0:1, n, 2], '-o',
                    markersize=2.0, color=colors[i])[0] for n in range(1, n_joints)]
        all_lines.append(lines_j)
        ax.set_title(titles[i])

    # dirty hack to get equal axes behaviour
    min_val = np.amin(pos[0], axis=(0, 1))
    max_val = np.amax(pos[0], axis=(0, 1))
    max_range = (max_val - min_val).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (max_val[0] + min_val[0])
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (max_val[1] + min_val[1])
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (max_val[2] + min_val[2])

    for ax in axes:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.view_init(elev=0, azim=-56)

    def on_move(event):
        # find which axis triggered the event
        source_ax = None
        for i in range(len(axes)):
            if event.inaxes == axes[i]:
                source_ax = i
                break

        # transfer rotation and zoom to all other axes
        if source_ax is None:
            return

        for i in range(len(axes)):
            if i != source_ax:
                axes[i].view_init(elev=axes[source_ax].elev, azim=axes[source_ax].azim)
                axes[i].set_xlim3d(axes[source_ax].get_xlim3d())
                axes[i].set_ylim3d(axes[source_ax].get_ylim3d())
                axes[i].set_zlim3d(axes[source_ax].get_zlim3d())
        fig.canvas.draw_idle()

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig_text = fig.text(0.05, 0.05, '')

    def update_frame(num, positions, lines, parents, colors):
        for l in range(len(positions)):
            k = 0
            pos = positions[l]
            points_j = lines[l]
            for i in range(1, len(parents)):
                a = pos[num, i]
                b = pos[num, parents[i]]
                p = np.vstack([b, a])
                points_j[k].set_data(p[:, :2].T)
                points_j[k].set_3d_properties(p[:, 2].T)
                if change_color_after_frame and change_color_after_frame[l] and num >= change_color_after_frame[l]:
                    points_j[k].set_color(_colors[2])
                else:
                    points_j[k].set_color(colors[l])

                k += 1
        time_passed = '{:>.2f} seconds passed'.format(1/60.0*num)
        fig_text.set_text(time_passed)

    # create the animation object, for animation to work reference to this object must be kept
    line_ani = animation.FuncAnimation(fig, update_frame, seq_length,
                                       fargs=(pos, all_lines, parents, colors + [colors[0]]),
                                       interval=1000/fps)

    writergif = animation.PillowWriter(fps=60) 
    line_ani.save('prima_prova.gif', writer=writergif)
    plt.show()
    plt.close()


if __name__ == '__main__':
    import os
    from configuration import CONSTANTS as C
    from data import LMDBDataset
    from fk import SMPLForwardKinematics

    valid_data = LMDBDataset(os.path.join(C.DATA_DIR, "validation"), transform=None)
    visualizer = Visualizer(SMPLForwardKinematics())
    for i in range(10):
        sample = valid_data[i]
        visualizer.visualize(sample.poses[:120], sample.poses[120:],
                             'random validation sample {}'.format(sample.seq_id))
