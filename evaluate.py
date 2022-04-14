"""
Evaluate a model on the test set.

Copyright ETH Zurich, Manuel Kaufmann
"""
import argparse
import numpy as np
import os
import pandas as pd
import torch
import utils as U

from configuration import Configuration
from configuration import CONSTANTS as C
from data import AMASSBatch
from data import LMDBDataset
from data_transforms import ToTensor
from fk import SMPLForwardKinematics
from models import create_model
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from visualize import Visualizer


def _export_results(eval_result, output_file):
    """
    Write predictions into a file that can be uploaded to the submission system.
    :param eval_result: A dictionary {sample_id => (prediction, seed)}
    :param output_file: Where to store the file.
    """

    def to_csv(fname, poses, ids, split=None):
        n_samples, seq_length, dof = poses.shape
        data_r = np.reshape(poses, [n_samples, seq_length * dof])
        cols = ['dof{}'.format(i) for i in range(seq_length * dof)]

        # add split id very last
        if split is not None:
            data_r = np.concatenate([data_r, split[..., np.newaxis]], axis=-1)
            cols.append("split")

        data_frame = pd.DataFrame(data_r,
                                  index=ids,
                                  columns=cols)
        data_frame.index.name = 'Id'

        if not fname.endswith('.gz'):
            fname += '.gz'

        data_frame.to_csv(fname, float_format='%.8f', compression='gzip')

    sample_file_ids = []
    sample_poses = []
    for k in eval_result:
        sample_file_ids.append(k)
        sample_poses.append(eval_result[k][0])

    to_csv(output_file, np.stack(sample_poses), sample_file_ids)


def load_model_weights(checkpoint_file, net, state_key='model_state_dict'):
    """Loads a pre-trained model."""
    if not os.path.exists(checkpoint_file):
        raise ValueError("Could not find model checkpoint {}.".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file)
    ckpt = checkpoint[state_key]
    net.load_state_dict(ckpt)


def get_model_config(model_id):
    model_id = model_id
    model_dir = U.get_model_dir(C.EXPERIMENT_DIR, model_id)
    model_config = Configuration.from_json(os.path.join(model_dir, 'config.json'))
    return model_config, model_dir


def load_model(model_id):
    model_config, model_dir = get_model_config(model_id)
    net = create_model(model_config)

    net.to(C.DEVICE)
    print('Model created with {} trainable parameters'.format(U.count_parameters(net)))

    # Load model weights.
    checkpoint_file = os.path.join(model_dir, 'model.pth')
    load_model_weights(checkpoint_file, net)
    print('Loaded weights from {}'.format(checkpoint_file))

    return net, model_config, model_dir


def evaluate_test(model_id, viz=False):
    """
    Load a model, evaluate it on the test set and save the predictions into the model directory.
    :param model_id: The ID of the model to load.
    :param viz: If some samples should be visualized.
    """
    net, model_config, model_dir = load_model(model_id)

    # No need to extract windows for the test set, since it only contains the seed sequence anyway.
    test_transform = transforms.Compose([ToTensor()])
    test_data = LMDBDataset(os.path.join(C.DATA_DIR, "test"), transform=test_transform)
    test_loader = DataLoader(test_data,
                             batch_size=model_config.bs_eval,
                             shuffle=False,
                             num_workers=model_config.data_workers,
                             collate_fn=AMASSBatch.from_sample_list)

    # Put the model in evaluation mode.
    net.eval()
    results = dict()
    with torch.no_grad():
        for abatch in test_loader:
            # Move data to GPU.
            batch_gpu = abatch.to_gpu()

            # Get the predictions.
            model_out = net(batch_gpu)

            for b in range(abatch.batch_size):
                results[batch_gpu.seq_ids[b]] = (model_out['predictions'][b].detach().cpu().numpy(),
                                                 model_out['seed'][b].detach().cpu().numpy())

    fname = 'predictions_in{}_out{}.csv'.format(model_config.seed_seq_len, model_config.target_seq_len)
    _export_results(results, os.path.join(model_dir, fname))

    if viz:
        fk_engine = SMPLForwardKinematics()
        visualizer = Visualizer(fk_engine)
        n_samples_viz = 10
        rng = np.random.RandomState(42)
        idxs = rng.randint(0, len(results), size=n_samples_viz)
        sample_keys = [list(sorted(results.keys()))[i] for i in idxs]
        for k in sample_keys:
            visualizer.visualize(results[k][1], results[k][0], title='Sample ID: {}'.format(k))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', required=True, help='Which model to evaluate.')
    config = parser.parse_args()
    evaluate_test(config.model_id, viz=True)

