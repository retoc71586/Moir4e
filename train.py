"""
The training script.

Copyright ETH Zurich, Manuel Kaufmann
"""
import collections
import glob
import numpy as np
import os
import sys
import time
import torch
import torch.optim as optim
import utils as U

from configuration import Configuration
from configuration import CONSTANTS as C
from data import AMASSBatch
from data import LMDBDataset
from data_transforms import ExtractWindow
from data_transforms import ToTensor
from evaluate import evaluate_test
from models import create_model
from motion_metrics import MetricsEngine
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def _log_loss_vals(loss_vals, writer, global_step, mode_prefix):
    for k in loss_vals:
        prefix = '{}/{}'.format(k, mode_prefix)
        writer.add_scalar(prefix, loss_vals[k], global_step)


def _evaluate(net, data_loader, metrics_engine):
    """
    Evaluate a model on the given dataset. This computes the loss, but does not do any backpropagation or gradient
    update.
    :param net: The model to evaluate.
    :param data_loader: The dataset.
    :param metrics_engine: MetricsEngine to compute metrics.
    :return: The loss value.
    """
    # Put the model in evaluation mode.
    net.eval()

    # Some book-keeping.
    loss_vals_agg = collections.defaultdict(float)
    n_samples = 0
    metrics_engine.reset()

    with torch.no_grad():
        for abatch in data_loader:
            # Move data to GPU.
            batch_gpu = abatch.to_gpu()

            # Get the predictions.
            model_out = net(batch_gpu)

            # Compute the loss.
            loss_vals, targets = net.backward(batch_gpu, model_out)

            # Accumulate the loss and multiply with the batch size (because the last batch might have different size).
            for k in loss_vals:
                loss_vals_agg[k] += loss_vals[k] * batch_gpu.batch_size

            # Compute metrics.
            metrics_engine.compute_and_aggregate(model_out['predictions'], targets)

            n_samples += batch_gpu.batch_size

    # Compute the correct average for the entire data set.
    for k in loss_vals_agg:
        loss_vals_agg[k] /= n_samples

    return loss_vals_agg


def main(config):
    # Fix seed.
    if config.seed is None:
        config.seed = int(time.time())

    # Define some transforms that are applied to each `AMASSSample` before they are collected into a batch.
    # You can define your own transforms in `data_transforms.py` and add them here.
    rng_extractor = np.random.RandomState(4313)
    window_size = config.seed_seq_len + config.target_seq_len
    train_transform = transforms.Compose([ExtractWindow(window_size, rng_extractor, mode='random'),
                                          ToTensor()])
    # Validation data is already in the correct length, so no need to extract windows.
    valid_transform = transforms.Compose([ToTensor()])

    # For the training data we pass in the `window_size` variable, so that all samples whose length is smaller than
    # `window_size` are rejected.
    train_data = LMDBDataset(os.path.join(C.DATA_DIR, "training"), transform=train_transform,
                             filter_seq_len=window_size)
    valid_data = LMDBDataset(os.path.join(C.DATA_DIR, "validation"), transform=valid_transform)

    train_loader = DataLoader(train_data,
                              batch_size=config.bs_train,
                              shuffle=True,
                              num_workers=config.data_workers,
                              collate_fn=AMASSBatch.from_sample_list)
    valid_loader = DataLoader(valid_data,
                              batch_size=config.bs_eval,
                              shuffle=False,
                              num_workers=config.data_workers,
                              collate_fn=AMASSBatch.from_sample_list)

    # Load some data statistics, but they are not used further.
    # You may use these stats if you want, but you can also compute them yourself.
    stats = np.load(os.path.join(C.DATA_DIR, "training", "stats.npz"), allow_pickle=True)['stats'].tolist()

    # Set the pose size in the config as models use this later.
    setattr(config, 'pose_size', 135)

    # Create the model.
    net = create_model(config)
    net.to(C.DEVICE)
    print('Model created with {} trainable parameters'.format(U.count_parameters(net)))

    # Prepare metrics engine.
    me = MetricsEngine(C.METRIC_TARGET_LENGTHS)
    me.reset()

    # Create or a new experiment ID and a folder where to store logs and config.
    experiment_id = int(time.time())
    experiment_name = net.model_name()
    model_dir = U.create_model_dir(C.EXPERIMENT_DIR, experiment_id, experiment_name)

    # Save code as zip and config as json into the model directory.
    code_files = glob.glob('./*.py', recursive=False)
    U.export_code(code_files, os.path.join(model_dir, 'code.zip'))
    config.to_json(os.path.join(model_dir, 'config.json'))

    # Save the command line that was used.
    cmd = sys.argv[0] + ' ' + ' '.join(sys.argv[1:])
    with open(os.path.join(model_dir, 'cmd.txt'), 'w') as f:
        f.write(cmd)

    # Create a checkpoint file for the best model.
    checkpoint_file = os.path.join(model_dir, 'model.pth')
    print('Saving checkpoints to {}'.format(checkpoint_file))

    # Create Tensorboard logger.
    writer = SummaryWriter(os.path.join(model_dir, 'logs'))

    # Define the optimizer.
    optimizer = optim.SGD(net.parameters(), lr=config.lr)

    # Training loop.
    global_step = 0
    best_valid_loss = float('inf')
    for epoch in range(config.n_epochs):

        for i, abatch in enumerate(train_loader):
            start = time.time()
            optimizer.zero_grad()

            # Move data to GPU.
            batch_gpu = abatch.to_gpu()

            # Get the predictions.
            model_out = net(batch_gpu)

            # Compute gradients.
            train_losses, targets = net.backward(batch_gpu, model_out)

            # Update params.
            optimizer.step()

            elapsed = time.time() - start

            # Write training stats to Tensorboard.
            _log_loss_vals(train_losses, writer, global_step, 'train')
            writer.add_scalar('lr', config.lr, global_step)

            if global_step % (config.print_every - 1) == 0:
                loss_string = ' '.join(['{}: {:.6f}'.format(k, train_losses[k]) for k in train_losses])
                print('[TRAIN {:0>5d} | {:0>3d}] {} elapsed: {:.3f} secs'.format(
                    i + 1, epoch + 1, loss_string, elapsed))
                me.reset()
                me.compute_and_aggregate(model_out['predictions'], targets)
                me.to_tensorboard_log(me.get_final_metrics(), writer, global_step, 'train')

            if global_step % (config.eval_every - 1) == 0:
                # Evaluate on validation.
                start = time.time()
                net.eval()
                valid_losses = _evaluate(net, valid_loader, me)
                valid_metrics = me.get_final_metrics()
                elapsed = time.time() - start

                # Log to console.
                loss_string = ' '.join(['{}: {:.6f}'.format(k, valid_losses[k]) for k in valid_losses])
                print('[VALID {:0>5d} | {:0>3d}] {} elapsed: {:.3f} secs'.format(
                    i + 1, epoch + 1, loss_string, elapsed))
                print('[VALID {:0>5d} | {:0>3d}] {}'.format(i+1, epoch+1, me.get_summary_string(valid_metrics)))

                # Log to tensorboard.
                _log_loss_vals(valid_losses, writer, global_step, 'valid')
                me.to_tensorboard_log(valid_metrics, writer, global_step, 'valid')

                # Save the current model if it's better than what we've seen before.
                if valid_losses['total_loss'] < best_valid_loss:
                    best_valid_loss = valid_losses['total_loss']
                    torch.save({
                        'iteration': i,
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_losses['total_loss'],
                        'valid_loss': valid_losses['total_loss'],
                    }, checkpoint_file)

                # Make sure the model is in training mode again.
                net.train()

            global_step += 1

    # After the training, evaluate the model on the test and generate the result file that can be uploaded to the
    # submission system. The submission file will be stored in the model directory.
    evaluate_test(experiment_id)


if __name__ == '__main__':
    main(Configuration.parse_cmd())
