"""
Define common constants that are used globally.

Also define a configuration object whose parameters can be set via the command line or loaded from an existing
json file. Here you can add more configuration parameters that should be exposed via the command line. In the code,
you can access them via `config.your_parameter`. All parameters are automatically saved to disk in JSON format.

Copyright ETH Zurich, Manuel Kaufmann
"""
import argparse
import json
import os
import pprint
import torch


class Constants(object):
    """
    This is a singleton.
    """
    class __Constants:
        def __init__(self):
            # Environment setup.
            self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.DTYPE = torch.float32
            self.DATA_DIR = os.environ['MP_DATA']
            self.EXPERIMENT_DIR = os.environ['MP_EXPERIMENTS']
            self.METRIC_TARGET_LENGTHS = [5, 10, 19, 24]  # @ 60 fps, in ms: 83.3, 166.7, 316.7, 400

    instance = None

    def __new__(cls, *args, **kwargs):
        if not Constants.instance:
            Constants.instance = Constants.__Constants()
        return Constants.instance

    def __getattr__(self, item):
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)


CONSTANTS = Constants()


class Configuration(object):
    """Configuration parameters exposed via the commandline."""

    def __init__(self, adict):
        self.__dict__.update(adict)

    def __str__(self):
        return pprint.pformat(vars(self), indent=4)

    @staticmethod
    def parse_cmd():
        parser = argparse.ArgumentParser()

        #Additionals
        parser.add_argument('--opt', type=str, choices={"adam", "sgd"}, default="adam", help='Type of optimizer') 
        parser.add_argument('--use_lr_decay', default=True, help='Use LR decay', action = "store_true")
        parser.add_argument('--lr_decay_rate', type=float, default=0.98, help='Learning rate decay rate.')
        parser.add_argument('--lr_decay_step', type=float, default=330, help='Learning rate decay step.')
        parser.add_argument('--clip_gradient', help='Use gradient clipping to l2 norm max_norm', action = "store_true")
        parser.add_argument('--max_norm', type=float, default=1,help='max norm for gradient clipping')
        parser.add_argument('--nr_dct_dim', type=int, default=64, help='number of dct dimension')
        parser.add_argument('--kernel_size', type=int, default=40, help='number of past frames to look to predict the future')
        parser.add_argument('--model', type=str, default=None, help='Defines the model to train on.')

        # model dct_att_gcn --n_epochs 1000 --lr 0.0005 --use_lr_decay --lr_decay_step 330 --bs_train 128 
        # --bs_eval 128 --nr_dct_dim 64 --loss_ABtype avg_l1 --lr_decay_rate 0.98 --opt adam --kernel_size 40
        # --clip_gradient --max_norm 1

        # General.
        parser.add_argument('--data_workers', type=int, default=4, help='Number of parallel threads for data loading.')
        parser.add_argument('--print_every', type=int, default=200, help='Print stats to console every so many iters.')
        parser.add_argument('--eval_every', type=int, default=400, help='Evaluate validation set every so many iters.')
        parser.add_argument('--tag', default='', help='A custom tag for this experiment.')
        parser.add_argument('--seed', type=int, default=None, help='Random number generator seed.')

        # Data.
        parser.add_argument('--seed_seq_len', type=int, default=120, help='Number of frames for the seed length.')
        parser.add_argument('--target_seq_len', type=int, default=24, help='How many frames to predict.')

        # Learning configurations.
        parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate.')
        parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs.')
        parser.add_argument('--bs_train', type=int, default=128, help='Batch size for the training set.')
        parser.add_argument('--bs_eval', type=int, default=128, help='Batch size for valid/test set.')

        config = parser.parse_args()
        return Configuration(vars(config))

    @staticmethod
    def from_json(json_path):
        """Load configurations from a JSON file."""
        with open(json_path, 'r') as f:
            config = json.load(f)
            return Configuration(config)

    def to_json(self, json_path):
        """Dump configurations to a JSON file."""
        with open(json_path, 'w') as f:
            s = json.dumps(vars(self), indent=2, sort_keys=True)
            f.write(s)
