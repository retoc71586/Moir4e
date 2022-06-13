"""
Neural networks for motion prediction.

Copyright ETH Zurich, Manuel Kaufmann
"""
import torch
import torch.nn as nn
from data import AMASSBatch
from losses import *
import random
import losses
import numpy as np
import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch._six import with_metaclass
from configuration import CONSTANTS as C
from utils import get_dct_matrix
from torch._C import _ImperativeEngine as ImperativeEngine
from gcn import GCN

def create_model(config):
    print('using device: ', C.DEVICE)
    return DCT_ATT_GCN(config)

class BaseModel(nn.Module):
    """A base class for neural networks that defines an interface and implements a few common functions."""

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.pose_size = config.pose_size
        self.create_model()
        self.is_test = False
        self.loss_fun = losses.avg_l1

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        """Create the model, called automatically by the initializer."""
        raise NotImplementedError("Must be implemented by subclass.")

    def forward(self, batch: AMASSBatch):
        """The forward pass."""
        raise NotImplementedError("Must be implemented by subclass.")

    def backward(self, batch: AMASSBatch, model_out):
        """The backward pass."""
        raise NotImplementedError("Must be implemented by subclass.")

    def model_name(self):
        """A summary string of this model. Override this if desired."""
        return '{}-lr{}'.format(self.__class__.__name__, self.config.lr)

class VariableMeta(type):
    def __instancecheck__(cls, other):
        return isinstance(other, torch.Tensor)

# mypy doesn't understand torch._six.with_metaclass
class Variable(with_metaclass(VariableMeta, torch._C._LegacyVariableBase)):  # type: ignore[misc]
    pass

Variable._execution_engine = ImperativeEngine()

"""
This code is an adaptation from the paper
Wei Mao, Miaomiao Liu, Mathieu Salzmann, Hongdong Li. 
Learning Trajectory Dependencies for Human Motion Prediction. In ICCV 19.
link : https://github.com/wei-mao-2019/LearnTrajDep

The following model uses the Discrete Cosine Transform (DCT) 
to encode temporal information, together with a modelling of the 
spatial dependencies between the joints, with
graph convolutional networks.
"""

class DCT_ATT_GCN(BaseModel):
  

    def __init__(self, config):
        self.seed_seq_len   = config.seed_seq_len
        self.target_seq_len = config.target_seq_len
        self.input_size     = config.pose_size

        assert config.kernel_size % 2 == 0

        self.kernel_size    = config.kernel_size
        self.hidden_feature = 256  
        self.gcn_p_dropout  = 0.3  
        self.gcn_num_stage  = 12  
        self.itera          = 1    
        
        self.n_dct_freq = self.kernel_size + config.target_seq_len 

        # Compute DCT matrices once
        dct_mat, idct_mat = get_dct_matrix(self.kernel_size + self.target_seq_len)
        self.dct_mat  = Variable(torch.from_numpy(dct_mat)).float().to(C.DEVICE)
        self.idct_mat = Variable(torch.from_numpy(idct_mat)).float().to(C.DEVICE)

        
        super(DCT_ATT_GCN, self).__init__(config)


    def create_model(self):
        
        self.convQ = nn.Sequential(nn.Conv1d(in_channels=self.input_size,
                                             out_channels=self.hidden_feature,
                                             kernel_size=(self.kernel_size//2 + 1),
                                             bias=False),
                                             nn.ReLU(),
                                             nn.Conv1d(in_channels=self.hidden_feature,
                                                        out_channels=self.hidden_feature,
                                                        kernel_size=(self.kernel_size//2),
                                                        bias=False),
                                             nn.ReLU())

        self.convK = nn.Sequential(nn.Conv1d(in_channels=self.input_size,
                                             out_channels=self.hidden_feature,
                                             kernel_size=(self.kernel_size//2 + 1),
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=self.hidden_feature,
                                             out_channels=self.hidden_feature,
                                             kernel_size=(self.kernel_size//2),
                                             bias=False),
                                   nn.ReLU())
    
        self.gcn = GCN(input_feature=(self.n_dct_freq)*2,
                       hidden_feature=self.hidden_feature,
                       p_dropout=self.gcn_p_dropout,
                       num_stage=self.gcn_num_stage,
                       node_n=self.input_size) # Number of Joints * Degree of freedom

    def forward(self, batch: AMASSBatch):
        """
        The forward pass.
        :param batch: Current batch of data.
        :return: Each forward pass must return a dictionary with keys {'seed', 'predictions'}.
        """
        model_out = {'seed': batch.poses[:, :self.config.seed_seq_len],
                     'predictions': None}
        batch_size = batch.batch_size
        input_series = batch.poses[:, :self.seed_seq_len, :]
        src_tmp = input_series.clone()
        src_key_tmp   = src_tmp.transpose(1, 2)[:, :, :(self.seed_seq_len - self.target_seq_len)].clone()
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()
        vn = self.seed_seq_len - self.kernel_size - self.target_seq_len + 1
        vl = self.kernel_size + self.target_seq_len
        idx_subsequences = np.expand_dims(np.arange(vl), axis=0) + np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx_subsequences].clone().reshape([batch_size * vn, vl, -1])      
        src_value_tmp = torch.matmul(self.dct_mat[:self.n_dct_freq].unsqueeze(dim=0), src_value_tmp) # to DCT
        src_value_tmp = src_value_tmp.reshape([batch_size, vn, self.n_dct_freq, -1])
        src_value_tmp = src_value_tmp.transpose(2, 3)
        src_value_tmp = src_value_tmp.reshape([batch_size, vn, -1])
        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * self.target_seq_len
        key_tmp = self.convK(src_key_tmp)
        query_tmp = self.convQ(src_query_tmp)
        score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15
        att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
        dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape([batch_size, -1, self.n_dct_freq])
        input_gcn = src_tmp[:, idx]
        dct_in_tmp = torch.matmul(self.dct_mat[:self.n_dct_freq].unsqueeze(dim=0), input_gcn).transpose(1, 2)
        dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)
        # feed to GCN
        dct_out_tmp = self.gcn(dct_in_tmp)
        output_series = torch.matmul(self.idct_mat[:, :self.n_dct_freq].unsqueeze(dim=0),
                               dct_out_tmp[:, :, :self.n_dct_freq].transpose(1, 2))
        if self.training:
            model_out['predictions'] = output_series
        else:
            model_out['predictions'] = output_series[:, self.kernel_size:, :]
        return model_out

    def backward(self, batch: AMASSBatch, model_out):
        """
        The backward pass.
        :param batch: The same batch of data that was passed into the forward pass.
        :param model_out: Whatever the forward pass returned.
        :return: The loss values for book-keeping, as well as the targets for convenience.
        """
        predictions = model_out['predictions']
        if self.training:
            targets = batch.poses[:, -(self.kernel_size + self.target_seq_len):, :]
        else:
            targets = batch.poses[:, -self.target_seq_len:, :]
        total_loss = self.loss_fun(predictions, targets)
        # If you have more than just one loss, just add them to this dict and they will automatically be logged.
        loss_vals = {'total_loss': total_loss.cpu().item()}
        if self.training:
            # We only want to do backpropagation in training mode, as this function might also be called when evaluating
            # the model on the validation set.
            total_loss.backward()
        return loss_vals, targets
