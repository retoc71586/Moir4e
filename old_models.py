"""
Neural networks for motion prediction.

Copyright ETH Zurich, Manuel Kaufmann
"""
import torch
import torch.nn as nn

from data import AMASSBatch
from losses import mse


def create_model(config):
    # This is a helper function that can be useful if you have several model definitions that you want to
    # choose from via the command line. For now, we just return the Dummy model.
    return DummyModel(config)


class BaseModel(nn.Module):
    """A base class for neural networks that defines an interface and implements a few common functions."""

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.pose_size = config.pose_size
        self.create_model()

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


class DummyModel(BaseModel):
    """
    This is a dummy model. It provides basic implementations to demonstrate how more advanced models can be built.
    """

    def __init__(self, config):
        self.n_history = 10
        super(DummyModel, self).__init__(config)

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        # In this model we simply feed the last time steps of the seed to a dense layer and
        # predict the targets directly.
        self.dense = nn.Linear(in_features=self.n_history * self.pose_size,
                               out_features=self.config.target_seq_len * self.pose_size)

    def forward(self, batch: AMASSBatch):
        """
        The forward pass.
        :param batch: Current batch of data.
        :return: Each forward pass must return a dictionary with keys {'seed', 'predictions'}.
        """
        model_out = {'seed': batch.poses[:, :self.config.seed_seq_len],
                     'predictions': None}
        batch_size = batch.batch_size
        model_in = batch.poses[:, self.config.seed_seq_len-self.n_history:self.config.seed_seq_len]
        pred = self.dense(model_in.reshape(batch_size, -1))
        model_out['predictions'] = pred.reshape(batch_size, self.config.target_seq_len, -1)
        return model_out

    def backward(self, batch: AMASSBatch, model_out):
        """
        The backward pass.
        :param batch: The same batch of data that was passed into the forward pass.
        :param model_out: Whatever the forward pass returned.
        :return: The loss values for book-keeping, as well as the targets for convenience.
        """
        predictions = model_out['predictions']
        targets = batch.poses[:, self.config.seed_seq_len:]

        total_loss = mse(predictions, targets)

        # If you have more than just one loss, just add them to this dict and they will automatically be logged.
        loss_vals = {'total_loss': total_loss.cpu().item()}

        if self.training:
            # We only want to do backpropagation in training mode, as this function might also be called when evaluating
            # the model on the validation set.
            total_loss.backward()

        return loss_vals, targets


class S2sModel(BaseModel):
    """
    A dummy RNN model.
    """
    def __init__(self, config, data_pl, mode, reuse,**kwargs):
        super(S2sModel, self).__init__(config, data_pl, mode, reuse, **kwargs)

        # Extract some config parameters specific to this model
        self.cell_type = self.config["cell_type"]
        self.cell_size = self.config["cell_size"]
        self.num_layers  = self.config["num_layers"]
        self.input_hidden_size = self.config.get("input_hidden_size")
        self.loss_to_use = self.config["loss_to_use"]
        self.architecture = self.config["architecture"]
        
        # Prepare some members that need to be set when creating the graph.
        self.cell = None  # The recurrent cell.
        self.initial_states = None  # The intial states of the RNN.
        self.rnn_outputs = None  # The outputs of the RNN layer.
        self.rnn_state = None  # The final state of the RNN layer.
        self.inputs_hidden = None  # The inputs to the recurrent cell.

        # How many steps we must predict.
        if self.is_training:
            self.sequence_length = self.source_seq_len + self.target_seq_len - 1 #143
        else:
            self.sequence_length = self.target_seq_len

        self.prediction_inputs = self.data_inputs[:, :self.source_seq_len, :]  # Pose input.
        self.prediction_targets = self.data_inputs[:, 1:, :]  # The target poses for every time step.
        self.prediction_seq_len = torch.ones((torch.shape(self.prediction_targets)[0]), dtype=torch.int32)*self.sequence_length
        
        if self.is_test:
            self.encoder_inputs = self.data_inputs[:, :self.source_seq_len-1, :]
            var = torch.ones([torch.shape(self.encoder_inputs)[0],self.target_seq_len-1,self.input_size])
            self.decoder_inputs = torch.concat([self.data_inputs[:, self.source_seq_len-1:, :],var],1)
            self.decoder_outputs = self.data_inputs[:, self.source_seq_len:, 0:self.input_size]
            self.encoder_prev = self.data_inputs[:, :self.source_seq_len, :]
            
        else:
            self.encoder_inputs = self.data_inputs[:, :self.source_seq_len-1, :]
            self.decoder_inputs = self.data_inputs[:, self.source_seq_len-1:self.source_seq_len+self.target_seq_len-1, :]
            self.decoder_outputs = self.data_inputs[:, self.source_seq_len:, 0:self.input_size]
            self.encoder_prev = self.data_inputs[:, :self.source_seq_len, :]
            
        encoder_inputs=[]
        decoder_inputs=[]
        decoder_outputs=[]
        encoder_prev = []
        encoder_inputs.append(self.encoder_inputs)
        decoder_inputs.append(self.decoder_inputs)
        decoder_outputs.append(self.decoder_outputs)
        encoder_prev.append(self.encoder_prev)
        #self.encoder_inputs = enc_in
        #self.decoder_inputs = dec_in
        #self.decoder_outputs = dec_out
        
        encoder_inputs[-1] = torch.reshape(torch.transpose(encoder_inputs[-1], [1, 0, 2]),[-1, self.input_size])
        decoder_inputs[-1] = torch.reshape(torch.transpose(decoder_inputs[-1], [1, 0, 2]),[-1, self.input_size])
        decoder_outputs[-1]= torch.reshape(torch.transpose(decoder_outputs[-1], [1, 0, 2]),[-1, self.input_size])
        encoder_prev[-1]= torch.reshape(torch.transpose(encoder_prev[-1], [1, 0, 2]),[-1, self.input_size])
        
        #self.encoder_inputs = torch.reshape(self.encoder_inputs, [-1, self.input_size])
        #self.decoder_inputs = torch.reshape(self.decoder_inputs, [-1, self.input_size])
        #self.decoder_outputs = torch.reshape(self.decoder_outputs, [-1, self.input_size])
        if self.is_test:

            self.enc_in = torch.split(encoder_inputs[-1] , self.source_seq_len-1, axis=0)
            self.dec_in = torch.split(decoder_inputs[-1],self.target_seq_len, axis=0)
            self.dec_out = torch.split(decoder_outputs[-1], self.target_seq_len, axis=0)
            self.enc_prev = torch.split(encoder_prev[-1] , self.source_seq_len, axis=0)
        else:
            self.enc_in = torch.split(encoder_inputs[-1] , self.source_seq_len-1, axis=0)
            self.dec_in = torch.split(decoder_inputs[-1],self.target_seq_len, axis=0)
            self.dec_out = torch.split(decoder_outputs[-1], self.target_seq_len, axis=0)
            self.enc_prev = torch.split(encoder_prev[-1] , self.source_seq_len, axis=0)
        
        # Sometimes the batch size is available at compile time.
        self.tf_batch_size = self.prediction_inputs.shape.as_list()[0]
        if self.tf_batch_size is None:
            # Sometimes it isn't. Use the dynamic shape instead.
            self.tf_batch_size = torch.shape(self.prediction_inputs)[0]        
           
    def build_input_layer(self):
        """
        Here we can do some stuff on the inputs before passing them to the recurrent cell. The processed inputs should
        be stored in `self.inputs_hidden`.
        """
        # We could e.g. pass them through a dense layer
        if self.input_hidden_size is not None:
            with torch.variable_scope("input_layer", reuse=self.reuse):
                self.inputs_hidden = torch.layers.dense(self.prediction_inputs, self.input_hidden_size,
                                                     torch.nn.relu, self.reuse)
        else:
            self.inputs_hidden = self.prediction_inputs

    def build_cell(self,residual_velocities=True):
        """Create recurrent cell."""
        cell = torch.contrib.rnn.LSTMCell(self.cell_size,reuse=self.reuse,initializer=torch.orthogonal_initializer(),forget_bias=0.2)
        cell = torch.contrib.rnn.AttentionCellWrapper(cell, attn_length=64, state_is_tuple=True)
        cell = torch.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-0.5)
        #cell = torch.contrib.rnn.BidirectionalGridLSTMCell(self.cell_size,reuse=self.reuse,initializer=torch.orthogonal_initializer(),forget_bias=0.2)
        if self.num_layers > 1:
            cell = torch.contrib.rnn.MultiRNNCell( [torch.contrib.rnn.LSTMCell(1024,reuse=self.reuse,initializer=torch.orthogonal_initializer(),forget_bias=0.2) for _ in range(self.num_layers)] )
        cell = torch.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-0.5)
        # CELL extension 
        cell = rnn_cell_extensions.LinearSpaceDecoderWrapper( cell, self.input_size,reuse=self.reuse)
        if residual_velocities:
            cell = rnn_cell_extensions.ResidualWrapper(cell)
        
        
        '''
        with torch.variable_scope("rnn_cell", reuse=self.reuse):
            if self.cell_type == C.LSTM:
                cell = torch.nn.rnn_cell.LSTMCell(self.cell_size, reuse=self.reuse)
            elif self.cell_type == C.GRU:
                cell = torch.nn.rnn_cell.GRUCell(self.cell_size, reuse=self.reuse)
            else:
                raise ValueError("Cell type '{}' unknown".format(self.cell_type))
        '''
        self.cell = cell

    def build_network(self):
        """Build the core part of the model."""
        #self.build_input_layer()
        self.build_cell()

        #self.initial_states = self.cell.zero_state(batch_size=self.tf_batch_size, dtype=torch.float32)
        lf = None
        if self.loss_to_use == "sampling_based":
            def lf(prev, i): # function for sampling_based loss
                return prev
        elif self.loss_to_use == "supervised":
            pass
        else:
            raise(ValueError, "unknown loss: %s" % loss_to_use)
        
        if self.architecture == "basic":
            # Basic RNN does not have a loop function in its API, so copying here.
            with vs.variable_scope("basic_rnn_seq2seq"):
                #torch.reset_default_graph()
                self.outputs_prev, self.enc_state = torch.contrib.rnn.static_rnn(self.cell, self.enc_in, dtype=torch.float32) # Encoder
                self.outputs, self.states = torch.contrib.legacy_seq2seq.rnn_decoder( self.dec_in, self.enc_state, self.cell, loop_function=lf ) # Decoder
        elif self.architecture == "tied":
            self.outputs, self.states = torch.contrib.legacy_seq2seq.tied_rnn_seq2seq( self.enc_in, self.dec_in, self.cell, loop_function=lf )
        else:
            raise(ValueError, "Uknown architecture: %s" % architecture )
        #self.outputs = outputs
        '''
        with torch.variable_scope("rnn_layer", reuse=self.reuse):
            self.rnn_outputs, self.rnn_state = torch.nn.dynamic_rnn(self.cell,
                                                                 self.inputs_hidden,
                                                                 sequence_length=self.prediction_seq_len,
                                                                 initial_state=self.initial_states,
                                                                 dtype=torch.float32)
            self.prediction_representation = self.rnn_outputs
        '''
        #self.build_output_layer()
        if self.is_training or self.is_eval:
            self.build_loss()
        
        
    def build_loss(self):
        with torch.name_scope("loss_angles"):
            
            target = torch.reshape(self.dec_out,[-1,self.input_size])
            out_put = torch.reshape(self.outputs,[-1,self.input_size])
            print("out_put_shape",torch.shape(out_put))
            target = torch.reshape(target,[-1,3])
            out_put = torch.reshape(out_put,[-1,3])
            
            angle1 = torch.sqrt(torch.reduce_sum(torch.square(out_put),1))
            angle1 = torch.reshape(angle1,[-1,1])
            angle = torch.concat([angle1,angle1],axis=1)
            angle = torch.concat([angle, angle1],axis=1)
            axis1 = out_put/angle
            
            angle2 = torch.sqrt(torch.reduce_sum(torch.square(target),1))
            angle2 = torch.reshape(angle2,[-1,1])
            angle_ = torch.concat([angle2,angle2],axis=1)
            angle_ = torch.concat([angle_, angle2],axis=1)
            axis2 = target/angle_
            
            def axis_angle_to_matrix(angle,axis):
                sin_axis = torch.sin(angle) * axis
                cos_angle = torch.cos(angle)
                cos1_axis = (1.0-cos_angle) * axis
                _, axis_y, axis_z = torch.unstack(axis, axis=-1)
                cos1_axis_x, cos1_axis_y, _ = torch.unstack(cos1_axis, axis=-1)
                sin_axis_x, sin_axis_y,sin_axis_z = torch.unstack(sin_axis,axis=-1)
                tmp = cos1_axis_x * axis_y
                m01 = tmp - sin_axis_z
                m10 = tmp + sin_axis_z
                tmp = cos1_axis_x * axis_z
                m02 = tmp + sin_axis_y
                m20 = tmp - sin_axis_y
                tmp = cos1_axis_y * axis_z
                m12 = tmp - sin_axis_x
                m21 = tmp + sin_axis_x
                diag = cos1_axis * axis +cos_angle
                diag_x, diag_y, diag_z = torch.unstack(diag, axis=-1)
                matrix = torch.stack((diag_x, m01, m02,
                                   m10, diag_y, m12,
                                   m20, m21, diag_z),axis=-1)
                output_shape = torch.concat((torch.shape(input=axis)[:-1],
                                          (3,3)),axis=-1)
                return torch.reshape(matrix,shape=output_shape)
            R1 = axis_angle_to_matrix(angle1,axis1)
            R2 = axis_angle_to_matrix(angle2,axis2)
            R1_t = torch.transpose(R1,[0,2,1])
            diff = (torch.trace(torch.matmul(R1_t,R2))-1.0)/2.0
            diff = torch.reshape(diff,[-1,1])
            cond1 = torch.cast((diff>=1), torch.float32)
            cond2 = torch.cast((diff<=-1),torch.float32)
            cond3 = torch.cast((torch.abs(diff)<1),torch.float32)
            diff_angle = torch.acos(torch.sign(diff)*(torch.abs(diff)-0.0001))
            self.loss = 180.0* torch.reduce_mean(torch.abs(diff_angle))/3.1415926

    def step(self, session):
        """
        Run a training or validation step of the model.
        Args:
          session: Tensorflow session object.
        Returns:
          A triplet of loss, summary update and predictions.
        """
        if self.is_training:
            # Training step.
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs,
                           self.parameter_update]
            outputs = session.run(output_feed)
            return outputs[0], outputs[1], outputs[2]
        else:
            # Evaluation step (no backprop).
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs]
            outputs = session.run(output_feed)
            return outputs[0], outputs[1], outputs[2]

    def sampled_step(self, session):
        """
        Generates a sequence by feeding the prediction of time step t as input to time step t+1. This still assumes
        that we have ground-truth available.
        Args:
          session: Tensorflow session object.
        Returns:
          Prediction with shape (batch_size, self.target_seq_len, feature_size), ground-truth targets, seed sequence and
          unique sample IDs.
        """
        assert self.is_eval, "Only works in evaluation mode."

        # Get the current batch.
        #batch = session.run(self.data_placeholders)
        #data_id = batch[C.BATCH_ID]
        #data_sample = batch[C.BATCH_INPUT]
        #targets = data_sample[:, self.source_seq_len:]

        #seed_sequence = data_sample[:, :self.source_seq_len]
        predictions = self.sample(session)

        return predictions[0], predictions[1], predictions[3], predictions[2]

    def predict(self, session):
        """
        Generates a sequence by feeding the prediction of time step t as input to time step t+1. This assumes no
        ground-truth data is available.
        Args:
            session: Tensorflow session object.

        Returns:
            Prediction with shape (batch_size, self.target_seq_len, feature_size), seed sequence and unique sample IDs.
        """
        # `sampled_step` is written such that it works when no ground-truth data is available, too.
        output_feed = [self.outputs,
                       self.prediction_inputs,
                       self.data_ids]
        outputs = session.run(output_feed)
        return outputs[0], outputs[1], outputs[2]

    def sample(self, session):
        """
        Generates `prediction_steps` may poses given a seed sequence.
        Args:
            session: Tensorflow session object.
            seed_sequence: A tensor of shape (batch_size, seq_len, feature_size)
            prediction_steps: How many frames to predict into the future.
            **kwargs:
        Returns:
            Prediction with shape (batch_size, prediction_steps, feature_size)
        """
        assert self.is_eval, "Only works in sampling mode."
        output_feed = [self.loss,
                       self.summary_update,
                       self.outputs,
                       self.decoder_outputs]
        outputs = session.run(output_feed)
        return outputs[0],outputs[1],outputs[2],outputs[3]
        '''
        one_step_seq_len = np.ones(seed_sequence.shape[0])

        # Feed the seed sequence to warm up the RNN.
        feed_dict = {self.prediction_inputs: seed_sequence,
                     self.prediction_seq_len: np.ones(seed_sequence.shape[0])*seed_sequence.shape[1]}
        state, prediction = session.run([self.rnn_state, self.outputs], feed_dict=feed_dict)

        # Now create predictions step-by-step.
        prediction = prediction[:, -1:]
        predictions = [prediction]
        for step in range(prediction_steps-1):
            # get the prediction
            feed_dict = {self.prediction_inputs: prediction,
                         self.initial_states: state,
                         self.prediction_seq_len: one_step_seq_len}
            state, prediction = session.run([self.rnn_state, self.outputs], feed_dict=feed_dict)
            predictions.append(prediction)
        return np.concatenate(predictions, axis=1)
        '''
        
