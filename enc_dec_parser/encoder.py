"""Encoder class of the seq2seq model.

Author: Trang Tran and Shubham Toshniwal
Contact: ttmt001@uw.edu, shtoshni@ttic.edu
Date: April, 2017
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import embedding_ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import core_rnn_cell as rnn_cell
from tensorflow.python.ops import variable_scope


class Encoder(object):
    """Encoder class that encodes input sequence."""

    def __init__(self, isTraining, enc_attribs):
        """Initializer for encoder class.

        Args:
            isTraining: Whether the network is in training mode or not. This
                would affect whether dropout is used or not.
            enc_attribs: A dictionary of attributes used by encoder like:
                hidden_size: Hidden size of LSTM cell used for encoding
                num_layers: Number of hidden layers used
                vocab_size: Vocabulary size of input symbols
                emb_size: Embedding size used to feed in input symbols
                out_prob(Optional): (1 - Dropout probability)
        """
        self.isTraining = isTraining
        # Update the parameters
        self.__dict__.update(enc_attribs)
        # Create the LSTM cell using the hidden size attribute
        self.cell = rnn_cell.BasicLSTMCell(self.hidden_size,
                                           state_is_tuple=True)
        if self.isTraining:
            # During training a dropout wrapper is used
            self.cell = rnn_cell.DropoutWrapper(self.cell,
                                                output_keep_prob=self.out_prob)
        if self.num_layers > 1:
            self.cell = rnn_cell.MultiRNNCell([self.cell] * self.num_layers,
                                              state_is_tuple=True)

    def _cnn_word_process(self, filter_size):
        # TensorArray read operation
        #print (filter_size)
        #print (filter_size.value)
        def loop_function(t, speech_input, cnn_output):
            with variable_scope.variable_scope("conv_mxpool_%d" % filter_size):
                speech_input_t = speech_input.read(t)
                filter_shape = [filter_size, self.feat_dim, 1,
                                self.num_filters]
                W = variable_scope.get_variable("W_%d" % filter_size,
                                                filter_shape)
                b = variable_scope.get_variable("b_%d" % filter_size,
                                                self.num_filters)
                feats_conv = tf.expand_dims(speech_input_t, -1)
                conv = tf.nn.conv2d(feats_conv, W, strides=[1, 1, 1, 1],
                                    padding="VALID", name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(
                    h, ksize=[1, self.fixed_word_length-filter_size+1, 1, 1],
                    strides=[1, 1, 1, 1], padding='VALID', name="pool")
                cnn_output = cnn_output.write(t, pooled)
                t = t + 1
            return t, speech_input, cnn_output

        return loop_function

    def encode_input(self, encoder_inp, seq_len):
        """Run the encoder on gives input.

        Args:
            encoder_inp: Input IDs that are time major i.e. TxB. These IDs are
                first passed through embedding layer before feeding to first
                LSTM layer.
            seq_len: Actual length of input time sequences.
        Returns:
            attention_states: Final encoder output for every input timestep.
                This tensor is used by attention-enabled decoders.
            final_state: Final state of encoder LSTM
        """
        with variable_scope.variable_scope("encoder"):
            comb_encoder_inputs = None
            embedding = {}
            # Necessary to sort so that the order of encoder_inputs is
            # maintained
            for idx, key in enumerate(sorted(encoder_inp.iterkeys())):
                print (key)
                if key == "speech_frames":
                    continue
                elif key == "word_dur":
                    cur_inputs = encoder_inp[key]
                    # No embedding for word duration - so just extend dim.
                    cur_inputs = tf.expand_dims(cur_inputs, -1)
                else:
                    embedding[key] = variable_scope.get_variable(
                        "emb_" + key, [self.vocab_size[key],
                                       self.embedding_size[key]])

                    cur_inputs = embedding_ops.embedding_lookup(
                        embedding[key], encoder_inp[key])
                if comb_encoder_inputs is None:
                    comb_encoder_inputs = cur_inputs
                else:
                    comb_encoder_inputs = tf.concat([comb_encoder_inputs,
                                                     cur_inputs], 2)
            if "speech_frames" in encoder_inp:
                cnn_outputs = []
                max_words = tf.reduce_max(seq_len)
                for i, filter_size in enumerate(self.filter_sizes):
                    acoustic_input_ta = tf.TensorArray(
                        size=0, dtype=tf.float32, dynamic_size=True)
                    acoustic_input_ta = acoustic_input_ta.unstack(
                        encoder_inp["speech_frames"])
                    cur_filter_size_output_array = tf.TensorArray(
                        size=0, dtype=tf.float32, dynamic_size=True)
                    _, _, cur_filter_size_output = tf.while_loop(
                        cond=lambda time_idx, a_t, _: time_idx < max_words,
                        body=self._cnn_word_process(filter_size),
                        loop_vars=(tf.constant(0),
                                   acoustic_input_ta,
                                   cur_filter_size_output_array
                                   )
                        )
                    # Convert the TensorArray to Tensor
                    cur_filter_size_output = cur_filter_size_output.stack()
                    cnn_outputs.append(cur_filter_size_output)

                # T * B * filter_sizes * 1 * num_filters
                cnn_features = tf.concat(cnn_outputs, 2)
                num_filters_total = self.num_filters * len(self.filter_sizes)

                time_dim = array_ops.shape(cnn_features)[0]
                batch_size = array_ops.shape(cnn_features)[1]
                cnn_features = tf.reshape(cnn_features,
                                          array_ops.stack([time_dim, batch_size, num_filters_total]))

                comb_encoder_inputs = tf.concat([comb_encoder_inputs,
                                                cnn_features], 2)

            encoder_outputs, encoder_state = rnn.dynamic_rnn(
                self.cell, comb_encoder_inputs, sequence_length=seq_len,
                dtype=tf.float32, time_major=True)
            # Make the attention states batch major
            attention_states = tf.transpose(encoder_outputs, [1, 0, 2])

        return attention_states, encoder_state
