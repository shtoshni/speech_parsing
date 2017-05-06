"""Decoder class of seq2seq model.

Author: Trang Tran and Shubham Toshniwal
Contact: ttmt001@uw.edu, shtoshni@ttic.edu
Date: April, 2017
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import core_rnn_cell as rnn_cell
from tensorflow.python.ops import variable_scope
import tensorflow as tf

from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear\
 as linear


class Decoder(object):
    """Base class for decoder in encoder-decoder framework."""

    def __init__(self, isTraining, dec_attribs):
        """The initializer for decoder class.

        Args:
            isTraining: Whether the network is in training mode or not. This
                would affect whether dropout and sampling are used or not.
            **attribs: A dictionary of attributes used by encoder like:
                hidden_size: Hidden size of LSTM cell used for decoding
                num_layers: Number of hidden layers used
                num_decoder_symbols: Vocabulary size of output symbols
                embedding_size: Embedding size used to feed in input symbols
                out_prob (Optional): (1 - Dropout probability)
                samp_prob (Optional): Sampling probability for sampling output
                    of previous step instead of using ground truth during
                    training.
                max_output (Optional): Maximum length of output sequence.
                    Assumed to be 100 if not specified.
        """
        self.isTraining = isTraining
        self.__dict__.update(dec_attribs)
        if self.isTraining:
            self.out_prob = dec_attribs['out_prob']
            self.isSampling = False
            if ("samp_prob" in dec_attribs) and dec_attribs["samp_prob"] > 0.0:
                self.isSampling = True
                self.samp_prob = dec_attribs['samp_prob']

        self.hidden_size = dec_attribs['hidden_size']
        self.num_layers = dec_attribs['num_layers']
        self.vocab_size = dec_attribs['vocab_size']
        self.cell = self.set_cell_config()

        self.embedding_size = dec_attribs['embedding_size']

        self.max_output = 100   # Maximum length of output
        if 'max_output' in dec_attribs:
            self.max_output = dec_attribs['max_output']

    def set_cell_config(self):
        """Create the LSTM cell used by decoder."""
        # Use the BasicLSTMCell - https://arxiv.org/pdf/1409.2329.pdf
        cell = rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        if self.isTraining:
            # During training we use a dropout wrapper
            cell = rnn_cell.DropoutWrapper(cell,
                                           output_keep_prob=self.out_prob)
        if self.num_layers > 1:
            # If RNN is stacked then we use MultiRNNCell class
            cell = rnn_cell.MultiRNNCell([cell] * self.num_layers,
                                         state_is_tuple=True)

        # Use the OutputProjectionWrapper to project cell output to output
        # vocab size. This projection is fine for a small vocabulary output
        # but would be bad for large vocabulary output spaces.
        cell = rnn_cell.OutputProjectionWrapper(cell, self.vocab_size)
        return cell

    def prepare_decoder_input(self, decoder_inputs):
        """Do this step before starting decoding.

        This step converts the decoder IDs to vectors and
        Args:
            decoder_inputs: Time major decoder IDs
        Returns:
            embedded_inp: Embedded decoder input.
            loop_function: Function for getting next timestep input.
        """
        with variable_scope.variable_scope("decoder"):
            # Create an embedding matrix
            embedding = variable_scope.get_variable(
                "embedding", [self.vocab_size, self.embedding_size],
                initializer=tf.random_uniform_initializer(-1.0, 1.0))
            # Embed the decoder input via embedding lookup operation
            embedded_inp = embedding_ops.embedding_lookup(embedding,
                                                          decoder_inputs)

        if self.isTraining:
            if self.isSampling:
                # This loop function samples the output from the posterior
                # and embeds this output.
                loop_function = self._sample_argmax(embedding)
            else:
                loop_function = None
        else:
            # Get the loop function that would embed the maximum posterior
            # symbol. This funtion is used during decoding in RNNs
            loop_function = self._get_argmax(embedding)

        return (embedded_inp, loop_function)

    def decode(self, decoder_inputs, seq_len,
               encoder_hidden_states, final_state, seq_len_inp):
        """Abstract method that needs to be extended by Inheritor classes.

        Args:
            decoder_inputs: Time major decoder IDs, TxB that contain ground tr.
                during training and are dummy value holders at test time.
            seq_len: Output sequence length for each input in minibatch.
                Useful to limit the computation to the max output length in
                a minibatch.
            encoder_hidden_states: Batch major output, BxTxH of encoder RNN.
                Useful with attention-enabled decoders.
            final_state: Final hidden state of encoder RNN. Useful for
                initializing decoder RNN.
            seq_len_inp: Useful with attention-enabled decoders to mask the
                outputs corresponding to padding symbols.
        Returns:
            outputs: Time major output, TxBx|V|, of decoder RNN.
        """
        decoder_inputs, loop_function = self.prepare_decoder_input(
            decoder_inputs)

        output_size = self.cell.output_size

        with variable_scope.variable_scope("attention_decoder"):
            batch_size = array_ops.shape(decoder_inputs)[1]
            embedding_size = decoder_inputs.get_shape()[2].value

            attn_length = tf.shape(encoder_hidden_states)[1]
            attn_size = encoder_hidden_states.get_shape()[2].value

            # To calculate W1 * h_t we use a 1-by-1 convolution, need to
            # reshape before.
            hidden = tf.expand_dims(encoder_hidden_states, 2)

            attention_vec_size = 64

            k = variable_scope.get_variable(
                "AttnW", [1, 1, attn_size, attention_vec_size])
            hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            v = variable_scope.get_variable("AttnV", [attention_vec_size])
            if self.use_conv:
                F = variable_scope.get_variable(
                    "AttnF", [self.conv_filter_width, 1, 1,
                              self.conv_num_channels])
                U = variable_scope.get_variable("AttnU",
                                                [1, 1, self.conv_num_channels,
                                                 attention_vec_size])

            batch_attn_size = array_ops.stack([batch_size, attn_size])
            attn = array_ops.zeros(batch_attn_size, dtype=tf.float32)
            attn.set_shape([None, attn_size])

            batch_alpha_size = array_ops.stack([batch_size, attn_length, 1, 1])
            alpha = array_ops.zeros(batch_alpha_size, dtype=tf.float32)

            # Assumes Time major arrangement
            inputs_ta = tf.TensorArray(size=400, dtype=tf.float32,
                                       dynamic_size=True)
            inputs_ta = inputs_ta.unstack(decoder_inputs)

            attn_mask = tf.sequence_mask(tf.cast(seq_len_inp, tf.int32),
                                         dtype=tf.float32)

            def raw_loop_function(time, cell_output, state, loop_state):
                def attention(query, prev_alpha):
                    """Calculate attention weights."""
                    with variable_scope.variable_scope("Attention"):
                        y = linear(query, attention_vec_size, True)
                        y = array_ops.reshape(y, [-1, 1, 1,
                                                  attention_vec_size])
                        if self.use_conv:
                            conv_features = nn_ops.conv2d(prev_alpha, F,
                                                          [1, 1, 1, 1], "SAME")
                            feat_reshape = nn_ops.conv2d(conv_features, U,
                                                         [1, 1, 1, 1], "SAME")
                            s = math_ops.reduce_sum(
                                v * math_ops.tanh(hidden_features + y +
                                                  feat_reshape), [2, 3])
                        else:
                            s = math_ops.reduce_sum(
                                v * math_ops.tanh(hidden_features + y), [2, 3])

                        alpha = nn_ops.softmax(s) * attn_mask
                        sum_vec = tf.reduce_sum(alpha, reduction_indices=[1],
                                                keep_dims=True) + 1e-12
                        norm_term = tf.tile(sum_vec,
                                            tf.stack([1, tf.shape(alpha)[1]]))
                        alpha = alpha / norm_term
                        alpha = tf.expand_dims(alpha, 2)
                        alpha = tf.expand_dims(alpha, 3)
                        # Now calculate the attention-weighted vector d.
                        d = math_ops.reduce_sum(alpha * hidden, [1, 2])
                        d = array_ops.reshape(d, [-1, attn_size])

                    return tuple([d, alpha])

                # If loop_function is set, we use it instead of decoder_inputs.
                elements_finished = (time >= seq_len)
                finished = tf.reduce_all(elements_finished)

                if cell_output is None:
                    next_state = final_state
                    output = None
                    loop_state = tuple([attn, alpha])
                    next_input = inputs_ta.read(time)
                else:
                    next_state = state
                    loop_state = attention(cell_output, loop_state[1])
                    with variable_scope.variable_scope("AttnOutputProjection"):
                        output = linear([cell_output, loop_state[0]],
                                        output_size, True)

                    if loop_function is not None:
                        simple_input = loop_function(output)
                        # print ("Yolo")
                    else:
                        simple_input = tf.cond(
                            finished,
                            lambda: tf.zeros([batch_size, embedding_size],
                                             dtype=tf.float32),
                            lambda: inputs_ta.read(time)
                        )

                    # Merge input and previous attentions into one vector of
                    # the right size.
                    input_size = simple_input.get_shape().with_rank(2)[1]
                    if input_size.value is None:
                        raise ValueError("Could not infer input size")
                    with variable_scope.variable_scope("InputProjection"):
                        next_input = linear([simple_input, loop_state[0]],
                                            input_size, True)

                return (elements_finished, next_input, next_state, output,
                        loop_state)

        outputs, state, _ = rnn.raw_rnn(self.cell, raw_loop_function)
        return outputs.concat()

    def _get_argmax(self, embedding):
        """Return a function that returns the previous output with max prob.

        Args:
            embedding : Embedding matrix for embedding the symbol
        Returns:
            loop_function: A function that returns the embedded output symbol
                with maximum probability (logit score).
        """
        def loop_function(logits):
            max_symb = math_ops.argmax(logits, 1)
            emb_symb = embedding_ops.embedding_lookup(embedding, max_symb)
            return emb_symb

        return loop_function

    def _sample_argmax(self, embedding):
        """Return a function that samples from posterior over previous output.

        Args:
            embedding : Embedding matrix for embedding the symbol
        Returns:
            loop_function: A function that samples the output symbol from
            posterior and embeds the sampled symbol.
        """
        def loop_function(prev):
            """The closure function returned by outer function.

            Args:
                prev: logit score for previous step output
            Returns:
                emb_prev: The embedding of output symbol sampled from
                    posterior over previous output.
            """
            # tf.multinomial performs sampling given the logit scores
            # Reshaping is required to remove the extra dimension introduced
            # by sampling for a batch size of 1.
            prev_symbol = tf.reshape(tf.multinomial(prev, 1), [-1])
            emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
            return emb_prev

        return loop_function
