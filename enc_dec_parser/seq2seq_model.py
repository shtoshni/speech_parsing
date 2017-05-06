"""Seq2Seq model class that creates the computation graph.

Author: Trang Tran and Shubham Toshniwal
Contact: ttmt001@uw.edu, shtoshni@ttic.edu
Date: April, 2017
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops

import data_utils
from encoder import Encoder
from decoder import Decoder


class Seq2SeqModel(object):
    """Implements the Encoder-Decoder model."""

    def __init__(self, buckets, isTraining, max_gradient_norm, batch_size,
                 learning_rate, learning_rate_decay_factor,
                 encoder_attribs, decoder_attribs):
        """Initializer of class that defines the computational graph.

        Args:
            buckets: List of input-output sizes that limit the amount of
                sequence padding (http://goo.gl/d8ybpl).
            isTraining: boolean that denotes training v/s evaluation.
            max_gradient_norm: Maximum value of gradient norm.
            batch_size: Minibatch size used for doing SGD.
            learning_rate: Initial learning rate of optimizer
            learning_rate_decay_factor: Multiplicative learning rate decay
                factor
            {encoder, decoder}_attribs: Dictionary containing attributes for
                {encoder, decoder} RNN.
        """
        self.buckets = buckets
        self.isTraining = isTraining
        self.batch_size = batch_size

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)

        # Number of gradient updates performed
        self.global_step = tf.Variable(0, trainable=False)
        # Number of epochs done
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_incr = self.epoch.assign(self.epoch + 1)

        self.encoder_attribs = encoder_attribs
        self.decoder_attribs = decoder_attribs
        # Placeholder for encoder input IDs
        self.encoder_inputs = {}
        for feat_type in self.encoder_attribs.feat_types:
            if feat_type == "speech_frames":
                self.encoder_inputs[feat_type] = tf.placeholder(
                    tf.float32,
                    # T * B * num_frame_per_word * frame_dimension
                    shape=[None, None, self.encoder_attribs.fixed_word_length,
                           self.encoder_attribs.feat_dim],
                    name=feat_type + '_encoder')
            elif feat_type == "word_dur":
                self.encoder_inputs[feat_type] = tf.placeholder(
                    tf.float32, shape=[None, None], name=feat_type + '_encoder')
            else:
                self.encoder_inputs[feat_type] = tf.placeholder(
                    tf.int32, shape=[None, None], name=feat_type + '_encoder')

        _batch_size = self.encoder_inputs["word"].get_shape()[1].value
        # Input sequence length placeholder
        self.seq_len = tf.placeholder(tf.int32, shape=[_batch_size],
                                      name="seq_len")
        # Output sequence length placeholder
        self.seq_len_target = tf.placeholder(tf.int32, shape=[_batch_size],
                                             name="seq_len_target")

        # Input to decoder RNN. This input has an initial extra symbol - GO -
        # that initiates the decoding process.
        self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, None],
                                             name="decoder")
        # Targets are decoder inputs shifted by one thus, ignoring GO symbol
        self.targets = tf.slice(self.decoder_inputs, [1, 0], [-1, -1])

        # Initialize the encoder and decoder RNNs
        self.encoder = Encoder(isTraining, encoder_attribs)
        self.decoder = Decoder(isTraining, decoder_attribs)

        # First encode input
        self.encoder_hidden_states, self.final_state = \
            self.encoder.encode_input(self.encoder_inputs, self.seq_len)
        # Then decode
        self.outputs = \
            self.decoder.decode(self.decoder_inputs, self.seq_len_target,
                                self.encoder_hidden_states, self.final_state,
                                self.seq_len)
        if isTraining:
            # Training outputs and losses.
            self.losses = self.seq2seq_loss(self.outputs, self.targets,
                                            self.seq_len_target)
            # Gradients and parameter updation for training the model.
            params = tf.trainable_variables()
            print ("\nModel parameters:\n")
            for var in params:
                print (("{0}: {1}").format(var.name, var.get_shape()))
            print
            # Initialize optimizer
            opt = tf.train.AdamOptimizer(self.learning_rate)
            # Get gradients from loss
            gradients = tf.gradients(self.losses, params)
            # Clip the gradients to avoid the problem of gradient explosion
            # possible early in training
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)
            self.gradient_norms = norm
            # Apply gradients
            self.updates = opt.apply_gradients(zip(clipped_gradients, params),
                                               global_step=self.global_step)

        # Model saver function
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

    @staticmethod
    def seq2seq_loss(logits, targets, seq_len_target):
        """Calculate the cross entropy loss w.r.t. given target.

        Args:
            logits: A 2-d tensor of shape (TxB)x|V| containing the logit score
                per output symbol.
            targets: 2-d tensor of shape TxB that contains the ground truth
                output symbols.
            seq_len_target: Sequence length of output sequences. Required to
                mask padding symbols in output sequences.
        """
        with ops.name_scope("sequence_loss", [logits, targets]):
            flat_targets = tf.reshape(targets, [-1])
            cost = nn_ops.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=flat_targets)

            # Mask this cost since the output sequence is padded
            batch_major_mask = tf.sequence_mask(seq_len_target,
                                                dtype=tf.float32)
            time_major_mask = tf.transpose(batch_major_mask, [1, 0])
            weights = tf.reshape(time_major_mask, [-1])
            mask_cost = weights * cost

            loss = tf.reshape(mask_cost, tf.shape(targets))
            # Average the loss for each example by the # of timesteps
            cost_per_example = tf.reduce_sum(loss, reduction_indices=0) /\
                tf.cast(seq_len_target, tf.float32)
            # Return the average cost over all examples
            return tf.reduce_mean(cost_per_example)

    def step(self, sess, encoder_inputs, seq_len, decoder_inputs,
             seq_len_target):
        """Perform 1 minibatch update/evaluation.

        Args:
            sess: Tensorflow session where computation graph is created
            encoder_inputs: List of a minibatch of input IDs
            seq_len: Input sequence length
            decoder_inputs: List of a minibatch of output IDs
            seq_len_target: Output sequence length
        Returns:
            Output of a minibatch updated. The exact output depends on
            whether the model is in training mode or evaluation mode.
        """
        # Pass inputs via feed dict method
        input_feed = {}
        for key in self.encoder_inputs:
            input_feed[self.encoder_inputs[key].name] = encoder_inputs[key]
        input_feed[self.decoder_inputs.name] = decoder_inputs
        input_feed[self.seq_len.name] = seq_len
        input_feed[self.seq_len_target.name] = seq_len_target

        if self.isTraining:
            # Important to have gradient updates as this operation is what
            # actually updates the parameters.
            output_feed = [self.updates,  self.gradient_norms, self.losses]
        else:
            # Evaluation
            output_feed = [self.outputs]

        outputs = sess.run(output_feed, input_feed)
        if self.isTraining:
            return outputs[1], outputs[2]
        else:
            return outputs[0]

    def _process_speech_input(self, speech_input, pad_size):
        feat_dim = self.encoder_attribs.feat_dim
        fixed_word_length = self.encoder_attribs.fixed_word_length

        speech_encoder_input = speech_input["frames"]
        partition = speech_input["partition"]

        speech_frames = []
        for frame_idx in partition:
            center_frame = int((frame_idx[0] + frame_idx[1])/2)
            start_idx = center_frame - int(fixed_word_length/2)
            end_idx = center_frame + int(fixed_word_length/2)
            raw_word_frames = speech_encoder_input[:,
                                                   frame_idx[0]:frame_idx[1]]
            # 6 * Number of frames
            raw_count = raw_word_frames.shape[1]
            if raw_count > fixed_word_length:
                # too many frames, choose wisely
                this_word_frames =\
                    speech_encoder_input[:, frame_idx[0]:frame_idx[1]]
                extra_ratio = int(raw_count/fixed_word_length)
                if extra_ratio < 2:  # delete things in the middle
                    mask = np.ones(raw_count, dtype=bool)
                    num_extra = raw_count - fixed_word_length
                    not_include = range(center_frame-num_extra,
                                        center_frame+num_extra)[::2]
                    # need to offset by beginning frame
                    not_include = [x-frame_idx[0] for x in not_include]
                    mask[not_include] = False
                else:  # too big, just sample
                    mask = np.zeros(raw_count, dtype=bool)
                    include = range(frame_idx[0], frame_idx[1])[::extra_ratio]
                    include = [x-frame_idx[0] for x in include]
                    if len(include) > fixed_word_length:
                        # still too many frames
                        num_current = len(include)
                        sub_extra = num_current - fixed_word_length
                        num_start = int((num_current - sub_extra)/2)
                        not_include = include[num_start:num_start+sub_extra]
                        for ni in not_include:
                            include.remove(ni)
                    mask[include] = True
                this_word_frames = this_word_frames[:, mask]
            else:  # not enough frames, choose frames extending from center
                this_word_frames =\
                    speech_encoder_input[:, max(0, start_idx):end_idx]
                if this_word_frames.shape[1] == 0:
                    # make random if no frame info
                    this_word_frames = np.zeros((feat_dim,
                                                 fixed_word_length))
                if start_idx < 0 and this_word_frames.shape[1] < fixed_word_length:
                    this_word_frames = np.hstack(
                        [np.zeros((feat_dim, -start_idx)),
                         this_word_frames])
                # still not enough frames
                if this_word_frames.shape[1] < fixed_word_length:
                    num_more = fixed_word_length - this_word_frames.shape[1]
                    this_word_frames = np.hstack(
                        [this_word_frames, np.zeros((feat_dim, num_more))])
            # flip frames within word
            this_word_frames = np.fliplr(this_word_frames)
            speech_frames.append(this_word_frames)
        padding = [np.zeros((feat_dim, fixed_word_length))
                   for _ in range(pad_size)]
        # flip words in sequence
        speech_stuff = list(reversed(speech_frames)) + padding
        return speech_stuff

    def get_batch(self, data):
        """Prepare minibatch from given data.

        Args:
            data: A list of datapoints (all from same bucket).
            bucket_id: Bucket ID of data. This is irrevelant for training but
                for evaluation we can limit the padding by the bucket size.
        Returns:
            Batched input IDs, input sequence length, output IDs & output
            sequence length
        """
        bucket_id = data.keys()[0]  # Only one key
        if not self.isTraining:
            # During evaluation the bucket size limits the amount of padding
            _, decoder_size = self.buckets[bucket_id]

        encoder_inputs, decoder_inputs = {}, []
        for feat_type in self.encoder_inputs:
            encoder_inputs[feat_type] = []
        batch_size = len(data[bucket_id])

        seq_len = np.zeros((batch_size), dtype=np.int64)
        seq_len_target = np.zeros((batch_size), dtype=np.int64)

        for i, sample in enumerate(data[bucket_id]):
            sent_id, encoder_input, decoder_input = sample
            seq_len[i] = len(encoder_input["word"])
            if not self.isTraining:
                seq_len_target[i] = decoder_size
            else:
                # 1 is added to output sequence length because the EOS token is
                # crucial for "halting" the decoder. Consider it the punctuation
                # mark of a English sentence. Both are necessary.
                #seq_len_target[i] = len(decoder_input) + 1

                # EOS is already part of processed file
                seq_len_target[i] = len(decoder_input)

        # Maximum input and output length which limit the padding till them
        max_len_source = max(seq_len)
        max_len_target = max(seq_len_target)

        for i, sample in enumerate(data[bucket_id]):
            sent_id, encoder_input, decoder_input = sample
            # Encoder inputs are padded and then reversed.
            # Encoder input is reversed - https://arxiv.org/abs/1409.3215
            for feat_type in self.encoder_inputs:
                encoder_pad_size = (max_len_source -
                                    len(encoder_input[feat_type]))
                encoder_pad = [data_utils.PAD_ID] * encoder_pad_size
                if feat_type != "speech_frames":
                    encoder_inputs[feat_type].append(
                        list(reversed(encoder_input[feat_type])) + encoder_pad)
                else:
                    # Separate pad sizes is computed since the speech corresponding to
                    # text could be missing, in which case we want to pad everything
                    pad_size = (max_len_source -
                            len(encoder_input["speech_frames"]["partition"]))
                    encoder_inputs["speech_frames"].append(
                        self._process_speech_input(
                            encoder_input["speech_frames"], pad_size))
            # 1 is added to decoder_input because GO_ID is considered a part of
            # decoder input. While EOS_ID is also added, it's really used by
            # the target tensor (self.tensor) in the core code above.
            #decoder_pad_size = max_len_target - (len(decoder_input) + 1)

            # EOS is already added as part of preprocessing
            decoder_pad_size = max_len_target - (len(decoder_input))
            decoder_inputs.append([data_utils.GO_ID] +
                                  decoder_input +
                                  #[data_utils.EOS_ID] +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Both the id sequences are made time major via transpose
        for feat_type in self.encoder_inputs:
            #print (encoder_inputs[feat_type])
            if feat_type == "speech_frames":
                encoder_inputs["speech_frames"] = np.transpose(
                    np.asarray(encoder_inputs["speech_frames"],
                               dtype=np.float32), (1, 0, 3, 2))
            elif feat_type == "word_dur":
                encoder_inputs[feat_type] = np.asarray(
                    encoder_inputs["word_dur"], dtype=np.float32).T
            else:
                encoder_inputs[feat_type] = np.asarray(
                    encoder_inputs[feat_type], dtype=np.int32).T
                # print (len(encoder_inputs["speech_frames"]))

        decoder_inputs = np.asarray(decoder_inputs, dtype=np.int32).T
        return encoder_inputs, seq_len, decoder_inputs, seq_len_target
