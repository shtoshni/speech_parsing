from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import random
import sys
import time

import numpy as np
import tensorflow as tf
import cPickle as pickle
import argparse
import operator
from bunch import bunchify
from copy import deepcopy

import data_utils
import seq2seq_model
import subprocess
from tree_utils import add_brackets, match_length, delete_empty_constituents, merge_sent_tree

# Use the following buckets:
_buckets = [(10, 55), (25, 110), (50, 200), (100, 350)]
FLAGS = object()


def parse_feat_types(feat_str):
    """Parse type of inputs."""
    feats = feat_str.split(",")
    feat_types = ['word']
    for feat in feats:
        if feat == 'pb':
            feat_types.append('pause_bef')
        elif feat == "pa":
            feat_types.append('pause_aft')
        elif feat == "p":
            feat_types.append('pause_bef')
            feat_types.append('pause_aft')
        elif feat == "wd":
            feat_types.append('word_dur')
        elif feat == "s":
            feat_types.append("speech_frames")

    return feat_types


def parse_filter_sizes(filter_size_str):
    """Parse filter sizes."""
    filter_sizes = [int(filter_size) for filter_size in
                    filter_size_str.split("-")]
    return filter_sizes


def parse_options():
    """Parse command line options."""
    parser = argparse.ArgumentParser()

    # Learning parameters
    parser.add_argument("-lr", "--learning_rate",
                        default=1e-3, type=float,
                        help="learning rate")
    parser.add_argument("-lr_decay", "--learning_rate_decay_factor",
                        default=0.9, type=float,
                        help="multiplicative decay factor for learning rate")
    parser.add_argument("-opt", "--optimizer", default="adam",
                        type=str, help="Optimizer")
    parser.add_argument("-bsize", "--batch_size",
                        default=64, type=int, help="Mini-batch Size")
    parser.add_argument("-max_gnorm", "--max_gradient_norm",
                        default=5.0, type=float,
                        help="Maximum allowed norm of gradients")
    parser.add_argument("-max_epochs", "--max_epochs",
                        default=50, type=int, help="Max epochs")
    parser.add_argument("-num_check", "--steps_per_checkpoint",
                        default=500, type=int,
                        help="Number of steps before updated model is saved")

    # Common encoder-decoder attribs
    parser.add_argument("-esize", "--embedding_size",
                        default=512, type=int, help="Embedding Size")
    parser.add_argument("-hsize", "--hidden_size",
                        default=256, type=int, help="Hidden layer size")
    parser.add_argument("-num_layers", "--num_layers",
                        default=3, type=int,
                        help="Number of stacked RNN layers")
    parser.add_argument("-out_prob", "--output_keep_prob",
                        default=0.7, type=float,
                        help="Output keep probability for dropout")

    # Encoder attrib
    parser.add_argument("-psize", "--pause_size",
                        default=32, type=int, help="Pause embedding size")
    parser.add_argument("-feat_types", "--feat_types",
                        default="", type=str, help="Types of feature")
    parser.add_argument("-num_filters", "--num_filters",
                        default=32, type=int,
                        help="Number of convolution filters")
    parser.add_argument("-filter_sizes", "--filter_sizes",
                        default="10-25-50", type=str,
                        help="Convolution filter sizes")
    parser.add_argument("-feat_dim", "--feat_dim",
                        default=6, type=int,
                        help="Dimension of acoustic features")
    parser.add_argument("-fixed_word_length", "--fixed_word_length",
                        default=100, type=int,
                        help="Number of speech frames per word")

    # Decoder attribs
    parser.add_argument("-use_conv", "--use_convolution",
                        default=False, action="store_true",
                        help="Use convolution feature in attention")
    parser.add_argument("-conv_filter", "--conv_filter_width",
                        default=40, type=int,
                        help="Convolution filter width dimension")
    parser.add_argument("-conv_channel", "--conv_num_channels",
                        default=5, type=int,
                        help="Number of output convolution channels")

    parser.add_argument("-tv_file", "--target_vocab_file",
                        default="vocab.parse", type=str,
                        help="Vocab file for target")
    parser.add_argument("-data_dir", "--data_dir",
                        default="/share/data/speech/shtoshni/research/sw_parsing/data/prosody_buckets/my_fmt",
                        type=str, help="Data directory")
    parser.add_argument("-tb_dir", "--train_base_dir",
                        default="/share/data/speech/shtoshni/research/sw_parsing/cmd_t2p/models",
                        type=str, help="Training directory")
    parser.add_argument("-bm_dir", "--best_model_dir",
                        default="/share/data/speech/shtoshni/research/sw_parsing/cmd_t2p/models/best_models",
                        type=str, help="Best model directory")
    parser.add_argument("-vocab_dir", "--vocab_dir",
                        default="/share/data/speech/shtoshni/research/sw_parsing/data/vocab_dir",
                        type=str, help="Vocab directory")
    parser.add_argument("-prefix", "--prefix",
                        default="nopunc", type=str, help="Type of data")

    parser.add_argument("-eval", "--eval_dev",
                        default=False, action="store_true",
                        help="Get dev set results using the last saved model")
    parser.add_argument("-test", "--test",
                        default=False, action="store_true",
                        help="Get test results using the last saved model")
    parser.add_argument("-core", "--core", default=False, action="store_true",
                        help="Core eval sets only")
    parser.add_argument("-run_id", "--run_id",
                        default=0, type=int, help="Run ID")

    args = parser.parse_args()
    arg_dict = vars(args)

    feat_types = parse_feat_types(arg_dict['feat_types'])
    arg_dict['feat_types'] = feat_types
    feat_type_string = ""
    for i, feat_type in enumerate(feat_types):
        suffix = "_"
        feat_type_string += feat_type + suffix

    conv_string = ""
    if arg_dict['use_convolution']:
        conv_string = ("use_conv_" + "filter_dim_" +
                       str(arg_dict['conv_filter_width']) + "_" +
                       "num_channel_" + str(arg_dict['conv_num_channels'])
                       + "_")

    psize_string = ""
    if arg_dict['pause_size'] != 32:
        psize_string = "psize_" + str(arg_dict['pause_size']) + "_"

    train_dir = ('lr' + '_' + str(arg_dict['learning_rate']) + '_' +
                 'bsize' + '_' + str(arg_dict['batch_size']) + '_' +
                 'esize' + '_' + str(arg_dict['embedding_size']) + '_' +
                 psize_string +
                 'hsize' + '_' + str(arg_dict['hidden_size']) + '_' +
                 'num_layers' + '_' + str(arg_dict['num_layers']) + '_' +
                 'out_prob' + '_' + str(arg_dict['output_keep_prob']) + '_' +
                 'run_id' + '_' + str(arg_dict['run_id']) + '_' +
                 conv_string +
                 feat_type_string +
                 arg_dict['prefix'] + "_" +
                 "cnn_eos"
                 )

    arg_dict['train_dir'] = os.path.join(arg_dict['train_base_dir'], train_dir)
    arg_dict['best_model_dir'] = os.path.join(
        arg_dict['best_model_dir'], train_dir)
    arg_dict['apply_dropout'] = False

    print (feat_types)
    source_vocab = {}
    arg_dict["source_vocab_file"] = {}
    arg_dict['input_vocab_size'] = {}
    for feat_type in feat_types:
        if feat_type == "speech_frames" or feat_type == "word_dur":
            continue
        elif "pause" in feat_type:
            vocab_file = os.path.join(arg_dict['vocab_dir'], "vocab.pause")
        else:
            vocab_file = os.path.join(arg_dict['vocab_dir'],
                                      "vocab." + feat_type)
        arg_dict["source_vocab_file"][feat_type] = vocab_file
        source_vocab[feat_type], _ =\
            data_utils.initialize_vocabulary(vocab_file)
        arg_dict['input_vocab_size'][feat_type] = len(source_vocab[feat_type])

    target_vocab_path = os.path.join(arg_dict['vocab_dir'],
                                     arg_dict['target_vocab_file'])
    target_vocab, _ = data_utils.initialize_vocabulary(target_vocab_path)
    arg_dict['output_vocab_size'] = len(target_vocab)

    # common_attribs contains attribute values common to encoder and decoder
    # RNNs such as number of hidden units, number of layers etc
    common_attribs = {}
    common_attribs['out_prob'] = arg_dict['output_keep_prob']
    common_attribs['hidden_size'] = arg_dict['hidden_size']
    common_attribs['num_layers'] = arg_dict['num_layers']

    encoder_attribs = deepcopy(common_attribs)
    encoder_attribs["feat_types"] = arg_dict["feat_types"]
    encoder_attribs["vocab_size"] = arg_dict["input_vocab_size"]
    encoder_attribs["embedding_size"] = {}
    for feat_type in feat_types:
        if "pause" in feat_type:
            encoder_attribs["embedding_size"][feat_type] \
                = arg_dict["pause_size"]
        elif feat_type == "word_dur" or feat_type == "speech_frames":
            continue
        else:
            encoder_attribs["embedding_size"][feat_type] \
                = arg_dict["embedding_size"]
    encoder_attribs["feat_dim"] = arg_dict["feat_dim"]
    encoder_attribs["fixed_word_length"] = arg_dict["fixed_word_length"]
    encoder_attribs["num_filters"] = arg_dict["num_filters"]
    encoder_attribs["filter_sizes"] = parse_filter_sizes(
        arg_dict["filter_sizes"])
    arg_dict["encoder_attribs"] = encoder_attribs

    decoder_attribs = deepcopy(common_attribs)
    decoder_attribs["vocab_size"] = arg_dict["output_vocab_size"]
    decoder_attribs["embedding_size"] = arg_dict["embedding_size"]
    decoder_attribs["use_conv"] = arg_dict["use_convolution"]
    decoder_attribs["conv_num_channels"] = arg_dict["conv_num_channels"]
    decoder_attribs["conv_filter_width"] = arg_dict["conv_filter_width"]
    arg_dict["decoder_attribs"] = decoder_attribs

    if not arg_dict['test'] and not arg_dict['eval_dev']:
        arg_dict['apply_dropout'] = True
        if not os.path.exists(arg_dict['train_dir']):
            os.makedirs(arg_dict['train_dir'])
        if not os.path.exists(arg_dict['best_model_dir']):
            os.makedirs(arg_dict['best_model_dir'])

        # Sort the arg_dict to create a parameter file
        parameter_file = 'parameters.txt'
        sorted_args = sorted(arg_dict.items(), key=operator.itemgetter(0))

        with open(os.path.join(arg_dict['train_dir'], parameter_file), 'w') as g:
            for arg, arg_val in sorted_args:
                sys.stdout.write(arg + "\t" + str(arg_val) + "\n")
                sys.stdout.flush()
                g.write(arg + "\t" + str(arg_val) + "\n")

    options = bunchify(arg_dict)
    return options


def load_eval_data(split="dev", isCore=False):
    """Load evaluation data."""
    if not isCore:
        dev_data_path = os.path.join(FLAGS.data_dir,
                                     split + '_' + FLAGS.prefix + '.pickle')
    else:
        dev_data_path = os.path.join(FLAGS.data_dir, "core_" + split
                                     + '_' + FLAGS.prefix + '.pickle')
    dev_set = pickle.load(open(dev_data_path))

    return dev_set


def load_train_data():
    """Load train data."""
    swtrain_data_path = os.path.join(FLAGS.data_dir,
                                     'train_' + FLAGS.prefix + '.pickle')
                                     #'test_' + FLAGS.prefix + '.pickle')
    train_sw = pickle.load(open(swtrain_data_path))

    train_bucket_sizes = [len(train_sw[b]) for b in xrange(len(_buckets))]
    print(train_bucket_sizes)
    print ("# of instances: %d" % (sum(train_bucket_sizes)))
    sys.stdout.flush()

    train_bucket_offsets = [np.arange(0, x, FLAGS.batch_size)
                            for x in train_bucket_sizes]
    offset_lengths = [len(x) for x in train_bucket_offsets]
    tiled_buckets = [[i]*s for (i, s) in
                     zip(range(len(_buckets)), offset_lengths)]
    all_bucks = [x for sublist in tiled_buckets for x in sublist]
    all_offsets = [x for sublist in list(train_bucket_offsets)
                   for x in sublist]
    train_set = zip(all_bucks, all_offsets)

    return train_sw, train_set


# evalb paths
evalb_path = '/share/data/speech/Data/ttran/parser_misc/EVALB/evalb'
prm_file = '/share/data/speech/Data/ttran/parser_misc/EVALB/seq2seq.prm'


def create_model_graph(session, isTraining):
    """Create the model graph by creating an instance of Seq2SeqModel."""
    return seq2seq_model.Seq2SeqModel(
        _buckets, isTraining, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
        FLAGS.encoder_attribs, FLAGS.decoder_attribs)


def get_model(session, isTraining=True, actual_eval=False, model_path=None):
    """Create the model graph/Restore from a prev. checkpoint."""
    model = create_model_graph(session, isTraining)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    ckpt_best = tf.train.get_checkpoint_state(FLAGS.best_model_dir)
    steps_done = 0
    if ckpt:
        steps_done = int(ckpt.model_checkpoint_path.split('-')[-1])
        if ckpt_best:
            steps_done_best =\
                int(ckpt_best.model_checkpoint_path.split('-')[-1])
            if (steps_done_best > steps_done) or actual_eval:
                ckpt = ckpt_best
                steps_done = steps_done_best
    elif ckpt_best:
        ckpt = ckpt_best

    if ckpt and (model_path is None):
        print("loaded from %d done steps" % (steps_done))
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        steps_done = int(ckpt.model_checkpoint_path.split('-')[-1])
        print("loaded from %d done steps" % (steps_done))
    elif ckpt and model_path is not None:
        model.saver.restore(session, model_path)
        steps_done = int(model_path.split('-')[-1])
        print("Reading model parameters from %s" % model_path)
        print("loaded from %d done steps" % (steps_done))
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        steps_done = 0
    return model, steps_done


def train():
    """Train a sequence to sequence parser."""
    # Prepare data
    print("Loading data from %s" % FLAGS.data_dir)
    train_sw, train_set = load_train_data()
    dev_set = load_eval_data()

    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2))\
            as sess:
        # Create model.
        print("Creating %d layers of %d units." %
              (FLAGS.num_layers, FLAGS.hidden_size))
        sys.stdout.flush()
        with tf.variable_scope("model", reuse=None):
            model, steps_done = get_model(sess, isTraining=True)
        with tf.variable_scope("model", reuse=True):
            model_dev = create_model_graph(sess, isTraining=False)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        epoch = model.epoch.eval()
        f_score_best = 0.0
        if steps_done > 0:
            # Some training has been done
            score_file = os.path.join(FLAGS.train_dir, "best.txt")
            # Check existence of such a file
            if os.path.isfile(score_file):
                try:
                    f_score_best = float(open(score_file).
                                         readline().strip("\n"))
                except ValueError:
                    f_score_best = 0.0

        print("Best F-Score: %.4f" % f_score_best)

        while epoch < FLAGS.max_epochs:
            print("Epochs done:", epoch)
            sys.stdout.flush()
            np.random.shuffle(train_set)
            for bucket_id, bucket_offset in train_set:
                this_sample = train_sw[bucket_id][bucket_offset:bucket_offset+FLAGS.batch_size]
                # Get a batch and make a step.
                start_time = time.time()
                encoder_inputs, seq_len, decoder_inputs, seq_len_target \
                    = model.get_batch({bucket_id: this_sample})
                _, step_loss = model.step(sess, encoder_inputs, seq_len,
                                              decoder_inputs, seq_len_target)


                step_time += (time.time()-start_time)/FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1

                # Once in a while, we save checkpoint, print statistics, and
                # run evals.
                if current_step % FLAGS.steps_per_checkpoint == 0:
                    # Print statistics for the previous epoch.
                    perplexity = math.exp(loss) if loss < 300 else float('inf')
                    print ("global step %d learning rate %.4f step-time %.2f"
                           " perplexity %.2f" % (model.global_step.eval(),
                                                model.learning_rate.eval(),
                                                step_time, perplexity))
                    # Decrease learning rate if no improvement was seen over
                    # last 3 times.
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        if not (FLAGS.learning_rate_decay_factor >= 1.0):
                            sess.run(model.learning_rate_decay_op)

                    previous_losses.append(loss)
                    step_time, loss = 0.0, 0.0

                    f_score_cur = write_decode(model_dev, sess, dev_set)
                    print ("Current f-score: %.4f" % f_score_cur)
                    # Early stopping
                    if f_score_best < f_score_cur:
                        f_score_best = f_score_cur
                        # Save model
                        print("Best F-Score: %.4f" % f_score_best)
                        print("Saving updated model")
                        sys.stdout.flush()

                        # Save the best score
                        f = open(os.path.join(FLAGS.train_dir, "best.txt"),
                                 "w")
                        f.write(str(f_score_best))
                        f.close()

                        # Save the model in best model directory
                        checkpoint_path = os.path.join(FLAGS.best_model_dir,
                                                       "parse_nn.ckpt")
                        model.best_saver.save(sess, checkpoint_path,
                                              global_step=model.global_step,
                                              write_meta_graph=False)

                    else:
                        # Save the model in regular directory
                        print("Saving for the sake of record")
                        checkpoint_path = os.path.join(FLAGS.train_dir,
                                                       "parse_nn.ckpt")
                        model.saver.save(sess, checkpoint_path,
                                         global_step=model.global_step,
                                         write_meta_graph=False)

                    sys.stdout.flush()
            # Update epoch counter
            sess.run(model.epoch_incr)
            epoch += 1


def process_eval(out_lines, this_size, get_results=False):
    """Process the evaluation output."""
    # main stuff between outlines[3:-32]
    results = out_lines[3:-32]
    try:
        assert len(results) == this_size
        matched = 0
        gold = 0
        test = 0
        for line in results:
            m, g, t = line.split()[5:8]
            matched += int(m)
            gold += int(g)
            test += int(t)

        if get_results:
            return matched, gold, test, results
        else:
            return matched, gold, test
    except AssertionError:
        return 0, 0, 0


def write_decode(model_dev, sess, dev_set, get_results=False):
    """Perform evaluatio."""
    # Load vocabularies.
    sents_vocab_path = os.path.join(FLAGS.vocab_dir,
                                    FLAGS.source_vocab_file['word'])
    parse_vocab_path = os.path.join(FLAGS.vocab_dir, FLAGS.target_vocab_file)
    sents_vocab, rev_sent_vocab = data_utils.initialize_vocabulary(
        sents_vocab_path)
    _, rev_parse_vocab = data_utils.initialize_vocabulary(parse_vocab_path)

    gold_file_name = os.path.join(FLAGS.train_dir, 'gold.txt')
    # file with matched brackets
    decoded_br_file_name = os.path.join(FLAGS.train_dir, 'decoded.br.txt')
    # file filler XX help as well
    decoded_mx_file_name = os.path.join(FLAGS.train_dir, 'decoded.mx.txt')
    sent_id_file_name = os.path.join(FLAGS.train_dir, 'sent_id.txt')

    fout_gold = open(gold_file_name, 'w')
    fout_br = open(decoded_br_file_name, 'w')
    fout_mx = open(decoded_mx_file_name, 'w')
    fsent_id = open(sent_id_file_name, 'w')

    num_dev_sents = 0
    for bucket_id in xrange(len(_buckets)):
        bucket_size = len(dev_set[bucket_id])
        offsets = np.arange(0, bucket_size, FLAGS.batch_size)
        for batch_offset in offsets:
            all_examples = dev_set[bucket_id][batch_offset:batch_offset+FLAGS.batch_size]
            model_dev.batch_size = len(all_examples)
            sent_ids = [x[0] for x in all_examples]
            for sent_id in sent_ids:
                fsent_id.write(sent_id + "\n")

            token_ids = [x[1] for x in all_examples]
            gold_ids = [x[2] for x in all_examples]
            dec_ids = [[]] * len(token_ids)
            encoder_inputs, seq_len, decoder_inputs, seq_len_target =\
                model_dev.get_batch({bucket_id:
                                     zip(sent_ids, token_ids, dec_ids)})
            output_logits = model_dev.step(sess, encoder_inputs, seq_len,
                                           decoder_inputs, seq_len_target)

            outputs = np.argmax(output_logits, axis=1)
            outputs = np.reshape(outputs, (max(seq_len_target),
                                           model_dev.batch_size))  # T*B

            to_decode = np.array(outputs).T
            num_dev_sents += to_decode.shape[0]
            for sent_id in range(to_decode.shape[0]):
                parse = list(to_decode[sent_id, :])
                if data_utils.EOS_ID in parse:
                    parse = parse[:parse.index(data_utils.EOS_ID)]
                decoded_parse = []
                for output in parse:
                    if output < len(rev_parse_vocab):
                        decoded_parse.append(tf.compat.as_str(
                            rev_parse_vocab[output]))
                    else:
                        decoded_parse.append("_UNK")

                parse_br, valid = add_brackets(decoded_parse)
                # get gold parse, gold sentence
                gold_parse = [tf.compat.as_str(rev_parse_vocab[output])
                              for output in gold_ids[sent_id]]
                sent_text = [tf.compat.as_str(rev_sent_vocab[output])
                             for output in token_ids[sent_id]['word']]
                # parse with also matching "XX" length
                parse_mx = match_length(parse_br, sent_text)
                parse_mx = delete_empty_constituents(parse_mx)
                # account for EOS
                to_write_gold = merge_sent_tree(gold_parse[:-1], sent_text)
                to_write_br = merge_sent_tree(parse_br, sent_text)
                to_write_mx = merge_sent_tree(parse_mx, sent_text)

                fout_gold.write('{}\n'.format(' '.join(to_write_gold)))
                fout_br.write('{}\n'.format(' '.join(to_write_br)))
                fout_mx.write('{}\n'.format(' '.join(to_write_mx)))

    # Write to file
    fout_gold.close()
    fout_br.close()
    fout_mx.close()
    fsent_id.close()

    f_score_mx = 0.0
    correction_types = ["Bracket only", "Matched XX"]
    corrected_files = [decoded_br_file_name, decoded_mx_file_name]

    for c_type, c_file in zip(correction_types, corrected_files):
        cmd = [evalb_path, '-p', prm_file, gold_file_name, c_file]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        out, err = p.communicate()
        with open(os.path.join(FLAGS.train_dir, "log.txt"), "w") as log_f:
            log_f.write(out)
        out_lines = out.split("\n")
        vv = [x for x in out_lines if "Number of Valid sentence " in x]
        if len(vv) == 0:
            return 0.0
        s1 = float(vv[0].split()[-1])
        m_br, g_br, t_br = process_eval(out_lines, num_dev_sents)

        try:
            recall = float(m_br)/float(g_br)
            prec = float(m_br)/float(t_br)
            f_score = 2 * recall * prec / (recall + prec)
        except ZeroDivisionError:
            recall, prec, f_score = 0.0, 0.0, 0.0

        print("%s -- Num valid sentences: %d; p: %.4f; r: %.4f; f1: %.4f"
              % (c_type, s1, prec, recall, f_score))
        if "XX" in c_type:
            f_score_mx = f_score

    return f_score_mx


def decode(split='dev'):
    """Decode file sentence-by-sentence."""
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2))\
            as sess:
        # Create model and load parameters.
        with tf.variable_scope("model", reuse=None):
            model_dev, steps_done = get_model(
                sess, isTraining=False, actual_eval=True)
        print ("Epochs done: %d" % model_dev.epoch.eval())
        dev_set = load_eval_data(split, FLAGS.core)

        start_time = time.time()
        write_decode(model_dev, sess, dev_set)
        time_elapsed = time.time() - start_time
        print("Decoding time: ", time_elapsed)


if __name__ == "__main__":
    FLAGS = parse_options()
    if FLAGS.test:
        decode(split='test')
    elif FLAGS.eval_dev:
        decode()
    else:
        train()
