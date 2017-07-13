from __future__ import print_function
from __future__ import division

import sys
import time

import numpy as np
from copy import deepcopy

import tensorflow as tf
from attention_gru_cell import AttentionGRUCell

from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

import preload
from softmax_cell import SMGRUCell

import properties as p

class Config(object):
    """Holds model hyperparams and data information."""

    reset = False

    batch_size = 100

    max_epochs = 256
    max_answer_len = 2
    early_stopping = 20

    dropout = 0.9
    lr = 0.001
    l2 = 0.001

    cap_grads = False
    max_grad_val = 10
    noisy_grads = False

    word2vec_init = p.use_glove
    embedding_init = np.sqrt(3)

    # set to zero with strong supervision to only train gates
    strong_supervision = False
    beta = 1

    # NOTE not currently used hence non-sensical anneal_threshold
    anneal_threshold = 1000
    anneal_by = 1.5

    num_hops = 3
    num_attention_features = 4

    max_allowed_inputs = 130
    num_train = 900

    floatX = np.float32

    task_id = "1"
    babi_test_id = ""

    train_mode = True
    #softmax or rnn
    answer_prediction = p.answer_prediction


def _add_gradient_noise(t, stddev=1e-3, name=None):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks."""
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

# from https://github.com/domluna/memn2n


def _position_encoding(sentence_size, embedding_size):
    """Position encoding described in section 4.1 in "End to End Memory Networks" (http://arxiv.org/pdf/1503.08895v5.pdf)"""
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)


class Model(object):

    def __init__(self, config, glove):
        self.config = config
        self.glove = glove
        self.init_global()

    def init_global(self):
        self.variables_to_save = {}
        # load data from babi dataset
        self.load_data(debug=False)

        with tf.device('/%s' % p.device):
            # init memory
            self.add_placeholders()
            # init computational node
            # init model
            self.output = self.inference()
            # init prediction step
            self.pred = self.get_predictions(self.output)
            # init cost function
            self.calculate_loss = self.add_loss_op(self.output)
            # init gradient
            self.train_step = self.add_training_op(self.calculate_loss)
        self.merged = tf.summary.merge_all()

    def load_data(self, debug=False):
        """Loads train/valid/test data and sentence encoding"""
        if self.config.train_mode:
            self.train, self.valid, self.word_embedding, self.max_q_len, \
            self.max_input_len, self.max_sen_len, self.max_answer_len, \
            self.num_supporting_facts, self.vocab_size = preload.load_babi(self.config, self.glove, split_sentences=True, train_mode=self.config.train_mode)
        else:
            self.test, self.word_embedding, self.max_q_len, self.max_input_len, \
            self.max_sen_len, self.max_answer_len, self.num_supporting_facts, \
            self.vocab_size = preload.load_babi(self.config, self.glove, split_sentences=True, train_mode=self.config.train_mode)
        # plus one in max_answer_len for an addition eos character to comparison in cross_entropy
        # minus when perform rnn
        # self.max_answer_len = self.max_answer_len + 1
        self.encoding = _position_encoding(
            self.max_sen_len, p.embed_size)

    def add_placeholders(self):
        """add data placeholder to graph"""
        #placeholder for vector inputs
        self.question_placeholder = tf.placeholder(
            tf.int32, shape=(self.config.batch_size, self.max_q_len))
        # max_sen_len: number of word in sentence
        # max_input_len: number of sent in bag
        self.input_placeholder = tf.placeholder(tf.int32, shape=(
            self.config.batch_size, self.max_input_len, self.max_sen_len))  
        # remove 1 to get real answer length in RNN. Will plus one in first y0
        # this for rnn prediction
        self.answer_placeholder = tf.placeholder(tf.int64, shape=(self.config.batch_size, self.max_answer_len))
        
        # placeholder for length of rnn 
        self.question_len_placeholder = tf.placeholder(
            tf.int32, shape=(self.config.batch_size,))
        self.input_len_placeholder = tf.placeholder(
            tf.int32, shape=(self.config.batch_size,))
        self.answer_len_placeholder = tf.placeholder(
            tf.int64, shape=(self.config.batch_size,))

        # in case of using softmax in answer  
        # this for comparison in cross_entropy  
        if self.config.answer_prediction == "softmax":
            self.answer_label_placeholder = tf.placeholder(
                tf.int64, shape=(self.config.batch_size,))
        else:
            self.answer_label_placeholder = tf.placeholder(
                tf.int64, shape=(self.config.batch_size, self.max_answer_len))
        # placeholder for position 
        self.rel_label_placeholder = tf.placeholder(tf.int32, shape=(
            self.config.batch_size, self.num_supporting_facts))

        self.dropout_placeholder = tf.placeholder(tf.float32)

    def get_question_representation(self, embeddings):
        """Get question vectors via embedding and GRU"""
        questions = tf.nn.embedding_lookup(
            embeddings, self.question_placeholder)
            
        gru_cell = tf.contrib.rnn.GRUCell(p.hidden_size)
        _, q_vec = tf.nn.dynamic_rnn(gru_cell,
                                     questions,
                                     dtype=np.float32,
                                     sequence_length=self.question_len_placeholder
                                     )
        # output with b x d
        return q_vec

    def get_input_representation(self, embeddings):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        inputs = tf.nn.embedding_lookup(embeddings, self.input_placeholder)

        # use encoding to get sentence representation plus position encoding
        # (like fb represent)
        inputs = tf.reduce_sum(inputs * self.encoding, 2)

        forward_gru_cell = tf.contrib.rnn.GRUCell(p.hidden_size)
        backward_gru_cell = tf.contrib.rnn.GRUCell(p.hidden_size)
        # outputs with [batch_size, max_time, cell_bw.output_size]
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            forward_gru_cell,
            backward_gru_cell,
            inputs,
            dtype=np.float32,
            sequence_length=self.input_len_placeholder
        )

        # f<-> = f-> + f<-
        fact_vecs = tf.reduce_sum(tf.stack(outputs), axis=0)
        # apply dropout
        fact_vecs = tf.nn.dropout(fact_vecs, self.dropout_placeholder)
        # outputs with [batch_size, max_time, cell_bw.output_size = d]
        return fact_vecs

    def get_answer_representation(self, embeddings):
        """ Get word vector of answers """
        answers = tf.nn.embedding_lookup(embeddings, self.answer_placeholder)
        # answers will be batch_size x length x dimension
        return answers

    def get_attention(self, q_vec, prev_memory, fact_vec, reuse):
        """
            Use question vector and previous memory to create scalar attention for current fact
            g = HMN(z)
        """
        with tf.variable_scope("attention", reuse=False):

            features = [fact_vec, prev_memory, q_vec, \
                        fact_vec * q_vec, \
                        fact_vec * prev_memory, \
                        tf.abs(fact_vec - q_vec), \
                        tf.abs(fact_vec - prev_memory)]
            # feature_vec with size b x 4d
            feature_vec = tf.concat(features, 1)
            attention = tf.layers.dense(feature_vec,
                                        p.embed_size,
                                        activation=tf.nn.tanh,
                                        name="layer1",
                                        reuse=reuse)

            attention = tf.layers.dense(attention,
                                        1,
                                        activation=None,
                                        name="layer2",
                                        reuse=reuse)

        return attention

    def generate_episode(self, memory, q_vec, fact_vecs, hop_index):
        """Generate episode by applying attention to current fact vectors through a modified GRU"""
        # attentions is g
        # from g generate h/e
        # squeeze is remove all dimension that has 1-d (1,2,3, 1,4) => (2,3)
        attentions = [tf.squeeze(
            self.get_attention(q_vec, memory, fv, bool(hop_index) or bool(i)), axis=1)
            for i, fv in enumerate(tf.unstack(fact_vecs, axis=1))]

        attentions = tf.transpose(tf.stack(attentions))
        self.attentions.append(attentions)
        attentions = tf.nn.softmax(attentions)
        attentions = tf.expand_dims(attentions, axis=-1)

        reuse = True if hop_index > 0 else False

        # concatenate fact vectors and attentions for input into attGRU
        gru_inputs = tf.concat([fact_vecs, attentions], 2)
        # GRU - RNN to generate final e
        with tf.variable_scope('attention_gru', reuse=reuse):
            _, episode = tf.nn.dynamic_rnn(AttentionGRUCell(p.hidden_size),
                                           gru_inputs,
                                           dtype=np.float32,
                                           sequence_length=self.input_len_placeholder
                                           )
        # episode is latest memory
        return episode

    def add_answer_module(self, rnn_output, q_vec, embeddings):
        if self.config.answer_prediction == "rnn":
            # get prediction at timestep
            rnn_output = tf.nn.dropout(rnn_output, self.dropout_placeholder, name="output_dropout")
            y0 = tf.layers.dense(rnn_output,
                                self.vocab_size,
                                activation=None,
                                name="output_prediction_softmax")
            y0 = tf.expand_dims(y0, 1)
            if self.max_answer_len > 1:
                answer_embedding = self.get_answer_representation(embeddings)
                gru_cell = SMGRUCell(p.hidden_size, q_vec, rnn_output)
                outputs, _ = tf.nn.dynamic_rnn(gru_cell,
                                            answer_embedding,
                                            dtype=np.float32,
                                            initial_state=rnn_output,
                                            sequence_length=self.answer_len_placeholder)
                # outputs will be a array of words predict with batch_size x dimension 
                # need to dense each vector answer to vocab size 
                outputs = tf.unstack(outputs, axis=1)
                pred_outputs = list()
                pred_outputs.append(y0)
                for o in outputs[:-1]:
                    o_x = tf.expand_dims(o, 1)
                    dr_output = tf.nn.dropout(o_x, self.dropout_placeholder, name="output_dropout")
                    o_d = tf.layers.dense(dr_output,
                                    self.vocab_size,
                                    activation=None,
                                    name="output_prediction_softmax", 
                                    reuse=True)
                    pred_outputs.append(o_d)
                
                output = tf.concat(pred_outputs, 1)
            else: 
                output = y0
            # dr_output = tf.nn.dropout(pred_outputs, self.dropout_placeholder)
            # #output tensor shape: batch_size x length_of_answer x vocab_size_dimension
            # output = tf.layers.dense(dr_output,
            #                         self.vocab_size,
            #                         activation=None,
            #                         name="output_prediction_rnn")
        else:
            # currently use just 1 single output answer
            rnn_output = tf.nn.dropout(rnn_output, self.dropout_placeholder)
            output = tf.layers.dense(tf.concat([rnn_output, q_vec], 1),
                                    self.vocab_size,
                                    activation=None,
                                    name="output_prediction_softmax")
        return output

    def inference(self):
        """Performs inference on the DMN model"""

        # set up embedding
        embeddings = tf.Variable(
            self.word_embedding.astype(np.float32), name="Embedding")
            
        # input fusion module
        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get question representation')
            q_vec = self.get_question_representation(embeddings)

        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get input representation')
            fact_vecs = self.get_input_representation(embeddings)

        # keep track of attentions for possible strong supervision
        self.attentions = []

        # memory module
        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> build episodic memory')

            # generate n_hops episodes
            # loop to generate memory over multiple hops
            prev_memory = q_vec

            for i in range(self.config.num_hops):
                # get a new episode
                print('==> generating episode', i)
                episode = self.generate_episode(
                    prev_memory, q_vec, fact_vecs, i)

                # untied weights for memory update
                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode, q_vec], 1),
                                                  p.hidden_size,
                                                  activation=tf.nn.relu)

            output = prev_memory

        # pass memory module output through linear answer module
        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            output = self.add_answer_module(output, q_vec, embeddings)

        return output

    def add_loss_op(self, output):
        """Calculate loss"""
        # optional strong supervision of attention with supporting facts
        gate_loss = 0
        if self.config.strong_supervision:
            for i, att in enumerate(self.attentions):
                labels = tf.gather(tf.transpose(self.rel_label_placeholder), 0)
                gate_loss += tf.reduce_sum(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=att, labels=labels))
        if self.config.answer_prediction == "rnn":
            answer_flat = tf.reshape(self.answer_label_placeholder, shape=[-1])
            output = tf.reshape(output, shape=[self.config.batch_size * self.max_answer_len, self.vocab_size])
            # print("answer_flat", answer_flat)
            # print("output", output)
            """
                answer_flat Tensor("DMN/Reshape_2:0", shape=(200,), dtype=int64)
                output Tensor("DMN/Reshape_3:0", shape=(300, 24), dtype=float32)
            """
            loss = self.config.beta * tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=output, labels=answer_flat)) + gate_loss
        else:
            """ answer Tensor("DMN/Placeholder_6:0", shape=(100,), dtype=int64)
                output Tensor("DMN/answer/output_prediction_softmax/BiasAdd:0", shape=(?, 32), dtype=float32)
            """
            # print("answer", self.answer_label_placeholder)
            # print("output", output)
            loss = self.config.beta * tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=output, labels=self.answer_label_placeholder)) + gate_loss

        # add l2 regularization for all variables except biases
        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
                loss += self.config.l2 * tf.nn.l2_loss(v)

        tf.summary.scalar('loss', loss)

        return loss

    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        gvs = opt.compute_gradients(loss)

        # optionally cap and noise gradients to regularize
        if self.config.cap_grads:
            gvs = [(tf.clip_by_norm(grad, self.config.max_grad_val), var)
                   for grad, var in gvs]
        if self.config.noisy_grads:
            gvs = [(_add_gradient_noise(grad), var) for grad, var in gvs]

        train_op = opt.apply_gradients(gvs)
        return train_op
   
    def get_predictions(self, output):
        preds = tf.nn.softmax(output)
        if self.config.answer_prediction == "rnn":
            preds = tf.argmax(preds, -1)
        else:
            preds = tf.argmax(preds, 1)
        return preds

    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False):
        config = self.config
        dp = config.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        total_steps = len(data[0]) // config.batch_size
        total_loss = []
        accuracy = 0

        # shuffle data
        p = np.random.permutation(len(data[0]))
        if config.answer_prediction == "softmax":
            qp, ip, ql, il, im, a, r = data
            qp, ip, ql, il, im, a, r = qp[p], ip[p], ql[p], il[p], im[p], a[p], r[p]
        else:
            qp, ip, ql, il, im, a, al, alb, r = data
            qp, ip, ql, il, im, a, al, alb, r = qp[p], ip[p], ql[p], il[p], im[p], a[p], al[p], alb[p], r[p]

        for step in range(total_steps):
            index = range(step * config.batch_size,
                          (step + 1) * config.batch_size)
            feed = {self.question_placeholder: qp[index],
                    self.input_placeholder: ip[index],
                    self.question_len_placeholder: ql[index],
                    self.input_len_placeholder: il[index],
                    self.dropout_placeholder: dp}
            if config.answer_prediction == "softmax":
                feed[self.answer_label_placeholder] = a[index]
                answers = a[step *
                        config.batch_size:(step + 1) * config.batch_size]
            else:
                feed[self.answer_placeholder] = a[index]
                if not train:
                    print(a[index])
                feed[self.answer_label_placeholder] = alb[index]
                feed[self.answer_len_placeholder] = al[index]
                answers = alb[step *
                        config.batch_size:(step + 1) * config.batch_size]
            if config.strong_supervision:
                feed[self.rel_label_placeholder] = r[index]

            loss, pred, summary, _ = session.run(
                [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)

            if train_writer is not None:
                train_writer.add_summary(
                    summary, num_epoch * total_steps + step)

            if config.answer_prediction == "softmax":
                accuracy += np.sum(pred == answers) / float(len(answers))
            else:
                ans_length = len(np.hstack(answers))
                accuracy += np.sum(pred == answers) / float(ans_length)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
        avg_acc = 0.
        if total_steps:
            avg_acc = accuracy / float(total_steps)
        return np.mean(total_loss), avg_acc