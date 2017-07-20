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

from model import Config, Model, _add_gradient_noise, _position_encoding


class ModelSquad(Model):

    def __init__(self, config):
        self.config = config

    def set_data(self, train, valid, word_embedding, max_q_len, \
                max_input_len, max_answer_len, vocab_len):
        self.train = train
        self.valid = valid
        self.word_embedding = word_embedding
        self.max_q_len = max_q_len
        self.max_input_len = max_input_len
        self.max_sen_len = self.max_input_len
        self.max_answer_len = max_answer_len
        self.vocab_size = vocab_len

    def add_placeholders(self):
        """add data placeholder to graph """
        
        self.question_placeholder = tf.placeholder(
            tf.int32, shape=(self.config.batch_size, self.max_q_len))
    
        self.input_placeholder = tf.placeholder(tf.int32, shape=(
            self.config.batch_size, self.max_input_len))  

        # placeholder for length of rnn 
        self.question_len_placeholder = tf.placeholder(
            tf.int32, shape=(self.config.batch_size,))
        self.input_len_placeholder = tf.placeholder(
            tf.int32, shape=(self.config.batch_size,))
    
        # place holder for start vs end position

        self.start_placeholder = tf.placeholder(
            tf.int64, shape=(self.config.batch_size,))
        
        self.end_placeholder = tf.placeholder(
            tf.int64, shape=(self.config.batch_size,))
        
        self.dropout_placeholder = tf.placeholder(tf.float32)

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
        # get Gt from each input vector Ct
        attentions = [tf.squeeze(
            self.get_attention(q_vec, memory, fv, bool(hop_index) or bool(i)), axis=1)
            for i, fv in enumerate(tf.unstack(fact_vecs, axis=1))]
        # tf.stack(attentions) will be length x batch_size x dimension
        # transpose to batch_size x length x dimension
        attentions = tf.transpose(tf.stack(attentions))
        # self.attentions will mem_length x batch_size x length x dimension
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

    def add_answer_module(self, rnn_output, q_vec, embeddings, reuse=False):
        # currently use just 1 single output answer
        rnn_output = tf.nn.dropout(rnn_output, self.dropout_placeholder)
        output_s = tf.layers.dense(tf.concat([rnn_output, q_vec], 1),
                                self.max_input_len,
                                activation=None,
                                name="output_prediction_start",
                                reuse=reuse)
        output_e = tf.layers.dense(tf.concat([rnn_output, q_vec], 1),
                                self.max_input_len,
                                activation=None,
                                name="output_prediction_end",
                                reuse=reuse)
        return output_s, output_e

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
            a_s = tf.zeros_like(q_vec, dtype=self.config.floatX)
            a_e = tf.zeros_like(q_vec, dtype=self.config.floatX)
            output_s = tf.zeros_like(q_vec, dtype=self.config.floatX)
            output_e = tf.zeros_like(q_vec, dtype=self.config.floatX)
            
            for i in xrange(self.config.num_hops):
                # get a new episode
                print('==> generating episode', i)
                episode = self.generate_episode(
                    prev_memory, q_vec, fact_vecs, i)

                # in last time prediction, push real answer to memory
                if i == (self.config.num_hops - 1):
                    a_e = self.get_answer_representation(self.start_placeholder, embeddings)
                    a_s = self.get_answer_representation(self.start_placeholder, embeddings)
                
                # untied weights for memory update
                with tf.variable_scope("hop_%d" % i):
                    # at first time, use zeros vector
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode, q_vec, a_s, a_e], 1),
                                                  p.hidden_size,
                                                  activation=tf.nn.relu)
                    output = prev_memory
                
                # pass memory module output through linear answer module
                with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
                    output_s, output_e = self.add_answer_module(output, q_vec, embeddings, bool(i))
                    # if i still smaller than pen_final_hops, use predicted answer for next memory
                    if i < self.config.num_hops - 2:
                        pred_s, pred_e = self.get_predictions((output_s, output_e))
                        a_e = self.get_answer_representation(pred_s, embeddings)
                        a_s = self.get_answer_representation(pred_e, embeddings)
            
        # pass memory module output through linear answer module
        # with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
        #     output_s, output_e = self.add_answer_module(output, q_vec, embeddings)

        return (output_s, output_e)

    def get_input_representation(self, embeddings):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        inputs = tf.nn.embedding_lookup(embeddings, self.input_placeholder)
        # use encoding to get sentence representation plus position encoding
        # (like fb represent)
        forward_gru_cell = tf.contrib.rnn.LSTMCell(p.hidden_size)
        backward_gru_cell = tf.contrib.rnn.LSTMCell(p.hidden_size)
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

    def get_answer_representation(self, placeholder, embeddings):
        return tf.nn.embedding_lookup(embeddings, placeholder)

    def add_loss_op(self, output):
        output_s, output_e = output
        """Calculate loss"""
        # optional strong supervision of attention with supporting facts
        gate_loss = 0
        if self.config.strong_supervision:
            # like np.hstack but only get first index
            # labels = tf.gather(tf.transpose(self.rel_label_placeholder), 0)
            for i, att in enumerate(self.attentions):
                gate_loss += tf.reduce_sum(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=att, labels=self.start_placeholder))
                gate_loss += tf.reduce_sum(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=att, labels=self.end_placeholder))

        loss = (tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output_s, labels=self.start_placeholder)) + \
            tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output_e, labels=self.end_placeholder)))
        
        if self.config.beta != 1:
            loss *= self.config.beta
        if gate_loss != 0:
            loss += gate_loss

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
        output_s, output_e = output
        pred_s = tf.nn.softmax(output_s)
        pred_e = tf.nn.softmax(output_e)
        pred_s = tf.argmax(pred_s, 1)
        pred_e = tf.argmax(pred_e, 1)
        return pred_s, pred_e

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
      
        ct, ct_l, q, q_l, _, _, s, e = data
        ct, ct_l, q, q_l, s, e = np.asarray(ct, dtype=config.floatX), np.asarray(ct_l, dtype=config.floatX), \
                                        np.asarray(q, dtype=config.floatX), np.asarray(q_l, dtype=config.floatX), \
                                        np.asarray(s, dtype=config.floatX), np.asarray(e, dtype=config.floatX)
        ct, ct_l, q, q_l, s, e = ct[p], ct_l[p], q[p], q_l[p], s[p], e[p]
        for step in range(total_steps):
            index = range(step * config.batch_size,
                          (step + 1) * config.batch_size)
            feed = {self.question_placeholder: q[index],
                    self.question_len_placeholder: q_l[index],
                    self.input_placeholder: ct[index],
                    self.input_len_placeholder: ct_l[index],
                    self.start_placeholder: s[index],
                    self.end_placeholder: e[index],
                    self.dropout_placeholder: dp}
            
            start = s[step * config.batch_size:(step + 1) * config.batch_size]
            end = e[step * config.batch_size:(step + 1) * config.batch_size]
            
            # if config.strong_supervision:
            #     feed[self.rel_label_placeholder] = r[index]

            loss, pred, summary, _ = session.run(
                [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)

            if train_writer is not None:
                train_writer.add_summary(
                    summary, num_epoch * total_steps + step)
            
            pred_s, pred_e = pred

            accuracy += (np.sum(pred_s == start) + np.sum(pred_e == end)) / float(len(start) + len(end))
           
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
