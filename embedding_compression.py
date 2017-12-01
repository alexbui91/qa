"""
M=8 M=16 M=32 M=64
K=8 K=16 K=32 K=64
""" 

from __future__ import print_function
from __future__ import division

import sys

import numpy as np

import tensorflow as tf

import properties as p

class Compression(object):
    def __init__(self, word_embedding=None, M=64, K=64, batch_size=128, embedding_size=50, learning_rate=0.0001):
        self.M = M
        self.K = K
        self.hidden_layer_size = self.M * self.K // 2
        self.word_embedding = word_embedding
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.embedding_size))  

    def build_model(self):
        tf.set_random_seed(1234)
        rg = np.sqrt(6. / (self.embedding_size + self.K))
        #random A
        # hidden layer
        # embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")
        h_ = tf.layers.dense(self.input_placeholder,
                            self.hidden_layer_size,
                            # =tf.tanh,
                            name="hidden_layer")
        h_cube = tf.reshape(tf.tile(h_, [1, self.M]), [self.batch_size, self.M, self.hidden_layer_size])
        # alpha
        alp_value = tf.layers.dense(h_cube,
                                    self.K,
                                    activation=None, name="alpha_plus")
        alp_ = tf.nn.softplus(alp_value, name="alpha_softplus")
        # init noise term
        G = tf.log(-tf.log(tf.random_uniform([self.batch_size, self.M, self.K], 0, 1, dtype=tf.float32)))
        alp_log = tf.log(alp_) - G
        if  p.softmax_temperature and p.softmax_temperature != 1:
            alp_log /= p.softmax_temperature
        # one-hot => argmax to get c
        # need to extract this layer to Cw 
        # d_ B x M x K => M x B x K
        d_ = tf.nn.softmax(alp_log, name="one_hot")
        self.code_words = tf.argmax(d_, axis=2)
        # reconstruction embedding
        # need to extract weight of this layer -> code book 
        # code_book = M x K x H
        code_book = tf.Variable(tf.random_uniform([self.M, self.K, self.embedding_size], -rg, rg, dtype=tf.float32), name="code_book", trainable=True)
        d_dense = tf.matmul(tf.transpose(d_, perm=[1,0,2]), code_book)
        # d_dense = tf.layers.dense(d_, self.embedding_size, =None, name="code_book_matrix")
        e_ = tf.reduce_sum(tf.transpose(d_dense, perm=[1,0,2]), axis=1, name="final")
        print(e_.get_shape())
        return e_

    def add_loss_op(self, target):
        # using cosin distance
        # loss = tf.losses.cosine_distance(labels=self.input_placeholder, predictions=target)
        # using vector distance
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(target, self.input_placeholder)), reduction_indices=1))
        tf.summary.scalar('loss', loss)
        return loss

    def init_opts(self):
        with tf.device('/%s' % p.device):
            # init memory
            self.add_placeholders()
            # init computational node
            # init model
            self.output = self.build_model()
            # init cost function
            self.loss = self.add_loss_op(self.output)
            # init gradient
            self.train_step = self.add_training_op(self.loss)
        self.merged = tf.summary.merge_all()

    # using adam gradient
    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = opt.compute_gradients(loss)

        train_op = opt.apply_gradients(gvs)

        return train_op

    def update_weight(self):
        np.savetxt("W.csv", W_val, delimiter=",")

    def run_epoch(self, session, data, num_epoch=0, train_writer=None):
        total_loss = []
        total_steps = len(data) // self.batch_size
        vocabs = []
        for step in xrange(total_steps):
            index = range(step * self.batch_size, (step + 1) * self.batch_size)
            feed = {self.input_placeholder: self.word_embedding[index]}
            loss, code_words, summary = session.run([self.loss, self.code_words, self.merged], feed_dict=feed)
            if train_writer is not None:
                train_writer.add_summary(
                    summary, num_epoch * total_steps + step)
            for row in code_words:
                vocabs.append(row)
            total_loss.append(loss)
            sys.stdout.write('\r{} / {} : loss = {}'.format(
                step, total_steps, np.mean(total_loss)))
            sys.stdout.flush()
            sys.stdout.write('\r')
        return np.mean(total_loss), vocabs

    