from __future__ import print_function
from __future__ import division

import sys
import time

import numpy as np

import tensorflow as tf

from tensorflow.contrib.rnn import BasicLSTMCell

import properties as p

class ModelSentiment():

    def set_data(self, train, valid, word_embedding, max_input_len, using_compression=False):
        self.train = train
        self.valid = valid
        self.word_embedding = word_embedding
        self.max_input_len = max_input_len
        self.using_compression = False

    def init_ops(self):
        with tf.device('/%s' % p.device):
            # init memory
            self.add_placeholders()
            # init model
            self.output = self.inference()
            # init prediction step
            self.pred = self.get_predictions(self.output)
            # init cost function
            self.calculate_loss = self.add_loss_op(self.output)
            # init gradient
            self.train_step = self.add_training_op(self.calculate_loss)

        self.merged = tf.summary.merge_all()

    def add_placeholders(self):
        """add data placeholder to graph """
        
        self.input_placeholder = tf.placeholder(tf.int32, shape=(
            p.batch_size, self.max_input_len))  

        self.input_len_placeholder = tf.placeholder(
            tf.int32, shape=(p.batch_size,))
        self.pred_placeholder = tf.placeholder(
            tf.int32, shape=(p.batch_size,))
        # place holder for start vs end position

        self.dropout_placeholder = tf.placeholder(tf.float32)

    def inference(self):
        """Performs inference on the DMN model"""

        # set up embedding
        embeddings = tf.Variable(
            self.word_embedding.astype(np.float32), name="Embedding")
       
        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get input representation')
            word_reps = self.get_input_representation(embeddings)
            word_reps = tf.reduce_mean(word_reps, axis=1)
            # print(word_reps)

        with tf.variable_scope("hidden", initializer=tf.contrib.layers.xavier_initializer()):
            
            # output = tf.layers.dense(word_reps,
            #                         p.embed_size,
            #                         activation=tf.nn.tanh,
            #                         name="h1")

            output = tf.layers.dense(word_reps,
                                    p.hidden_size,
                                    activation=tf.nn.tanh,
                                    name="h2")
                                    
            output = tf.nn.dropout(output, self.dropout_placeholder)

            output = tf.layers.dense(output,
                                    p.sentiment_classes,
                                    name="fn")
        return output

    def get_input_representation(self, embeddings):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        inputs = tf.nn.embedding_lookup(embeddings, self.input_placeholder)
        # chunking_len = int(self.max_input_len / p.fixation)
        # inputs = tf.reshape(tf.reshape(inputs, [-1]), [p.batch_size, chunking_len, p.fixation * p.embed_size])
        # use encoding to get sentence representation plus position encoding
        # (like fb represent)
        gru_cell = BasicLSTMCell(p.embed_size)
        # outputs with [batch_size, max_time, cell_bw.output_size]
        outputs, _ = tf.nn.dynamic_rnn(
            gru_cell,
            inputs,
            dtype=np.float32,
            sequence_length=self.input_len_placeholder
        )

        return outputs


    def add_loss_op(self, output):
        """Calculate loss"""

        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output, labels=self.pred_placeholder))
        
        # add l2 regularization for all variables except biases
        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
                loss += p.l2 * tf.nn.l2_loss(v)

        tf.summary.scalar('loss', loss)

        return loss


    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(learning_rate=p.lr)
        gvs = opt.compute_gradients(loss)

        train_op = opt.apply_gradients(gvs)
        return train_op
   
    def get_predictions(self, output):
        pred = tf.nn.softmax(output)
        pred = tf.argmax(pred, 1)
        return pred

    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False):
        dp = p.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        total_steps = len(data[0]) // p.batch_size
        total_loss = []
        accuracy = 0

        # shuffle data
        r = np.random.permutation(len(data[0]))
        ct, ct_l, pr = data
        ct, ct_l, pr = np.asarray(ct, dtype=np.float32), np.asarray(ct_l, dtype=np.float32), np.asarray(pr, dtype=np.float32)
        ct, ct_l, pr = ct[r], ct_l[r], pr[r]
        for step in range(total_steps):
            index = range(step * p.batch_size,
                          (step + 1) * p.batch_size)
            feed = {self.input_placeholder: ct[index],
                    self.input_len_placeholder: ct_l[index] / p.fixation,
                    self.pred_placeholder: pr[index],
                    self.dropout_placeholder: dp}
            
            pred_labels = pr[step * p.batch_size:(step + 1) * p.batch_size]
            
            loss, pred, summary, _ = session.run(
                [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)

            if train_writer is not None:
                train_writer.add_summary(
                    summary, num_epoch * total_steps + step)
            
            accuracy += (np.sum(pred == pred_labels)) / float(len(pred_labels))
           
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
