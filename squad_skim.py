from __future__ import print_function
from __future__ import division

import sys

import numpy as np

import tensorflow as tf

from tensorflow.contrib.rnn import GRUCell

import properties as p

from pointer_cell import PointerCell, GateAttentionBase
from model import Config, Model


class SquadSkim(Model):

    def set_data(self, train, valid, word_embedding, max_input_len, max_question_len):
        self.train = train
        self.valid = valid
        self.word_embedding = word_embedding
        self.max_input_len = max_input_len
        self.max_question_len = max_question_len

    def init_ops(self):
        with tf.device('/%s' % p.device):
            # init memory
            self.add_placeholders()
            # init model
            st, ed = self.inference()
            # init prediction step
            self.pred_s = self.get_predictions(st)
            self.pred_e = self.get_predictions(ed)
            # init cost function
            self.calculate_loss = self.add_loss_op(st, ed)
            # init gradient
            self.train_step = self.add_training_op(self.calculate_loss)

        self.merged = tf.summary.merge_all()

    def add_placeholders(self):
        """add data placeholder to graph """
        
        self.input_placeholder = tf.placeholder(tf.int32, shape=(
            p.batch_size, self.max_input_len))  
        self.question_placeholder = tf.placeholder(tf.int32, shape=(
            p.batch_size, self.max_question_len))  
        # for quick skim
        self.input_len_placeholder = tf.placeholder(
            tf.int32, shape=(p.batch_size,))
        # for full scan
        self.question_len_placeholder = tf.placeholder(
            tf.int32, shape=(p.batch_size,))
        
        self.pred_len_placeholder = tf.placeholder(
            tf.int32, shape=(p.batch_size,))
        
        self.start_placeholder = tf.placeholder(
            tf.int32, shape=(p.batch_size,))
        self.end_placeholder = tf.placeholder(
            tf.int32, shape=(p.batch_size,))
        # place holder for start vs end position

        self.dropout_placeholder = tf.placeholder(tf.float32)

    def inference(self):
        epsilon = tf.constant(1e-10, dtype=tf.float32)
        """Performs inference on the DMN model"""
        embed = p.hidden_size
        embeddings = tf.Variable(
            self.word_embedding.astype(np.float32), name="Embedding")
       
        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> encode paragraph')
            word_reps = self.get_input_representation(embeddings, self.input_placeholder, self.input_len_placeholder)

        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> encode question')
            question_reps = self.get_input_representation(embeddings, self.question_placeholder, self.question_len_placeholder)

        # mix question information to each word in paragraph
        with tf.variable_scope("input_question_attention", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get input question attention')
            iqa_cell = GateAttentionBase(embed, contexts=question_reps)
            attention, _ = tf.nn.dynamic_rnn(iqa_cell,
                                word_reps,
                                dtype=np.float32,
                                sequence_length=self.input_len_placeholder
                                )
        
        # with tf.variable_scope("self_matching_attention", initializer=tf.contrib.layers.xavier_initializer()):
        #     print('==> get self paragraph attention')
        #     sma_cell = PointerCell(embed, contexts=attention)
        #     self_attention, _ = tf.nn.dynamic_rnn(sma_cell,
        #                         attention,
        #                         dtype=np.float32,
        #                         sequence_length=self.input_len_placeholder
        #                         )

        with tf.variable_scope("init_output_state", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get self question attention')
            io_cell = PointerCell(embed, contexts=question_reps)
            _, init_answer_attention = io_cell.get_max_pooling(question_reps)

        with tf.variable_scope("output_layer_start", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> final prediction')
            o_cell = PointerCell(embed, contexts=attention)
            _, start_p = o_cell.call(state=init_answer_attention)

        with tf.variable_scope("output_layer_end", initializer=tf.contrib.layers.xavier_initializer()):
            o_e_cell = PointerCell(embed, contexts=attention)
            _, end_p = o_e_cell.call(state=start_p)

        return start_p,  end_p

    def get_input_representation(self, embeddings, input_vectors, len_placeholder, cell=None):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        inputs = tf.nn.embedding_lookup(embeddings, input_vectors)
        # use encoding to get sentence representation plus position encoding
        # (like fb represent)
        if cell is None:
            cell = GRUCell(p.embed_size)
        # outputs with [batch_size, max_time, cell_bw.output_size]
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell,
            cell,
            inputs,
            dtype=np.float32,
            sequence_length=len_placeholder
        )
        # f<-> = f-> + f<-
        fact_vecs = tf.concat(outputs, axis=2)
        fact_vecs = tf.layers.dense(fact_vecs,
                            p.hidden_size,
                            activation=tf.nn.tanh)
        # To B x L x 1
        return fact_vecs

    def add_loss_op(self, output_s, output_e):
        """Calculate loss"""
        # get label of start and end inside boundary
        loss = (tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output_s, labels=self.start_placeholder)) + \
            tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output_e, labels=self.end_placeholder)))

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

        train_op = opt.apply_gradients(gvs)
        return train_op
   
    def get_predictions(self, output):
        pred = tf.nn.softmax(output)
        pred = tf.argmax(pred, 1)
        return pred

    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False):
        config = self.config
        dp = config.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        total_steps = len(data[0]) // p.batch_size
        total_loss = []
        accuracy = 0

        # shuffle data
        r = np.random.permutation(len(data[0]))
        ct, ct_l, q, ql, st, ed = data
        ct, ct_l, q, ql, st, ed = np.asarray(ct, dtype=config.floatX), np.asarray(ct_l, dtype=config.floatX),  \
                                np.asarray(q, dtype=config.floatX), np.asarray(ql, dtype=config.floatX), \
                                np.asarray(st, dtype=config.floatX), np.asarray(ed, dtype=config.floatX)
        ct, ct_l, q, ql, st, ed = ct[r], ct_l[r], q[r], ql[r], st[r], ed[r]
        for step in range(total_steps):
            index = range(step * p.batch_size,
                          (step + 1) * p.batch_size)
            feed = {self.input_placeholder: ct[index],
                    self.input_len_placeholder: ct_l[index],
                    self.question_placeholder: q[index],
                    self.question_len_placeholder: ql[index],
                    self.start_placeholder: st[index],
                    self.end_placeholder: ed[index],
                    self.pred_len_placeholder: np.full(p.batch_size, 2),
                    self.dropout_placeholder: dp}
            
            start = st[step * p.batch_size:(step + 1) * p.batch_size]
            end = ed[step * p.batch_size:(step + 1) * p.batch_size]
            # if config.strong_supervision:
            #     feed[self.rel_label_placeholder] = r[index]

            loss, pred_s, pred_e, summary, _ = session.run(
                [self.calculate_loss, self.pred_s, self.pred_e, self.merged, train_op], feed_dict=feed)
            if train_writer is not None:
                train_writer.add_summary(
                    summary, num_epoch * total_steps + step)
            
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
