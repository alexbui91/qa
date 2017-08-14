from __future__ import print_function
from __future__ import division

import sys

import numpy as np

import tensorflow as tf

from tensorflow.contrib.rnn import GRUCell

import properties as p

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
            start_offset, st, ed = self.inference()
            self.start_offset = start_offset
            # init prediction step
            self.pred_s = tf.add(start_offset, self.get_predictions(st))
            self.pred_e = tf.add(start_offset, self.get_predictions(ed))
            # init cost function
            self.calculate_loss = self.add_loss_op(st, ed)
            # init gradient
            self.train_step = self.add_training_op(self.calculate_loss)

        self.merged = tf.summary.merge_all()

    def add_placeholders(self):
        """add data placeholder to graph """
        
        self.input_placeholder = tf.placeholder(tf.int32, shape=(
            self.config.batch_size, self.max_input_len))  
        self.question_placeholder = tf.placeholder(tf.int32, shape=(
            self.config.batch_size, self.max_question_len))  
        # for quick skim
        self.input_len_placeholder = tf.placeholder(
            tf.int32, shape=(self.config.batch_size,))
        # for full scan
        self.input_scan_len_placeholder = tf.placeholder(
            tf.int32, shape=(self.config.batch_size,))

        self.question_len_placeholder = tf.placeholder(
            tf.int32, shape=(self.config.batch_size,))
        
        self.pred_len_placeholder = tf.placeholder(
            tf.int32, shape=(self.config.batch_size,))
        
        self.start_placeholder = tf.placeholder(
            tf.int32, shape=(self.config.batch_size,))
        self.end_placeholder = tf.placeholder(
            tf.int32, shape=(self.config.batch_size,))
        # place holder for start vs end position

        self.dropout_placeholder = tf.placeholder(tf.float32)

    def inference(self):
        epsilon = tf.constant(1e-10, dtype=tf.float32)
        """Performs inference on the DMN model"""
        embeddings = tf.Variable(
            self.word_embedding.astype(np.float32), name="Embedding")
       
        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> skim paragraph')
            word_reps = self.quick_skim_representation(embeddings, self.input_placeholder, self.max_input_len, self.input_len_placeholder)

        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> skim question')
            question_reps = self.quick_skim_representation(embeddings, self.question_placeholder, self.max_question_len, self.question_len_placeholder)

        # get answer position to scan
        with tf.variable_scope("first_look_up", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> predict start position')
            look_up_vector = tf.concat([word_reps, question_reps], 1)
            look_up_vector = tf.nn.dropout(look_up_vector, self.dropout_placeholder, name="start_point_do")
            look_up_pos = tf.layers.dense(look_up_vector,
                                        p.embed_size,
                                        activation=tf.nn.tanh,
                                        name="start_point")
            start_offset = self.get_predictions(look_up_pos)
            input_v = list()
            
            for para, s_ in zip(tf.unstack(self.input_placeholder), tf.unstack(start_offset)):
                a = tf.expand_dims(tf.gather(para, tf.range(s_, s_ + p.max_scanning)), 0)
                input_v.append(tf.reshape(a, [1, p.max_scanning]))
            
            # will be b x l x d
            # dig deep in question
        with tf.variable_scope("question_full_rep", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> deeply dig question meaning')
            questions, _ = self.get_question_representation(embeddings)

        with tf.variable_scope("input_full_rep", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get representation on span of text')
            input_vectors = self.get_input_representation(embeddings, tf.concat(input_v, 0))
        
        with tf.variable_scope("attention", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get attention with sum up pooling question')
            attention = self.get_attention_map(input_vectors, questions)

        with tf.variable_scope("pred", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> final prediction')
            cell = GRUCell(p.embed_size)
            start_p, state = self.scan_answer(cell, attention)
            end_p, _ = self.scan_answer(cell, attention, state, True)

        return start_offset, tf.add(start_p, epsilon),  tf.add(end_p, epsilon)

    def get_input_representation(self, embeddings, input_vectors):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        inputs = tf.nn.embedding_lookup(embeddings, input_vectors)
        # use encoding to get sentence representation plus position encoding
        # (like fb represent)
        cell = GRUCell(p.embed_size)
        # outputs with [batch_size, max_time, cell_bw.output_size]
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell,
            cell,
            inputs,
            dtype=np.float32,
            sequence_length=self.input_scan_len_placeholder
        )

        # f<-> = f-> + f<-
        fact_vecs = tf.reduce_sum(tf.stack(outputs), axis=0)
        # outputs with [batch_size, max_time, cell_bw.output_size = d]
        return fact_vecs

    def quick_skim_representation(self, embeddings, input_vectors, length, len_placeholder, reuse=None):
        pl = tf.div(len_placeholder, p.fixation)
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        chunking_len = int(length / p.fixation)
        inputs = tf.nn.embedding_lookup(embeddings, input_vectors)
        inputs = tf.reshape(tf.reshape(inputs, [-1]), [self.config.batch_size, chunking_len, p.fixation * p.embed_size])
        # use encoding to get sentence representation plus position encoding
        # (like fb represent)
        gru_cell = GRUCell(p.embed_size)
        # outputs with [batch_size, max_time, cell_bw.output_size]
        outputs, _ = tf.nn.dynamic_rnn(
            gru_cell,
            inputs,
            dtype=np.float32,
            sequence_length=pl
        )
        outputs = tf.transpose(outputs, perm=[0, 2, 1])
        outputs = tf.nn.dropout(outputs, self.dropout_placeholder, name="quick_skim_dense_do")
        outputs = tf.layers.dense(outputs, 1,
                                activation=tf.nn.tanh,
                                name='quick_skim_dense',
                                reuse=reuse)
        outputs = tf.squeeze(outputs)
        # output will be B x D 
        return outputs

    def get_attention_map(self, skimming_vectors, question_vectors):
        attentions = tf.unstack(skimming_vectors, axis=1)
        # l x b x d
        tmp = list()
        # e: b x d
        prev_ = tf.zeros((self.config.batch_size, p.embed_size), dtype=tf.float32)
        cell = GRUCell(p.embed_size)
        for index, a in enumerate(attentions):
            reuse = bool(index)
            _, c_ = self.get_max_pooling(question_vectors, a, reuse)
            _, prev_ = tf.nn.dynamic_rnn(cell,
                                tf.expand_dims(c_, 1),
                                dtype=np.float32,
                                sequence_length=self.pred_len_placeholder,
                                initial_state=prev_
                                )
            tmp.append(prev_)
        return tf.transpose(tf.stack(tmp), perm=[1,0,2])


    def scan_answer(self, cell, init_vector, init_state=None, reuse=False):
        # init_vector: B x L x D
        # question: B x M x D
        # e: b x d
        if init_state is None:
            init_state = tf.zeros((self.config.batch_size, p.embed_size), dtype=tf.float32)
        pred, c_ = self.get_max_pooling(init_vector, init_state, reuse)
        _, f_state = tf.nn.dynamic_rnn(cell,
                            tf.expand_dims(c_, 1),
                            dtype=np.float32,
                            sequence_length=self.pred_len_placeholder,
                            initial_state=init_state
                            )
        return pred, f_state

    def get_max_pooling(self, context, time_step_vector, reuse=False):
        """ sum over context to single vectzor"""
        context_ = tf.unstack(context, axis=1)
        tmp = list()
        for c_ in context_:
            tmp.append(tf.concat([c_, time_step_vector], axis=1))
        # B x L x D
        s_ = tf.transpose(tf.stack(tmp), perm=[1,0,2])

        # B x L x D
        s_ = tf.nn.dropout(s_, self.dropout_placeholder, name="attention_question_do")
        s_ = tf.layers.dense(s_,
                            p.embed_size,
                            activation=tf.nn.tanh,
                            name="attention_question", 
                            reuse=reuse)
        # To B x L x 1
        s_ = tf.layers.dense(s_,
                            1,
                            activation=None,
                            name="attention_question_softmax", 
                            reuse=reuse)
        # To B x L
        a_ = tf.squeeze(s_)
        tmp = list()

        for c_, a_v in zip(context_, tf.unstack(a_, axis=1)):
            tmp.append(tf.stack([e_c * e_a for e_c, e_a in zip(tf.unstack(c_), tf.unstack(a_v))]))
        u_ = tf.stack(tmp)
        # return prediction and agg vector
        # expect a_: BxL, agg: BxD
        return a_, tf.reduce_sum(u_, 0)

    def add_loss_op(self, output_s, output_e):
        """Calculate loss"""
        so = tf.cast(self.start_offset, tf.int32)
        st_label = tf.subtract(self.start_placeholder, so)
        ed_label = tf.subtract(self.end_placeholder, so)
        loss = (tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output_s, labels=st_label)) + \
            tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output_e, labels=ed_label)))

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
        total_steps = len(data[0]) // config.batch_size
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
            index = range(step * config.batch_size,
                          (step + 1) * config.batch_size)
            feed = {self.input_placeholder: ct[index],
                    self.input_len_placeholder: ct_l[index],
                    self.question_placeholder: q[index],
                    self.question_len_placeholder: ql[index],
                    self.start_placeholder: st[index],
                    self.end_placeholder: ed[index],
                    self.pred_len_placeholder: np.ones(config.batch_size),
                    self.input_scan_len_placeholder: np.full(config.batch_size, p.max_scanning),
                    self.dropout_placeholder: dp}
            
            start = st[step * config.batch_size:(step + 1) * config.batch_size]
            end = ed[step * config.batch_size:(step + 1) * config.batch_size]
            # if config.strong_supervision:
            #     feed[self.rel_label_placeholder] = r[index]

            loss, pred_s, pred_e, summary, _ = session.run(
                [self.calculate_loss, self.pred_s, self.pred_e, self.merged, train_op], feed_dict=feed)
            print('pred_s', pred_s)
            print('start', start)
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
