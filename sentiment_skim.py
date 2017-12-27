from __future__ import print_function
from __future__ import division

import sys
import time

import numpy as np

import tensorflow as tf

from tensorflow.contrib.rnn import BasicLSTMCell

import properties as p

class ModelSentiment():


    def __init__(self, word_embedding=None, max_input_len=None, using_compression=False, book=None, words=None, num_gpus=0, use_multiple_gpu=False):
        self.word_embedding = word_embedding
        self.max_input_len = max_input_len
        self.using_compression = using_compression
        self.book = book
        self.words = words
        self.num_gpus = num_gpus
        self.use_multiple_gpu = use_multiple_gpu
        self.iteration = tf.placeholder(tf.int32)
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def set_data(self, train, valid):
        self.train = train
        self.valid = valid
           
    def set_embedding(self):
        self.word_embedding = word_embedding
    
    def init_ops(self, input_vector=None, input_lens=None, labels=None):
        # init memory
        if not self.use_multiple_gpu:
            _, _, _ = self.add_placeholders()
        # init model
        output = self.inference()
        # init prediction step
        pred = self.get_predictions(output)
        # init cost function
        calculate_loss = self.add_loss_op(output, labels)
        # init gradient
        self.train_step = self.add_training_op(calculate_loss)
        if not elf.use_multiple_gpu:
            self.output, self.pred, self.calculate_loss,  = output, pred, calculate_loss
            self.merged = tf.summary.merge_all()    
        else:
            pred = tf.reduce_sum(tf.cast(tf.equal(pred, labels), tf.int32))
        return pred, calculate_loss
        
    def add_placeholders(self):
        """add data placeholder to graph """
        input_placeholder = tf.placeholder(tf.int32, shape=(
            p.batch_size, self.max_input_len))  

        input_len_placeholder = tf.placeholder(
            tf.int32, shape=(p.batch_size,))
        pred_placeholder = tf.placeholder(
            tf.int32, shape=(p.batch_size,))
        if not self.use_multiple_gpu:
            self.input_placeholder, self.input_len_placeholder, self.pred_placeholder = input_placeholder, input_len_placeholder, pred_placeholder
        return input_placeholder, input_len_placeholder, pred_placeholder

    def inference(self, input_vector=None, input_lens=None):
        """Performs inference on the DMN model"""

        # set up embedding
        embeddings = None
        if not self.using_compression:
            embeddings = tf.Variable(
                self.word_embedding.astype(np.float32), name="Embedding")
       
        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get input representation')
            if not self.use_multiple_gpu:
                word_reps = self.get_input_representation(embeddings)
            else:
                word_reps = self.get_input_representation(embeddings, input_vector, input_lens)
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

    def build_book(self):
        b = tf.Variable(self.book, name="book", trainable=False)
        w = tf.Variable(self.words, name="words", trainable=False)
        return b, w

    def get_input_representation(self, embeddings, input_vector=None, input_lens=None):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        if not self.use_multiple_gpu:
            input_lens = self.input_len_placeholder
            input_vector = self.input_placeholder
        inputs = None
        if self.using_compression:
            b_embedding, w_embedding = self.build_book()
            # from code words => build one hot
            # B x L x M: batchsize x length_sentence
            d = tf.nn.embedding_lookup(w_embedding, input_vector)
            # => B x L x M x K
            d_ = tf.reshape(d, [-1])
            d_ = tf.one_hot(d_, depth=p.code_size, axis=-1)
            d_ = tf.reshape(d_, [p.batch_size * self.max_input_len, p.book_size, p.code_size]);
            #  => M x B * L x K => B * L x K
            inputs = tf.reduce_sum(tf.matmul(tf.transpose(d_, perm=[1, 0, 2]), b_embedding), axis=0);
            inputs = tf.reshape(tf.reshape(inputs, [-1]), [p.batch_size, self.max_input_len, p.embed_size])
        else:
            # get word vectors from embedding
            inputs = tf.nn.embedding_lookup(embeddings, input_vector)
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
            sequence_length=input_lens,
        )

        return outputs


    def add_loss_op(self, output, labels):
        """Calculate loss"""

        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output, labels=labels))
        
        # add l2 regularization for all variables except biases
        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
                loss += p.l2 * tf.nn.l2_loss(v)

        tf.summary.scalar('loss', loss)

        return loss


    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        lr = tf.train.exponential_decay(learning_rate=p.lr, global_step=self.iteration, decay_steps=p.lr_depr, decay_rate=p.decay_rate)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        gvs = opt.compute_gradients(loss)

        train_op = opt.apply_gradients(gvs)
        return train_op
   
    def get_predictions(self, output):
        pred = tf.nn.softmax(output)
        pred = tf.argmax(pred, 1)
        return pred

    def shuffle_batch(self, data):
        ct, ct_l, pr = data
        total_steps = len(ct) // p.batch_size
        context, length, pred = tf.train.shuffle_batch(
                                [ct, ct_l, pr], 
                                p.batch_size, 
                                enqueue_many=True, 
                                num_threads=16,
                                capacity=total_steps)
        return tf.contrib.slim.prefetch_queue.prefetch_queue([context, length, pred], capacity =  2 * self.num_gpus)

    # prepare data for multiple gpu
    def prepare_gpu_data(self):
        self.batch_queue = self.shuffle_batch(self.train)
        self.batch_queue_valid = self.shuffle_batch(self.valid)
        self.input_placeholder, self.inputs_len, self.pred_placeholder = [], [], []
        for i in xrange(self.num_gpus):
            ip, il, pl = self.add_placeholders();
            self.input_placeholder.append(ip)
            self.input_len_placeholder.append(il),
            self.pred_placeholder.append(pl)
    
    def debug(self, step, total_steps, total_loss):
        sys.stdout.write('\r{} / {} : loss = {}'.format(
            step, total_steps, total_loss))
        sys.stdout.flush()

    # prepare machine to run multiple gpu
    def setup_multiple_gpu(self):
        with tf.variable(tf.get_variable_scope()):
            for i in xrange(self.num_gpus):
                with tf.device('/gpu:%i' % i):
                    ip = self.input_placeholder[i]
                    il = self.input_len_placeholder[i] 
                    pl = self.pred_placeholder[i]
                    loss, pred = self.init_ops(ip, il, pl)
                    feed = {
                        input_placeholder : context_batch,
                        input_len_placeholder: context_lens,
                        self.dropout_placeholder: dp,
                        pred_placeholder: pred_labels
                    }
                    tf.get_variable_scope().reuse_variables()
        summary = tf.summary.merge_all() 

    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None):
        dp = p.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        total_steps = len(data[0]) // p.batch_size
        total_loss = 0
        accuracy = 0

        if not self.use_multiple_gpu:
            ct, ct_l, pr = data
            r = np.random.permutation(len(ct))
            ct, ct_l, pr = np.asarray(ct, dtype=np.float32), np.asarray(ct_l, dtype=np.float32), np.asarray(pr, dtype=np.float32)
            ct, ct_l, pr = ct[r], ct_l[r], pr[r]
            for step in range(total_steps):
                index = range(step * p.batch_size,
                            (step + 1) * p.batch_size)
                feed = {self.input_pen_placeholder: ct_l[index],
                        self.pred_placeholder: pr[index],
                        self.dropout_placeholder: dp,
                        self.iteration: num_epoch}
                
                pred_labels = pr[step * p.batch_size:(step + 1) * p.batch_size]
                
                loss, pred, summary, _ = session.run(
                    [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)
                total_loss += loss
                if train_writer is not None:
                    train_writer.add_summary(
                        summary, num_epoch * total_steps + step)
                
                accuracy += (np.sum(pred == pred_labels)) / p.batch_size
                debug(step, total_steps, total_loss)
        else:
            if train_op:
                context_batch, context_lens, pred_labels = self.batch_queue.dequeue()
            else:
                context_batch, context_lens, pred_labels = self.batch_queue_valid.dequeue()
            loss, acc, _ = session.run([loss, pred, train_op], feed)
            total_loss += loss
            accuracy += acc

        sys.stdout.write('\r')
        avg_acc = 0.
        if total_steps:
            avg_acc = accuracy / float(total_steps)
        return total_loss, avg_acc
