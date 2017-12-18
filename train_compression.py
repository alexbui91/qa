import utils as u
import tensorflow as tf
import numpy as np
import os
import sys
import time
import argparse

from embedding_compression import Compression

import properties as p
import utils


def stringify_code_words(code_words):
    tmp = ""
    for row in code_words:
        for col in row:
            tmp += "%i " % col
        tmp += "\n"
    return tmp


def main(restore=0):
    print("==> Load Word Embedding")
    word_embedding = utils.load_glove(use_index=True)
    # init word embedding
    model = Compression(np.array(word_embedding), M=p.book_size, K=p.code_size)
    # model.set_encoding()
    model.init_opts()
    # model.init_data_node()
    # create model
    tconfig = tf.ConfigProto(allow_soft_placement=True)
    non_words = utils.load_file(p.glove_path + '/non_words_50d.txt', False)
    validation_data = []
    for w in non_words:
        w_ = w.replace('\n', '').split(' ')
        validation_data.append(int(w_[-1]))
    training_data = utils.sub(range(len(word_embedding)), validation_data)
    with tf.device('/%s' % p.device):
        print('==> initializing variables')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    with tf.Session(config=tconfig) as session:
        sum_dir = 'summaries/compression/' + time.strftime("%Y-%m-%d %H %M")
        if not utils.check_file(sum_dir):
            os.makedirs(sum_dir)
        train_writer = tf.summary.FileWriter(sum_dir, session.graph)
        session.run(init)
        best_val_loss = float('inf')
        best_epoch = 0
        if restore:
            print('==> restoring weights')
            saver.restore(session, 'weights/compression.weights')

        print('start compressing')
        for epoch in xrange(p.total_iteration):
            print('Epoch {}'.format(epoch))
            start = time.time()

            train_loss, code_words = model.run_epoch(session, training_data, epoch, train_writer)
            print('Training loss: {}'.format(train_loss))
            # run validation after each 1000 iteration
            if (epoch % p.validation_checkpoint) == 0:
                valid_loss, code_words_valid = model.run_epoch(session, validation_data)
                print('Validation loss: {}'.format(valid_loss))
                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_epoch = epoch
                    print('Saving weights')
                    code_words_str = stringify_code_words(code_words)
                    code_words_valid_str = stringify_code_words(code_words_valid)
                    utils.save_file("weights/code_words_training.txt", code_words_str, use_pickle=False)
                    utils.save_file("weights/code_words_valid.txt", code_words_valid_str, use_pickle=False)
                    saver.save(session, 'weights/compression.weights')
            if (epoch - best_epoch) >= p.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--restore",
                        help="restore previously trained weights (default=false)", type=int)
    
    args = parser.parse_args()
    
    main(args.restore)