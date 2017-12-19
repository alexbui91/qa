import utils as u
import tensorflow as tf
import numpy as np
import os
import sys
import time
import argparse
import math

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


def get_data(vocabs=""):
    print("==> Load Word Embedding")
    word_embedding = utils.load_glove(use_index=True)

    validation_data = []
    training_data = []
    if not vocabs:
        non_words = utils.load_file(p.non_word, False)
        for w in non_words:
            w_ = w.replace('\n', '').split(' ')
            validation_data.append(int(w_[-1]))
        training_data = utils.sub(range(len(word_embedding)), validation_data)
    else:
        vocabs_set = utils.load_file(vocabs)
        print("vc", len(vocabs_set))
        training_data = [w for _, w in vocabs_set.iteritems()]
        tm = range(len(word_embedding))
        validation_data = list(utils.sub(set(tm), set(training_data)))
        length = int(math.ceil(len(training_data) * 1.0 / p.compression_batch_size)) * p.compression_batch_size - len(training_data)
        print('before', 'vd', len(validation_data), 'td', len(training_data))
        if length:
            add_on = np.random.choice(validation_data, length)
            training_data += add_on.tolist()
            validation_data = utils.sub(set(validation_data), set(add_on))
        print('vd', len(validation_data), 'td', len(training_data))
    # utils.save_file(p.glove_path, training_data)
    return word_embedding, training_data, validation_data


def main(book, word, restore=0, vocabs=""):
    # init word embedding
    # create model
    tconfig = tf.ConfigProto(allow_soft_placement=True)
    model = Compression(M=book, K=word)
    # model.set_encoding()
    model.init_opts()
    # model.init_data_node()
    word_embedding, training_data, validation_data = get_data(vocabs)
    model.set_embedding(np.array(word_embedding))
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
        length = int(len(training_data) * 0.2)
        best_code_words, best_code_words_valid = None, None
        for epoch in xrange(p.total_iteration):
            print('Epoch {}'.format(epoch))
            start = time.time()

            train_loss, code_words = model.run_epoch(session, training_data, epoch, train_writer)
            print('Training loss: {}'.format(train_loss))
            # run validation after each 1000 iteration
            if (epoch % p.validation_checkpoint) == 0:
                np.random.shuffle(validation_data)
                validation_data = validation_data[:length]
                valid_loss, code_words_valid = model.run_epoch(session, validation_data)
                print('Validation loss: {}'.format(valid_loss))
                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_epoch = epoch
                    print('Saving weights')
                    best_code_words = code_words
                    best_code_words_valid = code_words_valid
                    saver.save(session, 'weights/compression.weights')
            if (epoch - best_epoch) >= p.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))
        if best_code_words and best_code_words_valid:
            code_words_str = stringify_code_words(code_words)
            code_words_valid_str = stringify_code_words(code_words_valid)
            utils.save_file("weights/code_words_training.txt", code_words_str, use_pickle=False)
            utils.save_file("weights/code_words_valid.txt", code_words_valid_str, use_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--restore",
                        help="restore previously trained weights (default=false)", type=int, default=0)
    parser.add_argument("-t", "--train_data",
                        help="get trained words (default=false)", type=int)
    parser.add_argument("-b", "--book",
                        help="book size", type=int, default=64)
    parser.add_argument("-w", "--word",
                        help="book size", type=int, default=64)
    parser.add_argument("-v", "--vocabs",
                        help="link vocabs to train")
    args = parser.parse_args()
    if args.train_data:
        _, _, _ = get_data(args.vocabs)
    else:
        main(args.book, args.word, args.restore, args.vocabs)