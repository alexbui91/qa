import utils as u
import tensorflow as tf
import numpy as np
import os
import sys
import time
import argparse

from model import Config
from model_squad import ModelSquad

import properties as p
import preload as pr
import utils
import train


def main(restore=False):
    word_embedding = None
    
    word_embedding_file = 'squad/word_embedding.pkl'
    if u.check_file(word_embedding_file):
        word_embedding = utils.load_file(word_embedding_file)
    else:
        print("==> Load vectors ...")
        vocabs = None
        ft_vc = 'squad/vocabs_fine_tuned.pkl'
        if u.check_file(ft_vc):
            vocabs = u.load_file(ft_vc)
        else:
            raise ValueError("Please check vocabs fine-tuned file")
        word2vec = utils.load_glove()
        word_embedding = np.zeros((len(vocabs), p.embed_size))
        for index, v in enumerate(vocabs):
            if v in word2vec:
                word_embedding[index] = word2vec[v]
            else:
                word_embedding[index] = pr.create_vector(v, word2vec, p.embed_size)
        del word2vec
        utils.save_file('squad/word_embedding.pkl', word_embedding)
        print("==> Done vectors ")
   
    # init word embedding
    contexts, contexts_len, questions, questions_len, answers, answers_len, start, end = utils.load_file('squad/doc_train_idx.pkl')

    data_len = int(np.floor(0.9 * len(contexts))) 
    
    train = contexts[:data_len], contexts_len[:data_len], questions[:data_len], \
            questions_len[:data_len], answers[:data_len], answers_len[:data_len], \
            start[:data_len], end[:data_len]
    dev = contexts[data_len:], contexts_len[data_len:], questions[data_len:], \
            questions_len[data_len:], answers[data_len:], answers_len[data_len:], \
            start[data_len:], end[data_len:]

    config = Config()
    config.strong_supervision = True
    model = ModelSquad(config)

    model.set_data(train, dev, word_embedding, np.shape(questions)[1], np.shape(contexts)[1], np.shape(answers)[1], len(word_embedding))
    model.set_encoding()
    model.init_ops()

     # tf.reset_default_graph()
    print('Start training DMN on squad')
    # model.init_data_node()
    best_overall_val_loss = float('inf')
    # create model
    tconfig = tf.ConfigProto(allow_soft_placement=True)

    with tf.device('/%s' % p.device):
        print('==> initializing variables')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    with tf.Session(config=tconfig) as session:

        sum_dir = 'summaries/train_squad/' + time.strftime("%Y-%m-%d %H %M")
        if not utils.check_file(sum_dir):
            os.makedirs(sum_dir)
        train_writer = tf.summary.FileWriter(sum_dir, session.graph)

        session.run(init)

        if restore:
            print('==> restoring weights')
            saver.restore(session, 'weights/squad.weights')

        print('==> starting test')
        start = time.time()

        valid_loss, valid_accuracy = model.run_epoch(session, model.valid)
        print('Validation loss: {}'.format(valid_loss))
        print('Validation accuracy: {}'.format(valid_accuracy))

        print('Total time: {}'.format(time.time() - start))


if __name__ == "__main__":
    # preload_vocabs()
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--restore",
                        help="restore previously trained weights (default=false)")
    
    args = parser.parse_args()
    
    main(args.restore)