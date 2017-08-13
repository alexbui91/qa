import utils as u
import tensorflow as tf
import numpy as np
import os
import sys
import time
import argparse

from model import Config
from squad_skim import SquadSkim

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
            questions_len[:data_len], start[:data_len], end[:data_len]
    dev = contexts[data_len:], contexts_len[data_len:], questions[data_len:], \
            questions_len[data_len:], start[data_len:], end[data_len:]

    config = Config()
    # config.strong_supervision = True
    model = SquadSkim(config)

    model.set_data(train, dev, word_embedding, np.shape(contexts)[1], np.shape(questions)[1])
    # model.set_encoding()
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

        best_val_epoch = 0
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_accuracy = 0.0

        if restore:
            print('==> restoring weights')
            saver.restore(session, 'weights/squad.weights')

        print('==> starting training')
        for epoch in range(config.max_epochs):
            print('Epoch {}'.format(epoch))
            start = time.time()

            train_loss, train_accuracy = model.run_epoch(
                session, model.train, epoch, train_writer,
                train_op=model.train_step, train=True)
            valid_loss, valid_accuracy = model.run_epoch(session, model.valid)
            print('Training loss: {}'.format(train_loss))
            print('Validation loss: {}'.format(valid_loss))
            print('Training accuracy: {}'.format(train_accuracy))
            print('Validation accuracy: {}'.format(valid_accuracy))

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = epoch
                if best_val_loss < best_overall_val_loss:
                    print('Saving weights')
                    best_overall_val_loss = best_val_loss
                    saver.save(session, 'weights/squad.weights')
            # anneal
            if train_loss > prev_epoch_loss * model.config.anneal_threshold:
                model.config.lr /= model.config.anneal_by
                print('annealed lr to %f' % model.config.lr)
            
            if best_val_accuracy < valid_accuracy:
                best_val_accuracy = valid_accuracy
                
            prev_epoch_loss = train_loss

            if epoch - best_val_epoch > config.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))

        print('Best validation accuracy:', best_val_accuracy)


if __name__ == "__main__":
    # preload_vocabs()
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--restore",
                        help="restore previously trained weights (default=false)")
    
    args = parser.parse_args()
    
    main(args.restore)