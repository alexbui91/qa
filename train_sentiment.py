import utils as u
import tensorflow as tf
import numpy as np
import os
import sys
import time
import argparse

from sentiment_skim import ModelSentiment

import properties as p
import preload as pr
import utils


def pad_row(data, maxlen):
    if len(data) < maxlen:
        data = np.pad(data, (0, maxlen - len(data)), 'constant', constant_values=0)
    elif len(data) > maxlen:
        data = data[:maxlen]
    return data


def pad_data(data, lens=400):
    contexts, contexts_len = list(), list()
    for sent in data:
        if len(sent) > lens:
            contexts_len.append(lens)
        else:
            contexts_len.append(len(sent))
        context = pad_row(sent, lens)
        contexts.append(context)
    return contexts, contexts_len


def prepare_data(data):
    output, output_l, pre = list(), list(), list()
    if len(data) > 1:
        t_n, t_n_l = pad_data(data[0])
        t_p, t_p_l = pad_data(data[1])
        t_p = zip(t_p, t_p_l, np.ones(len(t_p)))
        t_n = zip(t_n, t_n_l, np.zeros(len(t_n)))
        t = t_p + t_n
        np.random.shuffle(t)
        for x, y, z in t:
            output.append(x)
            output_l.append(y)
            pre.append(z)
    else:
        raise ValueError("dataset expect size 2 while %i" % len(data))
    return output, output_l, pre


def load_code_words(url):
    dat = utils.load_file(url, use_pickle=False)
    book = [];
    for d in dat:
        d = d.replace('\n', '')
        a = d.split(' ')
        a = [x for x in a if x]
        book.append(map(int, a))
    return book


def main(restore=False, b="", w="", prefix="", using_bidirection=False, forward_cell='', backward_cell=''):
    if prefix:
        prefix = prefix +  "_";
    word_embedding = None
    # if u.check_file(p.sentiment_embedding_path):
    #     word_embedding = utils.load_file(p.sentiment_embedding_path)
    # else:
    #     print("==> Load vectors ...")
    #     vocabs = None
    #     ft_vc = 'sentiment/vocabs_fine_tuned.pkl'
    #     if u.check_file(ft_vc):
    #         vocabs = u.load_file(ft_vc)
    #     else:
    #         raise ValueError("Please check vocabs fine-tuned file")
    #     word2vec = utils.load_glove()
    #     word_embedding = np.zeros((len(vocabs), p.embed_size))
    #     for index, v in enumerate(vocabs):
    #         if v in word2vec:
    #             word_embedding[index] = word2vec[v]
    #         else:
    #             word_embedding[index] = pr.create_vector(v, word2vec, p.embed_size)
    #     del word2vec
    #     utils.save_file('sentiment/word_embedding.pkl', word_embedding)
    #     print("==> Done vectors ")
   
    # init word embedding
    book, words = None, None
    using_compression = False
    if b and w:
        book = utils.load_file(b)
        words = load_code_words(w)
        using_compression = True
    elif u.check_file(p.sentiment_embedding_path):
        print("Loading dataset")
        word_embedding = utils.load_file(p.sentiment_embedding_path)
    else:
        raise ValueError("%s is not existed" % p.sentiment_embedding_path)
    
        # config.strong_supervision = True
    model = ModelSentiment(np.array(word_embedding), p.max_imdb_sent, using_compression, book, words, False, p.lr, False, using_bidirection, forward_cell, backward_cell)
    
    data = utils.load_file(p.dataset_imdb)
    train = data['train']
    test = data['test']
    train = prepare_data(train)
    test, test_l, pre_t = prepare_data(test)
    dev_l = int(len(test) * 0.2)
    dev = (test[:dev_l], test_l[:dev_l], pre_t[:dev_l])
    # test = (test[dev_l:], test_l[dev_l:], pre_t[dev_l:])
    test = (test, test_l, pre_t)
    
    model.set_data(train, dev)

    # model.init_data_node()
    with tf.device('/%s' % p.device):
        model.init_ops()
        print('==> initializing variables')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
    tconfig = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=tconfig) as session:

        sum_dir = 'summaries/train_sentiment/' + time.strftime("%Y-%m-%d %H %M")
        if not utils.check_file(sum_dir):
            os.makedirs(sum_dir)
        train_writer = tf.summary.FileWriter(sum_dir, session.graph)

        session.run(init)

        best_val_epoch = 0
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        best_overall_val_loss = float('inf')

        if restore:
            print('==> restoring weights')
            saver.restore(session, 'weights/%ssentiment.weights' % prefix)
        
        print('==> starting training')
        train_losses, train_accuracies = [], []
        val_losses, val_acces = [], []
        for epoch in xrange(p.total_iteration):
            print('Epoch {}'.format(epoch))
            start = time.time()

            train_loss, train_accuracy = model.run_epoch(
                session, model.train, epoch, train_writer,
                train_op=model.train_step, train=True)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            print('Training loss: {}'.format(train_loss))
            print('Training accuracy: {}'.format(train_accuracy))
            # anneal
            if train_loss > (prev_epoch_loss * p.anneal_threshold):
                p.lr /= p.anneal_by
                print('annealed lr to %f' % p.lr)
            prev_epoch_loss = train_loss
            # validation
            if epoch > p.skip_validation and (epoch % p.validation_checkpoint) == 0:
                valid_loss, valid_accuracy = model.run_epoch(session, model.valid)
                val_losses.append(valid_loss)
                val_acces.append(valid_accuracy)
                print('Validation loss: {}'.format(valid_loss))
                print('Validation accuracy: {}'.format(valid_accuracy))

                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_val_epoch = epoch
                    if best_val_loss < best_overall_val_loss:
                        print('Saving weights')
                        best_overall_val_loss = best_val_loss
                        saver.save(session, 'weights/%ssentiment.weights' % prefix)
            
                if best_val_accuracy < valid_accuracy:
                    best_val_accuracy = valid_accuracy

            if (epoch - best_val_epoch) > p.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))
        print('Best validation accuracy:', best_val_accuracy)
        print('=> Running test')
        print('==> Restoring best weights')
        saver.restore(session, 'weights/%ssentiment.weights' % prefix)
        _, test_accuracy = model.run_epoch(session, test)
        test_accuracy = test_accuracy * 100
        print('Test accuracy: %.5f' % test_accuracy)
        tmp = "Best validation accuracy: %.5f \n Test accuracy: %.5f" % (best_val_accuracy, test_accuracy)
        utils.save_file("%saccuracy.txt" % prefix, tmp, False)
        utils.save_file("logs/%slosses.pkl", {"train_loss": train_losses, "train_acc": train_accuracies, "valid_loss": val_losses, "valid_acc" : val_acces})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--restore", help="restore previously trained weights (default=false)", type=int)
    
    parser.add_argument("-b", "--book", help="code book url")
    parser.add_argument("-w", "--word", help="code words url")
    parser.add_argument("-bs", "--book_size", type=int, default=64)
    parser.add_argument("-ws", "--word_size", type=int, default=64)

    parser.add_argument("-p", "--prefix", help="prefix to save weighted files")
    parser.add_argument("-fw", "--forward_cell", default='basic')
    parser.add_argument("-bw", "--backward_cell", default='basic')
    parser.add_argument("-bd", "--bidirection", type=int)

    args = parser.parse_args()
    p.book_size = args.book_size
    p.code_size = args.word_size

    if not args.prefix:
        args.prefix = "%sx%s" % (args.book_size, args.word_size)
    main(args.restore, args.book, args.word, args.prefix, args.bidirection, args.forward_cell, args.backward_cell)
    