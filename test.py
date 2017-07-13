from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import time
import argparse

from model import Model, Config
import properties as p
import utils


def main(model):
    global config
    print('Testing task: %s ' % config.task_id)
    # create model
    with tf.device('/%s' % p.device):
        print('==> initializing variables')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
    tconfig = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=tconfig) as session:
        session.run(init)
        print('==> restoring weights')
        saver.restore(session, 'weights/task%s.weights' % model.config.task_id)
        print('==> running model')
        test_loss, test_accuracy = model.run_epoch(session, model.valid, verbose=0)
        print("Test accuracy: %0.2f" % test_accuracy)


def init_config(task_id):
    global config, word2vec, model
    if config.word2vec_init:
        if not word2vec:
            word2vec = utils.load_glove()
    else:
        word2vec = {}
    config.batch_size = 10
    config.strong_supervision = False
    config.train_mode = False
    config.task_id = task_id
    if config.reset:
        tf.reset_default_graph()
    if model is None:
        model = Model(config, word2vec)
    else:
        model.config = config
        model.init_global()
    config.reset = True
    main(model)


config = Config()

model = None
word2vec = None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_id",
                        help="specify babi task 1-20 (default=1)")
    args = parser.parse_args()
    
    init_config(args.task_id)