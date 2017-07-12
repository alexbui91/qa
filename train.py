from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import time
import argparse
import os
import properties as p

from model import Model, Config


def main(model, num_runs, restore):
    print('Start training DMN on babi task', config.task_id)
    
    # model.init_data_node()
    best_overall_val_loss = float('inf')

    # create model
    tconfig = tf.ConfigProto(allow_soft_placement=True)

    for run in range(num_runs):

        print('Starting run', run)

        with tf.device('/%s' % p.device):
            print('==> initializing variables')
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

        with tf.Session(config=tconfig) as session:

            sum_dir = 'summaries/train/' + time.strftime("%Y-%m-%d %H %M")
            if not os.path.exists(sum_dir):
                os.makedirs(sum_dir)
            train_writer = tf.summary.FileWriter(sum_dir, session.graph)

            session.run(init)

            best_val_epoch = 0
            prev_epoch_loss = float('inf')
            best_val_loss = float('inf')
            best_val_accuracy = 0.0

            if restore:
                print('==> restoring weights')
                saver.restore(session, 'weights/task' +
                            str(model.config.task_id) + '.weights')

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
                        best_val_accuracy = valid_accuracy
                        saver.save(session, 'weights/task' +
                                str(model.config.task_id) + '.weights')

                # anneal
                if train_loss > prev_epoch_loss * model.config.anneal_threshold:
                    model.config.lr /= model.config.anneal_by
                    print('annealed lr to %f' % model.config.lr)

                prev_epoch_loss = train_loss

                if epoch - best_val_epoch > config.early_stopping:
                    break
                print('Total time: {}'.format(time.time() - start))

            print('Best validation accuracy:', best_val_accuracy)


def init_config(task_id, restore=None, strong_supervision=None, l2_loss=None, num_runs=None):
    global config
    config.l2 = l2_loss if l2_loss is not None else 0.001
    config.strong_supervision = strong_supervision if strong_supervision is not None else False
    num_runs = num_runs if num_runs is not None else '1'
    if task_id is not None:
        if ',' in task_id:
            tn = get_task_num(task_id, num_runs.split(','))
            run_model(tn, restore)
        elif '-' in task_id:
            st_en = task_id.split('-')
            if len(st_en) < 2:
                raise ValueError("task id should be the forms of x,y,z,t or x-y or x")
            st = st_en[0]
            en = st_en[-1]
            tn = get_task_num(np.arange(st, en), num_runs.split(','))
            run_model(tn, restore)
        else:
            config.task_id = task_id
            model = Model(config, word2vec)
            main(model, int(num_runs[0]), restore)


def run_model(tasks, restore):
    global config
    for task, num in tasks:
        config.task_id = task
        model = Model(config, word2vec)
        main(model, num, restore)


def get_task_num(tasks, nums):
    nums_len = len(nums)
    n = 1
    task_id = 1
    tn = list()
    for index, t in enumerate(tasks):
        task_id = t
        if index < nums_len:
            n = int(nums[index])
        else: 
            n = int(nums[-1])
        tn.append((task_id, n))
    return tn


def load_glove():
    word2vec = {}
    print("==> loading glove")
    if not p.glove_file:
        with open(("%s/glove.6B.%id.txt") % (p.glove_path, p.embed_size)) as f:
            for line in f:
                l = line.split()
                word2vec[l[0]] = map(float, l[1:])
    else:
        with open(("%s/%s") % (p.glove_path, p.glove_file)) as f:
            for line in f:
                l = line.split()
                word2vec[l[0]] = map(float, l[1:])

    print("==> glove is loaded")

    return word2vec


config = Config()

word2vec = None
if config.word2vec_init:
    if not word2vec:
        word2vec = load_glove()
else:
    word2vec = {}

# init parameters in terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_id",
                        help="default=1. Use , to perform multitask (ex. 1,2,3). Use - to perform multitask in range (1-5)")
    parser.add_argument("-r", "--restore",
                        help="restore previously trained weights (default=false)")
    parser.add_argument("-s", "--strong_supervision",
                        help="use labelled supporting facts (default=false)")
    parser.add_argument("-l", "--l2_loss", type=float,
                        default=0.001, help="specify l2 loss constant")
    parser.add_argument("-n", "--num_runs",
                        help="Fixed value x or use respectively with task_id with form 3,4,2,...")

    args = parser.parse_args()

    init_config(args.task_id, args.restore, args.strong_supervision, args.l2_loss, args.num_runs)


