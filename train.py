from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import time
import argparse
import os
import properties as p

from model import Model, Config



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


def main(num_runs, restore):
    print('Start training DMN on babi task', config.task_id)

    best_overall_val_loss = float('inf')

    # create model
    tconfig = tf.ConfigProto(allow_soft_placement=True)
    with tf.device('/%s' % p.device):

        for run in range(num_runs):

            print('Starting run', run)

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
                    print('Vaildation accuracy: {}'.format(valid_accuracy))

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


def get_task_num(tasks, num_runs):
    tasks = tasks.split(',')
    nums = num_runs.split(',')
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


def init_config(task_id, restore=None, strong_supervision=None, l2_loss=None, num_runs=None):
    global config
    config.l2 = l2_loss if l2_loss is not None else 0.001
    config.strong_supervision = strong_supervision if strong_supervision is not None else False
    num_runs = num_runs if num_runs is not None else '1'
    if ',' not in task_id:
        model.config.task_id = task_id
        main(int(num_runs[0]), restore)
    else:
        tn = get_task_num(task_id, num_runs)
        for task, num in tn:
            model.config.task_id = task
            main(num, restore)


config = Config()
word2vec = None
if config.word2vec_init:
    if not word2vec:
        word2vec = load_glove()
else:
    word2vec = {}
model = Model(config, word2vec)


# init parameters in terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_id",
                        help="specify babi task 1-20 (default=1), use , to perform multitask")
    parser.add_argument("-r", "--restore",
                        help="restore previously trained weights (default=false)")
    parser.add_argument("-s", "--strong_supervision",
                        help="use labelled supporting facts (default=false)")
    parser.add_argument("-l", "--l2_loss", type=float,
                        default=0.001, help="specify l2 loss constant")
    parser.add_argument("-n", "--num_runs",
                        help="specify the number of model runs")

    args = parser.parse_args()

    config.l2 = args.l2_loss if args.l2_loss is not None else 0.001
    config.strong_supervision = args.strong_supervision if args.strong_supervision is not None else False
    num_runs = args.num_runs if args.num_runs is not None else '1'
    if args.task_id is not None:
        if ',' not in args.task_id:
            model.config.task_id = args.task_id
            main(int(num_runs[0]), args.restore)
        else:
            tn = get_task_num(args.task_id, args.num_runs)
            for task, num in tn:
                model.config.task_id = task
                main(num, restore)
            
    



