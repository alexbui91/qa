from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import time
import argparse
import os

from model import Model, Config

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--task_id",
                    help="specify babi task 1-20 (default=1)")
parser.add_argument("-r", "--restore",
                    help="restore previously trained weights (default=false)")
parser.add_argument("-s", "--strong_supervision",
                    help="use labelled supporting facts (default=false)")
parser.add_argument("-l", "--l2_loss", type=float,
                    default=0.001, help="specify l2 loss constant")
parser.add_argument("-n", "--num_runs", type=int,
                    help="specify the number of model runs")

args = parser.parse_args()

config = Config()

if args.task_id is not None:
    config.task_id = args.task_id

config.task_id = args.task_id if args.task_id is not None else str(1)
config.l2 = args.l2_loss if args.l2_loss is not None else 0.001
config.strong_supervision = args.strong_supervision if args.strong_supervision is not None else False
num_runs = args.num_runs if args.num_runs is not None else 1

print('Start training DMN on babi task', config.task_id)

best_overall_val_loss = float('inf')

# create model

config = Config()

model = Model(config)

for run in range(num_runs):

    print('Starting run', run)

    print('==> initializing variables')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:

        sum_dir = 'summaries/train/' + time.strftime("%Y-%m-%d %H %M")
        if not os.path.exists(sum_dir):
            os.makedirs(sum_dir)
        train_writer = tf.summary.FileWriter(sum_dir, session.graph)

        session.run(init)

        best_val_epoch = 0
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_accuracy = 0.0

        if args.restore:
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
