#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import utils
import shutil
import random
import numpy as np
from nltk.tokenize import RegexpTokenizer
import properties as pr

prefix = "/home/alex/Documents/nlp/code/lstm_sentiment/data/aclImdb";
vocabs_path = "./sentiment/vocabs_imdb.txt"
vocabs_pkl = "./sentiment/vocabs_imdb.pkl"
embedding_path = "./sentiment/embedding_imdb.pkl"
test = "/test"
train = "/train"
neg = "/neg"
pos = "/pos"
un = "/unsup"


def get_vocabs(appended, vocabs, path):
    txt = "";
    files = os.listdir(path)
    print("path %s: %i" % (path, len(files)))
    regex = RegexpTokenizer(r'\w+')
    for file in files:
        with open(os.path.join(path, file)) as f:
            data = f.readlines()
            for sent in data:
                words = sent.lower().split(' ')
                for w in words:
                    # if w in vocabs:
                    #     txt += w + "\n"
                    #     appended[w] = 1
                    # else:
                    a = ''.join([c for c in w if c.isalnum()]);
                    # word is special case like :) :(
                    if not a:
                        if w in vocabs:
                            if w not in appended:
                                txt += w + "\n"
                                appended[w] = 1
                    else:
                        # word is normal case
                        if a in vocabs:
                            if a not in appended:
                                txt += a + "\n"
                                appended[a] = 1
                        else:
                            # word has special characters
                            words = regex.tokenize(w)
                            if len(words):
                                w_ = '_'.join(words)
                                if w_ in vocabs:
                                    if w_ not in appended:
                                        txt += w_ + "\n"
                                        appended[w_] = 1
                                else:
                                    for w_ in words:
                                        if w_ in vocabs:
                                            if w_ not in appended:
                                                txt += w_ + "\n"
                                                appended[w_] = 1
                                        else:
                                            w_ = ''.join([c for c in w_ if c.isalnum()]);
                                            if w_ in vocabs:
                                                txt += w_ + "\n"
                                                appended[w_] = 1
                    # if w not in vocabs:
                    #     a = ''.join([c for c in w if c.isalnum()]);
                    #     if a:
                    #         if a not in vocabs:
                    #             txt += a + "\n"
                    #             vocabs[a] = 1
                    #     else:
                    #         txt += w + "\n"
                    #         vocabs[w] = 1
    return txt;


def build_embedding():
    vocabs = utils.load_file(vocabs_path, use_pickle=False)
    word_embedding = utils.load_glove()
    embedding = [];
    for w in vocabs:
        w = w.replace('\n', '');
        if w in word_embedding:
            embedding.append(word_embedding[w])
    utils.save_file(embedding_path, embedding)


def build_index(vocabs, path, tag=None):
    output = [];
    files = os.listdir(path)
    print("path %s: %i" % (path, len(files)))
    regex = RegexpTokenizer(r'\w+')
    for file in files:
        with open(os.path.join(path, file)) as f:
            sent = []
            data = f.readlines()
            for s in data:
                words = s.lower().split(' ')
                for w in words:
                    a = ''.join([c for c in w if c.isalnum()]);
                    # word is special case like :) :(
                    if not a:
                        if w in vocabs:
                            sent.append(vocabs[w])
                    else:
                        # word is normal case
                        if a in vocabs:
                            sent.append(vocabs[a])
                        else:
                            # word has special characters
                            words = regex.tokenize(w)
                            if len(words):
                                w_ = '_'.join(words)
                                if w_ in vocabs:
                                    sent.append(vocabs[w_])
                                else:
                                    for w_ in words:
                                        if w_ in vocabs:
                                            sent.append(vocabs[w_])
                                        else:
                                            w_ = ''.join([c for c in w_ if c.isalnum()]);
                                            if w_ in vocabs:
                                                sent.append(vocabs[w_])
            if sent:
                output.append(sent)
    print("output: %i" % len(output));
    return output


if __name__ == "__main__":
    test_p = prefix + test
    train_p = prefix + train
    # txt = ""
    # vocabs_ar = utils.load_file(vocabs_path, use_pickle=False)
    # vocabs = dict()
    # appended = dict()
    # for x in vocabs_ar:
    #     x = x.replace("\n", "")
    #     vocabs[x] = 1
    # # vocabs = OrderedDict(sorted(vocabs.items(), key=lambda t: t[0]))
    # txt = get_vocabs(appended, vocabs, test_p + neg)
    # txt += get_vocabs(appended, vocabs, test_p + pos)
    # txt += get_vocabs(appended, vocabs, train_p + neg)
    # txt += get_vocabs(appended, vocabs, train_p + pos)
    # txt += get_vocabs(appended, vocabs, train_p + un)
    # utils.save_file("vocabs.txt", txt, use_pickle=False)

    # build_embedding()
    # vocabs = utils.load_file(vocabs_path, use_pickle=False);
    # a = {}
    # for e, w in enumerate(vocabs):
    #     w = w.replace('\n', '')
    #     a[w] = e
    # utils.save_file(vocabs_pkl, a)

    vocabs = utils.load_file(vocabs_pkl);
    test = []
    n = build_index(vocabs, test_p + neg)
    p = build_index(vocabs, test_p + pos)
    test.append(n)
    test.append(p)
    train = []
    n = build_index(vocabs, train_p + neg)
    p = build_index(vocabs, train_p + pos)
    train.append(n)
    train.append(p)
    un_s = build_index(vocabs, train_p + un)
    fo = {
        'test' : test,
        'train' : train,
        'un' : un_s
    }
    utils.save_file(pr.dataset_imdb, fo)