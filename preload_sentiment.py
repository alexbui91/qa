import json
import utils as u
import numpy as np
import preload_squad as p
import properties as pr
# from pprint import pprint

def make_idx(vocabs, name):
    with open(name) as data_file:    
        max_input_len = 0
        data = data_file.readlines()
        default_del = '|999999|'
        results_y = list()
        sents = list()        
        for row in data:
            cols = row.split(default_del)
            c1 = int(cols[0])
            if c1 == 0:
                c1 = 0
            else:
                c1 = 1
            results_y.append(c1)
            sent = cols[-1].lower()
            words = p.get_idx(vocabs, sent)
            if max_input_len < len(words):
                max_input_len = len(words)
            sents.append(words)
             
    return pad_data(sents, results_y, max_input_len)


def pad_data(data, pred, max_input_len):
    contexts, contexts_len = list(), list()
    if max_input_len % pr.fixation != 0:
        max_input_len = (max_input_len / pr.fixation + 1) * pr.fixation

    for sent in data:
        contexts_len.append(len(sent))

        context = p.pad_row(sent, max_input_len)
        contexts.append(context)

    return contexts, contexts_len, pred


def load_file(vocabs, name):
    with open(name) as data_file:    
        data = data_file.readlines()
        default_del = '|999999|'
        for row in data:
            cols = row.split(default_del)
            c1 = int(cols[0])
            if c1 == 0:
                c1 = 0
            else:
                c1 = 1
            sent = cols[-1].lower()
            p.get_vocabs(vocabs, sent)


def preload_vocabs():
    vocabs = dict()
    load_file(vocabs, '/home/alex/Documents/nlp/code/data/training_twitter_full_.txt')
    load_file(vocabs, '/home/alex/Documents/nlp/code/data/test_twitter_.txt')
    u.save_file_utf8('%s/%s' % (folder, 'vocabs_text.txt'), p.convert_vocab_to_text(vocabs))

folder = 'sentiment'

if __name__ == "__main__":
    vocab_file = '%s/%s' % (folder, 'vocabs_fine_tuned.pkl')
    if not u.check_file(vocab_file):
        if not u.check_file('%s/vocabs_text.txt ' % folder):
            preload_vocabs()
        vocabs = p.get_vocabs_from_text(folder)
        u.save_file(vocab_file, vocabs)
    else: 
        vocabs = u.load_file(vocab_file)
        
    contexts, contexts_len, pred = make_idx(vocabs, '/home/alex/Documents/nlp/code/data/training_twitter_full.txt')
    u.save_file('sentiment/sentiment_chunking.pkl', (contexts, contexts_len, pred))