import json
import utils as u
from nltk.tokenize import RegexpTokenizer
import numpy as np
import properties as p

import argparse
# from pprint import pprint


def get_structure():
    with open('squad/dev-v1.1.json') as data_file:    
        data = json.load(data_file)
        context = data['data']
        print(context[0]['paragraphs'][0]['qas'][2]['question'])
        print(context[0]['paragraphs'][0]['qas'][2]['answers'])


def make_idx(vocabs, name, train_mode=True, limit=None):
    with open('%s/%s' % (folder, name)) as data_file:    
        max_input_len = max_q_len = max_ans_len = 0
        data = json.load(data_file)
        context = data['data']
        doc_idx = list()
        index = 0
        for pas in context:
            if not limit is None and index >= limit:
                break
            #one context has many paragraphs
            paragraphs = pas['paragraphs']
            # paras_idx = list()
            for para in paragraphs:
                if not limit is None and index >= limit:
                    break
                para_idx = get_idx(vocabs, para['context'])
                if max_input_len < len(para_idx):
                    max_input_len = len(para_idx)
                qas = para['qas']
                # qa_idx = list()
                for qa in qas:
                    if not limit is None and index >= limit:
                        break
                    #each paragraph has many questions and answers
                    quest_idx = get_idx(vocabs, qa['question'])
                    if max_q_len < len(quest_idx):
                        max_q_len = len(quest_idx)
                    answers = qa['answers']
                    answers_idx = list()
                    rel_idx = list()
                    # print("question", question)
                    for ans in answers:
                        # in dev we can have many answers with single questions
                        text = ans['text']
                        ans_idx = get_idx(vocabs, text)
                        rel = int(ans['answer_start'])
                        start = get_ans_pos(para['context'], rel)
                        if start != -1:
                            if max_ans_len < len(ans_idx):
                                max_ans_len = len(ans_idx)
                            end = start + len(ans_idx) - 1
                            answers_idx.append((ans_idx, start, end)),
                            rel_idx.append(rel)
                    if answers_idx:
                        doc_idx.append({
                            "context": para_idx,
                            "answers": answers_idx,
                            'rel': rel_idx,
                            "question": quest_idx,
                            "id" : qa['id']
                        })
                        index += 1
                # paras_idx.append({
                #     'context': para_idx,
                #     'qas': qa_idx
                # })
            # doc_idx.append(paras_idx)
        max_input_len = get_length_fixation(max_input_len)
        max_q_len = get_length_fixation(max_q_len)
        max_ans_len = get_length_fixation(max_ans_len)

    return pad_data(doc_idx, (max_input_len, max_q_len, max_ans_len), train_mode)


def get_ans_pos(context, rel):
    start = -1
    met_space = 0
    rel += 1
    for index, w in enumerate(context):
        if w == ' ':
            met_space += 1
        if index == rel:
            start = met_space
            break
        # if met < ans_len and ans_idx[met] == w:
        #     if met == 0:
        #         start = index
        #     elif met == ans_len:
        #         end = index
        #     met += 1
        # else:
        #     met = 0
        #     start = end = -1
                
    return start


def get_sentences_idx(vocabs, context):
    sentences = context.split('. ')
    idx = list()
    for sent in sentences:
        idx.append(get_idx(vocabs, sent))
    return idx


def get_idx(vocabs, sent):
    sent_idx = list()
    regex = RegexpTokenizer(r'\w+')
    words = sent.lower().split(' ')
    for w in words:
        w_ = regex.tokenize(w)
        w_ = '_'.join(w_)
        if w_ in vocabs:
            sent_idx.append(vocabs[w_])
        elif not w_ is '':
            sent_idx.append(0)
    return sent_idx


def get_length_fixation(length):
    if length % p.fixation != 0:
        length = (length / p.fixation + 1) * p.fixation
    return length

def pad_data(data, lens, train_mode=True):
    max_input_len, max_q_len, max_a_len = lens
    contexts, contexts_len, questions, questions_len, \
    answers, answers_len, start, end = list(), list(), list(), list(), list(), list(), list(), list()
    for para in data:
        context = para['context']
        contexts_len.append(len(context))

        context = pad_row(context, max_input_len)
        contexts.append(context)
        # process q and a
        question = para['question']
        questions_len.append(len(question))
        
        question = pad_row(question, max_q_len)
        questions.append(question)

        ans = para['answers']
        if train_mode:
            a_, s_, e_ = ans[0]
            answers_len.append(len(a_))
            a_ = pad_row(a_, max_a_len)
            answers.append(a_)
            start.append(s_)
            end.append(e_)
        else: 
            a_, a_l, s_, e_ = list(), list(), list(), list()
            
            for a_i, s_i, e_i in ans:
                a_l.append(len(a_i))
                a_.append(pad_row(a_i, max_a_len))
                s_.append(s_i)
                e_.append(e_i)
            
            answers.append(a_)
            answers_len.append(a_l)
            start.append(s_)
            end.append(e_)

    return contexts, contexts_len, questions, questions_len, answers, answers_len, start, end


def pad_row(data, maxlen):
    if len(data) < maxlen:
        data = np.pad(data, (0, maxlen - len(data)), 'constant', constant_values=0)
    elif len(data) > maxlen:
        raise ValueError("Please check max length")
    return data

def preload_vocabs():
    vocabs = dict()
    load_file(vocabs, 'train-v1.1.json')
    load_file(vocabs, 'dev-v1.1.json')
    u.save_file('%s/%s' % (folder, 'vocabs.pkl'), vocabs)
    u.save_file('%s/%s' % (folder, 'vocabs_text.txt'), convert_vocab_to_text(vocabs), use_pickle=False)


def load_file(vocabs, name):
    with open('%s/%s' % (folder, name)) as data_file:    
        data = json.load(data_file)
        context = data['data']
        for pas in context:
            #one context has many paragraphs
            paragraphs = pas['paragraphs']
            for para in paragraphs:
                ctx = para['context']
                get_vocabs(vocabs, ctx)
                qas = para['qas']
                for qa in qas:
                    #each paragraph has many questions and answers
                    question = qa['question']
                    get_vocabs(vocabs, question)
                    answers = qa['answers']
                    # print("question", question)
                    for ans in answers:
                        # in dev we can have many answers with single questions
                        text = ans['text']
                        get_vocabs(vocabs, text)


def get_vocabs(vocab, sent):
    words = sent.lower().split(' ')
    for w in words:
        if not w in vocab:
            vocab[w] = len(vocab)


def convert_vocab_to_text(vocabs):
    vocab_str = ""
    length = len(vocabs)
    i = 0
    vocab_idx = dict()
    vocab_lst = list()
    idx_file = '%s/%s' % (folder, 'vocabs_idx.pkl')
    if u.check_file(idx_file):
        vocab_idx = u.load_file(idx_file)
    else:
        for key, value in vocabs.iteritems():
            vocab_idx[value] = key
        u.save_file(idx_file, vocab_idx)
    lst_file = '%s/%s' % (folder, 'vocabs_list.pkl')
    if u.check_file(lst_file):
        vocab_lst = u.load_file(lst_file)
    else:
        for key in sorted(vocab_idx.iterkeys()):
            vocab_lst.append(vocab_idx[key])
        u.save_file(lst_file, vocab_lst)
    regex = RegexpTokenizer(r'\w+')
    for w in vocab_lst:
        words = regex.tokenize(w)
        if len(words) != 0:
            w_ = '_'.join(words)
            i += 1
            if i % 10000 == 0:
                print('Processed %i' % i)
                # break
            if i == length:
                vocab_str += '%s' % w_
            else: 
                vocab_str += '%s\n' % w_
    return vocab_str



def get_vocabs_from_text(folder):
    vocabs_text = u.load_file_utf8('%s/%s' % (folder, 'vocabs_text.txt'))
    vocabs = dict()
    vocabs['<unk>'] = 0
    for v in vocabs_text:
        v_ = v.replace('\n', '')
        if (not v_ is '') and (v_ not in vocabs):
            vocabs[v_] = len(vocabs)
    return vocabs


folder = 'squad'
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--limit", type=int,
                        help="Maximum questions load")
    
    args = parser.parse_args()
    
    # preload_vocabs()
    vocabs = dict()
    ft_vc = '%s/%s' % (folder, 'vocabs_fine_tuned.pkl')
    if u.check_file(ft_vc):
        vocabs = u.load_file(ft_vc)
    else:
        vocabs = get_vocabs_from_text(folder)
        u.save_file(ft_vc, vocabs)
    contexts, contexts_len, questions, questions_len, answers, answers_len, start, end = make_idx(vocabs, 'train-v1.1.json', limit=args.limit)
    contexts_d, contexts_len_d, questions_d, questions_len_d, answers_d, answers_len_d, start_d, end_d = make_idx(vocabs, 'dev-v1.1.json', False, limit=args.limit)
    u.save_file('%s/%s' % (folder, 'doc_train_idx.pkl'), (contexts, contexts_len, questions, questions_len, answers, answers_len, start, end))
    u.save_file('%s/%s' % (folder, 'doc_dev_idx.pkl'), (contexts_d, contexts_len_d, questions_d, questions_len_d, answers_d, answers_len_d, start_d, end_d ))
    # v_txt = convert_vocab_to_text(vocabs)
    # u.save_file_utf8('%s/%s' % (folder, 'vocabs_text.txt'), v_txt)