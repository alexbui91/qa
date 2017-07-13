from __future__ import division
from __future__ import print_function

import sys

import os as os
import numpy as np

import properties as p
# can be sentence or word
input_mask_mode = "sentence"

# adapted from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/

#make sure this will not be reloaded
word2vec = None


def init_babi(fname):

    print("==> Loading test from %s" % fname)
    tasks = []
    task = None
    length = 0
    for i, line in enumerate(open(fname)):
        length +=1
        id = int(line[0:line.find(' ')])
        if id == 1:
            # met starting point of bos
            task = {"C": "", "Q": "", "A": "", "S": ""}
            counter = 0
            id_map = {}

        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ') + 1:]
        # if not a question
        if line.find('?') == -1:
            task["C"] += line
            id_map[id] = counter
            counter += 1
        else:
            idx = line.find('?')
            tmp = line[idx + 1:].split('\t')
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
            task["S"] = []
            for num in tmp[2].split():
                task["S"].append(id_map[int(num.strip())])
            tasks.append(task.copy())
    # return bos
    return tasks


def get_babi_raw(id, test_id, train_mode=True):
    babi_map = {
        "1": "qa1",
        "2": "qa2",
        "3": "qa3",
        "4": "qa4",
        "5": "qa5",
        "6": "qa6",
        "7": "qa7",
        "8": "qa8",
        "9": "qa9",
        "10": "qa10",
        "11": "qa11",
        "12": "qa12",
        "13": "qa13",
        "14": "qa14",
        "15": "qa15",
        "16": "qa16",
        "17": "qa17",
        "18": "qa18",
        "19": "qa19",
        "20": "qa20",
        "MCTest": "MCTest",
        "19changed": "19changed",
        "joint": "allshuffled",
        "sh1": "../shuffled/qa1",
        "sh2": "../shuffled/qa2",
        "sh3": "../shuffled/qa3",
        "sh4": "../shuffled/qa4",
        "sh5": "../shuffled/qa5",
        "sh6": "../shuffled/qa6",
        "sh7": "../shuffled/qa7",
        "sh8": "../shuffled/qa8",
        "sh9": "../shuffled/qa9",
        "sh10": "../shuffled/qa10",
        "sh11": "../shuffled/qa11",
        "sh12": "../shuffled/qa12",
        "sh13": "../shuffled/qa13",
        "sh14": "../shuffled/qa14",
        "sh15": "../shuffled/qa15",
        "sh16": "../shuffled/qa16",
        "sh17": "../shuffled/qa17",
        "sh18": "../shuffled/qa18",
        "sh19": "../shuffled/qa19",
        "sh20": "../shuffled/qa20",
    }
    if (test_id == ""):
        test_id = id
    babi_name = babi_map[id]
    babi_test_name = babi_map[test_id]
    babi_valid_raw = None
    if train_mode:
        print('==> get train inputs')
        babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(
            __file__)), 'data/%s/%s_train.txt' % (p.train_folder, babi_name)))
        if 'valid' in p.train_folder:
            babi_valid_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(
                __file__)), 'data/%s/%s_valid.txt' % (p.train_folder, babi_name)))
        babi_raw = (babi_train_raw, babi_valid_raw)
    else:
        print('==> get test inputs')
        babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(
            __file__)), 'data/%s/%s_test.txt' % (p.train_folder, babi_test_name)))
        babi_raw = (babi_test_raw, babi_valid_raw)
    return babi_raw


def create_vector(word, word2vec=None, word_vector_size=None, silent=True):
    if word2vec is None:
        # create zeros vector
        vector = np.zeros((word_vector_size,))
        word2vec[word] = vector
    else:
        # if the word is missing from Glove, create some fake vector and store in
        # glove!
        vector = np.random.uniform(0.0, 1.0, (word_vector_size,))
        word2vec[word] = vector
    if (not silent):
        print("utils.py::create_vector => %s is missing" % word)
    return vector


def process_word(word, word2vec=None, vocab=None, ivocab=None, word_vector_size=None, to_return="word2vec", silent=True):
    if word2vec is None:
        # mean create zeros vector
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab:
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word

    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")


# data_raw = bag of sentence
def process_input(data_raw, floatX, word2vec, vocab, ivocab, embed_size, split_sentences=False, answer_prediction="softmax"):
    questions = []
    inputs = []
    answers = []
    input_masks = []
    relevant_labels = []
    for x in data_raw:
        #inp_vector is list of indexes of word_vector in vocabs
        inp = x["C"].lower().split(' . ')
        inp = [w for w in inp if len(w) > 0]
        inp = [i.split() for i in inp]

        inp_vector = list()
        for s in inp:
            inp_vector.append(process_sentence(s, word2vec, vocab, ivocab, embed_size))
                
        inputs.append(inp_vector)
        
        q = x["Q"].lower().split(' ')
        q = [w for w in q if len(w) > 0]

        q_vector = process_sentence(q, word2vec, vocab, ivocab, embed_size)

        #use vstack => generate a vector l x 1 => mapping to embedding => l x d
        #create a vector of question by vstack => map to embedding tensorflow
        questions.append(np.vstack(q_vector).astype(floatX))
        #process answer label
        if answer_prediction == "softmax":
            answers.append(process_word(word=x["A"],
                                        word2vec=word2vec,
                                        vocab=vocab,
                                        ivocab=ivocab,
                                        word_vector_size=embed_size,
                                        to_return="index"))
        else: 
            ans = [process_word(word=a,
                                word2vec=word2vec,
                                vocab=vocab,
                                ivocab=ivocab,
                                word_vector_size=embed_size,
                                to_return="index") for a in x["A"].lower().split(',')]
            answers.append(np.vstack(ans).astype(floatX))
        # if not split_sentences:
        #     if input_mask_mode == 'word':
        #         input_masks.append(
        #             np.array([index for index, w in enumerate(inp)], dtype=np.int32))
        #     elif input_mask_mode == 'sentence':
        #         input_masks.append(
        #             np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32))
        #     else:
        #         raise Exception("invalid input_mask_mode")

        relevant_labels.append(x["S"])
    return inputs, questions, answers, input_masks, relevant_labels


def process_sentence(sent, word2vec, vocab, ivocab, embed_size):
    s_v = list()
    for w in sent:
        s_v.append(process_word(word=w,
                                word2vec=word2vec,
                                vocab=vocab,
                                ivocab=ivocab,
                                word_vector_size=embed_size,
                                to_return="index"))
        process_characters(w, word2vec, vocab, ivocab, embed_size)
    return s_v


def process_characters(word, word2vec, vocab, ivocab, embed_size):
    for c in word:
        process_word(word=c,
                word2vec=word2vec,
                vocab=vocab,
                ivocab=ivocab,
                word_vector_size=embed_size,
                to_return="index")


def get_lens(inputs, split_sentences=False):
    lens = np.zeros((len(inputs)), dtype=int)
    for i, t in enumerate(inputs):
        lens[i] = t.shape[0]
    return lens


def get_sentence_lens(inputs):
    lens = np.zeros((len(inputs)), dtype=int)
    sen_lens = []
    max_sen_lens = []
    for i, t in enumerate(inputs):
        sentence_lens = np.zeros((len(t)), dtype=int)
        for j, s in enumerate(t):
            sentence_lens[j] = len(s)
        lens[i] = len(t)
        sen_lens.append(sentence_lens)
        max_sen_lens.append(np.max(sentence_lens))
    return lens, sen_lens, max(max_sen_lens)


def pad_inputs(inputs, lens, max_len, mode="", sen_lens=None, max_sen_len=None):
    if mode == "mask":
        padded = [np.pad(inp, (0, max_len - lens[i]), 'constant',
                         constant_values=0) for i, inp in enumerate(inputs)]
        return np.vstack(padded)

    elif mode == "split_sentences":
        padded = np.zeros((len(inputs), max_len, max_sen_len))
        for i, inp in enumerate(inputs):
            padded_sentences = [np.pad(s, (0, max_sen_len - sen_lens[i][j]),
                                       'constant', constant_values=0) for j, s in enumerate(inp)]
            # trim array according to max allowed inputs
            if len(padded_sentences) > max_len:
                padded_sentences = padded_sentences[(
                    len(padded_sentences) - max_len):]
                lens[i] = max_len
            padded_sentences = np.vstack(padded_sentences)
            padded_sentences = np.pad(padded_sentences, ((
                0, max_len - lens[i]), (0, 0)), 'constant', constant_values=0)
            padded[i] = padded_sentences
        return padded
    # incase of process answer and question
    padded = [np.pad(np.squeeze(inp, axis=1), (0, max_len - lens[i]),
                     'constant', constant_values=0) for i, inp in enumerate(inputs)]
    return np.vstack(padded)


def create_embedding(word2vec, ivocab, embed_size):
    embedding = np.zeros((len(ivocab), embed_size))
    for i in range(len(ivocab)):
        word = ivocab[i]
        embedding[i] = word2vec[word]
    return embedding


def get_max_len(config, data, maxs):
    max_input_len, max_sen_len, max_q_len, max_answer_len = maxs
    inputs, questions, answers, input_masks, rel_labels = data
    #context len
    input_lens, sen_lens, max_sen_len = get_sentence_lens(inputs)

    max_input_len = max(np.max(input_lens), max_input_len)
    # question len
    q_lens = get_lens(questions)
    max_q_len = max(np.max(q_lens), max_q_len)
    # answer len
    a_lens = None
    if config.answer_prediction == "rnn":
        a_lens = get_lens(answers)
        max_answer_len = max(np.max(a_lens), max_answer_len)
    else: 
        # change from list to matrix style
        max_answer_len = 1
    
    return input_lens, max_input_len, sen_lens, max_sen_len, q_lens, max_q_len, a_lens, max_answer_len


def process_data(config, data, lens, maxs, train_mode=True):
    max_input_len, max_sen_len, max_q_len, max_answer_len = maxs
    max_mask_len = max_sen_len

    input_lens, sen_lens, q_lens, a_lens = lens
    inputs, questions, answers, input_masks, rel_labels = data
    ans_label = None
    
    # pad answer
    if config.answer_prediction == "rnn":
        answers = pad_inputs(answers, a_lens, max_answer_len)
        ans_label = np.copy(answers)
        if not train_mode:
            answers = np.zeros_like(answers, dtype=config.floatX)
    else:
        answers = np.stack(answers)
    
    # pad out arrays to max
    inputs = pad_inputs(inputs, input_lens, max_input_len,
                        "split_sentences", sen_lens, max_sen_len)
    input_masks = np.zeros(len(inputs))
    
    questions = pad_inputs(questions, q_lens, max_q_len)
    rel_labels = np.zeros((len(rel_labels), len(rel_labels[0])))

    for i, tt in enumerate(rel_labels):
        rel_labels[i] = np.array(tt, dtype=int)
    
    return inputs, input_masks, questions, answers, ans_label, rel_labels


def load_babi(config, word2vec, split_sentences=False, train_mode=True):
    vocab = {}
    ivocab = {}
    
    # set word at index zero to be end of sentence token so padding with zeros
    # is consistent
    process_word(word="<unk>",
                 word2vec=None,
                 vocab=vocab,
                 ivocab=ivocab,
                 word_vector_size=p.embed_size,
                 to_return="index")
    process_word(word="<eos>",
                 word2vec=word2vec,
                 vocab=vocab,
                 ivocab=ivocab,
                 word_vector_size=p.embed_size,
                 to_return="index")

    babi_raw, babi_dev_raw = get_babi_raw(config.task_id, config.babi_test_id, config.train_mode)
    data = process_input(babi_raw, config.floatX, word2vec,
                        vocab, ivocab, p.embed_size, split_sentences, config.answer_prediction)
    dev = None
    if babi_dev_raw is not None:
        dev = process_input(babi_dev_raw, config.floatX, word2vec,
                            vocab, ivocab, p.embed_size, split_sentences, config.answer_prediction)
   
    if config.word2vec_init:
        word_embedding = create_embedding(word2vec, ivocab, p.embed_size)
    else:
        word_embedding = np.random.uniform(-config.embedding_init,
                                           config.embedding_init, (len(ivocab), p.embed_size))
    # get length of train
    input_lens, max_input_len, sen_lens, max_sen_len, q_lens, max_q_len, a_lens, max_answer_len = get_max_len(config, data, (0, 0, 0, 0))
    maxs = (max_input_len, max_sen_len, max_q_len, max_answer_len)
    # get length of dev
    if babi_dev_raw is not None:
        input_lens_d, max_input_len, sen_lens_d, max_sen_len, q_lens_d, max_q_len, a_lens_d, max_answer_len = get_max_len(config, dev, maxs)
        maxs = (max_input_len, max_sen_len, max_q_len, max_answer_len)
        lens_d = (input_lens_d, sen_lens_d, q_lens_d, a_lens_d)
        inputs_d, input_masks_d, questions_d, answers_d, ans_label_d, rel_labels_d = process_data(config, dev, lens_d, maxs, train_mode)
    lens = (input_lens, sen_lens, q_lens, a_lens)
    inputs, input_masks, questions, answers, ans_label, rel_labels = process_data(config, data, lens, maxs, train_mode)
    max_mask_len = max_sen_len
    if config.train_mode:
        #separate part of train data to valid
        if a_lens is not None:
            # for rnn
            if dev is None:
                train = questions[:config.num_train], inputs[:config.num_train], q_lens[:config.num_train], \
                        input_lens[:config.num_train], input_masks[:config.num_train], answers[:config.num_train],  \
                        a_lens[:config.num_train], ans_label[:config.num_train], rel_labels[:config.num_train]
                answers_valid = np.zeros_like(answers[config.num_train:], dtype=config.floatX)                        
                valid = questions[config.num_train:], inputs[config.num_train:], q_lens[config.num_train:], \
                    input_lens[config.num_train:], input_masks[config.num_train:], answers_valid,  \
                    a_lens[config.num_train:], ans_label[config.num_train:], rel_labels[config.num_train:]
            else:
                train = questions, inputs, q_lens, input_lens, input_masks, answers, a_lens, ans_label, rel_labels
                answers_valid = np.zeros_like(answers_d, dtype=config.floatX)
                valid = questions_d, inputs_d, q_lens_d, input_lens_d, input_masks_d, answers_valid, a_lens_d, ans_label_d, rel_labels_d
        else:
            # for softmax
            if dev is None:
                train = questions[:config.num_train], inputs[:config.num_train], q_lens[:config.num_train], \
                        input_lens[:config.num_train], input_masks[:config.num_train], answers[:config.num_train],  \
                        rel_labels[:config.num_train]
                valid = questions[config.num_train:], inputs[config.num_train:], q_lens[config.num_train:], \
                        input_lens[config.num_train:], input_masks[config.num_train:], answers[config.num_train:],  \
                        rel_labels[config.num_train:]
            else: 
                train = questions, inputs, q_lens, input_lens, input_masks, answers, rel_labels
                valid = questions_d, inputs_d, q_lens_d, input_lens_d, input_masks_d, answers_d, rel_labels_d
        return train, valid, word_embedding, max_q_len, max_input_len, max_mask_len, max_answer_len, rel_labels.shape[1], len(vocab)
    else:
        if a_lens is not None:
            # for rnn
            test = questions, inputs, q_lens, input_lens, input_masks, answers, a_lens, ans_label, rel_labels
        else:
            test = questions, inputs, q_lens, input_lens, input_masks, answers, rel_labels
        return test, word_embedding, max_q_len, max_input_len, max_mask_len, max_answer_len, rel_labels.shape[1], len(vocab)