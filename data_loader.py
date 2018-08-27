import os
import gensim
import json
import nltk
import random
import torch
from torch.autograd import Variable

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
USE_CUDA = True

PROJECT_DIR = os.path.dirname(os.path.abspath('__file__'))
TRAIN_DIR = PROJECT_DIR + "/tqa_dataset/train"

def do_question_preprocess(question):
    return nltk.word_tokenize(question.lower())

# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(dictionary, sentence):
    return [dictionary.token2id[word] for word in do_question_preprocess(sentence)] + [dictionary.token2id[EOS_TOKEN]]

# Pad a with the PAD symbol
def pad_seq(seq, max_length, dictionary):
    seq += [dictionary.token2id[PAD_TOKEN] for i in range(max_length - len(seq))]
    return seq

def random_batch(raw_data, batch_size, dictionary, print_pair=False):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(raw_data)
        if print_pair:
            print(pair)
        input_seqs.append(indexes_from_sentence(dictionary, pair[0]))
        target_seqs.append(indexes_from_sentence(dictionary, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    
    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths), dictionary) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths), dictionary) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    
    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()
        
    return input_var, input_lengths, target_var, target_lengths

def load_tqa_data():
    tqa_tuple_path = TRAIN_DIR + '/tqa_v1_train.tqa_tuple'
    raw_data = []
    dictionary = gensim.corpora.Dictionary([[PAD_TOKEN], [SOS_TOKEN, EOS_TOKEN], [UNK_TOKEN]])  # assert <PAD> -> 0
    with open(tqa_tuple_path, 'r', encoding='utf8') as fopen:
        for line in fopen:
            line_list = line.strip().split('\001')
            para, question, cor_answer, answers_json, img_path = line_list
            raw_data.append((para, question, cor_answer))
            dictionary.add_documents([do_question_preprocess(para), do_question_preprocess(question), do_question_preprocess(cor_answer)])
    return raw_data, dictionary

def do_sort(seqs):
    """
    sort and store original index
    #sort - #see http://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
    """
    argsort = lambda _: sorted(range(len(_)), key=_.__getitem__, reverse=True)
    original_lens = [len(seq) for seq in seqs]
    sort_idx = argsort(original_lens)
    undo_sort_idx = argsort(sort_idx)[::-1]

    seqs_padded = [pad_seq(_, max(original_lens), dictionary) for _ in seqs]
    seqs_var = Variable(torch.LongTensor(seqs_padded)[sort_idx])
    seqs_lens = [original_lens[_] for _ in sort_idx]
    return seqs_var, seqs_lens, undo_sort_idx

def random_tqa_batch(raw_data, dictionary, batch_size):
    paras = []
    questions = []
    answers = []
    for i in range(batch_size):
        p, q, a = random.choice(raw_data)
        paras.append(indexes_from_sentence(dictionary, p))
        questions.append(indexes_from_sentence(dictionary, q))
        answers.append(indexes_from_sentence(dictionary, a))

    # store original index, then sort & pad
    para_var, para_lens, para_undo_sort_idx = do_sort(paras)
    question_var, question_lens, question_undo_sort_idx = do_sort(questions)
    
    return para_var, para_lens, para_undo_sort_idx, question_var, question_lens, question_undo_sort_idx, answers
