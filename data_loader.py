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
