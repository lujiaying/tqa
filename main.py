import json
import math
import time
import random
import gensim
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim

from data_loader import do_question_preprocess, random_batch, indexes_from_sentence
from data_loader import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, USE_CUDA
from model import EncoderRNN, LuongAttnDecoderRNN, masked_cross_entropy

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def train(input_batches, input_lengths, target_batches, target_lengths,
          encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    batch_size = input_batches.size(1)

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_ID] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        target_lengths
    )
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0], ec, dc

def evaluate(input_sentence, dictionary, max_length=20):
    input_seq = indexes_from_sentence(dictionary, input_sentence)
    input_lengths = [len(input_seq)]
    input_seqs = [input_seq]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)

    if USE_CUDA:
        input_batches = input_batches.cuda()

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_ID]), volatile=True) # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder

    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        if ni == EOS_ID:
            decoded_words.append(EOS_TOKEN)
            break
        else:
            decoded_words.append(dictionary[ni.item()])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))
        if USE_CUDA:
            decoder_input = decoder_input.cuda()

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]

def evaluate_randomly():
    [input_sentence, target_sentence] = random.choice(raw_data)
    output_words, attentions = evaluate(input_sentence, dictionary)
    output_sentence = ' '.join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)

if __name__ == '__main__':
    # =============
    # Load Data
    # =============
    raw_data = []
    dictionary = gensim.corpora.Dictionary([[PAD_TOKEN], [SOS_TOKEN, EOS_TOKEN], [UNK_TOKEN]])  # assert <PAD> -> 0

    SOS_ID = dictionary.token2id[SOS_TOKEN]
    EOS_ID = dictionary.token2id[EOS_TOKEN]

    train_data = json.load(open('/media/drive/Jiaying/datasets/tqa_dataset/train/tqa_v1_train.json'))
    for lesson in train_data:
        for qid, _ in lesson['questions']['nonDiagramQuestions'].items():
            correct_answer_no = _['correctAnswer']['processedText']
            question = _['beingAsked']['processedText']
            if correct_answer_no not in _['answerChoices']:
                #print(lesson['lessonName'])
                #print(_)
                if  _['questionSubType'] == "True of False":  # get answer from other attr
                    correct_answer = _['correctAnswer']['rawText']
                else:
                    continue
            else:
                correct_answer = _["answerChoices"][correct_answer_no]['processedText']
            raw_data.append((question, correct_answer))
            dictionary.add_documents([do_question_preprocess(question), do_question_preprocess(correct_answer)])
    print('raw_data size', len(raw_data))

    # =============
    # Config
    # =============
    # Configure models
    attn_model = 'dot'
    hidden_size = 100
    n_layers = 2
    dropout = 0.1
    batch_size = 30

    # Configure training/optimization
    clip = 50.0
    teacher_forcing_ratio = 0.5
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_epochs = 3000
    epoch = 0
    #plot_every = 2
    print_every = 20
    evaluate_every = 60

    # Initialize models
    encoder = EncoderRNN(len(dictionary), hidden_size, n_layers, dropout=dropout)
    decoder = LuongAttnDecoderRNN(attn_model, hidden_size, len(dictionary), n_layers, dropout=dropout)

    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    criterion = nn.CrossEntropyLoss()

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    # =============
    # Begin!
    # =============
    print_loss_total = 0.0
    start = time.time()

    while epoch < n_epochs:
        epoch += 1

        # Get training data for this cycle
        input_batches, input_lengths, target_batches, target_lengths = random_batch(raw_data, batch_size, dictionary)

        # Run the train function
        loss, ec, dc = train(
            input_batches, input_lengths, target_batches, target_lengths,
            encoder, decoder,
            encoder_optimizer, decoder_optimizer, criterion
        )

        # Keep track of loss
        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs),
                                                   epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)

        if epoch % evaluate_every == 0:
            evaluate_randomly()
