import os
import sys
import math
import json
from string import punctuation
import nltk
import gensim

from data_loader import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, USE_CUDA

PROJECT_DIR = os.path.dirname(os.path.abspath('__file__'))
TRAIN_DIR = PROJECT_DIR + "/tqa_dataset/train"
TQA_train_json_path = TRAIN_DIR + "/tqa_v1_train.json"

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
STOPWORDS.update(set(punctuation))

def compute_sparse_vector_cosine(vec1, vec2):
    """
    Args:
        vec1, vec2: dict
    Returns:
        sim: float
    """
    denominator1 = 0.0
    denominator2 = 0.0
    numerator = 0.0
    for k, v in vec1.items():
        if k in vec2:
            numerator += (v * vec2[k])
        denominator1 += (v * v)
    for v in vec2.values():
        denominator2 += (v * v)
    sim = numerator / math.sqrt(denominator1 * denominator2)
    return sim

def get_question_answer_pair(nondiagram_dict, is_diagram_question=False):
    qa_pairs = []
    for qid, _ in nondiagram_dict.items():
        correct_answer_no = _['correctAnswer']['processedText']
        question = _['beingAsked']['processedText']
        if correct_answer_no not in _['answerChoices']:
            if  _['questionSubType'] == "true of false":  # get answer from other attr
                correct_answer = _['correctAnswer']['rawText']
            else:
                continue
        else:
            correct_answer = _["answerChoices"][correct_answer_no]['processedText']
        answers = {}
        for choice, answer_dict in _['answerChoices'].items():
            answers[choice] = answer_dict['processedText']
        if is_diagram_question:
            img_path = _['imagePath']
            qa_pairs.append((question, correct_answer, answers, img_path))
        else:
            qa_pairs.append((question, correct_answer, answers, ''))
    return qa_pairs

def do_sentence_tfidf_preprocess(sentence):
    """
    Return:
        sentence_clear_list: list of string
    """
    # word tokenize, lower case, stopwords filter
    sentence_clear_list = [_ for _ in nltk.word_tokenize(sentence.lower()) if _ not in STOPWORDS]
    return sentence_clear_list

def cal_idf():
    train_data = json.load(open(TQA_train_json_path))
    all_paragraph = []
    dictionary = gensim.corpora.Dictionary([[PAD_TOKEN], [SOS_TOKEN, EOS_TOKEN], [UNK_TOKEN]])  # assert <PAD> -> 0

    for lesson in train_data:
        # paragraphs
        for pid, topic in lesson['topics'].items():
            paragraph = topic['content']['text']
            paragraph_preprocessed = do_sentence_tfidf_preprocess(paragraph)
            all_paragraph.append(paragraph_preprocessed)
        # non-diagram question & answers
        for question, correct_answer, answers, img_path in get_question_answer_pair(lesson['questions']['nonDiagramQuestions']):
            dictionary.add_documents([do_sentence_tfidf_preprocess(question)])
        # diagram question & answers, only text part
        for question, correct_answer, answers, img_path in get_question_answer_pair(lesson['questions']['diagramQuestions']):
            dictionary.add_documents([do_sentence_tfidf_preprocess(question)])

    dictionary.add_documents(all_paragraph)
    corpus_model= [dictionary.doc2bow(p) for p in all_paragraph]
    tfidf_model = gensim.models.TfidfModel(corpus_model)
    return dictionary, tfidf_model

def sentence2tfidf(sentence, dictionary, tfidf_model):
    """
    Returns:
        tfidf_model[bow]: a list of tuple, e.g. [(word_id, word_tfidf), ()]
    """
    bow = dictionary.doc2bow(do_sentence_tfidf_preprocess(sentence))
    return tfidf_model[bow]

def generate_tqa_tuple():
    """
    Returns:
        tqa_tuple: a list of tuple, e.g. [(para, question, answer), ()]
    """
    empty_question_cnt = 0
    tqa_tuple = []
    tfidf_dictionary, tfidf_model = cal_idf()
    train_data = json.load(open(TQA_train_json_path))
    fwrite = open(TRAIN_DIR+'/tqa_v1_train.tqa_tuple', 'w', encoding='utf8')

    for lesson in train_data:
        paragraphs = []
        # paragraphs
        for pid, topic in lesson['topics'].items():
            paragraph = topic['content']['text']
            if not paragraph:
                #print("paragraph is none, topic_id=%s, topic_name=%s, lesson_name=%s" % (pid, topic['topicName'], lesson['lessonName']))
                continue
            paragraphs.append((paragraph, sentence2tfidf(paragraph, tfidf_dictionary, tfidf_model)))
        # non-diagram question & answers
        for question, correct_answer, answers, img_path in get_question_answer_pair(lesson['questions']['nonDiagramQuestions']) \
                + get_question_answer_pair(lesson['questions']['diagramQuestions'], is_diagram_question=True):
            question_tfidf = sentence2tfidf(question, tfidf_dictionary, tfidf_model)
            if not question_tfidf:
                empty_question_cnt += 1
                continue
            max_sim = -1.0
            for p, p_tfidf in paragraphs:
                sim = compute_sparse_vector_cosine(dict(p_tfidf), dict(question_tfidf))
                if sim >= max_sim:
                    most_relevant_para = p
                    max_sim = sim
            tqa_tuple.append((most_relevant_para, question, correct_answer))
            fwrite.write("%s\001%s\001%s\001%s\001%s\n" % (most_relevant_para, question, correct_answer, json.dumps(answers), img_path))
    print('empty_question_cnt: %d' % (empty_question_cnt))
    return tqa_tuple

if __name__ == '__main__':
    generate_tqa_tuple()
