import os
import math
import json
from string import punctuation
import nltk
import gensim

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

def do_sentence_preprocess(sentence):
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
    for lesson in train_data:
        for pid, topic in lesson['topics'].items():
            paragraph = topic['content']['text']
            paragraph_preprocessed = do_sentence_preprocess(paragraph)
            all_paragraph.append(paragraph_preprocessed)
    dictionary = gensim.corpora.Dictionary(all_paragraph)
    corpus_model= [dictionary.doc2bow(p) for p in all_paragraph]
    tfidf_model = gensim.models.TfidfModel(corpus_model)
    #print(tfidf_model)
    #first_para_tfidf = tfidf_model[corpus_model[0]]
    #print(first_para_tfidf)
    corpus_tfidf = tfidf_model[corpus_model]
    return dictionary, tfidf_model, corpus_tfidf

def sentence2tfidf(sentence, dictionary, tfidf_model):
    bow = dictionary.doc2bow(do_sentence_preprocess(sentence))
    return tfidf_model[bow]

if __name__ == '__main__':
    #sentence = 'the burning of fossil fuels contributes to global warming.'
    sentence = 'study of human effects on Earth'
    sentence_preprocessed = do_sentence_preprocess(sentence)
    print(sentence)
    print(sentence_preprocessed)

    dictionary, tfidf_model, corpus_tfidf = cal_idf()
    sentence_bow = dictionary.doc2bow(sentence_preprocessed)
    sentence_tfidf = tfidf_model[sentence_bow]
    sentence_tfidf_dict = dict(sentence_tfidf)
    print(sentence_bow)
    print(sentence_tfidf)

    geology_para = """Geology is the study of the solid Earth. Geologists study how rocks and minerals form. The way mountains rise up is part of geology. The way mountains erode away is another part. Geologists also study fossils and Earths history. There are many other branches of geology. There is so much to know about our home planet that most geologists become specialists in one area. For example, a mineralogist studies minerals, as seen in (Figure 1.11). Some volcanologists brave molten lava to study volcanoes. Seismologists monitor earthquakes worldwide to help protect people and property from harm (Figure 1.11). Paleontologists are interested in fossils and how ancient organisms lived. Scientists who compare the geology of other planets to Earth are planetary geologists. Some geologists study the Moon. Others look for petroleum. Still others specialize in studying soil. Some geologists can tell how old rocks are and determine how different rock layers formed. There is probably an expert in almost anything you can think of related to Earth! Geologists might study rivers and lakes, the underground water found between soil and rock particles, or even water that is frozen in glaciers. Earth scientists also need geographers who explore the features of Earths surface and work with cartographers, who make maps. Studying the layers of rock beneath the surface helps us to understand the history of planet Earth (Figure 1.12)."""
    oceanograpy_para = """Oceanography is the study of the oceans. The word oceanology might be more accurate, since ology is the study of. Graph is to write and refers to map making. But mapping the oceans is how oceanography started. More than 70% of Earths surface is covered with water. Almost all of that water is in the oceans. Scientists have visited the deepest parts of the ocean in submarines. Remote vehicles go where humans cant. Yet much of the ocean remains unexplored. Some people call the ocean the last frontier. Humans have had a big impact on the oceans. Populations of fish and other marine species have been overfished. Contaminants are polluting the waters. Global warming is melting the thick ice caps and warming the water. Warmer water expands and, along with water from the melting ice caps, causes sea levels to rise. There are many branches of oceanography. Physical oceanography is the study of water movement, like waves and ocean currents (Figure 1.13). Marine geology looks at rocks and structures in the ocean basins. Chemical oceanography studies the natural elements in ocean water. Marine biology looks at marine life."""
    climatology_para = """Meteorologists dont study meteors they study the atmosphere! The word meteor refers to things in the air. Meteorology includes the study of weather patterns, clouds, hurricanes, and tornadoes. Meteorology is very important. Using radars and satellites, meteorologists work to predict, or forecast, the weather (Figure 1.14). The atmosphere is a thin layer of gas that surrounds Earth. Climatologists study the atmosphere. These scientists work to understand the climate as it is now. They also study how climate will change in response to global warming. The atmosphere contains small amounts of carbon dioxide. Climatologists have found that humans are putting a lot of extra carbon dioxide into the atmosphere. This is mostly from burning fossil fuels. The extra carbon dioxide traps heat from the Sun. Trapped heat causes the atmosphere to heat up. We call this global warming (Figure 1.15)."""
    docs_bow = [ dictionary.doc2bow(do_sentence_preprocess(p)) for p in [geology_para, oceanograpy_para, climatology_para] ]
    docs_tfidf = tfidf_model[docs_bow]
    for d in docs_tfidf:
        sim = compute_sparse_vector_cosine(dict(sentence_tfidf), dict(d))
        print(sim)

    train_data = json.load(open(TQA_train_json_path))
    lesson1 = train_data[0]
    print("lesson1 contains %s paras" % (len(lesson1['topics'])))
    for pid, topic in lesson1['topics'].items():
        paragraph = topic['content']['text']
        para_tfidf = sentence2tfidf(paragraph, dictionary, tfidf_model)
        print(paragraph)
        print(compute_sparse_vector_cosine(sentence_tfidf_dict, dict(para_tfidf)))
