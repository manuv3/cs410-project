import pickle
from collections import Counter, defaultdict
import corpus
import os
from nltk import FreqDist
from nltk.corpus import brown, reuters
from math import log
import word_to_vec
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from topics import LdaBasedModel
from word_to_vec import WordVecVectorizer, get_prebuilt_word2vec
import numpy as np

def load_data():
    data = corpus._tokenize(corpus.get_raw_corpus_path())
    data = [x for x in data]
    X_train = pd.DataFrame(data)
    return X_train


class CLusterModel(LdaBasedModel):

    _model = None
    # Get pre-built dictionary
    _my_dictionary = corpus.get_prebuilt_dictionary()
    _my_dictionary[0]  # This is only to "load" the dictionary.
    _id2token = _my_dictionary.id2token

    # Get pre built TF-IDF model
    _my_tfidf_model = corpus.get_tfidf_model()

    # Get pre built corpus
    _my_corpus = corpus.get_prebuilt_corpus()

    #************* Model Parameters *********************
    _topic_count = None
    #****************************************************

    _terms_per_topic = []

    _docs = {}

    _topics_identifier = {}

    def __init__(self, build_model = False, topic_count = 16, relevance_parameter = 0.4):
        self._model = pickle.load(open(os.path.join(word_to_vec._model_path, 'kmeans'), 'rb'))
        self._topic_count = topic_count
        self._relevance_parameter = 0.4

        # Background language models
        _brown_text = brown.words()
        _brown_fdist = FreqDist(w.lower() for w in _brown_text)
        _reuters_text = reuters.words()
        _reuters_fdist = FreqDist(w.lower() for w in _reuters_text)

        X_train = load_data()
        wtv_vect = WordVecVectorizer(get_prebuilt_word2vec())
        X_train_wtv = wtv_vect.transform(X_train)
        self._model.fit(X_train_wtv)
        self.X_train_wtv = X_train_wtv

        # Topics and contained terms, after filtering high frequency background words
        words_in_topic = defaultdict(Counter)
        for idx, topic in enumerate(self._model.predict(X_train_wtv)):
            self._topics_identifier[topic.item()] = 'topic_' + str(topic + 1)
            for word in X_train[:idx + 1].values[0]:
                if word:
                    if (_brown_fdist.get(word, 0) < self._background_word_max_frequency and _reuters_fdist.get(word, 0) < self._background_word_max_frequency):
                        words_in_topic[topic.item()][word] += 1


        for topic, counter in words_in_topic.items():
            total_count = sum(counter.values())
            words = []
            for word, count in counter.items():
                wprob = count / total_count
                p_w = self._my_dictionary.cfs[self._my_dictionary.token2id[word]]
                relevance = (self._relevance_parameter * log(wprob)) + ((1 - self._relevance_parameter) * log(wprob / p_w))
                words.append((word, relevance))
            self._terms_per_topic.append([word for word, relevance in sorted(words, key = lambda item: item[1], reverse = True)][0: self._num_terms_per_topic_after_processing])


        # Build terms per document
        file_names = open('../../data/file_order.txt', 'r').read().splitlines()
        idx = 0
        for doc in self._my_corpus:
            doc_name = file_names[idx]
            # Update doc names index
            self._docs[idx] = doc_name
            idx += 1


    def _get_topics_for_doc_internal(self, doc_id):
        arr = [(i, 0) for i in range(self._topic_count)]
        cur_topic = self._model.predict(self.X_train_wtv)[doc_id]
        arr[cur_topic.item()] = (cur_topic.item(), 1)
        return [(topic, prob) for topic, prob in
                sorted(arr, key=lambda item: item[1], reverse=True)]
