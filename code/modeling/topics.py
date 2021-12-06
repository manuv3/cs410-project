import lda
import corpus
import os
import logging
from pprint import pprint
import numpy as np
from tabulate import tabulate
from gensim.models.coherencemodel import CoherenceModel
from matplotlib import pyplot
from nltk import FreqDist
from nltk.corpus import brown, reuters
from math import log
import csv																																																										

class LdaBasedModel:

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
	_relevance_parameter = 0.4

	_num_terms_per_topic_before_processing = 100

	_num_terms_per_topic_after_processing = 20

	_background_word_max_frequency = 30

	_topic_min_threshold_per_doc = 0.3
	#****************************************************

	_terms_per_topic = []

	_docs = {}

	_topics_identifier = {}

	def __init__(self, build_model = False, topic_count = 16):
		if build_model:
			self._model = lda.build_lda(ntop = topic_count)
		else:
			self._model = lda.get_prebuilt_model()

		# Background language models
		_brown_text = brown.words()
		_brown_fdist = FreqDist(w.lower() for w in _brown_text)
		_reuters_text = reuters.words()
		_reuters_fdist = FreqDist(w.lower() for w in _reuters_text)


		topic_matrix = self._model.get_topics()
		topic_size, vocab_size = topic_matrix.shape

		# Topics and contained terms, after filtering high frequency background words
		idx = 0
		for prob, topic in self._model.show_topics(num_topics = topic_size, formatted = False, num_words = self._num_terms_per_topic_before_processing):
			self._topics_identifier[idx] = 'topic_' + str(idx + 1)
			idx += 1
			words = []
			for word, wprob in topic:
				if (_brown_fdist.get(word, 0) < self._background_word_max_frequency and _reuters_fdist.get(word, 0) < self._background_word_max_frequency):
					p_w = self._my_dictionary.cfs[self._my_dictionary.token2id[word]]
					relevance = (self._relevance_parameter * log(wprob)) + ((1 - self._relevance_parameter) * log(wprob / p_w))
					words.append((word, relevance))
			self._terms_per_topic.append([word for word, relevance in sorted(words, key = lambda item: item[1], reverse = True)][0: self._num_terms_per_topic_after_processing])

		# Build terms per document
		file_names = [files for root, dirs, files in os.walk(corpus.get_raw_corpus_path())][0]
		idx = 0
		for doc in self._my_corpus:
			doc_name = file_names[idx]
			# Update doc names index
			self._docs[idx] = doc_name
			idx += 1

	def get_topics(self, term_count = 10):
		topics = []
		for idx, name in self._topics_identifier.items():
			topics.append(self.get_topic(idx, term_count))
		return topics

	def get_topic(self, topic_id, term_count = 10):
		topic = {}
		topic['id'] = topic_id
		topic['name'] = self._topics_identifier[topic_id]
		topic['terms'] = self._terms_per_topic[topic_id][0: term_count]
		return topic	

	def get_topics_for_doc(self, doc_id, topic_threshold = 0, term_count = 10):
		topics_in_doc = self._get_topics_for_doc_internal(doc_id, topic_threshold)
		doc_topics = {}
		doc_topics['id'] = doc_id
		doc_topics['name'] = self._docs[doc_id]
		doc_topics['topics'] = []
		for topic, prob in topics_in_doc:
			tmp_obj = {}
			tmp_obj['id'] = topic
			tmp_obj['name'] = self._topics_identifier[topic]
			tmp_obj['coverage'] = str(prob)
			tmp_obj['terms'] = self._get_terms_for_doc_topic(doc_id, topic, term_count)
			doc_topics['topics'].append(tmp_obj)
		return doc_topics

	def get_docs(self):
		docs = []
		for doc_id, doc_name in self._docs.items():
			doc = {}
			doc['id'] = doc_id
			doc['name'] = doc_name
			docs.append(doc)
		return docs

	def _get_topics_for_doc_internal(self, doc_id, topic_threshold):
		return [(topic, prob) for topic, prob in sorted(self._model[self._my_corpus[doc_id]], key = lambda item: item[1], reverse = True) if prob >= topic_threshold]	

	def _get_terms_for_doc_topic(self, doc_id, topic_id, term_count):
		doc_name = self._docs[doc_id]
		bow = self._my_dictionary.doc2bow([token for token in corpus._tokenize(os.path.join(corpus.get_raw_corpus_path(), doc_name))][0])
		tfidf = {}
		for term_id, score in self._my_tfidf_model[bow]:
			tfidf[term_id] = score
		terms_in_doc = [term for term in self._terms_per_topic[topic_id]]
		terms_in_doc = [term for term in sorted(terms_in_doc, key = lambda item: tfidf.get(self._my_dictionary.token2id[item], 0), reverse = True)][0: term_count]
		return terms_in_doc