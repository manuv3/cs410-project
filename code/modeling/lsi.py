import logging
from gensim.models import LsiModel
import corpus
from pprint import pprint
import os
from gensim.models.coherencemodel import CoherenceModel
from matplotlib import pyplot

_model_path = os.path.abspath('../../tmp/lsi')

num_topics = 20

def build_lsi(path = None, ntop = num_topics, save_model = True):
	if path:
		corpus.build_corpus(path)

	my_corpus = corpus.get_prebuilt_corpus()
	my_dictionary = corpus.get_prebuilt_dictionary()
	temp = my_dictionary[0]  # This is only to "load" the dictionary.
	id2word = my_dictionary.id2token

	# Set training parameters.


	lsi = LsiModel(
	    corpus = my_corpus,
	    id2word = id2word,
	    chunksize = 2000,
	    num_topics = ntop,
	    onepass=False
	)

	if save_model:
		lsi.save(_model_path)

	return lsi


def get_prebuilt_model():
	return LsiModel.load(_model_path)


build_lsi(ntop=11, save_model=True)