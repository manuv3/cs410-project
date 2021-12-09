import logging
from gensim.models import LdaModel
import corpus
from pprint import pprint
import os
from gensim.models.coherencemodel import CoherenceModel
from matplotlib import pyplot

_model_path = os.path.abspath('../../model/lda')

num_topics = 20

def build_lda(build_corpus = False, ntop = num_topics, save_model = False, model_path = None):
	print('Building lda model')
	if build_corpus:
		corpus.build_corpus()

	my_corpus = corpus.get_prebuilt_corpus()
	my_dictionary = corpus.get_prebuilt_dictionary()
	temp = my_dictionary[0]  # This is only to "load" the dictionary.
	id2word = my_dictionary.id2token

	# Set training parameters.

	chunksize = 2000
	passes = 40
	iterations = 800
	eval_every = None  # Don't evaluate model perplexity, takes too much time.


	lda = LdaModel(
	    corpus = my_corpus,
	    id2word = id2word,
	    chunksize = chunksize,
	    alpha = 'auto',
	    eta = 'auto',
	    iterations = iterations,
	    num_topics = ntop,
	    passes = passes,
	    eval_every = eval_every
	)

	if save_model and model_path != None:
		lda.save(os.path.abspath('../../model/' + model_path))

	return lda


def get_prebuilt_model(model_path):
	return LdaModel.load(os.path.abspath('../../model/' + model_path))