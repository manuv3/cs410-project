import logging
from gensim.models import LdaModel
import corpus
from pprint import pprint
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

_model_path = os.path.abspath('../../tmp/lda')

def build_lda(build_corpus):
	if build_corpus:
		corpus.build_corpus('../../data/transcripts')

	my_corpus = corpus.get_prebuilt_corpus()
	my_dictionary = corpus.get_prebuilt_dictionary()
	temp = my_dictionary[0]  # This is only to "load" the dictionary.
	id2word = my_dictionary.id2token

	# Set training parameters.
	num_topics = 50
	chunksize = 2000
	passes = 20
	iterations = 400
	eval_every = None  # Don't evaluate model perplexity, takes too much time.


	lda = LdaModel(
	    corpus = my_corpus,
	    id2word = id2word,
	    chunksize = chunksize,
	    alpha = 'auto',
	    eta = 'auto',
	    iterations = iterations,
	    num_topics = num_topics,
	    passes = passes,
	    eval_every = eval_every
	)

	#top_topics = lda.top_topics(my_corpus) #, num_words=20)
	# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
	#avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
	#print('Average topic coherence: %.4f.' % avg_topic_coherence)
	#pprint(top_topics)

	lda.save(_model_path)


def get_prebuilt_model():
	return LdaModel.load(_model_path)
