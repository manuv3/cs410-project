import logging
from gensim.models import LdaModel
import corpus as corpus
from pprint import pprint
import os
from gensim.models.coherencemodel import CoherenceModel
from matplotlib import pyplot

_model_path = os.path.abspath('../../tmp/lda')

num_topics = 20

def build_lda(path = None, ntop = num_topics, save_model = True):
	if path:
		corpus.build_corpus(path)

	my_corpus = corpus.get_prebuilt_corpus()
	my_dictionary = corpus.get_prebuilt_dictionary()
	try:
		temp = my_dictionary[0]  # This is only to "load" the dictionary.
	except Exception as e:
		print(e)

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

	if save_model:
		lda.save(_model_path)

	return lda


def get_prebuilt_model():
	return LdaModel.load(_model_path)


for ntop in range(10, 26):
	cm_11 = CoherenceModel(model=build_lda(ntop = 11, save_model = False), corpus=corpus.get_prebuilt_corpus(), coherence='u_mass')
	cm_18 = CoherenceModel(model=build_lda(ntop = 18, save_model = False), corpus=corpus.get_prebuilt_corpus(), coherence='u_mass')


#pyplot.plot(range(10, 26), coherence)
#pyplot.show()


lda_11 = build_lda(ntop = 11)
top_topics = lda_11.top_topics(corpus.get_prebuilt_corpus())
# # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / 11
print('Average topic coherence: %.4f.' % avg_topic_coherence)

pprint(top_topics)
