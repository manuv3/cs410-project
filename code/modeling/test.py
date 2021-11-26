import lda
import lsi
#import corpus
import os
import logging
from pprint import pprint
import numpy as nm
from tabulate import tabulate
from gensim.models.coherencemodel import CoherenceModel
from matplotlib import pyplot																																																																														


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARN)

_corpus_path = os.path.abspath('../../data/transcripts')

# Build model without building corpus again
#model = lda.build_lda(ntop=10)

# Build corpus and model
#model = lda.build_lda(_corpus_path)

# Get pre built model
model = lda.get_prebuilt_model()

# Get pre-built LSI model
#model = lsi.get_prebuilt_model()

# Get pre built corpus
my_corpus = corpus.get_prebuilt_corpus()

# Get pre-built dictionary
my_dictionary = corpus.get_prebuilt_dictionary()
#temp = my_dictionary[0]  # This is only to "load" the dictionary.
#id2word = my_dictionary.id2token


# top_topics = model.top_topics(corpus.get_prebuilt_corpus())
# # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
# avg_topic_coherence = sum([t[1] for t in top_topics]) / 11
# print('Average topic coherence: %.4f.' % avg_topic_coherence)

# topic_arr = []
# topic_ctr = 0
# for topic in top_topics:
# 	topic_ctr += 1
# 	words, coherence = topic
# 	tokens = []
# 	for word in words:
# 		prob, token = word
# 		tokens.append(token)
# 	topic_arr.append(['topic_' + str(topic_ctr), ' '.join(tokens)])


# print(tabulate(topic_arr, tablefmt='fancy_grid'))

# coherence_lda = []
# coherence_lsi = []

# for ntop in range(5, 16):
# 	cm_lda = CoherenceModel(model=lda.build_lda(ntop = ntop, save_model = False), corpus=my_corpus, coherence='u_mass')
# 	cm_lsi = CoherenceModel(model=lsi.build_lsi(ntop = ntop, save_model = False), corpus=my_corpus, coherence='u_mass')
# 	coherence_lda.append(cm_lda.get_coherence())
# 	coherence_lsi.append(cm_lsi.get_coherence())

# pyplot.plot(range(5, 16), coherence_lda, label = 'LDA')
# pyplot.plot(range(5, 16), coherence_lsi, label = 'LSI')
# pyplot.xlabel('Number of topics')
# pyplot.ylabel('Model coherence')
# pyplot.legend()
# pyplot.show()



topic_arr = []
topic_ctr = 0
for prob, topic in model.show_topics(formatted = False, num_words = 20):
	topic_ctr += 1
	words = []
	for word_tuple in topic:
		word, wprob = word_tuple
		words.append(word)
	topic_arr.append(['topic_' + str(topic_ctr), ' '.join(words)])

print(tabulate(topic_arr, tablefmt='fancy_grid'))


#print(model.show_topics(formatted = False))



# file_names = [files for root, dirs, files in os.walk(_corpus_path)][0]

# output = {}
# idx = 0
# for doc in my_corpus:
# 	doc_name = file_names[idx]
# 	idx += 1
# 	output[doc_name] = []
# 	for key, value in model[doc]:
# 		tokens = []
# 		for token, prob in model.show_topic(key):
# 			tokens.append(token)
# 		output[doc_name].append('+'.join(tokens))

# for key, value in output.items():
# 	val = '\n'.join(value)
# 	print('{} ==> {}'.format(key, val))
# 	print('================================')
