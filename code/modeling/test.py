import lda
import lsi
import corpus
import topics
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



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARN)

_corpus_path = os.path.abspath('../../data/transcripts')

# Build model without building corpus again
#model = lda.build_lda(ntop=15)

# Build corpus and model
#model = lda.build_lda(build_corpus= True, ntop = 20)

# Get pre built model
model = lda.get_prebuilt_model()

# Get pre-built LSI model
#model = lsi.get_prebuilt_model()

# Get pre built corpus
my_corpus = corpus.get_prebuilt_corpus()

# Get pre-built phrases
my_phrases = corpus.get_prebuilt_phrases()

# Get pre-built dictionary
my_dictionary = corpus.get_prebuilt_dictionary()
#temp = my_dictionary[0]  # This is only to "load" the dictionary.
#id2word = my_dictionary.id2token

my_tfidf_model = corpus.get_tfidf_model()


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


brown_text = brown.words()
brown_fdist = FreqDist(w.lower() for w in brown_text)

reuters_text = reuters.words()
reuters_fdist = FreqDist(w.lower() for w in brown_text)


#topics_per_document = {}

#Relevant phrases per document after filtering high frequency background words.
# phrase_dict = {}
# phrases_per_document = {}
# for root, dirs, files in os.walk(_corpus_path):
# 	for name in files:
# 		doc_level_phrases = {}
# 		sentences = corpus.get_sentences(os.path.join(root, name))
# 		for phrase, score in my_phrases.find_phrases(sentences).items():
# 			if (phrase in phrase_dict.keys()):
# 				continue
# 			words = phrase.split('_')
# 			phrase_fdist = 0
# 			for word in words:
# 				if (reuters_fdist.get(word, 0) > brown_fdist.get(word, 0)):
# 					phrase_fdist = phrase_fdist + reuters_fdist.get(word, 0)
# 				else:	
# 					phrase_fdist = phrase_fdist + brown_fdist.get(word, 0)
# 			if phrase_fdist < 350:
# 				phrase_dict[phrase] = score
# 				doc_level_phrases[phrase] = score
# 		phrases_per_document[name] = [key for key, val in sorted(doc_level_phrases.items(), key = lambda item: item[1], reverse = True)]
#pprint(phrases_per_document)
# with open('doc_phrases.csv', 'w') as csv_file:
#     wr = csv.writer(csv_file, delimiter=',')
#     for name, phrases in phrases_per_document.items():
#     	wr.writerow([name, ' '.join(phrases)])


#Relevant phrases per topic
# filter_size = 15
topic_matrix = model.get_topics()
topic_size, vocab_size = topic_matrix.shape
# phrases_per_topic = []
# for i in range(topic_size):
# 	zero_order_relevance_scores = {}
# 	for phrase, score in phrase_dict.items():
# 		zero_order_relevance_score = 0
# 		words = phrase.split('_')
# 		for word in words:
# 			token_id = my_dictionary.token2id[word]
# 			if token_id:
# 				prob_w_theta = topic_matrix[i, token_id]
# 				prob_w = my_dictionary.cfs[token_id]/ vocab_size
# 				zero_order_relevance_score += log(prob_w_theta / prob_w)
# 		zero_order_relevance_scores[phrase] = zero_order_relevance_score
# 	phrases_per_topic.append(sorted(zero_order_relevance_scores.items(), key = lambda item: item[1], reverse = True)[:filter_size])

# for phrases_in_topic in phrases_per_topic:
# 	pprint([key for key, val in phrases_in_topic])


# Topics and contained terms, after filtering high frequency background words
terms_per_topic = [topic_terms[0:10] for topic_terms in topics.get_topics()]
#print(terms_per_topic)
#pprint(terms_per_topic)

# with open('topics.csv', 'w') as csv_file:
#     wr = csv.writer(csv_file, delimiter=',')
#     for i in range(topic_size):
#     	wr.writerow(['topic_' + str(i), ' '.join(terms_per_topic[i])])


print(tabulate(terms_per_topic, tablefmt='fancy_grid', showindex = 'always'))


#Top 5 relevant phrases per document for top two topics
# file_names = [files for root, dirs, files in os.walk(_corpus_path)][0]
# output = {}
# idx = 0
# temp = my_dictionary[0]  # This is only to "load" the dictionary.
# id2token = my_dictionary.id2token
# for doc in my_corpus:
# 	doc_name = file_names[idx]
# 	idx += 1
# 	doc = my_dictionary.doc2bow([token for token in corpus._tokenize(os.path.join(_corpus_path, doc_name))][0])
# 	tfidf = {} 
# 	for term_id, score in my_tfidf_model[doc]:
# 		tfidf[term_id] = score
# 	terms = [term for topic, prob in model[doc] if prob > 0.3 for term in terms_per_topic[topic]]
# 	terms = [term for term in sorted(terms, key = lambda item: tfidf.get(my_dictionary.token2id[item], 0), reverse = True)]
# 	output[doc_name] = terms
# 	#output[doc_name] = [id2token[term_id] for term_id, score in sorted(my_tfidf_model[doc], key = lambda item: item[1], reverse = True)][0:10]
# pprint(output)


# with open('output2.csv', 'w') as csv_file:
#     wr = csv.writer(csv_file, delimiter=',')
#     for doc_file_name, data in output.items():
#     	wr.writerow([doc_file_name, ' '.join(data)])
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