import lda
import corpus
import os


model = lda.get_prebuilt_model()

new_corpus = corpus.get_corpus(os.path.abspath('../../data/transcripts/10_1_Text_Clustering_Motivation.txt'))

for key, value in model[new_corpus]:
	print(model.print_topic(key))