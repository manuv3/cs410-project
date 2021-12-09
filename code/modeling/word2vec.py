from gensim.test.utils import datapath
from gensim import utils
from gensim.models import Word2Vec, KeyedVectors
import corpus
import os

_word2vec_model_path = os.path.abspath('../../model/myword2vec.wordvectors')


class MyCorpus:

	path = None

	def __init__(self, path):
		self.path = path


	"""An iterator that yields sentences (lists of str)."""
	def __iter__(self):
		sentences = corpus.get_sentences(self.path)
		for line in sentences:
			yield line


#phrases = corpus.get_prebuilt_phrases()
#sentences = MyCorpus('../../data/transcripts')
#model = Word2Vec(phrases[sentences])
#model.wv.save(_word2vec_model_path)

wv = KeyedVectors.load(_word2vec_model_path, mmap='r')

print(wv.most_similar(positive=['pagerank'], topn=20))

