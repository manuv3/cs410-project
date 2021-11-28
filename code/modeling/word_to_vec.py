import os
from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count
import corpus

_model_path = os.path.abspath('../../data/models')


data = corpus._tokenize(corpus._local_docs_path)
data = [x for x in data]

model = Word2Vec(data, min_count = 0, workers=cpu_count())

print(model.wv['lecture'])
print(model.wv.most_similar('lecture'))

model.save(os.path.join(_model_path, 'word2vec'))