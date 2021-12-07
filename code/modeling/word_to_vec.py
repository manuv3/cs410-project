import os
from matplotlib import pyplot
import pickle
from gensim.models import CoherenceModel
from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count

import corpus

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

_model_path = os.path.abspath('../../data/models')

def build_word2vec():
    data = corpus._tokenize(corpus._local_docs_path)
    data = [x for x in data]
    model = Word2Vec(data, min_count=0, workers=cpu_count())
    model.save(os.path.join(_model_path, 'word2vec'))

def get_prebuilt_word2vec():
    return Word2Vec.load(os.path.join(_model_path, 'word2vec'))


data = corpus._tokenize(corpus._local_docs_path)
data = [x for x in data]
X_train = pd.DataFrame(data)
sample = X_train.sample(frac=0.5)


class WordVecVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = 300

    def transform(self, X):
        return np.array([np.mean([self.word2vec.wv[w] for w in texts if w in self.word2vec.wv.key_to_index] or
                                 [np.zeros(self.dim)], axis=0) for texts in X.values])

wtv_vect = WordVecVectorizer(get_prebuilt_word2vec())
X_train_wtv = wtv_vect.transform(sample)
print(X_train_wtv.shape)

km = KMeans(
    n_clusters=16, init='random',
    n_init=10, max_iter=300,
    tol=1e-04, random_state=0
)

y_km = km.fit_predict(X_train_wtv)
df = pd.DataFrame({'docs' : [x for x in sample.values if x is not None], 'topic_cluster' :y_km})
print(df)

pickle.dump(km, open(os.path.join(_model_path, 'kmeans'), 'wb'))