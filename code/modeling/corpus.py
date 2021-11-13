import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.corpora import Dictionary, MmCorpus
from gensim.parsing.preprocessing import STOPWORDS
import tempfile
import os
import re
import smart_open

nltk.data.path = [os.path.abspath('../../data/nltk_data')] + nltk.data.path
_tokenizer = RegexpTokenizer(r'\w+')
_lemmatizer = WordNetLemmatizer()
_stoplist = set('for a of the and to in'.split())
_local_docs_path = os.path.abspath('../../data/transcripts')
_dictionary_path = os.path.abspath('../../tmp/dictionary.dict')
_corpus_path = os.path.abspath('../../tmp/corpus.mm')

def _get_docs(path):
  for root, dirs, files in os.walk(path):
    for name in files:
      with open(os.path.join(root, name), 'r') as doc:
      	yield re.sub('\n', ' ', doc.read())

def _tokenize_doc(doc):
	doc = doc.lower()  # Convert to lowercase.
	doc = _tokenizer.tokenize(doc)  # Split into words.
	doc = [token for token in doc if token not in stopwords.words('english')]
	doc = [token for token in doc if token not in STOPWORDS]
	doc = [token for token in doc if not token.isnumeric()]
	doc = [token for token in doc if len(token) > 1]
	return [_lemmatizer.lemmatize(topic) for topic in doc]

def _tokenize(path):
	for doc in _get_docs(path):
	  yield _tokenize_doc(doc)

# def _add_phrases(docs_path, tokens_file):
# 	bigram = Phrases(_get_docs(docs_path), min_count=1, threshold = 1, connector_words=ENGLISH_CONNECTOR_WORDS)
# 	for doc in (re.sub(os.linesep, '', line).split(',') for line in tokens_file.readlines()):
# 	  for token in bigram[doc]:
# 	  	if '_' in token:
# 	  		print(token)
# 	  		doc.append(token) # Token is a bigram, add to document.
# 	  yield doc

def _generate_dict(tokens_file):
	return Dictionary([re.sub(os.linesep, '', line).split(',') for line in tokens_file.readlines()])

def _generate_corpus(tokens_file, dictionary):
	for doc in (re.sub(os.linesep, '', line).split(',') for line in tokens_file.readlines()):
		yield dictionary.doc2bow(doc)


def build_corpus(path):
	with tempfile.TemporaryFile(mode = 'w+t') as fp:
		for doc in _tokenize(path):
			fp.write(','.join(doc))
			fp.write(os.linesep)
		fp.seek(0)
		dictionary = _generate_dict(fp)
		dictionary.save(_dictionary_path)
		fp.seek(0)
		MmCorpus.serialize(_corpus_path, _generate_corpus(fp, dictionary))

def get_prebuilt_dictionary():
	return Dictionary.load(_dictionary_path)	

def get_prebuilt_corpus():
	return MmCorpus(_corpus_path)