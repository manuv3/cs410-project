import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.word2vec import Text8Corpus
import tempfile
import os
import re
import json

nltk.data.path = [os.path.abspath('../../data/nltk_data')] + nltk.data.path
_tokenizer = RegexpTokenizer(r'\w+')
_sentence_tokenizer = PunktSentenceTokenizer()
_lemmatizer = WordNetLemmatizer()
_local_docs_path = os.path.abspath('../../data/transcripts')
_dictionary_path = os.path.abspath('../../model/dictionary.dict')
_corpus_path = os.path.abspath('../../model/corpus.mm')
_tfidf_model_path = os.path.abspath('../../model/tfidf')
_phrases_path = os.path.abspath('../../model/phrases.pkl')
_data_path = os.path.abspath('../../data')

def _get_docs(path):
	if os.path.isfile(path):
		yield _get_doc(path)
	else:			
		for root, dirs, files in os.walk(path):
			for name in files:
				yield _get_doc(os.path.join(root, name))

def _get_doc(file):
	with open(file, 'r') as doc:
			if file.endswith('.json'):
				return ' '.join([value for key, value in json.loads(doc.read()).items() if key != '0'])
			else:
				return doc.read()	

def _tokenize_doc(doc):
	doc = re.sub('\n', ' ', doc)
	doc = doc.lower()  # Convert to lowercase.
	doc = _tokenizer.tokenize(doc)  # Split into words.	
	doc = [token for token in doc if token not in stopwords.words('english')]
	doc = [token for token in doc if token not in STOPWORDS]
	doc = [token for token in doc if not token.isnumeric()]
	doc = [token for token in doc if len(token) > 2]
	return [_lemmatizer.lemmatize(topic) for topic in doc]

def _tokenize(path):
	for doc in _get_docs(path):
	  yield _tokenize_doc(doc)

def get_sentences(path):
	for doc in _get_docs(path):
		for sentence in _sentence_tokenizer.tokenize(doc):
			yield _tokenize_doc(sentence)

def _generate_phrases(path, min_count = 5):
	return Phrases(get_sentences(path), min_count=min_count, connector_words=ENGLISH_CONNECTOR_WORDS)

def _generate_dict(tokens_file):
	return Dictionary([re.sub('\n', ',', line).split(',') for line in tokens_file.readlines()])

def _generate_corpus(tokens_file, dictionary):
	for doc in (re.sub('\n', ',', line).split(',') for line in tokens_file.readlines()):
		yield dictionary.doc2bow(doc)


def build_corpus(build_phrases = True, min_colocation_count = 5):
	phrases = None
	if build_phrases:
			phrases = Phrases(get_sentences(os.path.join(_data_path, 'transcripts')), min_count= min_colocation_count, connector_words=ENGLISH_CONNECTOR_WORDS)
			phrases.add_vocab(get_sentences(os.path.join(_data_path, 'slides_raw_text')))
			phrases.save(_phrases_path)

	with tempfile.TemporaryFile(mode = 'w+t') as fp:
		for root, dirs, files in os.walk(os.path.join(_data_path, 'transcripts')):
			for name in files:
				doc = []
				sentences = get_sentences(os.path.join(root, name))
				for sent in sentences:
					doc.extend(sent)
					if build_phrases:
						phrases_in_sent = [phrase for phrase_arr in phrases[[sent]] for phrase in phrase_arr if "_" in phrase]
						doc.extend(phrases_in_sent)
				slide = os.path.join(_data_path, 'slides_raw_text', re.sub('.txt', '.json', name))
				if os.path.isfile(slide):
					doc.extend([term for term_arr in _tokenize(slide) for term in term_arr])
				fp.write(','.join(doc))
				fp.write('\n')
		fp.seek(0)
		dictionary = _generate_dict(fp)
		dictionary.save(_dictionary_path)
		fp.seek(0)
		MmCorpus.serialize(_corpus_path, _generate_corpus(fp, dictionary))
		fp.seek(0)
		TfidfModel(get_prebuilt_corpus()).save(_tfidf_model_path)


def get_prebuilt_dictionary():
	return Dictionary.load(_dictionary_path)	

def get_prebuilt_corpus():
	return MmCorpus(_corpus_path)

def get_prebuilt_phrases():
	return Phrases.load(_phrases_path)

def get_tfidf_model():
	return TfidfModel.load(_tfidf_model_path)

def get_raw_corpus_path():
	return _local_docs_path