import threading


class Indexer:
	_term_index = {}
	_topic_index = {}
	_model = {}
	_indexer_lock = threading.Lock()

	def __init__(self, model):
		self._model = model

	def index(self):
		self._indexer_lock.acquire()
		try:
			self._term_index = {}
			self._topic_index = {}
			for doc in self._model.get_docs():
				for topic in self._model.get_topics_for_doc(doc_id = doc['id'])['topics']:
					if float(topic['coverage']) >= 0.2:
						if topic['name'] not in self._topic_index:
							self._topic_index[topic['name']] = []
						self._topic_index[topic['name']].append(doc)
						for term in topic['terms']:
							if term not in self._term_index:
								self._term_index[term] = []
							self._term_index[term].append(doc)
			self._term_index = dict(sorted(self._term_index.items(), key = lambda item: item[0]))
			self._topic_index = dict(sorted(self._topic_index.items(), key = lambda item: item[0]))
		finally:
			self._indexer_lock.release()

	def get_index(self):
		lock_acquired = self._indexer_lock.acquire(timeout=2)
		if lock_acquired:
		  try:
		  	return {'term_index': self._term_index, 'topic_index': self._topic_index}
		  finally:
		  	self._indexer_lock.release()
		else:
			return None
