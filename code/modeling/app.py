from flask import Flask, jsonify, request
from topics import LdaBasedModel

app = Flask(__name__)



model = LdaBasedModel()



@app.route("/topics")
def topics():
	term_count = int(request.args.get('terms', '10'))
	return jsonify(model.get_topics(term_count = term_count))

@app.route("/topics/<int:topic_id>")
def terms_for_topic(topic_id):
	term_count = int(request.args.get('terms', '10'))
	return jsonify(model.get_topic(topic_id, term_count = term_count))

@app.route("/documents")
def documents():
	return jsonify(model.get_docs())

@app.route("/documents/<int:doc_id>/topics")
def topics_for_document(doc_id):
	term_count = int(request.args.get('terms', '10'))
	topic_threshold = float(request.args.get('topic_threshold', '0.2'))
	topics = model.get_topics_for_doc(doc_id, topic_threshold = topic_threshold, term_count = term_count)
	print(topics)
	return jsonify(topics)