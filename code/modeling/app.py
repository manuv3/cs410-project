from flask import Flask, jsonify, request
from topics import LdaBasedModel

app = Flask(__name__)


lda_model = LdaBasedModel()

model_names = ['LdaBasedModel']
model_objs = [lda_model]

curr_model_idx = 0

@app.route("/models")
def models():
	models_arr = []
	idx = 0
	for model in model_names:
		model_obj = {}
		model_obj['id'] = idx
		model_obj['name'] = model
		models_arr.append(model_obj)
	return jsonify(models_arr)

@app.route("/models/current")
def current_model():
	model_obj = {}
	model_obj['id'] = curr_model_idx
	model_obj['name'] = model_names[curr_model_idx]
	return model_obj


@app.route("/topics")
def topics():
	term_count = int(request.args.get('terms', '10'))
	return jsonify(model_objs[curr_model_idx].get_topics(term_count = term_count))

@app.route("/topics/<int:topic_id>")
def terms_for_topic(topic_id):
	term_count = int(request.args.get('terms', '10'))
	return jsonify(model_objs[curr_model_idx].get_topic(topic_id, term_count = term_count))

@app.route("/documents")
def documents():
	return jsonify(model_objs[curr_model_idx].get_docs())

@app.route("/documents/<int:doc_id>/topics")
def topics_for_document(doc_id):
	term_count = int(request.args.get('terms', '10'))
	topic_threshold = float(request.args.get('topic_threshold', '0.2'))
	topics = model_objs[curr_model_idx].get_topics_for_doc(doc_id, topic_threshold = topic_threshold, term_count = term_count)
	print(topics)
	return jsonify(topics)