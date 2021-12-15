from flask import Flask, jsonify, request, render_template, abort
from topics import LdaBasedModel, get_prebuilt_model
import threading
#from model_factory import Factory


app = Flask(__name__)

def _build_model(target):
	model_builder = Factory()
	builder_thread = threading.Thread(target = model_builder.build, args = (target,))
	builder_thread.setDaemon(True)
	builder_thread.start()
	return model_builder	


# model_objs = [LdaBasedModel(model_path = 'lda'), LdaBasedModel(model_path = 'lda', relevance_parameter = 1.0), LdaBasedModel(model_path = 'lda_10', relevance_parameter = 0.4)]
model_objs = [get_prebuilt_model('model_0.obj'), get_prebuilt_model('model_1.obj'), get_prebuilt_model('model_2.obj')]
model_names = ['Lda Based Model: topic_count: 16, relevance_param: 0.4', 'Lda Based Model: topic_count: 16, relevance_param: 1.0', 'Lda Based Model: topic_count: 10, relevance_param: 0.4']


@app.route("/models")
def models():
	models_arr = []
	idx = 0
	for model in model_names:
		model_obj = {}
		model_obj['id'] = idx
		model_obj['name'] = model
		models_arr.append(model_obj)
		idx += 1
	return jsonify(models_arr)

@app.route("/models/<int:model_id>/topics")
def topics(model_id):
	term_count = int(request.args.get('terms', '10'))
	temp = model_objs[model_id].get_topics(term_count = term_count)
	return jsonify(temp)

@app.route("/models/<int:model_id>/topics/<int:topic_id>")
def terms_for_topic(model_id, topic_id):
	term_count = int(request.args.get('terms', '10'))
	return jsonify(model_objs[model_id].get_topic(topic_id, term_count = term_count))

@app.route("/models/<int:model_id>/documents")
def documents(model_id):
	return jsonify(model_objs[model_id].get_docs())

@app.route("/models/<int:model_id>/documents/<int:doc_id>/topics")
def topics_for_document(model_id, doc_id):
	term_count = int(request.args.get('terms', '10'))
	topics = model_objs[model_id].get_topics_for_doc(doc_id, term_count = term_count)
	return jsonify(topics)

@app.route('/models/<int:model_id>/documents/<int:doc_id>/ui')
def document_ui(model_id, doc_id):
	doc = list(filter(lambda item: item['id'] == doc_id, model_objs[model_id].get_docs()))
	if len(doc) > 0:
		return render_template('lesson_page.html', model_id = model_id, doc_id = doc_id, doc_name = doc[0]['name'], doc_url = doc[0]['url'])
	else:
		print('Invalid doc id {}'.format(doc_id))
		abort(400)

@app.route('/models/summary/ui')
def topic_summary_ui():
	return render_template('topic_summary.html', model_names = model_names)

@app.route('/models/<int:model_id>/index')
def index(model_id):
	index = model_objs[model_id].get_index()
	if (index == None):
		abort(409)
	return index

if __name__ == "__main__":
	app.run(debug=True)
