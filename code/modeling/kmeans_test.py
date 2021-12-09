import pickle
import cluster_topics
from flask import jsonify, Flask

#model = pickle.load(open('/Users/sofiagodovykh/CS410/text_project/cs410-project/data/models/clustermodel', 'rb'))
from cluster_topics import CLusterModel


model = CLusterModel()
term_count = 10
print(model._topics_identifier)
temp = model.get_topics(term_count = term_count)

