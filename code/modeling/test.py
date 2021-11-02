import lda
import corpus
import os
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARN)

_corpus_path = os.path.abspath('../../data/transcripts')

# Build model without building corpus again
#my_lda = lda.build_lda()

# Build corpus and model
#my_lda = lda.build_lda(_corpus_path)

# Get pre built model
my_lda = lda.get_prebuilt_model()

# Get pre built corpus
my_corpus = corpus.get_prebuilt_corpus()

file_names = [files for root, dirs, files in os.walk(_corpus_path)][0]

output = {}
idx = 0
for doc in my_corpus:
	doc_name = file_names[idx]
	idx += 1
	output[doc_name] = []
	for key, value in my_lda[doc]:
		output[doc_name].append(my_lda.print_topic(key, topn = 50))

for key, value in output.items():
    print('{} ==> {}'.format(key, value))
    print('================================')
