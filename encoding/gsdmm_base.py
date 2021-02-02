from gsdmm import MovieGroupProcess
import numpy as np
import nltk
nltk.download('punkt')

class GsdmmBase:
    def __init__(self, topic_num):
        self.topic_num = topic_num

    def encode(self, text_list, dataset_name=None):
        # Get a list of a set of tokens, tokenized by nltk
        list_of_word_sets = [set(nltk.word_tokenize(x)) for x in text_list]

        # Make the vocab set, as a set of all words occuring in the data
        vocab = set()
        [vocab.update(x) for x in list_of_word_sets]

        # Do the topic modelling
        mgp = MovieGroupProcess(K=self.topic_num, alpha=0.1, beta=0.1, n_iters=20)
        mgp.fit(list_of_word_sets, len(vocab))

        # Score each document, getting a probability distribution over all topics
        embeddings = [mgp.score(word_set) for word_set in list_of_word_sets]
        embeddings = np.stack(embeddings)

        return embeddings
