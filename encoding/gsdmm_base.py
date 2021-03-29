from gsdmm import MovieGroupProcess
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

class GsdmmBase:
    def __init__(self, topic_num):
        self.topic_num = topic_num
        self.stopwords = set(stopwords.words('english') + list(string.punctuation))

    def encode(self, text_list, dataset_name=None):

        # Get a list of a set of tokens, tokenized by nltk
        list_of_word_sets = [set(self.tokenize_clean(x)) for x in text_list]

        # Make the vocab set, as a set of all words occuring in the data
        vocab = set()
        [vocab.update(x) for x in list_of_word_sets]

        # Do the topic modelling
        mgp = MovieGroupProcess(K=self.topic_num, alpha=0.1, beta=0.1, n_iters=30)
        mgp.fit(list_of_word_sets, len(vocab))

        # Score each document, getting a probability distribution over all topics
        embeddings = [mgp.score(word_set) for word_set in list_of_word_sets]
        embeddings = np.stack(embeddings)

        return embeddings

    def tokenize_clean(self, sentence):
        return [i for i in nltk.word_tokenize(sentence.lower()) if i not in self.stopwords]
