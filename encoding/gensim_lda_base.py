from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import numpy as np
import nltk
nltk.download('punkt')

class GensimLdaBase:
    def __init__(self, topic_num):
        self.topic_num = topic_num

    def encode(self, text_list, dataset_name=None):
        # Create a corpus from a list of texts
        toks_list = [nltk.word_tokenize(x) for x in text_list]
        common_dictionary = Dictionary(toks_list)
        common_corpus = [common_dictionary.doc2bow(toks) for toks in toks_list]

        # Train the model on the corpus.
        lda = LdaModel(common_corpus, num_topics=self.topic_num, minimum_probability=0)

        # Get the topic probability distribution across all topics
        return np.array([self.make_topic_vector_from_signed_tuple(lda[text], self.topic_num) for text in common_corpus])


    def make_topic_vector_from_signed_tuple(self, signed_tuple, num_topics):
        # Makes the output of lda[text], which is [(topic_num, probability),...] into a list of probabilities, ordered by topic number (I.e. [probability1, probability2, ...])

        # If a topic is not in the signed_tuple, then the probability is 0
        topic_vector = [0] * num_topics

        for i, val in signed_tuple:
            topic_vector[i] = val

        return topic_vector
