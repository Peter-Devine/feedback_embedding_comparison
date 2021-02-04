from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from biterm.utility import vec_to_biterms
from biterm.btm import oBTM

class BitermBase:
    def __init__(self, topic_num):
        self.topic_num = topic_num

    def encode(self, text_list, dataset_name=None):

        # Vectorize text
        vec = CountVectorizer(stop_words='english')
        X = vec.fit_transform(text_list).toarray()

        # Get the vocab of vectorized text
        vocab = np.array(vec.get_feature_names())

        # Make vectors into biterms
        biterms = vec_to_biterms(X)

        # Prepare biterm model with settings
        btm = oBTM(num_topics=self.topic_num, V=vocab)

        # Run biterm topic modelling
        topic_distribution = btm.fit_transform(biterms, iterations=50)

        # If there is a div(0) error, then the whole encoding is nan. We then simply say that there is an equally likely chance that this piece of text is part of any topic
        topic_distribution[np.isnan(topic_distribution).any(axis=1)] = 1 / self.topic_num

        return topic_distribution
