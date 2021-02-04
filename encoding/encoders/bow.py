from sklearn.feature_extraction.text import CountVectorizer

class Encoder:
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def encode(self, text_list, dataset_name=None):
        return self.vectorizer.fit_transform(text_list).toarray()
