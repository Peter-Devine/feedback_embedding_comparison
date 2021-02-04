from sklearn.feature_extraction.text import TfidfVectorizer

class Encoder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def encode(self, text_list, dataset_name=None):
        return self.vectorizer.fit_transform(text_list).toarray()
