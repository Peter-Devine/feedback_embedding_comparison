from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

class Encoder:
    def __init__(self):
        stop_words = stopwords.words('english')
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)

    def encode(self, text_list, dataset_name=None):
        return self.vectorizer.fit_transform(text_list).toarray()
