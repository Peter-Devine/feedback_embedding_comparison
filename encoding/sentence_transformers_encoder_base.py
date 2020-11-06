from sentence_transformers import SentenceTransformer

class SentenceTransformerEncoderBase:
    def __init__(self, dataset_name):
        self.model = SentenceTransformer(dataset_name)

    def encode(self, text_list):
        return self.model.encode(text_list, show_progress_bar=True)
