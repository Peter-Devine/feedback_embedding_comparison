import pandas as pd
import numpy as np
import os

class Encoder:
    def __init__(self):
        pass

    def encode(self, text_list, dataset_name):
        use_dir = os.path.join(".", "data", "encoding", dataset_name, "distiluse_base_multilingual_cased_v2.csv")
        use_data = pd.read_csv(use_dir, index_col = 0)
        use_embedding = use_data.drop("labels", axis=1).values

        sbert_dir = os.path.join(".", "data", "encoding", dataset_name, "bert_large_nli_mean_tokens.csv")
        sbert_data = pd.read_csv(sbert_dir, index_col = 0)
        sbert_embedding = sbert_data.drop("labels", axis=1).values

        use_embedding = self.normalize(use_embedding)
        sbert_embedding = self.normalize(sbert_embedding)

        return np.concatenate([use_embedding, sbert_embedding], axis=1)

    def normalize(self, enc):
        # We normalize embeddings so that the average magnitude of vectors is 1
        return  enc / (np.linalg.norm(enc, axis=1).mean())
