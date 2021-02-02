import pandas as pd
import numpy as np
import os
import numpy as np
from sklearn.decomposition import TruncatedSVD


class Encoder:
    def __init__(self):
        pass

    def encode(self, text_list, dataset_name):
        use_dir = os.path.join(".", "data", "encoding", dataset_name, "use_large.csv")
        use_data = pd.read_csv(use_dir, index_col = 0)
        use_embedding = use_data.drop("labels", axis=1).values

        sbert_dir = os.path.join(".", "data", "encoding", dataset_name, "bert_large_nli_mean_tokens.csv")
        sbert_data = pd.read_csv(sbert_dir, index_col = 0)
        sbert_embedding = sbert_data.drop("labels", axis=1).values

        use_embedding = self.normalize(use_embedding)
        sbert_embedding = self.normalize(sbert_embedding)

        embedding = np.concatenate([use_embedding, sbert_embedding], axis=1)

        return self.reduce(embedding, n_components=50)

    def normalize(self, enc):
        # We normalize embeddings so that the average magnitude of vectors is 1
        return  enc / (np.linalg.norm(enc, axis=1).mean())

    def reduce(self, enc, n_components):
        n_components = min([n_components, enc.shape[0], enc.shape[1]])
        svd = TruncatedSVD(n_components=n_components)
        return svd.fit_transform(enc)
