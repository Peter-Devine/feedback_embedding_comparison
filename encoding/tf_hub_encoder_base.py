import tensorflow_hub as hub
import tensorflow_text
import numpy as np
import tqdm as tq

class TfHubEncoderBase:
    def __init__(self, model_url):
        self.model = hub.load(model_url)
        self.model_url = model_url
        self.batch_size = 64

    def encode(self, text_list, dataset_name=None):
        embedding = [self.model(text_list[i:i+self.batch_size]).numpy() for i in tq.tqdm(range(0, len(text_list), self.batch_size), desc=f"{self.model_url} embed")]
        embedding = np.vstack(embedding)
        return embedding
