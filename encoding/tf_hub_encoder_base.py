import tensorflow_hub as hub

class TfHubEncoderBase:
    def __init__(self, model_url):
        self.model = hub.load(model_url)

    def encode(self, text_list, dataset_name=None):
        return self.model(text_list).numpy()
