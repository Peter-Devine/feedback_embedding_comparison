from encoding.tf_hub_encoder_base import TfHubEncoderBase

class Encoder(TfHubEncoderBase):
    def __init__(self):
        model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        super(Encoder, self).__init__(model_url)
