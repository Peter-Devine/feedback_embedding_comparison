from encoding.sentence_transformers_encoder_base import SentenceTransformerEncoderBase

class Encoder(SentenceTransformerEncoderBase):
    def __init__(self):
        model_name = "nq-distilbert-base-v1"
        super(Encoder, self).__init__(model_name)
