from encoding.sentence_transformers_encoder_base import SentenceTransformerEncoderBase

class Encoder(SentenceTransformerEncoderBase):
    def __init__(self):
        model_name = "distilbert-base-nli-stsb-quora-ranking"
        super(Encoder, self).__init__(model_name)