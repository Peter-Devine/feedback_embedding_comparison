from encoding.sentence_transformers_encoder_base import SentenceTransformerEncoderBase

class Encoder(SentenceTransformerEncoderBase):
    def __init__(self):
        model_name = "Capreolus/birch-bert-large-msmarco_mb"
        super(Encoder, self).__init__(model_name)
