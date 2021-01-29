from encoding.sentence_transformers_encoder_base import SentenceTransformerEncoderBase

class Encoder(SentenceTransformerEncoderBase):
    def __init__(self):
        model_name = "paraphrase-xlm-r-multilingual-v1"
        super(Encoder, self).__init__(model_name)
