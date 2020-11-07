from encoding.sentence_transformers_encoder_base import SentenceTransformerEncoderBase

class Encoder(SentenceTransformerEncoderBase):
    def __init__(self):
        model_name = "xlm-r-100langs-bert-base-nli-mean-tokens"
        super(Encoder, self).__init__(model_name)