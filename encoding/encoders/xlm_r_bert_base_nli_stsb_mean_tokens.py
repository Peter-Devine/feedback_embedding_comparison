from encoding.sentence_transformers_encoder_base import SentenceTransformerEncoderBase

class Encoder(SentenceTransformerEncoderBase):
    def __init__(self):
        model_name = "xlm-r-bert-base-nli-stsb-mean-tokens"
        super(Encoder, self).__init__(model_name)