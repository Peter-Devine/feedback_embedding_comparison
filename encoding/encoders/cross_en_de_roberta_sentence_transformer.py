from encoding.sentence_transformers_encoder_base import SentenceTransformerEncoderBase

class Encoder(SentenceTransformerEncoderBase):
    def __init__(self):
        model_name = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
        super(Encoder, self).__init__(model_name)