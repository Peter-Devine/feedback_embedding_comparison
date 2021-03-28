from encoding.sentence_transformers_encoder_base import SentenceTransformerEncoderBase

class Encoder(SentenceTransformerEncoderBase):
    def __init__(self):
        model_name = "clip-ViT-B-32"
        super(Encoder, self).__init__(model_name)
