from encoding.sentence_transformers_encoder_base import SentenceTransformerEncoderBase

class Encoder(SentenceTransformerEncoderBase):
    def __init__(self):
        model_name = "average_word_embeddings_levy_dependency"
        super(Encoder, self).__init__(model_name)