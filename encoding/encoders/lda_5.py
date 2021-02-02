from encoding.gensim_lda_base import GensimLdaBase

class Encoder(GensimLdaBase):
    def __init__(self):
        topic_num = 5
        super(Encoder, self).__init__(topic_num)
