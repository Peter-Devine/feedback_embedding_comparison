from encoding.gsdmm_base import GsdmmBase

class Encoder(GsdmmBase):
    def __init__(self):
        topic_num = 5
        super(Encoder, self).__init__(topic_num)
