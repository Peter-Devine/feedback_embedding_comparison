from encoding.biterm_base import BitermBase

class Encoder(BitermBase):
    def __init__(self):
        topic_num = 5
        super(Encoder, self).__init__(topic_num)
