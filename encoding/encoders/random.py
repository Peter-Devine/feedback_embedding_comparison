from numpy import random

class Encoder:
    def __init__(self):
        self.embed_size = 50

    def encode(self, text_list, dataset_name=None):
        data_size = len(text_list)
        return random.rand(data_size, self.embed_size)
