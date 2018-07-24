import numpy as np
from chainer import Chain


class AutoEncoder(Chain):
    def __init__(self):
        super(AutoEncoder, self).__init__()

    def encode(self):
        raise NotImplementedError("encode() not implemented")

    def decode(self):
        raise NotImplementedError("decode() not implemented")
