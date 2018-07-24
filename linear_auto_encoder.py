from auto_encoder import AutoEncoder
import chainer.functions as F
import chainer.links as L


class LinearAutoEncoder(AutoEncoder):
    def __init__(self):
        super(LinearAutoEncoder, self).__init__()
        self.name = "Linear auto encoder"
        with self.init_scope():
            self.l1 = L.Linear(None, 784)
            self.l2 = L.Linear(None, 512)
            self.l3 = L.Linear(None, 256)
            self.l4 = L.Linear(None, 128)
            self.l5 = L.Linear(None, 256)
            self.l6 = L.Linear(None, 512)
            self.l7 = L.Linear(None, 784)

    def encode(self, x):
        h = F.relu(self.l1(x))
        h = F.dropout(h)
        h = F.relu(self.l2(h))
        h = F.dropout(h)
        h = F.relu(self.l3(h))
        h = F.dropout(h)
        return F.relu(self.l4(h))

    def decode(self, x):
        h = F.relu(self.l5(x))
        h = F.dropout(h)
        h = F.relu(self.l6(h))
        h = F.dropout(h)
        return F.sigmoid(self.l7(h))
