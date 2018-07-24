from auto_encoder import AutoEncoder
import chainer.functions as F
import chainer.links as L


class ConvolutionalAutoEncoder(AutoEncoder):
    def __init__(self):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.name = "Convolutional auto encoder"
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels=1, out_channels=4, ksize=5, stride=1)
            self.conv2 = L.Convolution2D(
                in_channels=4, out_channels=8, ksize=5, stride=1)
            self.conv3 = L.Convolution2D(
                in_channels=8, out_channels=16, ksize=4, stride=1)
            self.l4 = L.Linear(None, 128)
            self.l5 = L.Linear(None, 4624)
            self.deconv6 = L.Deconvolution2D(
                in_channels=16, out_channels=8, ksize=4, stride=1)
            self.deconv7 = L.Deconvolution2D(
                in_channels=8, out_channels=4, ksize=5, stride=1)
            self.deconv8 = L.Deconvolution2D(
                in_channels=4, out_channels=1, ksize=5, stride=1)

    def encode(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        return F.relu(self.l4(h))

    def decode(self, x):
        h = F.relu(self.l5(x))
        h = F.reshape(h, (-1, 16, 17, 17))
        h = F.relu(self.deconv6(h))
        h = F.relu(self.deconv7(h))
        return F.sigmoid(self.deconv8(h))

    def __call__(self, x):
        return self.decode(self.encode(x))
