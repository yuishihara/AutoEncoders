from auto_encoder import AutoEncoder
import chainer.functions as F
import chainer.links as L


class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, encode_dim=20):
        super(VariationalAutoEncoder, self).__init__()
        self.name = "Variational Auto Encoder"
        with self.init_scope():
            self.l1 = L.Linear(None, 784)
            self.l2 = L.Linear(None, 256)
            self.lmu = L.Linear(None, encode_dim)
            self.lvar = L.Linear(None, encode_dim)
            self.l4 = L.Linear(None, 256)
            self.l5 = L.Linear(None, 784)

    def encode(self, x):
        mu, ln_var = self._latent_distribution(x)
        return self._sample(mu, ln_var)

    def decode(self, x):
        h = F.relu(self.l4(x))
        return F.sigmoid(self.l5(h))

    def loss(self, x, y):
        batch_size = len(x)
        mu, ln_var = self._latent_distribution(x)

        z = self._sample(mu, ln_var)

        reconstruction_loss = F.mean_squared_error(x, self.decode(z))
        latent_loss = 0.0005 * F.gaussian_kl_divergence(mu, ln_var) / batch_size
        loss = reconstruction_loss + latent_loss

        return loss

    def _sample(self, mu, ln_var):
        return F.gaussian(mu, ln_var)

    def _latent_distribution(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(x))
        mu = self.lmu(h)
        ln_var = F.relu(self.lvar(h))

        return mu, ln_var
