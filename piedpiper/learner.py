from ._config import *
from selfmod import VNet


class Learner:
    def __init__(self, 
                 model, 
                 loss_fn, 
                 key=None):
        if key is None:
            raise ValueError("You must provide a key for the learner.")
        self.key = key

        self.model = model
        self.loss_fn = loss_fn

    def save_learner(self, path):
        assert path[-1] == "/", "ERROR: Invalid path provided. The path must end with /"
        eqx.tree_serialise_leaves(path+"model.eqx", self.model)

    def load_learner(self, path):
        assert path[-1] == "/", "ERROR: Invalidn parovided. The path must end with /"
        self.model = eqx.tree_deserialise_leaves(path+"model.eqx", self.model)









class Encoder(eqx.Module):
    """ Set encoder for on-the-grid images """

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d

    def __init__(self, C, H, W, kernel_size=3, *, key):
        super().__init__()
        keys = jax.random.split(key, 2)
        self.conv1 = eqx.nn.Conv2d(1, 1, kernel_size, padding="same", key=keys[0])
        self.conv2 = eqx.nn.Conv2d(C, C-1, kernel_size, padding="same", key=keys[1])

    def __call__(self, Mc, Ic):
        h0 = self.conv1(Mc)
        Zc = Mc * Ic
        h = self.conv2(Zc) / (h0)
        return jnp.concatenate([h0, h], axis=0)


class Decoder(eqx.Module):
    vnet: eqx.Module
    mlp: eqx.nn.Conv2d

    def __init__(self, C, H, W, in_chans=3, out_chans=8, kernel_size=5, levels=4, depth=8, *, key):
        super().__init__()
        keys = jax.random.split(key, 4)

        self.vnet = VNet(input_shape=(in_chans, H, W), 
                         output_shape=(out_chans, H, W), 
                         levels=levels, 
                         depth=depth,
                         kernel_size=kernel_size,
                         activation=jax.nn.relu,
                         final_activation=lambda x:x,
                         batch_norm=False,
                         dropout_rate=0.,
                         key=keys[0],)

        self.mlp = eqx.nn.Conv2d(out_chans, 2*C, 1, padding="same", key=keys[2])

    def __call__(self, hc):
        h = self.vnet(hc)
        ft = self.mlp(h)
        return ft


