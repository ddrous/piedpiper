from ._config import *
from selfmod import VNet


class Learner:
    def __init__(self, 
                 model, 
                 loss_fn,
                 images=True):
        self.model = model
        self.loss_fn = loss_fn
        self.images = images

    def save_learner(self, path):
        assert path[-1] == "/", "ERROR: Invalid path provided. The path must end with /"
        eqx.tree_serialise_leaves(path+"model.eqx", self.model)

    def load_learner(self, path):
        assert path[-1] == "/", "ERROR: Invalidn parovided. The path must end with /"
        if os.path.exists(path+"model.eqx"):
            # print(f"\nLoading model from {path} folder ...\n")
            self.model = eqx.tree_deserialise_leaves(path+"model.eqx", self.model)
        elif os.path.exists(path+"best_model.eqx"):
            print("WARNING: No model found in the provided path. Using the best model found.")
            self.model = eqx.tree_deserialise_leaves(path+"best_model.eqx", self.model)
        else:
            raise FileNotFoundError("ERROR: No model found in the provided path.")










class Encoder(eqx.Module):
    """ Set encoder for on-the-grid images """

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d

    def __init__(self, C, H, W, kernel_size=3, *, key):
        super().__init__()
        keys = jax.random.split(key, 2)
        self.conv1 = eqx.nn.Conv2d(1, 1, kernel_size, padding="same", key=keys[0])
        self.conv2 = eqx.nn.Conv2d(C, C-1, kernel_size, padding="same", key=keys[1])

    def __call__(self, Ic, Mc):
        h0 = self.conv1(Mc)
        Zc = Mc * Ic
        h = self.conv2(Zc) / (h0)
        return jnp.concatenate([h0, h], axis=0)


class Decoder(eqx.Module):
    vnet: eqx.Module
    mlp: eqx.nn.Conv2d

    def __init__(self, C, H, W, in_chans=3, latent_chans=8, kernel_size=5, levels=4, depth=8, *, key):
        super().__init__()
        keys = jax.random.split(key, 4)

        self.vnet = VNet(input_shape=(in_chans, H, W), 
                         output_shape=(latent_chans, H, W), 
                         levels=levels, 
                         depth=depth,
                         kernel_size=kernel_size,
                         activation=jax.nn.relu,
                         final_activation=lambda x:x,
                         batch_norm=False,
                         dropout_rate=0.,
                         key=keys[0],)

        self.mlp = eqx.nn.Conv2d(latent_chans, 2*C, 1, padding="same", key=keys[2])

    def __call__(self, hc):
        h = self.vnet(hc)
        ft = self.mlp(h)
        return ft




def neg_log_likelihood(mu, sigma, y):
    # return jnp.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2
    # return 0.5 * ((y - mu)) ** 2
    return 0.5 * jnp.log(2*jnp.pi*sigma) + 0.5 * ((y - mu) / sigma) ** 2

def mse(mu, sigma, y_true):
    return jnp.mean((y_true - mu) ** 2)

def psnr(mu, sigma, y_true):
    return 20 * jnp.log10(1.0 / jnp.sqrt(jnp.mean((y_true - mu) ** 2)))

def ssim(mu, sigma, y_true):
    return jax.image.ssim(mu, y_true, 1.0)

def fid(mu, sigma, y_true):
    return jax.image.fid(mu, y_true, 1.0)