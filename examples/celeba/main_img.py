#%%
from selfmod import CelebADataset, NumpyLoader, make_image, VNet
# from selfmod import *
# from archs import VNet
from piedpiper import Encoder, Decoder, ImageDataset

import jax
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import equinox as eqx
import optax
from matplotlib import pyplot as plt
import time
import numpy as np


## For reproducibility
seed = 2022

## Dataloader hps
k_shots = 10
resolution = (32, 32)
H, W, C = (*resolution, 3)

data_folder="../../Self-Mod/examples/celeb-a/data/"
# data_folder="/Users/ddrous/Projects/Self-Mod/examples/celeb-a/data/"
shuffle = False
num_workers = 0
latent_chans = 16

envs_batch_size = 1
envs_batch_size_all = envs_batch_size
num_batches = 8*1

init_lr = 5e-5
nb_epochs = 7500
print_every = 500
sched_factor = 1.0
eps = 1e-6  ## Small value to avoid division by zero




#%%

if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)

adapt_folder = setup_run_folder(run_folder, os.path.basename(__file__))


# os.listdir(data_folder)

#%% 

mother_key = jax.random.PRNGKey(seed)
data_key, model_key, trainer_key, test_key = jax.random.split(mother_key, num=4)

##### Numpy Loaders
train_dataset = ImageDataset(data_folder, 
                            data_split="train",
                            num_shots=k_shots, 
                            order_pixels=False, 
                            resolution=resolution,
                            max_envs=envs_batch_size*num_batches,
                            )
train_dataloader = NumpyLoader(train_dataset, 
                              batch_size=envs_batch_size, 
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)


#%% 

gen_train_dataloader = iter(train_dataloader)

# dat = next(iter(all_shots_train_dataloader))
# dat_few_shots = next(iter(train_dataloader))

# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# img = make_image(dat[0][0], dat[1][0], img_size=(*resolution, 3))
# axs[0].imshow(img)

# img_fs = make_image(dat_few_shots[0][0], dat_few_shots[1][0], img_size=(*resolution, 3))
# axs[1].imshow(img_fs)


#%%


class ConvCNP(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    positivity: callable

    def __init__(self, latent_chans=8, key=None):
        super().__init__()
        ## From the ConvCNP paper, Figure 1c
        keys = jax.random.split(key, 2)
        self.encoder = Encoder(C, H, W, key=keys[0])    ## E
        self.decoder = Decoder(C, H, W, in_chans=C, out_chans=latent_chans, key=keys[1])    ## rho
        # self.positivity = lambda x: jax.nn.softplus(x)
        self.positivity = lambda x: jnp.clip(jax.nn.softplus(x), eps, 1)

    def preprocess(self, X, Y):
        img = jnp.zeros((C, H, W))
        mask = jnp.zeros((1, H, W))
        i_locs = (X[:, 0] * H).astype(int)
        j_locs = (X[:, 1] * W).astype(int)
        img = img.at[:, i_locs, j_locs].set(jnp.clip(Y, 0., 1.).T)
        mask = mask.at[:, i_locs, j_locs].set(1.)
        return img, mask

    def postprocess(self, mu, sigma):
        # mu = jnp.transpose(mu, (1, 2, 0)).reshape(-1, C)
        # sigma = jnp.transpose(sigma, (1, 2, 0)).reshape(-1, C)
        mu = jnp.transpose(mu, (1, 2, 0))
        sigma = jnp.transpose(sigma, (1, 2, 0))
        return mu, sigma

    def __call__(self, Xs, Ys):
        def predict(X, Y):
            Ic, Mc = self.preprocess(X, Y)   ## Context pixels and their location
            hc = self.encoder(Mc, Ic)   ## Normalized convolution

            ft = self.decoder(hc)
            mu, sigma = jnp.split(ft, 2, axis=0)
            # jax.debug.print("sigma is {}", sigma)
            sigma = self.positivity(sigma)

            mu, sigma = self.postprocess(mu, sigma)  ## Reshape into 2D arrays = (H, W, C)

            return mu, sigma    ## Shape: (H, W, C)

        return eqx.filter_vmap(predict)(Xs, Ys)


model = ConvCNP(latent_chans=latent_chans, key=model_key)

def loss_fn(model, batch):
    (Xc, Yc), (Xt, Yt) = batch
    # Xc shape: (B, K, 2), Yc shape: (B, K, C), Yt shape: (B, 1024, C)

    @eqx.filter_vmap
    def make_targets(X, Y):
        I, _ = model.preprocess(X, Y)
        return I.transpose(1, 2, 0)
    ys = make_targets(Xt, Yt)

    mus, sigmas = model(Xc, Yc)
    # mu, sigma shape: (B, H, W, C)

    # @eqx.filter_vmap
    # @eqx.filter_vmap
    # @eqx.filter_vmap
    # @eqx.filter_vmap
    def neg_log_likelihood(mu, sigma, y):
        # mu, sigma, y shape: (1)
        # return jnp.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2
        # return 0.5 * ((y - mu)) ** 2
        return 0.5 * jnp.log(2*jnp.pi*sigma) + 0.5 * ((y - mu) / sigma) ** 2

    losses = neg_log_likelihood(mus, sigmas, ys)
    # return losses.sum(axis=(1, 2)).mean()
    return losses.mean()


## Define the learner
learner = Learner(model, loss_fn)




## Define optimiser and train the model
total_steps = nb_epochs*train_dataloader.num_batches
bd_scales = {total_steps//3:sched_factor, 2*total_steps//3:sched_factor}
sched = optax.piecewise_constant_schedule(init_value=init_lr, boundaries_and_scales=bd_scales)
opt = optax.chain(optax.clip(1e-0), optax.adam(sched))

opt_state = opt.init(eqx.filter(model, eqx.is_array))

trainer = Trainer(learner, opt)



#%%
## Training loop
trainer.meta_train(train_dataloader)

## Plot the loss curve
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(np.clip(losses, None, 1))
ax.set_xlabel("Epochs")
ax.set_ylabel("Negative Log Likelihood")
# ax.set_yscale("log")
# ax.set_ylim(0, 10)
ax.set_title("Loss curve")

fig.savefig("loss_curve.png")



#%%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 6))
