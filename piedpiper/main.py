#%%
from selfmod import CelebADataset, NumpyLoader, make_image
# from selfmod import *
import jax
jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp
import equinox as eqx
import optax
from matplotlib import pyplot as plt
import time
import numpy as np


## For reproducibility
seed = 2022

## Dataloader hps
k_shots = 100
resolution = (32, 32)
data_folder="../../Self-Mod/examples/celeb-a/data/"
# data_folder="/Users/ddrous/Projects/Self-Mod/examples/celeb-a/data/"
shuffle = False
num_workers = 0
latent_chans = 32

envs_batch_size = 1
envs_batch_size_val = 1

init_lr = 1e-5
nb_epochs = 1000
print_every = 100
sched_factor = 0.1
eps = 1e-5  ## Small value to avoid division by zero

H, W, C = (*resolution, 3)

# os.listdir(data_folder)

#%% 

mother_key = jax.random.PRNGKey(seed)
data_key, model_key, trainer_key, test_key = jax.random.split(mother_key, num=4)

##### Numpy Loaders
train_dataset = CelebADataset(data_folder, 
                            data_split="train",
                            num_shots=k_shots, 
                            order_pixels=False, 
                            resolution=resolution,
                            # seed=seed
                            )
train_dataset.total_envs = envs_batch_size*1   ## 10 batches wanted
train_dataloader = NumpyLoader(train_dataset, 
                              batch_size=envs_batch_size, 
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

all_shots_train_dataset = CelebADataset(data_folder, 
                                    data_split="train",
                                    num_shots=np.prod(resolution), 
                                    order_pixels=False, 
                                    resolution=resolution,
                                    # seed=seed,
                                    )
all_shots_train_dataset.total_envs = envs_batch_size_val*1   ## 10 batches wanted
all_shots_train_dataloader = NumpyLoader(all_shots_train_dataset, 
                              batch_size=envs_batch_size_val, 
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)


#%% 
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
        self.encoder = Encoder(key=keys[0])    ## E
        self.decoder = Decoder(latent_chans, key=keys[1])    ## rho
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
        mu = jnp.transpose(mu, (1, 2, 0)).reshape(-1, C)
        sigma = jnp.transpose(sigma, (1, 2, 0)).reshape(-1, C)
        return mu, sigma

    def __call__(self, Xs, Ys):
        def predict(X, Y):
            Ic, Mc = self.preprocess(X, Y)   ## Context pixels and their location
            hc = self.encoder(Mc, Ic)   ## Normalized convolution

            ft = self.decoder(hc)
            mu, sigma = jnp.split(ft, 2, axis=-1)
            # jax.debug.print("sigma is {}", sigma)
            sigma = self.positivity(sigma)

            mu, sigma = self.postprocess(mu, sigma)  ## Reshape into 2D arrays = (H*W, C)
            return mu, sigma

        return eqx.filter_vmap(predict)(Xs, Ys)
    
class Encoder(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d

    def __init__(self, key):
        super().__init__()
        keys = jax.random.split(key, 2)
        self.conv1 = eqx.nn.Conv2d(1, 1, 4, padding="same", key=keys[0])
        self.conv2 = eqx.nn.Conv2d(C, C-1, 4, padding="same", key=keys[1])

    def __call__(self, Mc, Ic):
        h0 = self.conv1(Mc)
        # h0 = jnp.clip(h0, eps, None)
        # jax.debug.print("Ho is {}", h0)
        Zc = Mc * Ic
        h = self.conv2(Zc) / (h0)     ## nan to nums ?
        return jnp.concatenate([h0, h], axis=0)

class Decoder(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    mlp: eqx.nn.Conv2d

    def __init__(self, latent_chans, key):
        super().__init__()
        keys = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(C, latent_chans, 3, padding="same", key=keys[0])
        self.conv2 = eqx.nn.Conv2d(latent_chans, latent_chans, 3, padding="same", key=keys[1])
        self.mlp = eqx.nn.Conv2d(latent_chans, 2*C, 1, padding="same", key=keys[2])

    def __call__(self, hc):
        h = self.conv1(hc)
        h = self.conv2(h)
        # jax.debug.print("H is {}", h)
        ft = self.mlp(h)
        return ft


model = ConvCNP(latent_chans=latent_chans, key=model_key)

def loss_fn(model, batch):
    (Xc, Yc), (_, Yt) = batch
    # Xc shape: (B, K, 2), Yc shape: (B, K, C), Yt shape: (B, 1024, C)

    mus, sigmas = model(Xc, Yc)
    # mu, sigma shape: (B, 1024, C)

    @eqx.filter_vmap
    @eqx.filter_vmap
    @eqx.filter_vmap
    def neg_log_likelihood(mu, sigma, y):
        # mu, sigma, y shape: (1)
        # return jnp.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2
        # return 0.5 * ((y - mu)) ** 2
        return 0.5 * jnp.log(2*jnp.pi*sigma) + 0.5 * ((y - mu) / sigma) ** 2

    losses = neg_log_likelihood(mus, sigmas, Yt)
    # return losses.sum(axis=(1, 2)).mean()
    return losses.mean()

## Define optimiser and train the model
total_steps = nb_epochs*train_dataloader.num_batches
bd_scales = {total_steps//3:sched_factor, 2*total_steps//3:sched_factor}
sched = optax.piecewise_constant_schedule(init_value=init_lr, boundaries_and_scales=bd_scales)
opt = optax.chain(optax.clip(5.0), optax.adam(sched))

opt_state = opt.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def train_step(model, batch, opt_state):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, loss, opt_state


#%%
## Training loop

losses = []

## TODO: use tqdm for better progress bar
for epoch in range(nb_epochs):
    start_time_step = time.perf_counter()

    loss_epoch = 0.
    num_batches = 0
    for ctx_batch, tgt_batch in zip(train_dataloader, all_shots_train_dataloader):
        model, loss, opt_state = train_step(model, (ctx_batch, tgt_batch), opt_state)

        loss_epoch += loss
        num_batches += 1

    loss_epoch /= num_batches
    losses.append(loss_epoch)

    if epoch % print_every == 0 or epoch == nb_epochs-1:
        print(f"{time.strftime('%H:%M:%S')}      Epoch: {epoch:-3d}      Loss: {losses[-1]:-.8f}      Time/Epoch(s): {time.perf_counter()-start_time_step:-.4f}", flush=True, end="\n")


#%%
## Plot the loss curve
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(losses)
ax.set_xlabel("Epochs")
ax.set_ylabel("Negative Log Likelihood")
ax.set_yscale("log")
# ax.set_ylim(0, 10)
ax.set_title("Loss curve")


#%%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 6))

# Visualize an examploe prediction from the latest batch
Xc, Yc = ctx_batch
Xt, Yt = tgt_batch
mus, sigmas = model(Xc, Yc)
test_key = jax.random.PRNGKey(time.time_ns())
# Yt_hat = jax.random.normal(test_key, mus.shape) * sigmas + mus
Yt_hat = mus
print("Yt_hat shape: ", sigmas)

plt_idx = jax.random.randint(test_key, (1,), 0, envs_batch_size_val)[0]
img_true = make_image(Xt[plt_idx], Yt[plt_idx], img_size=(*resolution, 3))
ax1.imshow(img_true)
ax1.set_title(f"Target")

img_fw = make_image(Xc[plt_idx], Yc[plt_idx], img_size=(*resolution, 3))
ax2.imshow(img_fw)
ax2.set_title(f"Context Set")

img_pred = make_image(Xt[plt_idx], Yt_hat[plt_idx], img_size=(*resolution, 3))
ax3.imshow(img_pred)
ax3.set_title(f"Prediction")

img_std = make_image(Xt[plt_idx], sigmas[plt_idx], img_size=(*resolution, 3))
ax4.imshow(img_std)
ax4.set_title(f"Uncertainty")
