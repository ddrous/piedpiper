#%%[markdown]

# ## Simplest form of regression on images
# WHat I realise is that more data doesn't mean more flexibility. It's the oppositite. It takes ages, and gets stuck in a wird minima all only means.

#%%
import jax

print("Available devices:", jax.devices())

# from jax import config
# config.update("jax_debug_nans", True)

import jax.numpy as jnp

import numpy as np
import equinox as eqx

# import matplotlib.pyplot as plt
# from selfmod import NumpyLoader, make_image, make_run_folder, setup_run_folder, count_params
from selfmod import make_image, NumpyLoader
# from piedpiper import *
from neuralhub import *

import optax
import time
import torch

#%%

SEED = 2026
## Set numpy and torch, and jax random keys
main_key = jax.random.PRNGKey(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

## Training hps
nb_epochs = 15000*10
print_every = nb_epochs//20
batch_size = 16

## Optimiser hps
init_lr = 1e-3
# transition_steps = nb_epochs//100
transition_steps = 150

train = True

## Data hps
data_folder = "./data/" if train else "../../data/"
# run_folder = "./runs/241223-181951-Test/" if train else "./"
run_folder = None if train else "./"


#%%
### Create and setup the run folder
if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)
_ = setup_run_folder(run_folder, os.path.basename(__file__))


#%%

train_data = np.load(data_folder+"example_data.npz")['context'][0:16]
train_data_tgt = np.load(data_folder+"example_data.npz")['target'][0:16]

print("Train data shape:", train_data.shape)

## Define a data loader to load the data and the target
class DataSet(torch.utils.data.Dataset):
    def __init__(self, context, target):
        self.context = context
        self.target = target

        self.total_len = len(self.context)
    def __len__(self):
        return len(self.context)
        # return 16

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx]

        # ## get a number between 0 and len(context), based on idx
        # img_idx = np.random.randint(0, len(self.context))
        # return self.context[img_idx], self.target[img_idx]



train_set = DataSet(train_data, train_data_tgt)
train_loader = NumpyLoader(train_set, batch_size=batch_size, shuffle=True)


# %%


# class Model(eqx.Module):
#     mlp: eqx.nn.MLP
#     def __init__(self, key=None):
#         self.mlp = eqx.nn.MLP(2, 
#                             3, 
#                             64, 
#                             3, 
#                             use_bias=True, 
#                             activation=jax.nn.softplus,
#                             final_activation=jax.nn.sigmoid,
#                             key=key)
    # def __call__(self, xs):
    #     """ Forward call of the Model """
    #     # ctx_pred = eqx.filter_vmap(self.mlp)(xs)
    #     tgt_pred = eqx.filter_vmap(self.mlp)(xs)
    #     return tgt_pred


class Model(eqx.Module):
    encoder: eqx.nn.MLP
    decoder: eqx.nn.MLP
    def __init__(self, key=None):
        self.encoder = eqx.nn.MLP(2+3, 
                                    128, 
                                    128, 
                                    3, 
                                    use_bias=True, 
                                    activation=jax.nn.softplus,
                                    key=key)
        self.decoder = eqx.nn.MLP(2+128, 
                                    3*2, 
                                    128, 
                                    5, 
                                    use_bias=True, 
                                    activation=jax.nn.softplus,
                                    final_activation=jax.nn.softplus,
                                    key=key)

    def __call__(self, xys_ctx, xs_tgt):
        """ Forward call of the Model """
        z_rep = eqx.filter_vmap(self.encoder)(xys_ctx).mean(axis=0)
        # ctx_pred = eqx.filter_vmap(self.mlp)(xs)
        ys_tgt = eqx.filter_vmap(lambda x: self.decoder(jnp.concatenate([x, z_rep])))(xs_tgt)
        mu_tgt, sigma_tgt = ys_tgt[..., :3], ys_tgt[..., 3:]
        return mu_tgt, jnp.clip(sigma_tgt, 1e-6, 1)




# %%

model_keys = jax.random.split(main_key, num=2)

model = Model(key=model_keys[0])

## Print the total number of learnable paramters in the model components
print(f"Number of learnable parameters in the model: {count_params(model)/1000:3.1f} k")

# %%

# def loss_fn(model, batch, key):
#     ctx, tgt = batch
#     X, Y = ctx[..., :2], ctx[..., 2:]
#     X_tgt, Y_tgt = tgt[..., :2], tgt[..., 2:]
#     # print("Data shapes in loss_fn:", X.shape, Y.shape)
#     Y_hat = model(X_tgt)
#     return jnp.mean((Y_tgt - Y_hat)**2), (jnp.min(Y_hat), jnp.max(Y_hat))

def loss_fn(model, batch, key):
    XY_ctx, XY_tgt = batch
    X_tgt, Y_tgt = XY_tgt[..., :2], XY_tgt[..., 2:]
    # print("Data shapes in loss_fn:", X.shape, Y.shape)

    # Y_hat = eqx.filter_vmap(model)(XY_ctx, X_tgt)
    # return jnp.mean((Y_tgt - Y_hat)**2), (jnp.min(Y_hat), jnp.max(Y_hat))

    mu, sigma = eqx.filter_vmap(model)(XY_ctx, X_tgt)
    Y_hat = mu
    print("Shapes of mu, sigma, Y_tgt:", mu.shape, sigma.shape, Y_tgt.shape)
    losses = 0.5 * jnp.log(2*jnp.pi*sigma) + 0.5 * ((Y_tgt - mu) / sigma) ** 2
    return jnp.mean(losses), (jnp.min(Y_hat), jnp.max(Y_hat))


@eqx.filter_jit
def train_step(model, batch, opt_state, key):
    print('\nCompiling function "train_step" ...')

    (loss, aux_data), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, batch, key)

    updates, opt_state = opt_cnp.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss, aux_data




#%%

if train:
    sched_cnp = optax.exponential_decay(init_value=init_lr, transition_steps=transition_steps, decay_rate=0.99)
    opt_cnp = optax.adam(sched_cnp)
    opt_state_cnp = opt_cnp.init(eqx.filter(model, eqx.is_array))

    train_key, _ = jax.random.split(main_key)
    losses_cnp = []

    print(f"\n\n=== Beginning Training ... ===")
    start_time = time.time()

    for epoch in range(nb_epochs):

        loss_epoch = 0
        for i, batch in enumerate(train_loader):
            model, opt_state_cnp, loss, (min_pixel, max_pixel) = train_step(model, batch, opt_state_cnp, train_key)
            loss_epoch += loss

        loss = loss_epoch / (i+1)
        # model, opt_state_cnp, loss, (min_pixel, max_pixel) = train_step(model, (train_data, train_data_tgt), opt_state_cnp, train_key)

        losses_cnp.append(loss)

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            print(f"    Epoch: {epoch:-5d}      LossNode: {loss:.12f}      Min Pixel: {min_pixel:.12f}      Max Pixel: {max_pixel:.12f}", flush=True)
            eqx.tree_serialise_leaves(run_folder+"model.eqx", model) ## just for checkpoint elsewhere

    wall_time = time.time() - start_time
    print("\nTotal GD training time: %d hours %d mins %d secs" % (wall_time//3600, (wall_time%3600)//60, wall_time%60))

    print(f"Training Complete, saving model to folder: {run_folder}")
    eqx.tree_serialise_leaves(run_folder+"model_final.eqx", model)
    np.save(run_folder+"losses.npy", np.array(losses_cnp))

else:
    model = eqx.tree_deserialise_leaves(run_folder+"model.eqx", model)
    try:
        losses_cnp = np.load(run_folder+"losses.npy")
    except:
        losses_cnp = []

    print("Model loaded from folder")


# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax = sbplot(np.array(losses_cnp), x_label='Epoch', y_label='L2', y_scale="linear", label='Losses Node', ax=ax, dark_background=False);
plt.legend()
plt.draw();
plt.savefig(run_folder+"loss.png", dpi=100, bbox_inches='tight')


# %%

W, H = 32, 24
# xs, ys = jnp.unravel_index(jnp.arange(H*W), (W, H))
# X_grid = jnp.vstack((xs, ys)).T / jnp.array((W, H))

train_data, train_data_tgt = next(iter(train_loader))
plt_id = np.random.randint(0, len(train_data))

## Turn the model prediction into an image and visualise it
XY_ctx = train_data
X_tgt, Y_tgt = train_data_tgt[..., :2], train_data_tgt[..., 2:]

Y_test, UQ_test = eqx.filter_vmap(model)(XY_ctx, X_tgt)
Y_test = Y_test[plt_id]
UQ_test = UQ_test[plt_id]
# Y_test = eqx.filter_vmap(model)(XY_ctx, X_grid[None])[plt_id]
X_test = X_tgt[plt_id]

fig, ax = plt.subplots(1, 3, figsize=(7, 4))
# img_new = make_image(Xt_new, Yt_new, img_size=(*resolution, 3))
img_true = make_image(X_test, Y_tgt[plt_id], img_size=(H, W, 3))
img_pred = make_image(X_test, Y_test, img_size=(H, W, 3))
img_uq = make_image(X_test, UQ_test, img_size=(H, W, 3))
ax[0].imshow(img_true)
ax[0].set_title("True")
ax[1].imshow(img_pred)
ax[1].set_title("Prediction")
ax[2].imshow(img_uq)
ax[2].set_title("Uncertainty")

plt.draw()
plt.savefig(run_folder+"prediction.png", dpi=100, bbox_inches='tight')

# print("UQ_test shape:", UQ_test)