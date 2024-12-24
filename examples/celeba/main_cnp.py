#%%
%load_ext autoreload
%autoreload 2

import os
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '.5'
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
import jax.numpy as jnp

from selfmod import NumpyLoader, make_image, make_run_folder, setup_run_folder, count_params
from piedpiper import *
# import jax_dataloader as jdl

## For reproducibility
seed = 2025

## Dataloader hps
resolution = (24, 32)
k_shots = int(np.prod(resolution) * 0.8)
H, W, C = (resolution[1], resolution[0], 3)

data_folder="../../../Self-Mod/examples/celeb-a/data/"
# data_folder="./data/"
shuffle = False
num_workers = 24
latent_size = 128
hidden_mlp_size = 64

# envs_batch_size = 1627//10
envs_batch_size = 1627*10*5
# envs_batch_size = 1
envs_batch_size_all = envs_batch_size
# num_batches = 10*100
num_batches = 1

init_lr = 1e-3
sched_factor = 1.0
nb_epochs = 400
print_every = 10
validate_every = 10
eps = 1e-6  ## Small value to avoid division by zero

meta_train = True
run_folder = "./runs/241223-181951-Test/"
# run_folder = None


#%%

if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)

_ = setup_run_folder(run_folder, os.path.basename(__file__))


# os.listdir(data_folder)

#%% 

mother_key = jax.random.PRNGKey(seed)
data_key, model_key, trainer_key, test_key = jax.random.split(mother_key, num=4)

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

gen_train_dataloader = iter(train_dataloader)
batch = next(gen_train_dataloader)

## Save teh context and target data for a naive prediction task later on
dat_context, dat_target = batch
np.savez("data/example_data.npz", context=dat_context, target=dat_target)

#%% 


batch = next(gen_train_dataloader)
print([dat.shape for dat in batch])

dat_context, dat_target = batch
Xc, Yc = dat_context[0,..., :2], dat_context[0,..., 2:]
Xt, Yt = dat_target[0,..., :2], dat_target[0,..., 2:]
print("Context shape:", Xc.shape, Yc.shape)

fig, axs = plt.subplots(1, 2, figsize=(7, 3))
img = make_image(Xt, Yt, img_size=(*resolution, 3))
axs[0].imshow(img)
axs[0].set_title("Target Set")

img_fs = make_image(Xc, Yc, img_size=(*resolution, 3))
axs[1].imshow(img_fs)
axs[1].set_title("Context Set")
plt.suptitle("Example target and context sets", y=1.05);


#%%


class CNP(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    positivity: callable

    def __init__(self, latent_size, key=None):
        super().__init__()
        ## From the ConvCNP paper, Figure 1c
        keys = jax.random.split(key, 2)
        self.encoder = eqx.nn.MLP(in_size=2+3,  ## 2 for the location, 3 for the RGB values
                                  out_size=latent_size,
                                  width_size=hidden_mlp_size,
                                  depth=3,
                                  activation=jax.nn.softplus,
                                  key=keys[0])
        self.decoder = eqx.nn.MLP(in_size=2,    ## 2 for the location
                                  out_size=3*2,
                                  width_size=hidden_mlp_size,
                                  depth=3,
                                  activation=jax.nn.softplus,
                                  final_activation=jax.nn.sigmoid,
                                  key=keys[0])

        self.positivity = lambda x: jnp.clip(jax.nn.softplus(x), eps, 1)

    def preprocess(self, XY):
        X, Y = XY[..., :2], XY[..., 2:]
        img = jnp.zeros((C, H, W))
        mask = jnp.zeros((1, H, W))
        i_locs = (X[:, 0] * H).astype(int)
        j_locs = (X[:, 1] * W).astype(int)
        img = img.at[:, i_locs, j_locs].set(jnp.clip(Y, 0., 1.).T)
        mask = mask.at[:, i_locs, j_locs].set(1.)
        return img, mask

    def preprocess_channel_last(self, XY):
        img, mask = self.preprocess(XY)
        # return img.transpose(1, 2, 0), mask.transpose(1, 2, 0)
        return img.transpose(2, 1, 0), mask.transpose(2, 1, 0)

    def postprocess_img(self, X, Y):
        img = jnp.zeros((W, H, Y.shape[-1]))
        i_locs = (X[:, 0] * W).astype(int)
        j_locs = (X[:, 1] * H).astype(int)
        img = img.at[i_locs, j_locs, :].set(Y)
        return img

    def __call__(self, ctx_imgs, tgt_xys):
        # ctx_imgs, tgt_xys = ctx_imgs
        # ## Use meshgrid to get the xs, ys coordinates
        # xs, ys = jnp.meshgrid(jnp.arange(W), jnp.arange(H), indexing='ij')
        # xys = jnp.stack([xs, ys], axis=-1) / jnp.array((W, H))
        # xys = jnp.reshape(xys, (-1, 2))

        def decoding_fn(z, xy):
            xyz = jnp.concatenate([xy, z], axis=0)
            return self.decoder(xyz)

        def predict(ctx_img, tgt_xys):
            # zs = eqx.filter_vmap(self.encoder)(ctx_img)
            # z = jnp.mean(zs, axis=0)
            # # jax.debug.print("z representation: {}", z)

            # decoded = eqx.filter_vmap(decoding_fn, in_axes=(None, 0))(z, xys)
            # # decoded = einops.rearrange(decoded, '(W H) C -> W H C', H=H)

            decoded = eqx.filter_vmap(self.decoder)(tgt_xys)

            decoded = self.postprocess_img(tgt_xys, decoded)
            mu, sigma = jnp.split(decoded, 2, axis=-1)
            sigma = self.positivity(sigma)

            # jax.debug.print("mean predicted is: {}", mu)

            return mu, sigma    ## Shape: (H, W, C)

        return eqx.filter_vmap(predict)(ctx_imgs, tgt_xys)

    # def __call__(self, ctx_img):
    #     ## In this call, we want to return the context image, o=untouched
    #     img = eqx.filter_vmap(self.postprocess_img)(ctx_img[..., :2], ctx_img[..., 2:])
    #     return img, jnp.ones_like(img)


model = CNP(latent_size=latent_size, key=model_key)

def loss_fn(model, batch):
    ctx_data, tgt_data = batch
    # Xc shape: (B, K, 2), Yc shape: (B, K, C), Yt shape: (B, 1024, C)
    ys, _ = eqx.filter_vmap(model.preprocess_channel_last)(tgt_data)

    mus, sigmas = model(ctx_data, tgt_data[...,:2])    ## mu, sigma shape: (B, H, W, C)

    # losses = neg_log_likelihood(mus, sigmas, ys)
    losses = jnp.mean((mus - ys)**2)

    # return losses.sum(axis=(1, 2)).mean()
    # return losses.mean()
    return losses

## Define the learner
learner = Learner(model, loss_fn)

print("Total number of learnable parameters:", count_params(model))
print("Data context shape is:", dat_context.shape)

#%%
## Define optimiser and train the model
# total_steps = nb_epochs*train_dataloader.num_batches
total_steps = nb_epochs*num_batches
bd_scales = {total_steps//3:sched_factor, 2*total_steps//3:sched_factor}
sched = optax.piecewise_constant_schedule(init_value=init_lr, boundaries_and_scales=bd_scales)
opt = optax.adam(sched)


trainer = Trainer(learner, opt)

## Training loop
if meta_train:
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    trainer.meta_train(train_dataloader,
                        nb_epochs=nb_epochs,
                        print_every=print_every,
                        save_checkpoints=True,
                        save_path=run_folder,
                        val_dataloader=train_dataloader,
                        val_criterion="NLL",
                        validate_every=validate_every,
                        )
else:
    trainer.restore_trainer(run_folder)


#%%

# print("Training losses:", jnp.stack(trainer.train_losses))

visualtester = VisualTester(trainer)
visualtester.visualize_losses(run_folder+"loss_curve.png", log_scale=False, ylim=1)

#%%
test_dataset = ImageDataset(data_folder, 
                            data_split="train",
                            num_shots=k_shots, 
                            order_pixels=False, 
                            resolution=resolution,
                            max_envs=envs_batch_size*1,
                            seed=None,
                            )
test_dataloader = NumpyLoader(test_dataset, 
                              batch_size=envs_batch_size, 
                              shuffle=False,
                              num_workers=num_workers,
                              drop_last=False)

visualtester.visualize_images(test_dataloader, nb_envs=None, key=None, save_path=run_folder+"predictions_test.png", interp_method="linear", plot_ids=range(8))




#%%
# ## Test if my ordering in the model is the right one

# fig, axs = plt.subplots(1, 3, figsize=(7, 3))
# img = make_image(Xt, Yt, img_size=(*resolution, 3))
# axs[0].imshow(img)
# axs[0].set_title("Target Set")

# img_fs = make_image(Xc, Yc, img_size=(*resolution, 3))
# axs[1].imshow(img_fs)
# axs[1].set_title("Context Set")
# plt.suptitle("Example target and context sets", y=1.05);

# Yt_new =  Yt       
# xs, ys = jnp.unravel_index(jnp.arange(H*W), (W, H))
# Xt_new = jnp.vstack((xs, ys)).T / jnp.array((W, H))

# ## Generate pixel values, where the intensity of the pixel is the proportional to the coordinate (0,0)->(0.,0.,0.) and (1,1)->(1.,1.,1.)
# pix = Xt_new[:,0:1]*Xt_new[:,1:2]
# Yt_prop = jnp.concatenate([pix, pix, pix], axis=-1)
# # img_new = make_image(Xt_new, Yt_new, img_size=(*resolution, 3))
# img_new = make_image(Xt_new, Yt_prop, img_size=(*resolution, 3))
# axs[2].imshow(img_new)
# axs[2].set_title("Reordered Target Set")


# # Yt_newnew = einops.rearrange(Yt_new, '(W H) C -> W H C', H=H)


xs, ys = jnp.unravel_index(jnp.arange(H*W), (W, H))
xys = jnp.vstack((xs, ys)).T / jnp.array((W, H))

#%%


#%%
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    os.system(f"cp nohup.log {run_folder}")
