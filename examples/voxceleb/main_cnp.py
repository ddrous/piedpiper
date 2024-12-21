#%%[markdown]
# # CNPs for Video Dataset
# - The encoders and decoders of the convCNP take time as input, as well as coordinates
# - We want to take the mean representation, across spatial dimensions and across frames
# - This will later be used for ConvCNP


#%%
# %load_ext autoreload
# %autoreload 2

from selfmod import NumpyLoader, make_run_folder, setup_run_folder, count_params
from piedpiper import *
# jax.config.update("jax_debug_nans", True)

## For reproducibility
seed = 2026

## Dataloader hps
resolution = (32, 24)
k_shots = int(np.prod(resolution) * 0.1)
T, H, W, C = (3**2, resolution[1], resolution[0], 3)

print("==== Main properties of the dataset ====")
print(" Number of context shots:", k_shots)
print(" Number of frames:", T)
print(" Resolution:", resolution)
print("===================================")

shuffle = True
num_workers = 24
latent_size = 64*2
hidden_mlp_size = 64

video_dataset = 'vox2'
envs_batch_size = num_workers if video_dataset=='vox2' else 1
envs_batch_size_all = envs_batch_size

init_lr = 3e-4
nb_epochs = 10
print_every = 1
validate_every = 1
eps = 1e-6  ## Small value to avoid division by zero

meta_train = True
data_folder="./data/" if meta_train else "../../data/"
# run_folder = None if meta_train else "./"
run_folder = "./runs/241108-213626-Test/" if meta_train else "./"


#%%

if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)

_ = setup_run_folder(run_folder, os.path.basename(__file__))
mother_key = jax.random.PRNGKey(seed)
data_key, model_key, trainer_key, test_key = jax.random.split(mother_key, num=4)


#%%

## List every dir in the data folder

# train_dataset = VideoDataset(data_folder, 
#                       data_split="train", 
#                       num_shots=k_shots, 
#                       num_frames=T, 
#                       resolution=resolution, 
#                       order_pixels=False, 
#                       max_envs=-1,
#                       dataset=video_dataset,
#                       seed=seed)

train_dataset = PreprocessedVideoDataset(
                      data_path=data_folder+"preprocessed/", 
                      num_shots=k_shots, 
                      num_frames=T, 
                      resolution=resolution, 
                      seed=seed)

# print("Total number of environments in the training dataset:", len(train_dataset))
train_dataloader = NumpyLoader(train_dataset, 
                               batch_size=envs_batch_size, 
                               shuffle=shuffle, 
                               num_workers=num_workers,
                               drop_last=False)

ctx_videos, tgt_videos = next(iter(train_dataloader))
vt = VisualTester(None)
vt.visualize_video_frames(tgt_videos[0], resolution)
vt.visualize_video_frames(ctx_videos[0], resolution)


#%%


class CNP(eqx.Module):
    """ The model is a sequence of ConvCNPs """
    img_shape: tuple

    encoder: eqx.Module
    decoder: eqx.Module

    positivity: callable

    def __init__(self, latent_size, epsilon=1e-6, key=None):
        keys = jax.random.split(key, 3)
        self.img_shape = (C, H, W)
        self.positivity = lambda x: jnp.clip(jax.nn.softplus(x), epsilon, 1)

        self.encoder = eqx.nn.MLP(in_size=3+3,   ## 3 for (t, x, y), 3 for RGB
                                  out_size=latent_size,
                                  width_size=hidden_mlp_size,
                                  depth=3,
                                  activation=jax.nn.softplus,
                                  key=keys[0])

        self.decoder = eqx.nn.MLP(in_size=latent_size+3,        ## 3 for (t, x, y)
                                  out_size=3*2,                 ## RGB means and stds
                                  width_size=hidden_mlp_size,
                                  depth=5,
                                  activation=jax.nn.softplus,
                                  key=keys[0])

    def preprocess(self, XY):
        """ Preprocess a single context frame into the appropriate format """
        C, H, W = self.img_shape
        X, Y = XY[..., :2], XY[..., 2:]
        img = jnp.zeros((W, H, Y.shape[1]))
        mask = jnp.zeros((W, H, 1))
        i_locs = (X[:, 0] * W).astype(int)  ## Because the XY dataset is in W,H format
        j_locs = (X[:, 1] * H).astype(int)
        img = img.at[i_locs, j_locs, :].set(jnp.clip(Y, 0., 1.))
        mask = mask.at[i_locs, j_locs, :].set(1.)
    
        img = einops.rearrange(img, 'w h c -> c h w')
        mask = einops.rearrange(mask, 'w h c -> c h w')

        return img, mask    ## Shapes: (C, H, W), (1, H, W)

    def prepend_time(self, XY, t):
        """ Preprocess a single context frame by appending the time """
        t = jnp.ones_like(XY[..., :1]) * t
        XY = jnp.concatenate([t, XY], axis=-1)
        return XY    ## Shapes: (H*W, 6)

    def align_labels(self, XY):
        """ Align the labels to the model's output format - usefull in the loss function """
        img, mask = self.preprocess(XY)   # c h w
        # return img.transpose(1, 2, 0), mask.transpose(1, 2, 0)      ## img shape: (H, W, C)
        img = einops.rearrange(img, 'c h w -> w h c')
        mask = einops.rearrange(mask, 'c h w -> w h c')
        return img, mask

    def __call__(self, ctx_data):
        ts = jnp.linspace(0, 1, T)
        txy_ctx = eqx.filter_vmap(self.prepend_time)(ctx_data, ts)      ## XY_ctx shape: (T, H*W, 6)

        ## Encode the context frames
        zs = eqx.filter_vmap(eqx.filter_vmap(self.encoder))(txy_ctx)      ## zs shape: (T, H*W, latent_size)

        ## Aggretate the latent representations by taking the mean along the first two dimensions
        z = zs.mean(axis=(0, 1))                                         ## z shape: (latent_size,)

        # ## Define the decoding function
        # def decoding_fn(x, z):
        #     xz = jnp.concatenate([z, x], axis=-1)
        #     return self.decoder(xz)

        ## Define the decoding function
        def decoding_fn(t, xy, z):
            txyz = jnp.concatenate([z, t, xy], axis=-1)
            return self.decoder(txyz)

        ### Brute force strategy 1 ###
        # ## Create X_target (t, x, y) for each pixel in the target video
        # xs = jnp.linspace(0, 1, W)
        # ys = jnp.linspace(0, 1, H)
        # # txy_tgt = jnp.array(jnp.meshgrid(ts, xs, ys, indexing='ij')).transpose(1, 2, 3, 0).reshape(-1, 3)
        # txy_tgt = einops.rearrange(jnp.array(jnp.meshgrid(ts, xs, ys, indexing='ij')), 'c t w h -> (t w h) c')
        # out = eqx.filter_vmap(decoding_fn, in_axes=(0, None))(txy_tgt, z)                   ## out shape: (T*H*W, 6)
        # tgt_vid = einops.rearrange(out, '(t w h) c -> t w h c', t=T, h=H, w=W, c=3*2)
        # mus, sigmas = tgt_vid[..., :3], tgt_vid[..., 3:]
        # sigmas = self.positivity(sigmas)
        #===========================

        ### Brute force strategy 2 ###
        # txy_tgt = einops.rearrange(jnp.array(jnp.meshgrid(ts, xs, ys, indexing='ij')), 'c t w h -> t (w h) c')                  ## Shape: (T, H*W, 3)
        # out = eqx.filter_vmap(eqx.filter_vmap(decoding_fn, in_axes=(0, None)), in_axes=(0, None))(txy_tgt, z)                   ## out shape: (T, H*W, 6)
        # XY_tRGB = jnp.concatenate([txy_tgt[:, :, jnp.array([1,2,0])], out], axis=-1)      ## XY_tRGB shape: (T, H*W, 9), with time at the third position 
        # tgt_vid, _ = eqx.filter_vmap(self.align_labels)(XY_tRGB)      ## tgt_vid shape: (T, W, H, 1+3*2)
        # mus, sigmas = tgt_vid[..., 1:4], tgt_vid[..., 4:]
        # sigmas = self.positivity(sigmas)
        #===========================

        ### Strategy 3 ###
        xs, ys = jnp.unravel_index(jnp.arange(H*W), (W, H))
        xys = jnp.vstack((xs, ys)).T / jnp.array((W, H))        ## Shape: (H*W, 2)
        out = eqx.filter_vmap(eqx.filter_vmap(decoding_fn, in_axes=(None, 0, None)), in_axes=(0, None, None))(ts[:, None], xys, z)                   ## out shape: (T, H*W, 6)
        mus, sigmas = out[..., :3], out[..., 3:]

        # xys_long = jnp.tile(xys[None, ...], (T, 1, 1))      ## Shape: (T, H*W, 2)
        # mus = eqx.filter_vmap(self.align_labels)(jnp.concatenate([xys_long, mus], axis=-1))
        # sigmas = eqx.filter_vmap(self.align_labels)(jnp.concatenate([xys_long, sigmas], axis=-1))

        mus = einops.rearrange(mus, 't (w h) c -> t w h c', h=H)
        sigmas = einops.rearrange(sigmas, 't (w h) c -> t w h c', h=H)

        sigmas = self.positivity(sigmas)
        #===========================

        ctx_vid = eqx.filter_vmap(self.align_labels)(ctx_data)

        return (mus, sigmas), ctx_vid


def loss_fn(model, batch):
    ctx_data, tgt_data = batch
    # Xc shape: (B, T, K, 2), Yc shape: (B, T, K, C), Yt shape: (B, T, 1024, C)

    ys, _ = eqx.filter_vmap(eqx.filter_vmap(model.align_labels))(tgt_data)  # ys shape: (B, T, H, W, C)

    (mus, sigmas), _ = eqx.filter_vmap(model)(ctx_data)                     ## mu, sigma shape: (B, T, H, W, C)

    losses = neg_log_likelihood(mus, sigmas, ys)
    return losses.mean()

model = CNP(latent_size=latent_size, key=model_key)
learner = Learner(model, loss_fn, images=False)

print(f"Number of learnable parameters in the model: {count_params(model)/1000:3.1f} k")
print("====================================================\n")

## Define optimiser and train the model
sched = optax.exponential_decay(init_lr, transition_steps=10, decay_rate=0.90)
# opt = optax.chain(optax.clip(1e2), optax.adam(sched))
opt = optax.adam(sched)

trainer = Trainer(learner, opt)


## Training loop
if meta_train:
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


vt = VisualTester(trainer)
vt.visualize_losses(run_folder+"losses.png", log_scale=False, ylim=1.1)

test_dataset = VideoDataset(data_folder, 
                      data_split="train", 
                      num_shots=k_shots, 
                      num_frames=T, 
                      resolution=resolution, 
                      order_pixels=False, 
                      max_envs=envs_batch_size_all*1,
                      dataset=video_dataset,
                      seed=None)
test_dataloader = NumpyLoader(test_dataset, 
                               batch_size=envs_batch_size, 
                               shuffle=False, 
                               num_workers=num_workers)

vt.visualize_videos(test_dataloader, 
                    nb_envs=1, 
                    save_path=run_folder+"sample_predictions_nobt.png", 
                    bootstrap=False, 
                    save_video=True, 
                    video_prefix=run_folder+"sample_nobt")



#%%
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    os.system(f"cp nohup.log {run_folder}")
