
#%%
%load_ext autoreload
%autoreload 2

from selfmod import NumpyLoader, make_image, make_run_folder, setup_run_folder, count_params
from piedpiper import *

## For reproducibility
seed = 2026

## Dataloader hps
resolution = (128, 128)
k_shots = int(np.prod(resolution) * 0.1)
T, H, W, C = (20, *resolution, 3)

data_folder="../../data/"
shuffle = True
num_workers = 24
latent_chans = 32

envs_batch_size = 12
envs_batch_size_all = envs_batch_size
num_batches = 82//12

init_lr = 5e-5
nb_epochs = 10000
print_every = 500
validate_every = 100
sched_factor = 1.0
eps = 1e-6  ## Small value to avoid division by zero

run_folder = None
# run_folder = "./"

meta_train = True


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

train_dataset = VideoDataset(data_folder, 
                      data_split="train", 
                      num_shots=k_shots, 
                      num_frames=T, 
                      resolution=resolution, 
                      order_pixels=False, 
                      max_envs=envs_batch_size_all*num_batches,
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

class Cell(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    positivity: callable
    mem_gate: eqx.nn.Conv2d

    def __init__(self, C, H, W, latent_chans, key=None):
        super().__init__()
        ## From the ConvCNP paper, Figure 1c
        keys = jax.random.split(key, 3)
        self.encoder = Encoder(C, H, W, key=keys[0])    ## E
        self.decoder = Decoder(C, H, W, in_chans=C, latent_chans=latent_chans, key=keys[1])    ## rho
        # self.positivity = lambda x: jax.nn.softplus(x)
        self.positivity = lambda x: jnp.clip(jax.nn.softplus(x), eps, 1)

        self.mem_gate = eqx.nn.Conv2d(2*C, C, 3, padding="same", key=keys[2])

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
        return img.transpose(1, 2, 0), mask.transpose(1, 2, 0)

    def postprocess(self, mu, sigma):
        mu = jnp.transpose(mu, (1, 2, 0))
        sigma = jnp.transpose(sigma, (1, 2, 0))
        return mu, sigma

    def __call__(self, Inp, Mem):
        Ic, Mc = self.preprocess(Inp)   ## Context pixels and their location
        hc = self.encoder(Ic, Mc)   ## Normalized convolution

        Mem = Mem.transpose(2, 0, 1)    ## Shape: (C, H, W)
        mem_h = self.mem_gate(jnp.concatenate([hc, Mem], axis=0))      ## TODO The key.

        ft = self.decoder(mem_h)
        mu, sigma = jnp.split(ft, 2, axis=0)
        sigma = self.positivity(sigma)

        mu, sigma = self.postprocess(mu, sigma)  ## Reshape into 2D arrays = (H, W, C)

        return (mu, sigma), mem_h.transpose(1, 2, 0)    ## Shape: (H, W, C)




class RConvCNP(eqx.Module):
    cell: eqx.Module
    img_utils: tuple
    num_shots: int
    bootstrap_mode: bool
    seed: int

    def __init__(self, C, H, W, latent_chans, num_shots, key=None):
        super().__init__()
        ## From the ConvCNP paper, Figure 1c
        keys = jax.random.split(key, 2)
        self.cell = Cell(C, H, W, latent_chans, key=keys[0])    ## E
        self.img_utils = (H, W, C)
        self.bootstrap_mode = True
        self.num_shots = num_shots
        self.seed = int(key[0])

    def sample_ctx(self, full_frame, sigma_frame, key):

        idxs = jnp.arange(full_frame.shape[0])

        sigma_frame = sigma_frame.transpose(1, 2, 0)
        probas = sigma_frame.reshape((full_frame.shape[0], -1)).mean(axis=1)    ## TODO mean to feel the outliers, which we want to sample from
        probas = probas / probas.sum()

        Xc = jax.random.choice(key=key, a=idxs, shape=(self.num_shots,), replace=False, p=probas)

        Yc = full_frame[Xc] ## or close
        return Yc

    def bootstrap_predict(self, full_vid):
        """ Predict the full video by bootstrapping based on the uncertainties """
        pixel_count = np.prod(self.img_utils)
        assert full_vid.shape[1] == pixel_count // 3

        hidden = jnp.zeros(self.img_utils)
        sigma_0 = jnp.ones(self.img_utils) / pixel_count
        key = jax.random.PRNGKey(self.seed)

        def f(carry, full_frame):
            sigma, mem, key = carry
            new_key, _ = jax.random.split(key) 
            ctx_frame = self.sample_ctx(full_frame, sigma, new_key)
            (new_mu, new_sigma), new_mem = self.cell(ctx_frame, mem)
            return (new_sigma, new_mem, new_key), (new_mu, new_sigma, ctx_frame)

        _, (mus, sigmas, ctx_vid) = jax.lax.scan(f, (sigma_0, hidden, key), full_vid)

        return (mus, sigmas), ctx_vid
    
    def naive_predict(self, ctx_vid):
        """ Predict the full video by only based on predifined context frame pixels. DECOMPRESS """
        assert ctx_vid.shape[1] == self.num_shots

        hidden = jnp.zeros(self.img_utils)

        def f(carry, ctx_frame):
            (new_mu, new_sigma), new_mem = self.cell(ctx_frame, carry)
            return new_mem, (new_mu, new_sigma)

        _, (mus, sigmas) = jax.lax.scan(f, hidden, ctx_vid)

        return (mus, sigmas), ctx_vid

    def __call__(self, vids):
        # if self.bootstrap_mode:
        #     ## TODO Video frames are assumed to be full (all locataions available and we can sample from) !!!
        #     # keys = jax.random.split(key, len(vids))
        #     return eqx.filter_vmap(self.bootstrap_predict)(vids)
        # else:
        #     return eqx.filter_vmap(self.naive_predict)(vids)

        return eqx.filter_vmap(self.naive_predict)(vids)

def loss_fn(model, batch):
    ctx_data, tgt_data = batch
    # Xc shape: (B, T, K, 2), Yc shape: (B, T, K, C), Yt shape: (B, T, 1024, C)

    ys, _ = eqx.filter_vmap(eqx.filter_vmap(model.cell.preprocess_channel_last))(tgt_data)  #ys shape: (B, T, H, W, C)

    # keys = jax.random.split(key, ctx_data.shape[0])
    (mus, sigmas), ctx_vids = eqx.filter_vmap(model.bootstrap_predict)(tgt_data)              ## mu, sigma shape: (B, T, H, W, C)

    losses = neg_log_likelihood(mus, sigmas, ys)
    return losses.mean()

model = RConvCNP(C, H, W, num_shots=k_shots, latent_chans=latent_chans, key=model_key)
learner = Learner(model, loss_fn, images=False)


## Define optimiser and train the model
total_steps = nb_epochs*train_dataloader.num_batches
bd_scales = {total_steps//3:sched_factor, 2*total_steps//3:sched_factor}
sched = optax.piecewise_constant_schedule(init_value=init_lr, boundaries_and_scales=bd_scales)
opt = optax.chain(optax.clip(1e-0), optax.adam(sched))

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

# print("Training losses:", jnp.stack(trainer.train_losses))

vt = VisualTester(trainer)
vt.visualize_losses(run_folder+"losses.png", log_scale=False, ylim=1)

test_dataset = VideoDataset(data_folder, 
                      data_split="test", 
                      num_shots=k_shots, 
                      num_frames=T, 
                      resolution=resolution, 
                      order_pixels=False, 
                      max_envs=envs_batch_size_all*1,
                      seed=seed)
test_dataloader = NumpyLoader(test_dataset, 
                               batch_size=envs_batch_size, 
                               shuffle=False, 
                               num_workers=num_workers)

# ctx_videos, tgt_videos = next(iter(test_dataloader))
# plt_idx = 0
# (pred_video, _), _ = learner.model.bootstrap_predict(tgt_videos[plt_idx])
# # (pred_video, _), _ = model.naive_predict(ctx_videos[plt_idx])

# print("Context shape:", pred_video.shape, ctx_videos[plt_idx].shape)

# vt.visualize_video_frames(tgt_videos[plt_idx], resolution, title="Target Set", save_path=run_folder+"target_set.png")
# vt.visualize_video_frames(ctx_videos[plt_idx], resolution, title="Context Set", save_path=run_folder+"context_set.png")
# vt.visualize_video_frames(pred_video, resolution, title="Prediction", save_path=run_folder+"prediction.png")

# plt.savefig(f"{run_folder}predictions.png")


vt.visualize_videos(test_dataloader, 
                    nb_envs=4, 
                    save_path=run_folder+"sample_predictions_nobt.png", 
                    bootstrap=False, 
                    save_video=True, 
                    video_prefix="sample_nobt")




#%%
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    os.system(f"cp nohup.log {run_folder}")
