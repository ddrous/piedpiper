#%%
# %load_ext autoreload
# %autoreload 2

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '.85'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
import jax.numpy as jnp
x = jnp.linspace(0, 1, 100)
# print(x)

from selfmod import NumpyLoader, make_image, make_run_folder, setup_run_folder, count_params
from piedpiper import *
# import jax_dataloader as jdl

## For reproducibility
seed = 2026

## Dataloader hps
resolution = (128, 128)
# resolution = (32, 32)
k_shots = int(np.prod(resolution) * 0.03)
H, W, C = (*resolution, 3)
T = 9

# data_folder="../../../Self-Mod/examples/celeb-a/data/"
data_folder="../../data/"
shuffle = False
num_workers = 24
latent_chans = 32

# envs_batch_size = 1627//10
# envs_batch_size_all = envs_batch_size
# num_batches = 10*100
envs_batch_size = 32
envs_batch_size_all = envs_batch_size
num_batches = 10*100

video_dataset = 'vox2'

init_lr = 1e-4
sched_factor = 0.2
nb_epochs = 1000
print_every = 1
validate_every = 1
eps = 1e-6  ## Small value to avoid division by zero

meta_train = False
run_folder = "./"
# run_folder = None


#%%

if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)

if meta_train == False:
    _ = setup_run_folder(run_folder, os.path.basename(__file__))

# os.listdir(data_folder)

#%% 

mother_key = jax.random.PRNGKey(seed)
data_key, model_key, trainer_key, test_key = jax.random.split(mother_key, num=4)

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




#%%


class ConvCNP(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    positivity: callable

    def __init__(self, latent_chans, key=None):
        super().__init__()
        ## From the ConvCNP paper, Figure 1c
        keys = jax.random.split(key, 2)
        self.encoder = Encoder(C, H, W, key=keys[0])    ## E
        self.decoder = Decoder(C, H, W, in_chans=C, latent_chans=latent_chans, key=keys[1])    ## rho
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
        return img.transpose(1, 2, 0), mask.transpose(1, 2, 0)

    def align_labels(self, XY):
        return self.preprocess_channel_last(XY)

    def postprocess(self, mu, sigma):
        mu = jnp.transpose(mu, (1, 2, 0))
        sigma = jnp.transpose(sigma, (1, 2, 0))
        return mu, sigma

    def __call__(self, ctx_imgs):

        def predict(ctx_img):
            Ic, Mc = self.preprocess(ctx_img)   ## Context pixels and their location
            hc = self.encoder(Ic, Mc)           ## Normalized convolution

            ft = self.decoder(hc)
            mu, sigma = jnp.split(ft, 2, axis=0)
            # jax.debug.print("sigma is {}", sigma)
            sigma = self.positivity(sigma)

            mu, sigma = self.postprocess(mu, sigma)  ## Reshape into 2D arrays = (H, W, C)

            return mu, sigma    ## Shape: (H, W, C)

        return eqx.filter_vmap(predict)(ctx_imgs), None



def loss_fn(model, batch):
    ctx_data, tgt_data = batch
    # Xc shape: (B, T, K, 2), Yc shape: (B, T, K, C), Yt shape: (B, T, W*H, C)

    ys, _ = eqx.filter_vmap(eqx.filter_vmap(model.preprocess_channel_last))(tgt_data)

    (mus, sigmas), _ = eqx.filter_vmap(model)(ctx_data)    ## mu, sigma shape: (B, T, H, W, C)

    ## Let's only pick the first and last frames  TODO: Randomly pick frames
    idx = jnp.array([0, -1])
    mus, sigmas, ys = mus[:, idx], sigmas[:, idx], ys[:, idx]

    losses = neg_log_likelihood(mus, sigmas, ys)
    # return losses.sum(axis=(1, 2)).mean()
    return losses.mean()

## Define the learner
model = ConvCNP(latent_chans=latent_chans, key=model_key)
learner = Learner(model, loss_fn, images=False)

print(f"Number of learnable parameters in the model: {count_params(model)/1000:3.1f} k")
print("====================================================\n")

#%%
## Define optimiser and train the model
# total_steps = nb_epochs*train_dataloader.num_batches
total_steps = nb_epochs*num_batches
bd_scales = {total_steps//3:sched_factor, 2*total_steps//3:sched_factor}
sched = optax.piecewise_constant_schedule(init_value=init_lr, boundaries_and_scales=bd_scales)
opt = optax.chain(optax.clip(1e+2), optax.adam(sched))


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
                        val_criterion="MSE",
                        validate_every=validate_every,
                        )
else:
    trainer.restore_trainer(run_folder)


#%%

vt = VisualTester(trainer)
vt.visualize_losses(run_folder+"losses.png", log_scale=False, ylim=1.1, ylim_val=0.1)

#%%
test_dataset = VideoDataset(data_folder, 
                      data_split="train", 
                      num_shots=k_shots, 
                      num_frames=T, 
                      resolution=resolution, 
                      order_pixels=False, 
                      max_envs=envs_batch_size_all*1,
                      dataset=video_dataset,
                      seed=time.time_ns()%1000)
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
