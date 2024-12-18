#%%[markdown]
# # Intependent ConvCNPs for Video Dataset
# - Each frame is treated as a separate environment with its own ConvCNP
# - The model is a sequence of ConvCNPs (with independent weights). but parllelised with Equinox
# - Next step is to condition the weights on the contexts


#%%
# %load_ext autoreload
# %autoreload 2

from selfmod import NumpyLoader, make_image, make_run_folder, setup_run_folder, count_params
from piedpiper import *
from functools import partial
# jax.config.update("jax_debug_nans", True)

## For reproducibility
seed = 2024

## Dataloader hps
resolution = (32, 32)
k_shots = int(np.prod(resolution) * 0.1)
T, H, W, C = (10, resolution[1], resolution[0], 3)
c, h, w = (3, 4, 4)

print("==== Main properties of the dataset ====")
print(" Number of context shots:", k_shots)
print(" Number of frames:", T)
print(" Resolution:", resolution)
print("===================================")

data_folder="./data/"
shuffle = False
num_workers = 24
latent_chans = 32

envs_batch_size = 41
envs_batch_size_all = envs_batch_size
num_batches = 82//41

init_lr = 3e-4
nb_epochs = 500*2
print_every = 10
validate_every = 10
sched_factor = 1.0
eps = 1e-6  ## Small value to avoid division by zero

run_folder = None
# run_folder = "./runs/241108-213626-Test/"

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
                      dataset="vox2",
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

## Batch-eval a model
@eqx.filter_vmap(in_axes=(eqx.if_array(0), 0))
def batch_eval(model, x):
    return model(x)

class Model(eqx.Module):
    """ The model is a sequence of ConvCNPs """
    conv_cnps: eqx.Module

    def __init__(self, C, H, W, c, h, w, latent_chans, key=None):
        keys = jax.random.split(key, T)

        def make_conv_cnp(key):
            return ConvCNP(C, H, W, latent_chans, epsilon=eps, key=key)

        self.conv_cnps = eqx.filter_vmap(make_conv_cnp)(keys)

    # def align_inputs(self, XY):
    #     """ Align the inputs for appropriate processing by the model """
    #     img, mask = self.conv_cnps.preprocess(XY)
    #     return img.transpose(1, 2, 0), mask.transpose(1, 2, 0)      ## img shape: (H, W, C)

    def align_labels(self, XY):
        """ Align the labels to the model's output format - usefull in the loss function """
        img, mask = self.conv_cnps.preprocess(XY)   # c h w
        # return img.transpose(1, 2, 0), mask.transpose(1, 2, 0)      ## img shape: (H, W, C)
        img = einops.rearrange(img, 'c h w -> w h c')
        mask = einops.rearrange(mask, 'c h w -> w h c')
        return img, mask

    def __call__(self, ctx_data):
        ## The context data given is in shape (T, H*W, 2+C)
        (mus, sigmas) = batch_eval(self.conv_cnps, ctx_data)    ## mus, sigmas shape: (T, H, W, C)

        ctx_vids = eqx.filter_vmap(self.align_labels)(ctx_data)

        return (mus, sigmas), ctx_vids

def loss_fn(model, batch):
    ctx_data, tgt_data = batch
    # Xc shape: (B, T, K, 2), Yc shape: (B, T, K, C), Yt shape: (B, T, 1024, C)

    ys, _ = eqx.filter_vmap(eqx.filter_vmap(model.align_labels))(tgt_data)  # ys shape: (B, T, H, W, C)

    (mus, sigmas), _ = eqx.filter_vmap(model)(ctx_data)              ## mu, sigma shape: (B, T, H, W, C)

    losses = neg_log_likelihood(mus, sigmas, ys)
    return losses.mean()

model = Model(C, H, W, c, h, w, latent_chans=latent_chans, key=model_key)
learner = Learner(model, loss_fn, images=False)


## Define optimiser and train the model
sched = optax.exponential_decay(init_lr, transition_steps=100, decay_rate=0.99)
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
                      dataset="vox2",
                      seed=10)
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
