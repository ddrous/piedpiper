#%%[markdown]
# # ConvCNP mixed with CAVIA for Video Dataset
# - Each frame is treated as a separate environment
# - The ConvCNP frame encoding is concatenated with the context (learned through CAVIA)... This should learn neighboring information via gradient descent
# - If this works then we can replace CAVIA with NCF

#%%
# %load_ext autoreload
# %autoreload 2

from selfmod import NumpyLoader, make_run_folder, setup_run_folder, count_params
from piedpiper import *
# jax.config.update("jax_debug_nans", True)

## For reproducibility
seed = 2023

## Set JAX, numpy and torch seeds
mother_key = jax.random.PRNGKey(seed)
np.random.seed(seed)
torch.manual_seed(seed)

## Dataloader hps
resolution = (128, 128)
k_shots = int(np.prod(resolution) * 0.1)
T, H, W, C = (3**2, resolution[1], resolution[0], 3)
print("==== Main properties of the dataset ====")
print(" Number of context shots:", k_shots)
print(" Number of frames:", T)
print(" Resolution:", resolution)
print("===================================")

shuffle = True
num_workers = 24
latent_chans = 32

video_dataset = 'vox2'
envs_batch_size = num_workers if video_dataset=='vox2' else 1
envs_batch_size_all = envs_batch_size

init_lr = 1e-4
nb_epochs = 300
transition_steps = 10
print_every = 1
validate_every = 10
eps = 1e-6  ## Small value to avoid division by zero

meta_train = True
data_folder="./data/" if meta_train else "../../data/"
run_folder = None if meta_train else "./"
# run_folder = "./runs/241108-213626-Test/" if meta_train else "./"


#%%

if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)

_ = setup_run_folder(run_folder, os.path.basename(__file__))
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


class CAVIAConvCNP(eqx.Module):
    """ The model is a sequence of ConvCNPs """
    conv_cnp: eqx.Module

    def __init__(self, latent_chans, key=None):
        self.conv_cnp = ConvCNP(C, H, W, latent_chans, epsilon=eps, key=key)

    # def align_inputs(self, XY):
    #     """ Align the inputs for appropriate processing by the model """
    #     img, mask = self.conv_cnps.preprocess(XY)
    #     return img.transpose(1, 2, 0), mask.transpose(1, 2, 0)      ## img shape: (H, W, C)

    def align_labels(self, XY):
        """ Align the labels to the model's output format - usefull in the loss function """
        img, mask = self.conv_cnp.preprocess(XY)   # c h w
        # return img.transpose(1, 2, 0), mask.transpose(1, 2, 0)      ## img shape: (H, W, C)
        img = einops.rearrange(img, 'c h w -> w h c')
        mask = einops.rearrange(mask, 'c h w -> w h c')
        return img, mask

    def __call__(self, ctx_data):
        ## The context data given is in shape (T, H*W, 2+C)
        (mus, sigmas) = eqx.filter_vmap(self.conv_cnp)(ctx_data)    ## mus, sigmas shape: (T, H, W, C)

        ctx_vids = eqx.filter_vmap(self.align_labels)(ctx_data)

        return (mus, sigmas), ctx_vids


def loss_fn(model, batch):
    ctx_data, tgt_data = batch
    # Xc shape: (B, T, K, 2), Yc shape: (B, T, K, C), Yt shape: (B, T, 1024, C)

    ys, _ = eqx.filter_vmap(eqx.filter_vmap(model.align_labels))(tgt_data)  # ys shape: (B, T, H, W, C)

    (mus, sigmas), _ = eqx.filter_vmap(model)(ctx_data)                     ## mu, sigma shape: (B, T, H, W, C)

    losses = neg_log_likelihood(mus, sigmas, ys)
    # losses = neg_log_likelihood(mus[:,0], sigmas[:,0], ys[:,0])
    return losses.mean()

model = CAVIAConvCNP(latent_chans=latent_chans, key=model_key)
learner = Learner(model, loss_fn, images=False)

print(f"Number of learnable parameters in the model: {count_params(model)/1000:3.1f} k")
print("====================================================\n")

## Define optimiser and train the model
sched = optax.exponential_decay(init_lr, transition_steps=transition_steps, decay_rate=0.99)
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
                      seed=time.time_ns()%1000000)
test_dataloader = NumpyLoader(test_dataset, 
                               batch_size=envs_batch_size, 
                               shuffle=True, 
                               num_workers=num_workers)

#%%
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
