#%%
# %load_ext autoreload
# %autoreload 2

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '.5'
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
k_shots = int(np.prod(resolution) * 0.03)
H, W, C = (*resolution, 3)

data_folder="../../../../../Self-Mod/examples/celeb-a/data/"
# data_folder="./data/"
shuffle = True
num_workers = 24
latent_chans = 32

envs_batch_size = 1627//10
envs_batch_size_all = envs_batch_size
num_batches = 10*100

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

# m_dataset = pygrain.MapDataset(train_dataset, lambda x: x)
# print("Number of records in the dataset:", len(m_dataset))
# # import orbax as ob
# # print("Orbax version:", ob.__version__)
# sampler = pygrain.IndexSampler(num_records=50,
#                                 num_epochs=20,
#                                 shard_options=pygrain.NoSharding(),
#                                 shuffle=shuffle,
#                                 seed=seed,
#                             )
# class DummyOp(pygrain.MapTransform):
#   """A dummy operations."""
#   def map(self, element):
#     return element
# train_dataloader = pygrain.DataLoader(data_source=train_dataset,
#                                     operations=[DummyOp()],
#                                     sampler=sampler,
#                                     worker_count=0,  # Scale to multiple workers in multiprocessing
#                                 )

train_dataloader = NumpyLoader(train_dataset, 
                              batch_size=envs_batch_size, 
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

# train_dataloader = jdl.DataLoader(train_dataset, 
#                               batch_size=envs_batch_size, 
#                               backend='pytorch',
#                               shuffle=shuffle,
#                               num_workers=num_workers,
#                               drop_last=False)

gen_train_dataloader = iter(train_dataloader)
batch = next(gen_train_dataloader)
print([dat.shape for dat in batch])

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

        return eqx.filter_vmap(predict)(ctx_imgs)


model = ConvCNP(latent_chans=latent_chans, key=model_key)

def loss_fn(model, batch):
    ctx_data, tgt_data = batch
    # Xc shape: (B, K, 2), Yc shape: (B, K, C), Yt shape: (B, 1024, C)

    ys, _ = eqx.filter_vmap(model.preprocess_channel_last)(tgt_data)

    mus, sigmas = model(ctx_data)    ## mu, sigma shape: (B, H, W, C)

    losses = neg_log_likelihood(mus, sigmas, ys)
    # return losses.sum(axis=(1, 2)).mean()
    return losses.mean()

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

# print("Training losses:", jnp.stack(trainer.train_losses))

visualtester = VisualTester(trainer)
# visualtester.visualize_losses(run_folder+"loss_curve.png", log_scale=False, ylim=1)


## parse the nohup.log file to collect the losses. e.g, a line is like below.
# 19:33:42      Epoch:  31      Loss: -0.00397326      Time/Epoch(s): 179.0587
losses = []
with open("nohup.log", "r") as f:
    for line in f:
        if "Loss:" in line:
            losses.append(float(line.split("Loss:")[1].split()[0]))
plt.plot(losses)
plt.ylim(-0.1, 2)
print(losses)


#%%
test_dataset = ImageDataset(data_folder, 
                            data_split="test",
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

visualtester.visualize_images(test_dataloader, nb_envs=None, key=None, save_path=run_folder+"predictions_test.png", interp_method="linear", plot_ids=range(18))


#%%
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    os.system(f"cp nohup.log {run_folder}")
