
#%%
%load_ext autoreload
%autoreload 2

from selfmod import NumpyLoader, make_image, make_run_folder, setup_run_folder
from piedpiper import *

## For reproducibility
seed = 2026

## Dataloader hps
k_shots = 100
resolution = (32, 32)
T, H, W, C = (5, *resolution, 3)

data_folder="./data/"
shuffle = True
num_workers = 0
latent_chans = 32

envs_batch_size = 8
envs_batch_size_all = envs_batch_size
num_batches = 4*1

init_lr = 5e-5
nb_epochs = 20000
print_every = 500
validate_every = 1000
sched_factor = 1.0
eps = 1e-6  ## Small value to avoid division by zero

# run_folder = None
run_folder = "./runs/241023-110210-Test/"

meta_train = True


#%%

if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)

_ = setup_run_folder(run_folder, os.path.basename(__file__))



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
                               num_workers=num_workers)

ctx_videos, tgt_videos = next(iter(train_dataloader))
vt = VisualTester(None)
vt.visualize_video_frames(ctx_videos[0], resolution)
vt.visualize_video_frames(tgt_videos[0], resolution)

#%%

class Cell(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    positivity: callable
    mem_gate: eqx.nn.Conv2d

    def __init__(self, H, W, C, latent_chans=8, key=None):
        super().__init__()
        ## From the ConvCNP paper, Figure 1c
        keys = jax.random.split(key, 3)
        self.encoder = Encoder(C, H, W, key=keys[0])    ## E
        self.decoder = Decoder(C, H, W, in_chans=C, out_chans=latent_chans, key=keys[1])    ## rho
        # self.positivity = lambda x: jax.nn.softplus(x)
        self.positivity = lambda x: jnp.clip(jax.nn.softplus(x), eps, 1)

        self.mem_gate = eqx.nn.Conv2d(2*C, C, 3, padding="same", key=keys[2])

    def preprocess(self, X, Y):
        img = jnp.zeros((C, H, W))
        mask = jnp.zeros((1, H, W))
        i_locs = (X[:, 0] * H).astype(int)
        j_locs = (X[:, 1] * W).astype(int)
        img = img.at[:, i_locs, j_locs].set(jnp.clip(Y, 0., 1.).T)
        mask = mask.at[:, i_locs, j_locs].set(1.)
        return img, mask

    def postprocess(self, mu, sigma):
        mu = jnp.transpose(mu, (1, 2, 0))
        sigma = jnp.transpose(sigma, (1, 2, 0))
        return mu, sigma

    def __call__(self, Inp, Mem):
        X, Y = Inp

        Ic, Mc = self.preprocess(X, Y)   ## Context pixels and their location
        hc = self.encoder(Mc, Ic)   ## Normalized convolution

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

    def __init__(self, C, H, W, key=None):
        super().__init__()
        ## From the ConvCNP paper, Figure 1c
        keys = jax.random.split(key, 2)
        self.cell = Cell(C, H, W, key=keys[0])    ## E
        self.img_utils = (C, H, W)
        self.bootstrap_mode = True

    def sample_ctx(self, full_frame, sigma_frame, key):
        Xc = jnp.random.choice(full_frame, self.num_shots, replace=False, p=sigma_frame)
        Yc = full_frame[Xc] ## or close
        return Xc, Yc

    def bootstrap_predict(self, full_vid, key):
        """ Predict the full video by bootstrapping based on the uncertainties """
        assert full_vid.shape[1] = np.prod(self.img_utils)

        hidden = jnp.zeros(self.img_utils)
        sigma_0 = jnp.ones(self.img_utils) / jnp.prod(self.img_utils)

        def f(carry, full_frame):
            sigma, mem, key = carry
            new_key, _ = jax.random.split(key) 
            ctx_frame = self.sample_ctx(full_frame, sigma, new_key)
            (new_mu, new_sigma), new_mem = self.cell(ctx_frame, mem)
            return (new_mu, new_sigma, ctx_frame), (new_sigma, new_mem, new_key)

        _, (mus, sigmas, ctx_vid) = jax.lax.scan(f, (sigma_0, hidden, key), full_vid)

        return (mus, sigmas), ctx_vid
    
    def naive_predct(self, ctx_vid):
        """ Predict the full video by only based on predifined context frame pixels. DECOMPRESS """
        assert ctx_vid.shape[1] = self.num_shots

        hidden = jnp.zeros(self.img_utils)

        def f(carry, ctx_frame):
            (new_mu, new_sigma), new_mem = self.cell(ctx_frame, carry)
            return (new_mu, new_sigma), new_mem

        _, (mus, sigmas) = jax.lax.scan(f, hidden, ctx_vid)

        return (mus, sigmas), ctx_vid

    def __call__(self, vids, key):
        if self.bootstrap_mode:
            ## TODO Video frames are assumed to be full (all locataions available and we can sample from) !!!
            keys = jax.random.split(key, len(vids))
            return eqx.filter_vmap(self.bootstrap_predict)(vids, keys)
        else:
            return eqx.filter_vmap(self.naive_predict)(vids)
