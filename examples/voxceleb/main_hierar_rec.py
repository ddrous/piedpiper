#%%[markdown]
# # Hirarchical Encocoder and ConvCNPs for Video Dataset
# - We encode the video using a ConvCNP-like encoder
# - At ech hierachy, we have an encoder and a mask (using min max)
# - During decoding, we concatenate the video encoding with the frame context


#%%
# %load_ext autoreload
# %autoreload 2

from selfmod import NumpyLoader, make_run_folder, setup_run_folder, count_params
from piedpiper import *
# jax.config.update("jax_debug_nans", True)

## For reproducibility
seed = 2028

## Dataloader hps
resolution = (32, 32)
k_shots = int(np.prod(resolution) * 0.1)
T, H, W, C = (3**2, resolution[1], resolution[0], 3)

print("==== Main properties of the dataset ====")
print(" Number of context shots:", k_shots)
print(" Number of frames:", T)
print(" Resolution:", resolution)
print("===================================")

data_folder="./data/"
shuffle = False
num_workers = 24
latent_chans = 32

video_dataset = 'vox2'
envs_batch_size = 41 if video_dataset=='vox2' else 1
envs_batch_size_all = envs_batch_size
num_batches = 82//41 if video_dataset=='vox2' else 1

init_lr = 3e-4
nb_epochs = 10
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
                      dataset=video_dataset,
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
    img_shape: tuple
    vicinity: int
    nb_levels: int
    level_chans: dict

    # level_preencoders: dict
    level_encoders: dict
    frame_encoder: eqx.Module
    frame_decoder: eqx.Module

    positivity: callable

    def __init__(self, C, H, W, latent_chans, epsilon=1e-6, key=None):
        keys = jax.random.split(key, 3)
        self.img_shape = (C, H, W)

        self.vicinity = 3   ## The factor by which each level is contracted
        self.nb_levels = int(np.log(T) / np.log(self.vicinity))
        # self.level_chans = {l: C*(0+2) for l in range(self.nb_levels+1)}
        self.level_chans = {0: C}
        for l in range(1, self.nb_levels+1):
            # self.level_chans[l] = C*(0+1)
            self.level_chans[l] = 6

        print("===== Some of the atributes of the model ===")
        print("Number of levels:", self.nb_levels)
        print("Levels in/out channels:", self.level_chans)
        print("===========================================")

        ## Level 1 Encoder - Input sequence (T, C0*3, H, W) - Output (T/3, C1, H, W)
        ## --
        ## Level l Encoder - Input sequence (3, Cl-1*3, H, W) - Output (1, Cl, H, W)
        # self.level_preencoders = {}     ## Pre-encoder for each level - they maintain the chanels
        self.level_encoders = {}
        enc_keys = jax.random.split(keys[0], self.nb_levels)
        for l in range(1, self.nb_levels+1):
            # preenc = SimpleEncoder(in_chans=self.level_chans[l-1], out_chans=self.level_chans[l-1], kernel_size=3, key=enc_keys[l])
            # self.level_preencoders[l-1] = preenc

            enc = SimpleEncoder(in_chans=self.level_chans[l-1]*self.vicinity, out_chans=self.level_chans[l], kernel_size=3, key=enc_keys[l])
            self.level_encoders[l-1] = enc

        self.frame_encoder = SimpleEncoder(in_chans=C, out_chans=C, kernel_size=3, key=keys[1])         ## Single frame encoder TODO: it's output is a bottleneck. Increase it !?
        # self.frame_decoder = Decoder(C, H, W, in_chans=C, latent_chans=latent_chans, key=keys[2])       ## Single frame decoder
        self.frame_decoder = Decoder(C, H, W, in_chans=C+self.level_chans[self.nb_levels], latent_chans=latent_chans, key=keys[2])       ## Single frame decoder
        self.positivity = lambda x: jnp.clip(jax.nn.softplus(x), epsilon, 1)

    def preprocess(self, XY):
        """ Preprocess a single context frame into the appropriate format """
        C, H, W = self.img_shape
        X, Y = XY[..., :2], XY[..., 2:]
        img = jnp.zeros((W, H, C))
        mask = jnp.zeros((W, H, 1))
        i_locs = (X[:, 0] * W).astype(int)  ## Because the XY dataset is in W,H format
        j_locs = (X[:, 1] * H).astype(int)
        img = img.at[i_locs, j_locs, :].set(jnp.clip(Y, 0., 1.))
        mask = mask.at[i_locs, j_locs, :].set(1.)
    
        img = einops.rearrange(img, 'w h c -> c h w')
        mask = einops.rearrange(mask, 'w h c -> c h w')

        return img, mask    ## Shapes: (C, H, W), (1, H, W)

    def align_labels(self, XY):
        """ Align the labels to the model's output format - usefull in the loss function """
        img, mask = self.preprocess(XY)   # c h w
        # return img.transpose(1, 2, 0), mask.transpose(1, 2, 0)      ## img shape: (H, W, C)
        img = einops.rearrange(img, 'c h w -> w h c')
        mask = einops.rearrange(mask, 'c h w -> w h c')
        return img, mask

    def __call__(self, ctx_data):

        #### Let's construct the single embedding for the video ####

        ## 1. The Inp is the pixels and their location: shape (T, H*W, 2+C)
        Ic, Mc = eqx.filter_vmap(self.preprocess)(ctx_data)   ## Ic shape: (T, C, H, W), Mc shape: (T, 1, H, W)

        ## 2. Let's define the level transition funtion: takes input (Tl-1, Cl-1, H, W) and returns (Tl-1//3, Cl, H, W)
        # @eqx.filter_vmap(in_axes=(None, None, 0, 0))
        # def level_transition(preencoder, encoder, I, M):
        #     ## I shape: (V, Cl, H, W), M shape: (V, 1, H, W)
        #     H = eqx.filter_vmap(preencoder)(I, M)   ## H shape: (V, Cl-1, H, W)

        #     enc_in = einops.rearrange(H, 'v c h w -> (v c) h w') # enc_in shape (V*Cl-1, H, W); v for vicinity
        #     enc_mask = jnp.max(M, axis=0)   ## Mask shape: (1, H, W)
        #     enc_out = encoder(enc_in, enc_mask)  ## enc_out shape: (Cl, H, W)

        #     return enc_out, enc_mask

        @eqx.filter_vmap(in_axes=(None, 0, 0))
        def level_transition(l, I, M):
            # preencoder = self.level_preencoders[l]
            encoder = self.level_encoders[l]

            ## I shape: (V, Cl-1, H, W), M shape: (V, 1, H, W)
            # H = eqx.filter_vmap(preencoder)(I, M)   ## H shape: (V, Cl-1, H, W)
            H = I

            enc_in = einops.rearrange(H, 'v c h w -> (v c) h w') # enc_in shape (V*Cl-1, H, W); v for vicinity
            enc_mask = jnp.max(M, axis=0)   ## Mask shape: (1, H, W)
            enc_out = encoder(enc_in, enc_mask)  ## enc_out shape: (Cl, H, W)

            return enc_out, enc_mask


        ## 3. Let's do the level transitions
        current_seq = Ic
        current_mask = Mc
        for l in range(self.nb_levels):
            next_in_seq = einops.rearrange(current_seq, '(tnext vicinity) c h w -> tnext vicinity c h w', vicinity=self.vicinity)
            next_in_mask = einops.rearrange(current_mask, '(tnext vicinity) c h w -> tnext vicinity c h w', vicinity=self.vicinity)
            # current_seq, current_mask = level_transition(self.level_preencoders[l], self.level_encoders[l], next_in_seq, next_in_mask)
            current_seq, current_mask = level_transition(l, next_in_seq, next_in_mask)

        ## Current seq is of shape (1, Cl, H, W) and current mask is of shape (1, 1, H, W)
        current_seq = current_seq.squeeze(0)
        current_mask = current_mask.squeeze(0)

        # jax.debug.print("Current final mask is exactly: {}", current_mask)

        #### Let's combine the single video embedding with frames and let's decode ####
        @eqx.filter_vmap
        def frame_encdec(In, Mask):
            # frame_input = jnp.concatenate([In, current_seq], axis=0)  ## Shape: (C+Cl, H, W)
            # frame_mask = jnp.max(jnp.stack([Mask, current_mask], axis=0), axis=0)  ## Shape: (1, H, W)
            # hc = self.frame_encoder(frame_input, frame_mask)       ## Normalized convolution

            hc = self.frame_encoder(In, Mask)       ## Normalized convolution
            hc = jnp.concatenate([hc, current_seq], axis=0)

            ft = self.frame_decoder(hc)
            mu, sigma = jnp.split(ft, 2, axis=0)
            sigma = self.positivity(sigma)

            mu = einops.rearrange(mu, 'c h w -> w h c')
            sigma = einops.rearrange(sigma, 'c h w -> w h c')

            return mu, sigma  ## Shape: (W, H, C)

        ctx_vid = eqx.filter_vmap(self.align_labels)(ctx_data)
        return frame_encdec(Ic, Mc), ctx_vid


def loss_fn(model, batch):
    ctx_data, tgt_data = batch
    # Xc shape: (B, T, K, 2), Yc shape: (B, T, K, C), Yt shape: (B, T, 1024, C)

    ys, _ = eqx.filter_vmap(eqx.filter_vmap(model.align_labels))(tgt_data)  # ys shape: (B, T, H, W, C)

    (mus, sigmas), _ = eqx.filter_vmap(model)(ctx_data)                     ## mu, sigma shape: (B, T, H, W, C)

    losses = neg_log_likelihood(mus, sigmas, ys)
    return losses.mean()

model = Model(C, H, W, latent_chans=latent_chans, key=model_key)
learner = Learner(model, loss_fn, images=False)

print(f"Number of learnable parameters in the model: {count_params(model)/1000:3.1f} k")

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
                      dataset=video_dataset,
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
