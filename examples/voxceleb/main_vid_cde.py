
#%%
# %load_ext autoreload
# %autoreload 2

from selfmod import NumpyLoader, make_image, make_run_folder, setup_run_folder, count_params
from piedpiper import *
from functools import partial
# jax.config.update("jax_debug_nans", True)

## For reproducibility
seed = 2026

## Dataloader hps
resolution = (64, 48)
k_shots = int(np.prod(resolution) * 0.3)
T, H, W, C = (32, resolution[1], resolution[0], 3)
c, h, w = (3, 5, 5)     ## The window size for the vector field and control signal to interact !

data_folder="./data/"
shuffle = True
num_workers = 24
latent_chans = 32

envs_batch_size = 41
envs_batch_size_all = envs_batch_size
num_batches = 82//41

init_lr = 1e-3
nb_epochs = 5000
print_every = 10
validate_every = 100
sched_factor = 1.0
eps = 1e-6  ## Small value to avoid division by zero

run_folder = None
# run_folder = "./241108-213626-Test/"

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

class VectorField(eqx.Module):
    weights: eqx.Module
    img_shape: tuple
    window_shape: tuple       ## The windows to serach for our contraction

    def __init__(self, C, H, W, c, h, w, key=None):
        """  Let R = CxHxW be the size of the image. This modules would normally be R -> RxR as in the CDE case; 
        which implies that every pixel in the control signal influences every other in the memory y.
        Sore we prefer R -> Rxr, where r = cxhxw 
        For now, the weights are a CNN, but try a linear layer TODO
        """
        self.img_shape = (C, H, W)
        self.window_shape = (c, h, w)

        r = np.prod(self.window_shape)
        self.weights = VNet(input_shape=(2*C, H, W),    ## 2 because of the y is actually mu and sigma
                        output_shape=(2*C*(r + 1), H, W), 
                        levels=4, 
                        depth=8,
                        kernel_size=5,
                        activation=jax.nn.softplus,
                        final_activation=jax.nn.tanh,
                        batch_norm=False,
                        dropout_rate=0.,
                        key=key)

        # ## Weights is just a convolution
        # self.weights = eqx.nn.Conv2d(2*C, 2*C*(r+1), kernel_size=5, padding="SAME", key=key)

    # def __call__(self, y):
    #     return self.weights(y)

    def __call__(self, y_mu, y_sigma):
        r = np.prod(self.window_shape)

        y = jnp.concatenate([y_mu, y_sigma], axis=0)    ## y shape: (2*C, H, W)

        # y_conved = self.weights(y)        ## activation is already applied
        y_conved = jax.nn.tanh(self.weights(y))

        mu, sigma = jnp.split(y_conved, 2, axis=0)   ## mu of shape: (C*(r+1), H, W)

        ## Reshape mu and sigma to (r+1, C, H, W)
        mu = mu.reshape((r+1, C, H, W))
        sigma = sigma.reshape((r+1, C, H, W))

        return mu, sigma


class Control(eqx.Module):
    """ The signal dx/dt. Takes as input all ctx frames """
    # weights: eqx.Module
    img_shape : tuple
    window_shape : tuple

    def __init__(self, C, H, W, c, h, w, key=None):
        self.img_shape = (C, H, W)
        self.window_shape = (c, h, w)
        # self.weights = SimpleEncoder(in_chans=C, out_chans=C, key=key)

    def __call__(self, t, ts, xs, masks, weights):
        ## xs = ctx_frames of shape: (T, C, H, W) - although some are zeros
        ## ts = time points of shape: (T,)

        T, C, H, W = xs.shape

        ## Encode every frame independently with vmap - skip this and see the result
        # xs = eqx.filter_vmap(self.weights)(xs, masks)
        # xs = eqx.filter_vmap(weights)(xs, masks)

        # ## Add the time channel to the xs (CDE need these)
        # ts_3d = jnp.repeat(ts[:, None], H*W, axis=1).reshape((T, 1, H, W))
        # xs = jnp.concatenate([ts_3d, xs], axis=1)   ## xs shape: (T, C+1, H, W) - time is the first channel

        ## Put the time dimension last and channel first
        xs = xs.transpose(1, 2, 3, 0)

        # @eqx.filter_vmap(in_axes=(None, None, 0))
        @eqx.filter_vmap(in_axes=(None, 0))
        @eqx.filter_vmap(in_axes=(None, 0))
        @eqx.filter_vmap(in_axes=(None, 0))
        @partial(jax.grad)      # @eqx.filter_jacfwd ## @eqx.filter_grad
        def grad_interp_img(tau, xs_):
            # print("Shapes of ts, ys, t:", ts.shape, ys.shape, t.shape)
            # return jnp.interp(tau.squeeze(), ts, xs, left="extrapolate", right="extrapolate").squeeze()
            # tau = jnp.array([tau])
            # return jnp.interp(tau, ts, xs_, left="extrapolate", right="extrapolate").squeeze()
            return jnp.interp(tau, ts, xs_, left="extrapolate", right="extrapolate").squeeze()
    
        # t = jnp.array([t])
        grad_xs = grad_interp_img(t, xs).squeeze()        ## grad_xs shape: (C, H, W)

        # # Get change in time for later on - in the first channel
        delta_t = jax.grad(jnp.interp)(t, ts, ts, left="extrapolate", right="extrapolate")
        # delta_t = grad_xs[0, 0, 0]
        # grad_xs = grad_xs[1:]    ## grad_xs shape: (C, H, W)

        ## We do r convolutions of grad_xs, each with a different kernel: 1 at a specific location and 0 elsewhere. This just places the neighring pixels in a line, ready for my special product
        r = np.prod(self.window_shape)
        ones_hots = jax.nn.one_hot(jnp.arange(r), r)    ## shape: (r, r)
        @eqx.filter_vmap
        def convolve(kernel_1d):
            kernel_3d = kernel_1d.reshape(self.window_shape)
            return jax.scipy.signal.convolve(grad_xs, kernel_3d, mode="same")

        grad_xs = convolve(ones_hots)      ## shape: (r, C, H, W)

        ## Add time t to the grad_xs along the leading axis r
        delta_t = jnp.ones((1, C, H, W)) * delta_t              ## shape: (1, C, H, W)
        grad_xs = jnp.concatenate([delta_t, grad_xs], axis=0)    ## shape: (r+1, C, H, W)

        return grad_xs



class ODEFunc(eqx.Module):
    "Also called ControlTerm in Diffrax. --Special product-- of the vector field and the gradient of the control signal x"
    vector_field: eqx.Module
    control: eqx.Module

    def __init__(self, C, H, W, c, h, w, key=None):
        super().__init__()
        keys = jax.random.split(key, 2)
        self.vector_field = VectorField(C, H, W, c, h, w, key=keys[0])
        self.control = Control(C, H, W, c, h, w, key=keys[1])

    # def __call__(self, t, y, args):
    #     ts, xs, masks, control_encoder = args

    #     ## We take in all the xs, althgouh the future ones wont's be needed in the interpolation (most likely)- we can fix this
    #     ### We normally work with our data as (H, W, C). Let's make sure every input is channel first
    #     y = y.transpose(2, 0, 1)    ## y shape: (2*C, H, W)
    #     xs = xs.transpose(0, 3, 1, 2)    ## xs shape: (T, C, H, W)
    #     masks = masks.transpose(0, 3, 1, 2)    ## masks shape: (T, 1, H, W)

    #     mu, sigma = jnp.split(y, 2, axis=0)                     ## mu, sigma shape: (C, H, W)
    #     mu_big, sigma_big = self.vector_field(mu, sigma)        ## mu_big, sigma_big shape: (cxhxw+1, C, H, W)
    #     dx_dt_big = self.control(t, ts, xs, masks, control_encoder)                     ## dx_dt_big shape: (cxhxw+1, C, H, W)

    #     mu = jnp.sum(mu_big * dx_dt_big, axis=0)    ## mu shape: (C, H, W)
    #     sigma = jnp.sum(sigma_big * dx_dt_big, axis=0)    ## sigma shape: (C, H, W)

    #     next_y = jnp.concatenate([mu, sigma], axis=0)

    #     return next_y.transpose(1, 2, 0)    ## next_y shape: (H, W, 2*C)



    def __call__(self, t, y, args):
        """ Assume the data is in already channel first format """
        ts, xs, masks, control_encoder = args

        mu, sigma = jnp.split(y, 2, axis=0)                     ## mu, sigma shape: (C, H, W)
        mu_big, sigma_big = self.vector_field(mu, sigma)        ## mu_big, sigma_big shape: (cxhxw+1, C, H, W)
        dx_dt_big = self.control(t, ts, xs, masks, control_encoder)                     ## dx_dt_big shape: (cxhxw+1, C, H, W)

        mu = jnp.sum(mu_big * dx_dt_big, axis=0)    ## mu shape: (C, H, W)
        sigma = jnp.sum(sigma_big * dx_dt_big, axis=0)    ## sigma shape: (C, H, W)

        next_y = jnp.concatenate([mu, sigma], axis=0)

        return next_y    ## next_y shape: (2*C, H, W)




class Model(eqx.Module):
    """ The model is a ConvCNP with an ODETerm. The takes inputs in raw formar: (H*W, 5) and returns in channel last format """
    ode_func: eqx.Module
    init_cond: eqx.Module

    def __init__(self, C, H, W, c, h, w, latent_chans, key=None):
        keys = jax.random.split(key, 2)
        self.init_cond = ConvCNP(C, H, W, latent_chans, epsilon=eps, key=keys[0])
        self.ode_func = ODEFunc(C, H, W, c, h, w, key=keys[1])

    def align_inputs(self, XY):
        """ Align the inputs for appropriate processing by the model """
        img, mask = self.init_cond.preprocess(XY)
        return img.transpose(1, 2, 0), mask.transpose(1, 2, 0)      ## img shape: (H, W, C)

    def align_labels(self, XY):
        """ Align the labels to the model's output format - usefull in the loss function """
        img, mask = self.init_cond.preprocess(XY)   
        return img.transpose(2, 1, 0), mask.transpose(2, 1, 0)      ## img shape: (W, H, C)

    def __call__(self, ctx_data):
        ## Preprocess the context data given in shape (T, H*W, 2+C)
        xs, masks = eqx.filter_vmap(self.align_inputs)(ctx_data)    ## ctx_video shape: (T, H, W, C)  - TODO: we use the mask
        ts = jnp.linspace(0., 1., ctx_data.shape[0], endpoint=False)

        ## More efficient to preprocess the masks here
        repeats = 5
        def smoothen(corrupt_signal, mask):
            # kernel = jnp.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
            # kernel = jnp.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]]) / 173
            kernel = jnp.array([[1,1,1,1,1], [1,1,1,1,1], [1,1,0,1,1], [1,1,1,1,1], [1,1,1,1,1]])[..., None] / 24.0
            response = jax.scipy.signal.convolve(corrupt_signal, kernel, mode='same')
            for _ in range(repeats):
                response = corrupt_signal*mask + response*(1-mask)
                response = jax.scipy.signal.convolve(response, kernel, mode='same')
            return response
        xs = eqx.filter_vmap(smoothen)(xs, masks)

        ## Put the time dimension first and channel first
        xs = xs.transpose(0, 3, 1, 2)    ## xs shape: (T, C, H, W)
        masks = masks.transpose(0, 3, 1, 2)    ## masks shape: (T, 1, H, W)

        ## Predict y0 with a ConvCNP
        mu_y0, sigma_y0 = self.init_cond(ctx_data[0])    ## each of shape: (H, W, C)
        y0 = jnp.concatenate([mu_y0, sigma_y0], axis=-1).transpose(2, 0, 1)    ## y0 shape: (2*C, H, W)

        sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self.ode_func),
                solver=diffrax.Euler(),
                args=(ts, xs, masks, None),
                t0=ts[0],
                t1=ts[-1],
                dt0=ts[1]-ts[0],
                y0=y0,
                # stepsize_controller=diffrax.PIDController(rtol=1e-2, atol=1e-4, dtmin=None),
                saveat=diffrax.SaveAt(ts=ts),
                adjoint=diffrax.RecursiveCheckpointAdjoint(),
                # adjoint=diffrax.DirectAdjoint(),
                max_steps=4096*1,
                throw=True,    ## Keep the nans and infs, don't throw and error !
            )

        ## Width first and channel last
        pred_vid = sol.ys.transpose(0, 3, 2, 1)    ## pred_vid shape: (T, W, H, C*2)

        mus, sigmas = jnp.split(pred_vid, 2, axis=-1)    ## mus, sigmas shape: (T, W, H, C)
        sigmas = self.init_cond.positivity(sigmas)

        return (mus, sigmas), xs

def loss_fn(model, batch):
    ctx_data, tgt_data = batch
    # Xc shape: (B, T, K, 2), Yc shape: (B, T, K, C), Yt shape: (B, T, 1024, C)

    ys, _ = eqx.filter_vmap(eqx.filter_vmap(model.align_labels))(tgt_data)  #ys shape: (B, T, H, W, C)

    (mus, sigmas), _ = eqx.filter_vmap(model)(ctx_data)              ## mu, sigma shape: (B, T, H, W, C)

    losses = neg_log_likelihood(mus, sigmas, ys)
    return losses.mean()

model = Model(C, H, W, c, h, w, latent_chans=latent_chans, key=model_key)
learner = Learner(model, loss_fn, images=False)


## Define optimiser and train the model
total_steps = nb_epochs*train_dataloader.num_batches
bd_scales = {total_steps//3:sched_factor, 2*total_steps//3:sched_factor}
sched = optax.piecewise_constant_schedule(init_value=init_lr, boundaries_and_scales=bd_scales)
opt = optax.chain(optax.clip(1e2), optax.adam(sched))

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

# ## Print the vector weights before and after training
# print("Vector weights before training:\n", model.ode_func.vector_field.weights.weight[:2, :2, :2, :2])
# print("Vector weights after training:\n", trainer.learner.model.ode_func.vector_field.weights.weight[:2, :2, :2, :2])

# ## Print the control weights before and after training
# # print("Control weights before training:\n", model.ode_func.control.weights.conv2.weight[:2, :2, :2, :2])
# # print("Control weights after training:\n", trainer.learner.model.ode_func.control.weights.conv2.weight[:2, :2, :2, :2])


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
                      seed=10)
test_dataloader = NumpyLoader(test_dataset, 
                               batch_size=envs_batch_size, 
                               shuffle=False, 
                               num_workers=num_workers)

vt.visualize_videos(test_dataloader, 
                    nb_envs=4, 
                    save_path=run_folder+"sample_predictions_nobt.png", 
                    bootstrap=False, 
                    save_video=True, 
                    video_prefix=run_folder+"sample_nobt")



#%%
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    os.system(f"cp nohup.log {run_folder}")
