import pickle
from typing import Any, Tuple

from .learner import *
from .visualtester import VisualTester
from ._config import *





#%%

class Trainer:
    def __init__(self, learner:Learner, optimiser):
        """ Base class for training the models"""

        if not isinstance(learner, Learner):
            raise ValueError("The learner must be an instance of Learner")
        else:
            self.learner = learner
        self.opt = optimiser

        self.opt_state = self.opt.init(eqx.filter(self.learner.model, eqx.is_array))

        self.train_losses = []
        self.val_losses = []

        self.val_criterion = None
 
    def save_trainer(self, path, ignore_losses=False):
        assert path[-1] == "/", "ERROR: The path must end with /"

        if not ignore_losses:
            if len(self.val_losses) > 0:
                np.savez(path+"train_histories.npz", train_losses=jnp.array(self.train_losses), val_losses=jnp.array(self.val_losses))
            else:
                np.savez(path+"train_histories.npz", train_losses=jnp.array(self.train_losses))
        pickle.dump(self.opt_state, open(path+"opt_state.pkl", "wb"))

        self.learner.save_learner(path)

    def restore_trainer(self, path):
        assert path[-1] == "/", "ERROR: Invalidn parovided. The path must end with /"
        print(f"\nLoading model and results from {path} folder ...\n")

        if os.path.exists(path+"train_histories.npz"):
            histories = np.load(path+"train_histories.npz")
        elif os.path.exists(path+"checkpoints/train_histories.npz"):
            print("WARNING: No training history found in the provided path. Using checkpointed ones.")
            histories = np.load(path+"checkpoints/train_histories.npz")
        else:
            print("WARNING: No training history found at all. Using tens.")
            histories = {'train_losses': jnp.inf*np.ones((1,)), 'val_losses': jnp.inf*np.ones((1,))}

        self.train_losses = histories['train_losses']
        if 'val_losses' in histories:
            self.val_losses = histories['val_losses']
        else:
            if os.path.exists(path+"val_losses.npy"):
                self.val_losses = [np.load(path+"val_losses.npy")]
            elif os.path.exists(path+"checkpoints/val_losses.npy"):
                print("WARNING: No validation history found in the provided path. Using checkpointed ones.")
                self.val_losses = [np.load(path+"checkpoints/val_losses.npy")]
            else:
                print("WARNING: No validation history found at all. Using ten.")
                self.val_losses = []

        if os.path.exists(path+"opt_state.pkl"):
            self.opt_state = pickle.load(open(path+"opt_state.pkl", "rb"))
        else:
            print("WARNING: No optimiser state found in the provided path.")

        self.learner.load_learner(path)


    def meta_train(self,
                    dataloader, 
                    nb_epochs,
                    print_every=1, 
                    save_checkpoints=False,
                    validate_every=100,
                    save_path=False, 
                    val_dataloader=None, 
                    val_criterion=None):
        """ Train the model using the provided dataloader """

        opt = self.opt
        opt_state = self.opt_state
        model = self.learner.model
        loss_fn = self.learner.loss_fn

        self.val_criterion = val_criterion

        if save_checkpoints:
            os.makedirs(save_path+"checkpoints", exist_ok=True)

        @eqx.filter_jit
        def train_step(model, batch, opt_state):
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
            updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
            model = eqx.apply_updates(model, updates)
            return model, loss, opt_state

        losses = []

        start_time = time.time_ns()

        ## TODO: use tqdm for better progress bar
        for epoch in range(nb_epochs):
            start_time_step = time.perf_counter()

            loss_epoch = 0.
            num_batches = 0
            for batch in dataloader:
                model, loss, opt_state = train_step(model, batch, opt_state)

                loss_epoch += loss
                num_batches += 1

            loss_epoch /= num_batches
            losses.append(loss_epoch)

            if epoch % print_every == 0 or epoch == nb_epochs-1:
                print(f"{time.strftime('%H:%M:%S')}      Epoch: {epoch:-3d}      Loss: {losses[-1]:-.8f}      Time/Epoch(s): {time.perf_counter()-start_time_step:-.4f}", flush=True, end="\n")

            if save_checkpoints and epoch % validate_every == 0:
                self.learner.model = model
                eqx.tree_serialise_leaves(save_path+f"checkpoints/model_{epoch:06d}.eqx", model)

            if val_dataloader is not None and epoch % validate_every == 0:
                if self.learner.images:
                    val_loss = self.meta_test_img(val_dataloader, val_criterion)
                else:
                    val_loss = self.meta_test_vid(val_dataloader, val_criterion)
                print(f"    Validation loss: {val_loss}")
                self.val_losses.append(val_loss)
                if self.val_losses[-1] == min(self.val_losses):
                    self.learner.model = model
                    eqx.tree_serialise_leaves(save_path+"best_model.eqx", model)
                    print(f"    Best model saved at epoch {epoch} with validation loss: {val_loss}")

        self.train_losses += losses
        self.learner.model = model
        if save_path:
            self.save_trainer(save_path)

        ## Print the total time taken for training in HH:MM:SS
        print(f"\nTotal training time: {time.strftime('%H:%M:%S', time.gmtime((time.time_ns()-start_time)//10**9))}\n")

    def meta_test_img(self, dataloader, criterion="NLL"):
        """ Test the model using the provided dataloader """

        model = self.learner.model

        @eqx.filter_jit
        def test_step(model, batch):
            ctx_data, tgt_data = batch
            ys, _ = eqx.filter_vmap(model.preprocess_channel_last)(tgt_data)
            mus, sigmas = model(ctx_data)

            if criterion == "NLL":
                losses = neg_log_likelihood(mus, sigmas, ys)
            elif criterion == "MSE":
                losses = mse(mus, sigmas, ys)
            elif criterion == "SSIM":
                losses = ssim(mus, sigmas, ys)
            elif criterion == "PSNR":
                losses = psnr(mus, sigmas, ys)

            return losses.mean()

        losses = []

        for batch in dataloader:
            loss = test_step(model, batch)
            losses.append(loss)

        return jnp.mean(jnp.array(losses))
    

    def meta_test_vid(self, dataloader, criterion="NLL"):
        """ Test the model using the provided dataloader """

        model = self.learner.model

        @eqx.filter_jit
        def test_step(model, batch):
            ctx_data, tgt_data = batch
            ys, _ = eqx.filter_vmap(eqx.filter_vmap(model.preprocess_channel_last))(tgt_data)  #ys shape: (B, T, H, W, C)

            # keys = jax.random.split(key, ctx_data.shape[0])
            (mus, sigmas), ctx_vids = eqx.filter_vmap(model.naive_predict)(ctx_data)              ## mu, sigma shape: (B, T, H, W, C)
            # (mus, sigmas), ctx_vids = eqx.filter_vmap(model.bootstrap_predict)(tgt_data)              ## mu, sigma shape: (B, T, H, W, C)

            if criterion == "NLL":
                losses = neg_log_likelihood(mus, sigmas, ys)
            elif criterion == "MSE":
                losses = mse(mus, sigmas, ys)
            elif criterion == "SSIM":
                losses = ssim(mus, sigmas, ys)
            elif criterion == "PSNR":
                losses = psnr(mus, sigmas, ys)

            return losses.mean()

        losses = []

        for batch in dataloader:
            loss = test_step(model, batch)
            losses.append(loss)

        return jnp.mean(jnp.array(losses))