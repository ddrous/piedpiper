import pickle
from typing import Any, Tuple

from .dataloader import DataLoader
from .learner import Learner
from .visualtester import VisualTester
from ._config import *





#%%

class Trainer:
    def __init__(self, learner:Learner, optimiser, key=None):
        """ Base class for training the models"""

        if key is None:
            raise ValueError("You must provide a key for the trainer")
        self.key = key      ## Default training key

        if not isinstance(learner, Learner):
            raise ValueError("The learner must be an instance of Learner")
        else:
            self.learner = learner
        self.opt = optimiser

        self.opt_state = self.opt.init(eqx.filter(self.learner.model, eqx.is_array))

        self.train_losses = []
        self.val_losses = []
 
    def save_trainer(self, path, ignore_losses=False):
        assert path[-1] == "/", "ERROR: The path must end with /"

        if not ignore_losses:
            if len(self.val_losses) > 0:
                np.savez(path+"train_histories.npz", train_losses=jnp.vstack(self.train_losses), val_losses=jnp.vstack(self.val_losses))
            else:
                np.savez(path+"train_histories.npz", train_losses=jnp.vstack(self.train_losses))
        pickle.dump(self.opt_state_model, open(path+"opt_state.pkl", "wb"))

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
            histories = {'train_losses': jnp.inf*np.ones((1,1))}
        self.train_losses = [histories['train_losses']]

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
                    dataloader: DataLoader, 
                    nb_epochs,
                    print_error_every=1, 
                    save_checkpoints=False,
                    validate_every=100,
                    save_path=False, 
                    val_dataloader=None, 
                    val_criterion=None):
        """ Train the model using the provided dataloader """


        @eqx.filter_jit
        def train_step(model, batch, opt_state):
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
            updates, opt_state = opt.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, loss, opt_state

        losses = []

        ## TODO: use tqdm for better progress bar
        for epoch in range(nb_epochs):
            start_time_step = time.perf_counter()

            loss_epoch = 0.
            num_batches = 0
            for (ctx_batch, tgt_batch) in dataloader:
                model, loss, opt_state = train_step(model, (ctx_batch, tgt_batch), opt_state)

                loss_epoch += loss
                num_batches += 1

            loss_epoch /= num_batches
            losses.append(loss_epoch)

            if epoch % print_every == 0 or epoch == nb_epochs-1:
                print(f"{time.strftime('%H:%M:%S')}      Epoch: {epoch:-3d}      Loss: {losses[-1]:-.8f}      Time/Epoch(s): {time.perf_counter()-start_time_step:-.4f}", flush=True, end="\n")

        eqx.tree_serialise_leaves("model.eqx", model)
