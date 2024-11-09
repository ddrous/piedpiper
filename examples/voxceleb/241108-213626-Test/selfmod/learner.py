from ._utils import *
from math import factorial
from jax.experimental.jet import jet



class Learner:
    def __init__(self, 
                 model, 
                 env_loss_fn, 
                 context_size, 
                 context_pool_size, 
                 pool_filling="NF", 
                 contexts=None, 
                 reuse_contexts=False,
                 loss_filling="NF", 
                 model_reg="l2",
                 context_reg="l1",
                 loss_contributors=-1,
                 key=None):
        if key is None:
            raise ValueError("You must provide a key for the learner.")
        self.key = key

        self.model = model
        self.context_size = context_size
        self.context_pool_size = context_pool_size
        self.pool_filling = pool_filling
        self.reuse_contexts = reuse_contexts
        self.loss_contributors = loss_contributors

        self.model_reg = model_reg
        self.context_reg = context_reg
        self.loss_filling = loss_filling

        if contexts is not None:
            self.contexts = contexts
        else:
            print("    No context template provides, using arrays ...")
            self.contexts = ArrayContextParams(nb_envs=1, context_size=context_size)

        def moe_cv_loss_fn(model, ctxs):
            # If the model is a mixture of experts, then minimize its coefficient of variation
            is_moe = False
            gating_function = None
            if hasattr(model, "vectorfield") and model.vectorfield.neuralnet.is_moe:
                is_moe = True
                network = model.vectorfield.neuralnet
            elif hasattr(model, "neuralnet") and model.neuralnet.is_moe:
                is_moe = True
                network = model.neuralnet

            if is_moe:
                print("    Minimizing coefficient of variation in the loss function ...")
                batched_gates = eqx.filter_vmap(network.gating_function)(ctxs)
                # jax.debug.print("The batched gates are:  {}  ", batched_gates)

                ## count the non-zeros along axis 0
                non_zeros = jnp.count_nonzero(batched_gates, axis=0, keepdims=True)
                non_zeros = jnp.where(non_zeros==0, 1, non_zeros)
                importances = jnp.sum(batched_gates, axis=0) / non_zeros

                cv = jnp.var(importances) / jnp.mean(importances)**2

            else:
                cv = 0.0
                importances = None

            # jax.debug.print("The coefficient of variation is:  {} {}  ", cv, importances)

            return 0., batched_gates          ## TODO remove this
            # return 1.*cv, batched_gates




        def loss_fn_gates(gates, ctxs, key):
            print("    Compiling gating loss function - coefficient of variation ...")
            batched_gates = eqx.filter_vmap(gates["function"], in_axes=(None, 0))(gates, ctxs.params)

            ## count the non-zeros along axis 0
            non_zeros = jnp.count_nonzero(batched_gates, axis=0, keepdims=True)
            non_zeros = jnp.where(non_zeros==0, 1, non_zeros)
            importances = jnp.sum(batched_gates, axis=0) / non_zeros

            cv = jnp.var(importances) / jnp.mean(importances)**2

            return cv, (batched_gates, )

        self.loss_fn_gates = loss_fn_gates



        def env_loss_fn_(model, batch, ctx, ctxs, key):
            """ Wrapping the loss function before vectorizing it below """
            X, Y = batch

            if self.pool_filling=="RA":         ## Randomly fill the context pool
                ind = jax.random.permutation(key, ctxs.shape[0])[:self.context_pool_size]
                ctx_pool = ctxs[ind, :]
            elif self.pool_filling=="NF":       ## Fill the context with the nearest first
                # dists = jnp.mean(jnp.abs(ctxs-ctx), axis=1)
                dists = jnp.mean((ctxs-ctx)**2, axis=1)     ## TODO test with L2 norm
                ind = jnp.argsort(dists)[:self.context_pool_size]
                ctx_pool = ctxs[ind, :]
            elif self.pool_filling=="NF*":      ## Same as NF, but excluding the current context
                dists = jnp.mean(jnp.abs(ctxs-ctx), axis=1)
                ind = jnp.argsort(dists)[1:self.context_pool_size+1]
                ctx_pool = ctxs[ind, :]
            elif self.pool_filling=="SF":       ## Smallest contexts first
                dists = jnp.mean(jnp.abs(ctxs), axis=1)
                ind = jnp.argsort(dists)[:self.context_pool_size]
                ctx_pool = ctxs[ind, :]
            else:
                raise ValueError("Invalid pool filling strategy provided. Use one of 'RA', 'NF', 'NF*', 'SF'.")

            # if isinstance(X, tuple) or isinstance(X, list):    ## The input X comes with time steps
            #     X = (X[0], jnp.broadcast_to(X[1], (X[0].shape[0], X[1].shape[0])))

            Y_hat = jax.vmap(model, in_axes=(None, None, 0))(X, ctx, ctx_pool)
            Y_new = jnp.broadcast_to(Y, Y_hat.shape)

            return env_loss_fn(model, ctx, Y_new, Y_hat)

        # print("    Using all environments to estimate the global loss function ...")
        def loss_fn_full(model, contexts, batch, weightings, key):
            keys = jax.random.split(key, num=contexts.params.shape[0])

            losses, (term1, terms2, terms3) = jax.vmap(env_loss_fn_, in_axes=(None, 0, 0, None, 0))(model, batch, contexts.params, contexts.params, keys)
            base_loss = jnp.mean(losses)

            # cv, importances = moe_cv_loss_fn(model, contexts.params)
            # terms3 = (terms3, importances)
            # base_loss += cv

            # return jnp.sum(losses*weightings), (term1, terms2, terms3, np.arange(contexts.params.shape[0]))
            return base_loss, (term1, terms2, terms3, np.arange(contexts.params.shape[0]))

        if loss_contributors > 0:
            print(f"\nUsing {loss_contributors} environments to estimate the global training loss function ...")
            def loss_fn(model, contexts, batch, prev_losses, key):
                keys = jax.random.split(key, num=loss_contributors)

                if self.loss_filling=="RA":         ## Randomly pick contributors to the loss function
                    indices = jax.random.permutation(key, contexts.params.shape[0])[:loss_contributors]
                elif self.loss_filling=="FO":       ## Pick the first environments, based on their loss (no randomness at all)
                    indices = jnp.arange(loss_contributors)
                elif self.loss_filling=="NF":       ## Pick one at random and then the nearest to it
                    rnd_env = jax.random.randint(key, (1,), 0, contexts.params.shape[0])[0]
                    # dists = jnp.mean(jnp.abs(contexts.params-contexts.params[rnd_env]), axis=1)
                    dists = jnp.mean((contexts.params-contexts.params[rnd_env])**2, axis=1)    ## TODO test with L2 norm
                    indices = jnp.argsort(dists)[:loss_contributors]
                elif self.loss_filling=="NF-W":       ## Weighted. We Pick one of the environments we want to focus on
                    probas = prev_losses / jnp.sum(prev_losses)
                    rnd_env = jax.random.choice(key, a=contexts.params.shape[0], shape=(1,), p=probas)[0]
                    dists = jnp.mean(jnp.abs(contexts.params-contexts.params[rnd_env]), axis=1)
                    indices = jnp.argsort(dists)[:loss_contributors]
                elif self.loss_filling=="NF-iW":       ## inversely Weighted.
                    inv_losses = 1/prev_losses
                    probas = inv_losses / jnp.sum(inv_losses)
                    # jax.debug.print("These are the probabilities:  {}  ", probas)
                    rnd_env = jax.random.choice(key, a=contexts.params.shape[0], shape=(1,), p=probas)[0]
                    dists = jnp.mean(jnp.abs(contexts.params-contexts.params[rnd_env]), axis=1)
                    indices = jnp.argsort(dists)[:loss_contributors]
                    # jax.debug.print("These are the indices:  {}  ", indices)
                else:
                    raise ValueError("Invalid loss filling strategy provided. Use one of 'RA', 'NF'.")

                random_contexts = contexts.params[indices, :]

                # random_batch = (batch[0][indices], batch[1][indices])

                ## the full batch is now a pytree, the input is a tuple itself
                random_batch = jax.tree_map(lambda x: x[indices], batch)

                # keys = keys[indices]

                losses, (term1, terms2, terms3) = jax.vmap(env_loss_fn_, in_axes=(None, 0, 0, None, 0))(model, random_batch, random_contexts, random_contexts, keys)
                base_loss = jnp.sum(losses) / loss_contributors  ## TODO testing CV
                # base_loss = 0.

                # cv, importances = moe_cv_loss_fn(model, random_contexts)
                # base_loss += cv
                # terms3 = (terms3, importances)

                return base_loss, (term1, terms2, terms3, indices)

        else:
            print("    Using all environments to estimate the global training loss function ...")
            loss_fn = loss_fn_full

        self.loss_fn = loss_fn                  ## Meta loss function
        self.loss_fn_full = loss_fn_full        ## Base loss function in full
        self.env_loss_fn = env_loss_fn_         ## Base loss function


    def save_learner(self, path):
        assert path[-1] == "/", "ERROR: Invalid path provided. The path must end with /"
        eqx.tree_serialise_leaves(path+"model.eqx", self.model)
        if hasattr(self, "contexts"):
            eqx.tree_serialise_leaves(path+"contexts.eqx", self.contexts)

    def load_learner(self, path):
        assert path[-1] == "/", "ERROR: Invalidn parovided. The path must end with /"
        self.model = eqx.tree_deserialise_leaves(path+"model.eqx", self.model)
        if os.path.exists(path+"contexts.eqx") and hasattr(self, "contexts"):
            self.contexts = eqx.tree_deserialise_leaves(path+"contexts.eqx", self.contexts)



    def reset_model(self, taylor_order, verbose=True):
        if hasattr(self.model, "taylor_order") and taylor_order==self.model.taylor_order:
            model = self.model
        else:
            if verbose:
                print(f"    Creating a new model with taylor order {taylor_order} ...")
            if isinstance(self.model, NeuralContextFlow):
                model = NeuralContextFlow(neuralnet=self.model.neuralnet, 
                                            taylor_order=taylor_order)
            elif isinstance(self.model, NeuralNeuralContextFlow):
                if taylor_order != 0:
                    model = NeuralNeuralContextFlow(neuralnet=self.model.neuralnet, 
                                                    flownet=self.model.flownet)
                else:
                    model = NeuralNeuralContextFlow(neuralnet=self.model.neuralnet, 
                                                    flownet=None)
            elif isinstance(self.model, NeuralContextFlowAdaptiveTaylor):
                model = NeuralContextFlow(neuralnet=self.model.neuralnet, 
                                            taylor_order=taylor_order,
                                            taylor_scale=self.model.taylor_scale,
                                            taylor_weight_init=self.model.taylor_weight[0])
            elif isinstance(self.model, NeuralODE):
                model = NeuralODE(neuralnet=self.model.vectorfield.neuralnet, 
                                    taylor_order=taylor_order,
                                    taylor_ad_mode=self.model.taylor_ad_mode, 
                                    ivp_args=self.model.ivp_args,
                                    t_eval=self.model.t_eval)
            elif isinstance(self.model, BatchedNeuralContextFlow):
                if hasattr(self.model, "taylor_scale"):
                    model = BatchedNeuralContextFlow(neuralnet=self.model.neuralnet, 
                                                    taylor_order=taylor_order,
                                                    taylor_scale=self.model.taylor_scale,
                                                    taylor_weight_init=self.model.taylor_weight[0])
                else:
                    model = BatchedNeuralContextFlow(neuralnet=self.model.neuralnet, 
                                                    taylor_order=taylor_order)
            else:
                raise ValueError("The model type is not supported")
        return model


    # def reset_contexts(self, nb_envs):
    #     if hasattr(self.model.vectorfield.neuralnet, "ctx_utils"):
    #         mlp_utils = self.model.vectorfield.neuralnet.ctx_utils[3]
    #         contexts = InfDimContextParams(nb_envs=nb_envs, 
    #                                 context_size=self.context_size,
    #                                 hidden_size=mlp_utils[1],
    #                                 depth=mlp_utils[2], 
    #                                 key=None)
    #     else:
    #         contexts = ArrayContextParams(nb_envs=nb_envs, 
    #                                     context_size=self.context_size)

    #     return contexts

    def reset_contexts(self, nb_envs):
        if isinstance(self.contexts, InfDimContextParams):
            if hasattr(self.model, "vectorfield"):
                mlp_utils = self.model.vectorfield.neuralnet.ctx_utils[3]
            else:
                mlp_utils = self.model.neuralnet.ctx_utils[3]
            # contexts = InfDimContextParams(nb_envs=nb_envs, 
            #                         context_size=self.context_size,
            #                         hidden_size=mlp_utils[1],
            #                         depth=mlp_utils[2], 
            #                         key=None)
            input_dim, output_dim, hidden_size, depth, activation = mlp_utils
            key = self.contexts.key
            if key is not None:
                key, _ = jax.random.split(key)
            contexts = InfDimContextParams(nb_envs=nb_envs, 
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    hidden_size=hidden_size,
                                    depth=depth, 
                                    activation=activation,
                                    key=key)
        elif isinstance(self.contexts, ArrayContextParams):
            contexts = ArrayContextParams(nb_envs=nb_envs, 
                                        context_size=self.context_size)
        elif isinstance(self.contexts, GaussianContextParams):
            contexts = GaussianContextParams(nb_envs=nb_envs, 
                                        nb_gaussians_per_env=self.context_size//GAUSSIAN_ATTRIBUTE_COUNT_2D,
                                        img_shape=self.contexts.img_shape,
                                        key=self.contexts.key)
        elif isinstance(self.contexts, ConvContextParams):
            input_chans, output_chans, hidden_chans, kernel_size, depth, activation = self.model.neuralnet.ctx_utils[3]
            key = self.contexts.key
            if key is not None:
                key, _ = jax.random.split(key)
            contexts = ConvContextParams(nb_envs=nb_envs,
                                        input_chans=input_chans,
                                        output_chans=output_chans,
                                        hidden_chans=hidden_chans,
                                        kernel_size=kernel_size,
                                        depth=depth,
                                        activation=activation,
                                        key=key)
        else:
            print("COntexts is", self.contexts)
            raise ValueError(f"The context type {type(self.contexts)} is not supported")

        return contexts


    # @eqx.filter_jit
    # def batch_predict(self, model, contexts, batch):
    #     """ Predict Y_hat for a batch issued from a dataloader
    #         CSM may or may not be deleted from the model; 
    #         as this function ensures the deactivation of CSM"""
    #     X, Y = batch
    #     Y_hat = eqx.filter_vmap(model, in_axes=(0, 0, 0))(X, contexts.params, contexts.params)
    #     return X, Y, Y_hat



    # @eqx.filter_jit
    def batch_predict(self, model, contexts, batch, max_envs=-1):
        """ Predict Y_hat for a batch issued from a dataloader
            CSM may or may not be deleted from the model; 
            as this function ensures the deactivation of CSM"""
        ## Predict in in a single batched call if possible, or a maximum sub-batches to avoid OOM

        X, Y = batch
        batched_model = eqx.filter_vmap(model, in_axes=(0, 0, 0))

        if max_envs==-1 or max_envs>=Y.shape[0] or self.loss_contributors==-1:
            Y_hat = batched_model(X, contexts.params, contexts.params)

        elif max_envs == None:
            sub_batch_size = self.loss_contributors
            print(f"    Too many environments to predict in a single batch, predicting in {sub_batch_size} environments ...")
            X_list = []
            Y_list = []
            Y_hat = []
            for i in range(0, Y.shape[0], sub_batch_size):
                contexts_ = contexts.params[i:i+sub_batch_size]
                Y_hat.append(batched_model(X[i:i+sub_batch_size], contexts_, contexts_))

                X_list.append(X[i:i+sub_batch_size])
                Y_list.append(Y[i:i+sub_batch_size])

                # break   ## TODO 1 sub-batch is enough ?

            Y_hat = jnp.concatenate(Y_hat, axis=0)
            X = jnp.concatenate(X_list, axis=0)
            Y = jnp.concatenate(Y_list, axis=0)

        else:
            contexts_ = contexts.params[:max_envs]
            # Y_hat = batched_model(X[:max_envs], contexts_, contexts_)
            # X = X[:max_envs]

            if isinstance(X, tuple) or isinstance(X, list):
                # X = (X[0], jnp.broadcast_to(X[1], (X[1].shape[0], X[0].shape[1], X[1].shape[1])))
                # X = (X[0], jnp.repeat(X[1], X[0].shape[1], axis=1))
                X = jax.tree_map(lambda x: x[:max_envs], X)
            else:
                X = X[:max_envs]

            Y_hat = batched_model(X, contexts_, contexts_)
            Y = Y[:max_envs]

        return X, Y, Y_hat



    # # @eqx.filter_jit
    # def batch_predict_multi(self, model, contexts, batch, max_envs=-1):
    #     """ Predict multiple Y_hats for a batch issued from a dataloader
    #         CSM should be active in the model;
    #         max_envs=6 means do not predict more than 6 environments, even if we have more in the batch
    #         """

    #     X, Y = batch
    #     batched_model = eqx.filter_vmap(model, in_axes=(0, 0, 0))

    #     if max_envs==-1 or max_envs>=X.shape[0] or self.loss_contributors==-1:
    #         Y_hat = []
    #         for e in range(contexts.params.shape[0]):
    #             X_ctx = jnp.broadcast_to(X[e:e+1], X.shape)
    #             ctxs = jnp.broadcast_to(contexts.params[e:e+1], contexts.params.shape)
    #             Y_hat.append(batched_model(X_ctx, ctxs, contexts.params))

    #     else:
    #         X = X[:max_envs]
    #         Y = Y[:max_envs]
    #         contexts_ = contexts.params[:max_envs]

    #         Y_hat = []
    #         for e in range(contexts_.shape[0]):
    #             X_ctx = jnp.broadcast_to(X[e:e+1], X.shape)
    #             ctxs = jnp.broadcast_to(contexts_[e:e+1], contexts_.shape)
    #             Y_hat.append(batched_model(X_ctx, ctxs, contexts_))

    #     return X, Y, jnp.stack(Y_hat, axis=0)



    # @eqx.filter_jit
    def batch_predict_multi(self, model, contexts, batch, max_envs=-1, uq_train_contexts=-1):
        """ Predict multiple Y_hats for a batch issued from a dataloader
            CSM should be active in the model;
            max_envs=6 means do not predict more than 6 environments, even if we have more in the batch
            uq_train_contexts is the number of training contexts to use for uncertainty quantification later on
            Upon return, the first result in Y_hat is the prediction for the context itself
            """

        X, Y = batch
        batched_model = eqx.filter_vmap(model, in_axes=(0, 0, 0))

        if uq_train_contexts != -1:
            train_contexts = self.contexts
            assert uq_train_contexts <= train_contexts.params.shape[0], "The number of UQ contexts must be less than the number of training contexts."
            assert uq_train_contexts > 1, "The number of UQ contexts must be greater than 1."
            ## Select the max_envs closest to each of the given contexts for prediction
            neighbors = []
            for e in range(contexts.params.shape[0]):
                dists = jnp.mean(jnp.abs(train_contexts.params-contexts.params[e]), axis=1)
                indices = jnp.argsort(dists)[:uq_train_contexts-1]      ## -1 because we will append the context itself
                # indices = jnp.argsort(dists)[-uq_train_contexts+1:]   ## TODO UQ is much too pronounced if we take the farthest 
                neigh_e = jnp.concat((contexts.params[e:e+1], train_contexts.params[indices]))
                neighbors.append(neigh_e)
        else:
            ## Reuse the given contexts as the neighbors (rearange so that 0 is the context itself)
            neighbors = []
            for e in range(contexts.params.shape[0]):
                neigh_e = jnp.concatenate((contexts.params[e:e+1], contexts.params[:e], contexts.params[e+1:]))
                neighbors.append(neigh_e)
        neighbors = jnp.stack(neighbors, axis=0)

        ### Now the prediction of a maximum of max_envs environments
        if max_envs==-1 or max_envs>=X.shape[0] or self.loss_contributors==-1:
            Y_hat = []
            for e in range(contexts.params.shape[0]):
                X_ctx = jnp.broadcast_to(X[e:e+1], (neighbors[e].shape[0], *X.shape[1:]))
                ctxs = jnp.broadcast_to(contexts.params[e:e+1], neighbors[e].shape)
                Y_hat.append(batched_model(X_ctx, ctxs, neighbors[e]))
        else:
            X = X[:max_envs]
            Y = Y[:max_envs]
            contexts_ = contexts.params[:max_envs]

            Y_hat = []
            for e in range(contexts_.shape[0]):
                X_ctx = jnp.broadcast_to(X[e:e+1], (neighbors[e].shape[0], *X.shape[1:]))
                ctxs = jnp.broadcast_to(contexts_[e:e+1], neighbors[e].shape)
                Y_hat.append(batched_model(X_ctx, ctxs, neighbors[e]))

        return X, Y, jnp.stack(Y_hat, axis=0)





class MLP(eqx.Module):
    """ An MLP """
    layers: jnp.ndarray

    def __init__(self, in_size, out_size, hidden_size, depth, activation, key=None):
        keys = jax.random.split(key, num=depth+1)

        self.layers = []

        for i in range(depth):
            if i==0:
                layer = eqx.nn.Linear(in_size, hidden_size, use_bias=True, key=keys[i])
            elif i==depth-1:
                layer = eqx.nn.Linear(hidden_size, out_size, use_bias=True, key=keys[i])
            else:
                layer = eqx.nn.Linear(hidden_size, hidden_size, use_bias=True, key=keys[i])

            self.layers.append(layer)

            if i != depth-1:
                self.layers.append(activation)

    def __call__(self, x):
        """ Returns y such that y = MLP(x) """
        y = x
        for layer in self.layers:
            y = layer(y)
        return y


class ConvNet(eqx.Module):
    """ An MLP """
    layers: jnp.ndarray

    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, depth, activation, key=None):
        keys = jax.random.split(key, num=depth+1)

        self.layers = []

        for i in range(depth):
            if i==0:
                layer = eqx.nn.Conv2d(in_channels, hidden_channels, kernel_size, padding='SAME', key=keys[i])
            elif i==depth-1:
                layer = eqx.nn.Conv2d(hidden_channels, out_channels, kernel_size, padding='SAME', key=keys[i])
            else:
                layer = eqx.nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding='SAME', key=keys[i])

            self.layers.append(layer)

            if i != depth-1:
                self.layers.append(activation)

    def __call__(self, x):
        """ Returns y such that y = ConvNet(x) """
        y = x
        for layer in self.layers:
            y = layer(y)
        return y





# class ArrayContextParams(eqx.Module):
#     params: jnp.ndarray
#     def __init__(self, nb_envs, context_size):
#         self.params = jnp.zeros((nb_envs, context_size))
#     def __call__(self):
#         return self.params


class ArrayContextParams(eqx.Module):
    """ A context initialised with gaussian """
    params: jnp.ndarray
    eff_context_size: int


    def __init__(self, nb_envs, context_size, key=None):
        if key is None:
            self.params = jnp.zeros((nb_envs, context_size))
        else:
            self.params = jax.random.normal(key, (nb_envs, context_size))
        self.eff_context_size = context_size

    def __call__(self):
        return self.params


class GaussianContextParams(eqx.Module):
    """ A context initialised with gaussian """
    params: jnp.ndarray
    eff_context_size: int
    key: jnp.ndarray        ## If we want the gaussian to be always initialised the same
    img_shape: tuple

    def __init__(self, nb_envs, nb_gaussians_per_env, img_shape=None, key=None):
        self.eff_context_size = nb_gaussians_per_env*GAUSSIAN_ATTRIBUTE_COUNT_2D
        self.key = key
        self.img_shape = img_shape

        if key is None:
            self.params = jnp.zeros((nb_envs, self.eff_context_size))
        else:
            if img_shape is None:
                raise ValueError("You must provide the intended rendered image shape to properly initialise the Gaussians.")
            gaussians = init_gaussians(key, img_shape, nb_envs*nb_gaussians_per_env,)
            self.params = jnp.reshape(gaussians, (nb_envs, self.eff_context_size))

    def __call__(self):
        # return jnp.reshape(self.params, (-1, 9))        ## Returns the gaussians
        return self.params                                ## Returns the flattened gaussians


class InfDimContextParams(eqx.Module):
    params: list
    ctx_utils: any
    eff_context_size: int     ## The effective/actual size of a context vector (flattened neural network)
    key: jnp.ndarray

    def __init__(self, nb_envs, input_dim, output_dim, hidden_size, depth, activation=jax.nn.softplus, key=None):
        if key is None:
            self.key = None
            keys = jax.random.split(jax.random.PRNGKey(0), nb_envs)
        else:
            self.key = key
            keys = jax.random.split(key, nb_envs)

        all_contexts = [MLP(input_dim, output_dim, hidden_size, depth, activation, key=keys[i]) for i in range(nb_envs)]

        mlp_utils = (input_dim, output_dim, hidden_size, depth, activation)

        ex_params, ex_static = eqx.partition(all_contexts[0], eqx.is_array)
        ex_ravel, ex_shapes, ex_treedef = flatten_pytree(ex_params)
        self.ctx_utils = (ex_shapes, ex_treedef, ex_static, mlp_utils)

        all_params_1D = [flatten_pytree(eqx.filter(context, eqx.is_array))[0] for context in all_contexts]

        self.eff_context_size = sum(x.size for x in jax.tree_util.tree_leaves(ex_params) if x is not None)

        if key is None:
            self.params = jnp.zeros_like(jnp.stack(all_params_1D, axis=0))
        else:
            self.params = jnp.stack(all_params_1D, axis=0)



class ConvContextParams(eqx.Module):
    params: list
    ctx_utils: any
    eff_context_size: int     ## The effective/actual size of a context vector (flattened neural network)
    key: jnp.ndarray

    def __init__(self, nb_envs, input_chans, output_chans, hidden_chans, kernel_size, depth, activation=jax.nn.relu, key=None):

        if key is None:
            self.key = None
            keys = jax.random.split(jax.random.PRNGKey(0), nb_envs)
        else:
            self.key = key
            keys = jax.random.split(key, nb_envs)

        all_contexts = [ConvNet(input_chans, output_chans, hidden_chans, kernel_size, depth, activation, key=keys[i]) for i in range(nb_envs)]

        mlp_utils = (input_chans, output_chans, hidden_chans, kernel_size, depth, activation)

        ex_params, ex_static = eqx.partition(all_contexts[0], eqx.is_array)
        ex_ravel, ex_shapes, ex_treedef = flatten_pytree(ex_params)
        self.ctx_utils = (ex_shapes, ex_treedef, ex_static, mlp_utils)

        all_params_1D = [flatten_pytree(eqx.filter(context, eqx.is_array))[0] for context in all_contexts]

        self.eff_context_size = sum(x.size for x in jax.tree_util.tree_leaves(ex_params) if x is not None)

        if key is None:
            self.params = jnp.zeros_like(jnp.stack(all_params_1D, axis=0))
        else:
            self.params = jnp.stack(all_params_1D, axis=0)




class NeuralContextFlow(eqx.Module):
    neuralnet: eqx.Module
    taylor_order: int
    taylor_weight: jnp.ndarray
    taylor_scale: int

    def __init__(self, neuralnet, taylor_order, taylor_weight_init=0., taylor_scale=100):
        ############# NCF without the possibility to ignore Taylor expansion #############
        self.neuralnet = neuralnet
        self.taylor_order = taylor_order

        ## Taylor weight and scale are only included for backward compatibility
        self.taylor_weight = jnp.array([taylor_weight_init]).squeeze()
        self.taylor_scale = taylor_scale


    def __call__(self, xs, ctx, ctx_):

        def point_predict(x):

            vf = lambda xi: self.neuralnet(x, xi)

            if self.taylor_order==0:
                return vf(ctx)

            elif self.taylor_order==1:
                gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
                return vf(ctx_) + 1.0*gradvf(ctx_)

            elif self.taylor_order==2:
                gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
                scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]
                return vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term

            else:
                h0 = ctx_
                h1 = ctx-ctx_
                h2 = jnp.zeros_like(h0)

                hs = [h1, h2]
                coeffs = [1, 0.5]
                for order in range(2+1, self.taylor_order+1):
                    hs.append(jnp.zeros_like(h0))
                    coeffs.append(1 / factorial(order))

                f0, fs = jet(vf, (h0,), (hs,))
                taylor_exp = f0 + jnp.sum(jnp.stack(fs, axis=-1) * jnp.array(coeffs)[None,:], axis=-1)

                return taylor_exp

        ys = eqx.filter_vmap(point_predict)(xs)

        return ys



class NeuralNeuralContextFlow(eqx.Module):
    neuralnet: eqx.Module
    flownet: eqx.Module

    def __init__(self, neuralnet, flownet=None):
        ############# NCF with a flow network instead of Taylor expansion #############
        self.neuralnet = neuralnet
        self.flownet = flownet

    def __call__(self, xs, ctx, ctx_):

        def point_predict(x):

            vf = lambda xi: self.neuralnet(x, xi)

            if self.flownet is None:
                return vf(ctx)

            else:
                out_main = vf(ctx_)
                correction = self.flownet(out_main, ctx_, vf(ctx), ctx)
                # return vf(ctx_) + correction      ## TODO use different variations of the input/outputs to the flow network
                return out_main + correction

        ys = eqx.filter_vmap(point_predict)(xs)

        return ys






class NeuralContextFlowAdaptiveTaylor(eqx.Module):
    neuralnet: eqx.Module
    taylor_order: int
    taylor_weight: jnp.ndarray
    taylor_scale: int

    def __init__(self, neuralnet, taylor_order, taylor_weight_init=0., taylor_scale=100):
        """ Neural Context Flow with an additional parameter to select the weight of the Taylor expansion """

        self.neuralnet = neuralnet

        self.taylor_order = taylor_order
        self.taylor_weight = jnp.array([taylor_weight_init])
        self.taylor_scale = taylor_scale


    def __call__(self, xs, ctx, ctx_):

        def point_predict(x):

            ############# With possibility to ignore Taylor expansion #############
            vf = lambda xi: self.neuralnet(x, xi)
            alpha = jax.nn.sigmoid(self.taylor_scale*self.taylor_weight[0])

            if self.taylor_order==0:
                return (alpha)*vf(ctx)       ## Could be (1.-alpha)*vf(ctx), but problem when resetting the model with different alpha

            elif self.taylor_order==1:
                gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
                taylor_exp = vf(ctx_) + 1.0*gradvf(ctx_)

                return (1.-alpha)*vf(ctx) + (alpha)*taylor_exp

            elif self.taylor_order==2:
                gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
                scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]
                taylor_exp = vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term

                return (1.-alpha)*vf(ctx) + (alpha)*taylor_exp

            else:
                # raise NotImplementedError("Higher order terms are not implemented yet.")

                h0 = ctx_
                h1 = ctx-ctx_
                h2 = jnp.zeros_like(h0)

                hs = [h1, h2]
                coeffs = [1, 0.5]
                for order in range(2+1, self.taylor_order+1):
                    hs.append(jnp.zeros_like(h0))
                    coeffs.append(1 / factorial(order))

                f0, fs = jet(vf, (h0,), (hs,))
                taylor_exp = f0 + jnp.sum(jnp.stack(fs, axis=-1) * jnp.array(coeffs)[None,:], axis=-1)

                return (1.-alpha)*vf(ctx) + (alpha)*taylor_exp


        ys = eqx.filter_vmap(point_predict)(xs)

        return ys






class BatchedNeuralContextFlow(eqx.Module):
    neuralnet: eqx.Module
    taylor_order: int

    def __init__(self, neuralnet, taylor_order):
        self.neuralnet = neuralnet
        self.taylor_order = taylor_order

    def __call__(self, xs, ctx, ctx_):

        vf = lambda xi: self.neuralnet(xs, xi)

        if self.taylor_order==0:
            return vf(ctx)

        elif self.taylor_order==1:
            gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
            taylor_exp = vf(ctx_) + 1.0*gradvf(ctx_)

            return taylor_exp

        elif self.taylor_order==2:
            gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
            scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]
            taylor_exp = vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term

            return taylor_exp

        else:
            # raise NotImplementedError("Higher order terms are not implemented yet.")
            h0 = ctx_
            h1 = ctx-ctx_
            h2 = jnp.zeros_like(h0)

            hs = [h1, h2]
            coeffs = [1, 0.5]
            for order in range(2+1, self.taylor_order+1):
                hs.append(jnp.zeros_like(h0))
                coeffs.append(1 / factorial(order))

            f0, fs = jet(vf, (h0,), (hs,))
            taylor_exp = f0 + jnp.sum(jnp.stack(fs, axis=-1) * jnp.array(coeffs)[None,:], axis=-1)

            return taylor_exp




class BatchedNeuralContextFlowAdaptiveTaylor(eqx.Module):
    neuralnet: eqx.Module

    taylor_order: int
    taylor_scale: int
    taylor_weight: jnp.ndarray

    def __init__(self, neuralnet, taylor_order, taylor_weight_init=0., taylor_scale=100):
        self.neuralnet = neuralnet

        self.taylor_order = taylor_order
        self.taylor_weight = jnp.array([taylor_weight_init])        ## We start with 50-50
        self.taylor_scale = taylor_scale                     ## Multiply by this before sigmoid

    def __call__(self, xs, ctx, ctx_):

        vf = lambda xi: self.neuralnet(xs, xi)
        alpha = jax.nn.sigmoid(self.taylor_scale*self.taylor_weight[0])

        if self.taylor_order==0:
            return (1.-alpha)*vf(ctx)

        elif self.taylor_order==1:
            gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
            taylor_exp = vf(ctx_) + 1.0*gradvf(ctx_)

            return (1.-alpha)*vf(ctx) + (alpha)*taylor_exp

        elif self.taylor_order==2:
            gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
            scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]
            taylor_exp = vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term

            return (1.-alpha)*vf(ctx) + (alpha)*taylor_exp

        else:
            # raise NotImplementedError("Higher order terms are not implemented yet.")
            h0 = ctx_
            h1 = ctx-ctx_
            h2 = jnp.zeros_like(h0)

            hs = [h1, h2]
            coeffs = [1, 0.5]
            for order in range(2+1, self.taylor_order+1):
                hs.append(jnp.zeros_like(h0))
                coeffs.append(1 / factorial(order))

            f0, fs = jet(vf, (h0,), (hs,))
            taylor_exp = f0 + jnp.sum(jnp.stack(fs, axis=-1) * jnp.array(coeffs)[None,:], axis=-1)

            return (1.-alpha)*vf(ctx) + (alpha)*taylor_exp




class SelfModVectorField(eqx.Module):
    """ A vector field with fixed Taylor order """
    neuralnet: eqx.Module
    taylor_order: int
    taylor_ad_mode: str

    def __init__(self, neuralnet, taylor_order, taylor_ad_mode):
        self.neuralnet = neuralnet
        self.taylor_order = taylor_order
        self.taylor_ad_mode = taylor_ad_mode

    def __call__(self, t, x, args):
        ctx, ctx_ = args

        vf = lambda xi: self.neuralnet(t, x, xi)

        if self.taylor_order==0:
            return vf(ctx)

        elif self.taylor_order==1:
            if self.taylor_ad_mode=="forward":
                gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
                taylor_exp = vf(ctx_) + 1.0*gradvf(ctx_)
            elif self.taylor_ad_mode=="reverse":
                jac = eqx.filter_jacrev(vf)(ctx_)
                taylor_exp = vf(ctx_) + jac @ (ctx-ctx_)
            else:
                raise ValueError("Invalid AD mode provided.")

            return taylor_exp

        elif self.taylor_order==2:
            if self.taylor_ad_mode=="forward":
                gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
                scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]
                taylor_exp = vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term
            elif self.taylor_ad_mode=="reverse":
                print("WARNING: Reverse-mode AD for 2nd order Taylor expansion materialises the Hessian and is unstable for the CAVIA algorithm. Consider reducing the Taylor order or using forward-mode AD.")
                jac = eqx.filter_jacrev(vf)(ctx_)
                hess = eqx.filter_jacrev(eqx.filter_jacrev(vf))(ctx_)
                taylor_exp = vf(ctx_) + jac @ (ctx-ctx_) + 0.5 * (hess @ (ctx-ctx_)) @ (ctx-ctx_)
            else:
                raise ValueError("Invalid AD mode provided.")

            return taylor_exp

        else:
            if self.taylor_ad_mode=="forward":
                h0 = ctx_
                h1 = ctx-ctx_
                h2 = jnp.zeros_like(h0)

                hs = [h1, h2]
                coeffs = [1, 0.5]
                for order in range(2+1, self.taylor_order+1):
                    hs.append(jnp.zeros_like(h0))
                    coeffs.append(1 / factorial(order))

                f0, fs = jet(vf, (h0,), (hs,))
                taylor_exp = f0 + jnp.sum(jnp.stack(fs, axis=-1) * jnp.array(coeffs)[None,:], axis=-1)
            else:
                raise ValueError("Higher order terms are only implemented for forward mode AD.")

            return taylor_exp



# class NeuralODE(eqx.Module):
#     vectorfield: eqx.Module
#     ivp_args: dict
#     taylor_order: int
#     taylor_ad_mode: str
#     t_eval: tuple

#     def __init__(self, neuralnet, taylor_order, ivp_args=None, t_eval=None, taylor_ad_mode="forward"):
#         self.ivp_args = ivp_args if ivp_args is not None else {}
#         self.vectorfield = SelfModVectorField(neuralnet, taylor_order=taylor_order, taylor_ad_mode=taylor_ad_mode)
#         self.taylor_order = taylor_order
#         self.taylor_ad_mode = taylor_ad_mode

#         if t_eval is None:
#             self.t_eval = (0., ivp_args.get("T", 1.))
#         else:
#             self.t_eval = t_eval

#     def __call__(self, xs, ctx, ctx_):

#         integrator = self.ivp_args.get("integrator", diffrax.Dopri5())

#         # if isinstance(integrator, type(eqx.Module)):
#         if not callable(integrator):
#             def integrate(y0):
#                 sol = diffrax.diffeqsolve(
#                         terms=diffrax.ODETerm(self.vectorfield),
#                         solver=integrator,
#                         args=(ctx, ctx_.squeeze()),
#                         t0=self.t_eval[0],
#                         t1=self.t_eval[-1],
#                         dt0=self.ivp_args.get("dt_init", 1e-2),
#                         y0=jnp.concat([y0, jnp.zeros((self.ivp_args.get("y0_pad_size", 0),))], axis=0),
#                         stepsize_controller=diffrax.PIDController(rtol=self.ivp_args.get("rtol", 1e-3), 
#                                                                     atol=self.ivp_args.get("atol", 1e-6)),
#                         saveat=diffrax.SaveAt(ts=jnp.array(self.t_eval)),
#                         adjoint=self.ivp_args.get("adjoint", diffrax.RecursiveCheckpointAdjoint()),
#                         max_steps=self.ivp_args.get("max_steps", 4096*1)
#                     )

#                 if self.ivp_args.get("return_traj", False):
#                     return sol.ys[:, :y0.shape[0]]
#                 else:
#                     return sol.ys[-1, :y0.shape[0]]

#         else:   ## Custom-made integrator
#             def integrate(y0):
#                 ys = integrator(fun=self.vectorfield, 
#                                 t_span=(self.t_eval[0], self.t_eval[-1]), 
#                                 y0=y0,
#                                 args=(ctx, ctx_.squeeze()),
#                                 t_eval=jnp.array(self.t_eval), 
#                                 **self.ivp_args
#                                 )
#                 if self.ivp_args.get("return_traj", False):
#                     return ys
#                 else:
#                     return ys[-1]

#         return eqx.filter_vmap(integrate)(xs)



class NeuralODE(eqx.Module):
    vectorfield: eqx.Module
    ivp_args: dict
    taylor_order: int
    taylor_ad_mode: str
    t_eval: tuple

    def __init__(self, neuralnet, taylor_order, ivp_args=None, t_eval=None, taylor_ad_mode="forward"):
        self.ivp_args = ivp_args if ivp_args is not None else {}
        self.vectorfield = SelfModVectorField(neuralnet, taylor_order=taylor_order, taylor_ad_mode=taylor_ad_mode)
        self.taylor_order = taylor_order
        self.taylor_ad_mode = taylor_ad_mode
        self.t_eval = t_eval


    def get_t_eval(self, y0):
        """ Determines the appropriate t-eval based on the input y0 """
        if self.t_eval is None:     
            if not self.ivp_args.get("return_traj", False):     ## User only cares for terminal state
                t_eval = jnp.array((0., self.ivp_args.get("T", 1.)))
            else:   ## Users cares for trajectory, but didn't provide it at initialisationt ime. User must now provide t_eval in every call!
                y0, t_eval = y0
        else:       ## User provides t_eval in the constructor
            t_eval = jnp.array(self.t_eval)

        return y0, t_eval


    def __call__(self, xs, ctx, ctx_):

        integrator = self.ivp_args.get("integrator", diffrax.Dopri5())

        # if isinstance(integrator, type(eqx.Module)):
        if not callable(integrator):
            def integrate(y0):
                y0, t_eval = self.get_t_eval(y0)

                sol = diffrax.diffeqsolve(
                        terms=diffrax.ODETerm(self.vectorfield),
                        solver=integrator,
                        args=(ctx, ctx_.squeeze()),
                        t0=t_eval[0],
                        t1=t_eval[-1],
                        dt0=self.ivp_args.get("dt_init", t_eval[1]-t_eval[0]),
                        y0=jnp.concat([y0, jnp.zeros((self.ivp_args.get("y0_pad_size", 0),))], axis=0),
                        stepsize_controller=diffrax.PIDController(rtol=self.ivp_args.get("rtol", 1e-3), 
                                                                    atol=self.ivp_args.get("atol", 1e-6),
                                                                    dtmin=self.ivp_args.get("dt_min", None)),
                        saveat=diffrax.SaveAt(ts=t_eval),
                        adjoint=self.ivp_args.get("adjoint", diffrax.RecursiveCheckpointAdjoint()),
                        max_steps=self.ivp_args.get("max_steps", 4096*1),
                        throw=True,    ## Keep the nans and infs, don't throw and error !
                    )
                # jax.debug.print("SOL {}", sol.ys)
                ys = sol.ys
                clip = self.ivp_args.get("clip_sol", None)
                if clip is not None:
                    ys = jnp.clip(ys, clip[0], clip[1])

                if self.ivp_args.get("return_traj", False):
                    return ys[:, :y0.shape[0]]
                else:
                    return ys[-1, :y0.shape[0]]

        else:   ## Custom-made integrator
            def integrate(y0):
                y0, t_eval = self.get_t_eval(y0)
                ys = integrator(fun=self.vectorfield, 
                                t_span=(t_eval[0], t_eval[-1]), 
                                y0=y0,
                                args=(ctx, ctx_.squeeze()),
                                t_eval=t_eval, 
                                **self.ivp_args
                                )
                if self.ivp_args.get("return_traj", False):
                    return ys
                else:
                    return ys[-1]

        if isinstance(xs, tuple) or isinstance(xs, list):
            xs = (xs[0], jnp.broadcast_to(xs[1][None,:], (xs[0].shape[0], xs[1].shape[0])))

        batched_results = eqx.filter_vmap(integrate)(xs)

        return jnp.nan_to_num(batched_results, nan=0., posinf=0., neginf=0.)




class Swish(eqx.Module):
    """ Swish activation function """
    beta: jnp.ndarray
    def __init__(self, key=None):
        self.beta = jax.random.uniform(key, shape=(1,), minval=0.01, maxval=1.0)
    def __call__(self, x):
        return x * jax.nn.sigmoid(self.beta * x)



class NeuroModulatedSwish(eqx.Module):
    """ NMN neuro-modulation layer with swish base activation function: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0227922 """
    beta: jnp.ndarray
    w_s: jnp.ndarray
    w_b: jnp.ndarray

    def __init__(self, latent_size, key=None):
        self.beta = jax.random.uniform(key, shape=(1,), minval=0.1, maxval=1.0)
        self.w_s = jnp.ones((latent_size, 1))
        self.w_b = jnp.zeros((latent_size, 1))

    def __call__(self, x, ctx):
        y = ctx.T @ (x * self.w_s + self.w_b)
        return y * jax.nn.sigmoid(self.beta * y)



class Upsample2D(eqx.Module):
    """ Upsample 2D image by a factor: https://docs.kidger.site/equinox/examples/unet/ """
    factor: int
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, y):
        C, H, W = y.shape
        y = jnp.reshape(y, [C, H, 1, W, 1])
        y = jnp.tile(y, [1, 1, self.factor, 1, self.factor])
        return jnp.reshape(y, [C, H * self.factor, W * self.factor])


class VAEDecoder(eqx.Module):
    """ Decoder with dense layers and deconvolutions"""
    img_size: list
    kernel_size: list
    latent_dim: int

    layers: list

    def __init__(self, img_size, kernel_size, latent_dim, key):
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

        layer_keys = jax.random.split(key, 4)
        H, W, C = self.img_size

        self.layers = [
            eqx.nn.Linear(latent_dim, 48, key=layer_keys[0]),
            eqx.nn.PReLU(init_alpha=0.),
            eqx.nn.Linear(48, 12*H*W//(4*4), key=layer_keys[1]),
            eqx.nn.PReLU(init_alpha=0.),
            lambda x: x.reshape((12, H//4, W//4)),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(12, 8, kernel_size, padding="SAME", key=layer_keys[2]),
            eqx.nn.PReLU(init_alpha=0.),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(8, C, kernel_size, padding="SAME", key=layer_keys[3]),
            jax.nn.sigmoid
        ]

    def __call__(self, z):
        x = z
        for layer in self.layers:
            x = layer(x)
        return x
    



class Decoder(eqx.Module):
    """ Decoder with dense layers and deconvolutions"""
    img_size: list
    kernel_size: list
    latent_dim: int

    layers: list

    def __init__(self, img_size, kernel_size, latent_dim, key):
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

        layer_keys = jax.random.split(key, 4)
        H, W, C = self.img_size

        self.layers = [
            eqx.nn.Linear(latent_dim, 1024, key=layer_keys[0]),
            eqx.nn.PReLU(init_alpha=0.),
            eqx.nn.Linear(1024, 64*H*W//(4*4), key=layer_keys[1]),
            eqx.nn.PReLU(init_alpha=0.),
            lambda x: x.reshape((64, H//4, W//4)),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(64, 16, kernel_size, padding="SAME", key=layer_keys[2]),
            eqx.nn.PReLU(init_alpha=0.),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(16, C, kernel_size, padding="SAME", key=layer_keys[3]),
            jax.nn.sigmoid
        ]

    def __call__(self, z):
        x = z
        for layer in self.layers:
            x = layer(x)
        return x



class FuncContextParams(eqx.Module):
    params: list
    img_size: list
    kernel_size: list
    latent_dim: int
    context_size: int

    ctx_utils: any

    def __init__(self, nb_envs, key=None):

        keys = jax.random.split(key, num=nb_envs)

        # all_contexts = [Decoder(img_size=[32, 32, 3], kernel_size=[3, 3], latent_dim=18, key=keys[i]) for i in range(nb_envs)]

        ## Load the decoders from 240101-193230-VAE/decoder.eqx
        all_contexts = [eqx.tree_deserialise_leaves("runs/240101-193230-VAE/decoder.eqx", Decoder(img_size=[32, 32, 3], kernel_size=[3, 3], latent_dim=18, key=keys[i])) for i in range(nb_envs)]

        self.img_size = all_contexts[0].img_size
        self.kernel_size = all_contexts[0].kernel_size
        self.latent_dim = all_contexts[0].latent_dim

        ex_params, ex_static = eqx.partition(all_contexts[0], eqx.is_array)
        ex_ravel, ex_shapes, ex_treedef = flatten_pytree(ex_params)
        # self.ctx_utils = (ex_shapes, ex_treedef, ex_static)
        non_empty_shapes = []
        for shape in ex_shapes:
            if shape == ():
                non_empty_shapes.append((1,))
            else:
                non_empty_shapes.append(shape)
        self.ctx_utils = (non_empty_shapes, ex_treedef, ex_static)

        all_params_1D = [flatten_pytree(eqx.filter(context, eqx.is_array))[0] for context in all_contexts]
        self.context_size = all_params_1D[0].shape[0]

        self.params = jnp.stack(all_params_1D, axis=0)
        # self.params = jnp.zeros_like(jnp.stack(all_params_1D, axis=0))


    def __call__(self, z):
        def unravel_and_call(ctx, z):
            context = jax.flatten_util.unravel_pytree(ctx, self.treedef)
            return context(z)
        return jax.vmap(unravel_and_call)(self.params, z)











########### Implementation of a Vnet model ###########

class DownsamplingLayer(eqx.Module):
    layer: eqx.Module
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, *, key):
        self.layer = eqx.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding="SAME", key=key)
    
    def __call__(self, x):
        return self.layer(x)

class UpsamplingLayer(eqx.Module):
    layer: eqx.Module

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, *, key):
        self.layer = eqx.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding="SAME", key=key)

    def __call__(self, x):
        return self.layer(x)

class DoubleConv(eqx.Module):
    layer_1: eqx.Module
    layer_2: eqx.Module
    activation: callable
    norm_layer: eqx.Module
    dropout_rate: float

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=jax.nn.relu, batch_norm=False, dropout_rate=0., *, key):
        k1, k2 = jax.random.split(key, 2)

        self.layer_1 = eqx.nn.Conv2d(in_channels, out_channels, kernel_size, padding='SAME', key=k1)
        self.layer_2 = eqx.nn.Conv2d(out_channels, out_channels, kernel_size, padding='SAME', key=k2)
        self.activation = activation
        if batch_norm:
            self.norm_layer = eqx.nn.BatchNorm(input_size=out_channels)
        else:
            self.norm_layer = None
        self.dropout_rate = dropout_rate

    def __call__(self, x):
        x = self.layer_1(x)
        x = self.activation(x)
        x = self.layer_2(x)
        x = self.activation(x)
        if self.norm_layer:
            x = self.norm_layer(x)
        if self.dropout_rate > 0.:
            x = eqx.nn.Dropout(self.dropout_rate)(x)
        return x



class VNet(eqx.Module):
    input_shape: tuple
    output_shape: tuple
    levels: int
    depth: int
    kernel_size: int
    activation: callable
    final_activation: callable
    batch_norm: bool
    dropout_rate: float

    ## Learnable params
    left_doubleconvs: dict
    right_doubleconvs: dict
    downsamplings: dict
    upsamplings: dict
    final_conv: eqx.Module


    def __init__(self, input_shape, output_shape, levels=5, depth=32, kernel_size=5, activation=jax.nn.relu, final_activation=jax.nn.sigmoid, batch_norm=True, dropout_rate=0., *, key):

        l_key, r_key, d_key, u_key, f_key = jax.random.split(key, 5)

        self.input_shape = input_shape      ## C, H, W
        self.output_shape = output_shape    ## C, H, W
        self.levels = levels
        self.depth = depth                  ## Number of filters in the first layer
        self.kernel_size = kernel_size
        self.activation = activation
        self.final_activation = final_activation
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        self.left_doubleconvs = {}
        self.right_doubleconvs = {}
        self.downsamplings = {}
        self.upsamplings = {}
        self.final_conv = eqx.nn.Conv2d(depth, output_shape[0], 1, padding="SAME", key=f_key)


        ## NOTE! The convolutions are not changing the number of channels, the downsampling and upsampling layers are

        d_keys = jax.random.split(d_key, levels-1)
        l_keys = jax.random.split(l_key, levels)

        self.left_doubleconvs[0] = DoubleConv(input_shape[0], depth, kernel_size, activation, batch_norm, dropout_rate, key=l_keys[0])
        for i in range(1, levels):
            self.downsamplings[i] = DownsamplingLayer(self.depth*2**(i-1), self.depth*2**(i), key=d_keys[i-1])
            self.left_doubleconvs[i] = DoubleConv(self.depth*2**(i), self.depth*2**(i), kernel_size, activation, batch_norm, dropout_rate, key=l_keys[i])

        u_keys = jax.random.split(u_key, levels-1)
        r_keys = jax.random.split(r_key, levels-1)

        for i in range(self.levels-2, -1, -1):
            self.upsamplings[i] = UpsamplingLayer(self.depth*2**(i+1), self.depth*2**i, key=u_keys[i])
            self.right_doubleconvs[i] = DoubleConv(self.depth*2**(i), self.depth*2**i, kernel_size, activation, batch_norm, dropout_rate, key=r_keys[i])


    def __call__(self, inputs):
        left = {}
        left[0] = self.left_doubleconvs[0](inputs)
        # print("     - left[0].shape =", left[0].shape)
        for i in range(1, self.levels):
            down = self.downsamplings[i](left[i-1])
            conv = self.left_doubleconvs[i](down)
            left[i] = down + conv
            # if i<self.levels-1:
            #     print(f"     - left[{i}].shape = ", left[i].shape)

        central = left[self.levels-1]
        # print(f"     - central.shape = ", central.shape)

        right = central
        for i in range(self.levels-2, -1,-1):
            up = self.upsamplings[i](right)
            add = left[i] + up
            conv = self.right_doubleconvs[i](add)
            right = up + conv
            # print(f"     - right[{i}].shape =", right.shape)

        return self.final_activation(self.final_conv(right))

############################################################################################################