#%%
## Build typical neural networks

import jax
import jax.numpy as jnp
import equinox as eqx


class LaplacianNet(eqx.Module):
    """
    Laplacian weighted by a leanable coaeffcient
    """

    coaf: float

    def __init__(self, N=None, key=None):
        pass

    def __call__(self, x):
        pass
