#%%
import numpy as np
import jax

## Use jax cpu
# jax.config.update("jax_platform_name", "cpu")


import os
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax.numpy as jnp
import time


# import os
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=32'    ## Trick to virtualise CPU for pmap

print("Jax version:", jax.__version__)
print("Jax devices are:", jax.devices())


@jax.jit
def f(x):  # function we're benchmarking (works in both NumPy & JAX)
  return x.T @ (x - x.mean(axis=0))

start = time.time()
for _ in range(1000):
  x_np = np.ones((1000, 1000), dtype=np.float32)  # same as JAX default dtype
f(x_np)  # measure NumPy runtime

# x_jax = jax.device_put(x_np)  # measure JAX device transfer time
# f_jit = jax.jit(f)
# f_jit(x_jax).block_until_ready()  # measure JAX compilation time
# f_jit(x_jax).block_until_ready()  # measure JAX runtime

end = time.time()
print("Total time:", end-start)
