#%%

import jax
print(jax.devices())

def dot_product_benhmark():
    size = 1000
    x = jax.random.normal(jax.random.PRNGKey(0), (size, ))
    M = jax.random.normal(jax.random.PRNGKey(1), (size, size))
    return jax.numpy.dot(M, x)

%timeit -n5 -r20 dot_product_benhmark()