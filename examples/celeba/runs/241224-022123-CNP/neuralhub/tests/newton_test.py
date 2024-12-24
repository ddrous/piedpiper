# %%
import jax
import jax.numpy as jnp
import numpy as np

# %%

def newton(x, f, gf, hf, lr=0.01, lr_decr=0.999, maxiter=10, tol=0.001):

    nit = 0

    ### Put everything above in a jax fori lopp
    def body_fn(carry):
        _, x, lr, errors, nit = carry
        gradient = gf(x)
        hessian = hf(x).squeeze()

        # x_new = x - lr*jnp.linalg.solve(hessian, gradient)
        x_new = x - lr * jnp.linalg.inv(hessian)@gradient

        # jax.debug.print("x_new {}", x)

        errors = errors.at[nit+1].set(jnp.linalg.norm(x_new - x))
        return x, x_new, lr*lr_decr, errors, nit+1

    def cond_fn(carry):
        _, _, _, errors, nit = carry
        return (errors[nit] >= tol) & (nit < maxiter)

    errors = jnp.zeros((maxiter,))
    errors = errors.at[0].set(2*tol)

    _, x, lr, errors, nit = jax.lax.while_loop(cond_fn, body_fn, (x, x, lr, errors, nit))

    return x, None, errors, nit

# %%

### This works in 2 steps only !!

@jax.jit
def f(x):   ## The function to minimize
    x1, x2 = x
    return (x1 - 6)**2 + x2**2

gf = jax.jacrev(f, argnums=0)
hf = jax.jacrev(gf, argnums=0)
x = jnp.array([0.5, 3.])

x, _, errors, nit = newton(x, f, gf, hf, tol=1e-10, maxiter=20, lr=1e-0, lr_decr=0.999)

assert errors[nit] < 1e-10
assert jnp.allclose(x, jnp.array([6., 0.]))

print("\nNewton results", x, nit, errors[:])

# %%


### A more complex example. This doens't work !

# Generate random positive definite matrix A
A = np.random.rand(1000, 1000)
A = np.dot(A, A.T)
# Generate random vector b
b = np.random.rand(1000, 1)

@jax.jit
def f(x):
    res = 0.5 * jnp.dot(x.T, jnp.dot(A, x)) - jnp.dot(b.T, x)
    return res.squeeze()

gf = jax.jacrev(f, argnums=0)
hf = jax.jacrev(gf, argnums=0)
x = np.random.rand(1000,1)


## Perform gradient descent to start with
for _ in range(100):
    x = x - 0.001*gf(x)

x, _, errors, nit = newton(x, f, gf, hf, tol=1e-10, maxiter=20, lr=1e-0, lr_decr=0.999)

print("\nNewton results", nit, errors[:])
# %%
