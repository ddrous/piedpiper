#%%
import jax
import jax.numpy as jnp
import numpy as np
jax.numpy.set_printoptions(precision=6)



# %% [markdown]

## Simple vs Augmented Method of Multipliers

# ## References
# - **Simple method**: Augmenting Physical Models with Deep Networks for Complex Dynamics Forecasting (Yin et al., 2021) (Algorithm 1, page 7)
# - **Augmented method**: Practical augmented Lagrangian methods for constrained optimization (Birgin and Martinez, 2014) (Algorithm 4.1, page 33)

# ## Conclusions
# - The augmented method is much more faster to convergence than the simple method
# - The augmented method enfores equality constraints much much better


# Cherry-picked example to highlight the benefits of the augmented method of multipliers from Birgin et al., 2014, which is compared to the simple method from https://arxiv.org/abs/2010.04456, 2021. Potential techniques to add to one's Deep Learning arsenal:  https://gist.github.com/ddrous/cd258baeaebdc29529e329f660ab3760

# The augmented method is much faster, while enforcing the equality constraint h(x,y)=0 in a stronger way.

# %%

#### Simple method of multipliers

@jax.jit
def f(x):   ## The function to minimize
    x1, x2 = x
    return (x1 - 6)**2 + x2**2

@jax.jit
def h(x):   ## The equality constraint
    x1, x2 = x
    return (x2-(x1/4)**2)**2 + ((x1/4)-1)**2 - 1

tau1 = 1e-3 ## Actual learning rate for both methods
tau2 = 1e-3 ## Multiplicative factor for the penalty term in the simple method, used as tau in the augmented method

nb_iter_out = 200
nb_iter_in = 100
tol = 1e-10

@jax.jit
def inner_train_step(x, lamb, tau1, tau2):
    return x - tau1 * (lamb*jax.grad(f)(x) + jax.grad(h)(x))

@jax.jit
def outer_train_step(x, lamb, tau1, tau2):
    return lamb + tau2 * h(x)


lamb = 1.2
x = jnp.array([0.5, 3.])
iter_count = 0
path = [x]

for out_iter in range(nb_iter_out):

    x_old = x

    for in_iter in range(nb_iter_in):
        x_new = inner_train_step(x, lamb, tau1, tau2)

        if jnp.linalg.norm(x_new - x) < tol:
            break

        iter_count += 1
        x = x_new
        path.append(x)

    lamb = outer_train_step(x, lamb, tau1, tau2)

    if jnp.linalg.norm(x_new - x_old) < tol:
        break

    print(f"iter {out_iter:-3d} :   x={x}   f(x)={f(x):.6f}     h(x)={h(x):+.6f}    lambda={lamb}")

print(f"\nTotal number of iterations to achieve a tol of {tol} is: {iter_count}")

path = jnp.vstack(path)

# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='notebook', style='ticks',
        font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})
plt.style.use("dark_background")
from matplotlib.animation import FuncAnimation

def plot_optimisation_path(path, title='Method of Multipliers'):

    # Create a grid of points for contour plot
    x = np.linspace(0, 10, 100)
    y = np.linspace(-3, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(np.vstack([X.ravel(), Y.ravel()]))

    # Reshape for contour plot
    Z = Z.reshape(X.shape)

    ## Set the size of the plt
    plt.figure(figsize=(16, 8))

    # Plot contours
    plt.contourf(X, Y, Z, levels=30, cmap='grey')
    plt.colorbar(label='Function Value')

    Z = h(np.vstack([X.ravel(), Y.ravel()]))
    Z = Z.reshape(X.shape)

    # Plot contours
    plt.contour(X, Y, Z, levels=4, cmap='Blues')
    plt.colorbar(label="Constraint's countrours")

    plt.plot(path[:, 0], path[:, 1], 'rX-', lw=3, markevery=100, label='Optimisation path')

    # Add labels and legend
    plt.title(title)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()

    # Show the plot
    plt.show();

# plot_optimisation_path(path)

# %%

#### Simple method of multipliers

@jax.jit
def L(x, lamb, rho):
    return f(x) + 0.5*rho*(h(x) + lamb/rho)**2

@jax.jit
def inner_train_step_(x, lamb, rho, tau1, tau2):
    return x - tau1 * jax.grad(L)(x, lamb, rho)

lamb_min, lamb_max = -10, 10
gamma = 0.95

lamb = 5.
rho = 1.

x = jnp.array([0.5, 3.])
path_ = [x]
iter_count = 0

for k in range(nb_iter_out):

    x_old = x

    for i in range(nb_iter_in):
        x_new = inner_train_step_(x, lamb, rho, tau1, tau2)
        
        if jnp.linalg.norm(x_new - x) < tol:
            break

        iter_count += 1
        x = x_new
        path_.append(x)

    lamb = jnp.clip(lamb + rho*h(x_new), lamb_min, lamb_max)

    norm_h_old = jnp.linalg.norm(h(x_old))
    norm_h = jnp.linalg.norm(h(x_new))

    if k==0 or norm_h_old < tau2*norm_h:
        rho = 2*rho
    else:
        rho = gamma*rho

    if jnp.linalg.norm(x_new - x_old) < tol:
        break

    print(f"iter: {k:-3d}   x={x}   f(x)={f(x):.6f}   h(x)={h(x):+.6f}   rho={rho:.6f}    lambda={lamb:.6f}")


print(f"\nTotal number of iterations to achieve a tol of {tol} is: {iter_count}")

path_ = jnp.vstack(path_)
plot_optimisation_path(path_, "Method of Augmented Multipliers")


# %%

def animate_optimization_path(paths, labels=["Simple", "Augmented"], title='Simple vs Augmented Multipliers'):
    # Create a grid of points for contour plot
    x = np.linspace(0, 10, 100)
    y = np.linspace(-3, 5, 100)
    X, Y = np.meshgrid(x, y)

    ## Function to minimize
    Z = f(np.vstack([X.ravel(), Y.ravel()]))
    Z = Z.reshape(X.shape)

    ## Constraint
    Z_ = h(np.vstack([X.ravel(), Y.ravel()]))
    Z_ = Z_.reshape(X.shape)
    
    # Set the size of the plt
    plt.figure(figsize=(12, 8))

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    skip = 100      ## TODO: change this for more fluid simulations

    def update(frame):
        plt.clf()  # Clear the plot for the next frame
        plt.contourf(X, Y, Z, levels=50, cmap='gray')
        plt.colorbar(label=r'Function value $f(x,y)$')

        contour = plt.contour(X, Y, Z_, levels=[0], cmap='Blues')
        plt.clabel(contour, inline=True, fontsize=8, fmt=r'$h(x,y)={:1.0f}$'.format(contour.levels[0]))

        for i, path in enumerate(paths):
            if frame < len(path):
                plt.plot(path[:frame*skip, 0], path[:frame*skip, 1], '-', color=colors[i%8], lw=4, markevery=100, label=labels[i])

                x_val, y_val = path[frame*skip, 0], path[frame*skip, 1]
                f_val = f(np.array([x_val, y_val]))
                h_val = h(np.array([x_val, y_val]))

                plt.text(0.5+i*(3.5), 4.5+i*(-4.5-3), r'$f*={:.3f}, \quad  h*={:+.5f}$'.format(f_val, h_val), fontsize=20, color=colors[i%8], ha='left', va='bottom', weight='bold')

        plt.title(title)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.legend()

    frames = max(len(path) for path in paths)//skip
    ani = FuncAnimation(plt.gcf(), update, frames=frames, interval=1000, repeat=False)

    duration = 5  # seconds

    # Save the animation as a GIF
    ani.save("data/multipliers.gif", writer='ffmpeg', fps=frames//duration, dpi=300)

    plt.show()

animate_optimization_path([path[:5000], path_[:5000]], labels=["Simple", "Augmented"], title='Simple vs. Augmented Method of Multipliers')
