import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from functools import partial
    from collections import namedtuple

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import animation

    import jax
    from jax import numpy as jnp
    from jax import random

    return jax, jnp, namedtuple, np, partial, plt, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Mixture of gaussians, simple demo distribution
    """)
    return


@app.cell
def _(jnp, np):
    def gaussian(x, norm, mu, sigma):
        """Single Gaussian distribution"""
        return norm * jnp.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * jnp.sqrt(2 * np.pi))


    def gmm(x, norm, mu, sigma):
        """Gaussian Mixture Model"""
        values = jnp.sum(gaussian(x, norm, mu, sigma), axis=0) / mu.shape[0]

        # later we compute the gradient, which requires a returning a scalar value
        if values.shape == (1,):
            return values[0]

        return values

    return (gmm,)


@app.cell
def _(gmm, jnp, plt):
    norm, mu, sigma = (jnp.array([1, 1])[:, None], jnp.array([-1, 1])[:, None], jnp.array([0.25, 0.25])[:, None])
    _x_plot = jnp.linspace(-2, 2, 1000)
    _y = gmm(_x_plot, norm, mu, sigma)
    _ax = plt.subplot()
    _ax.plot(_x_plot, _y)
    _ax.set_xlabel('x')
    _ax.set_ylabel('p(x)')
    return mu, norm, sigma


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using jax we can easily sample from this:
    """)
    return


@app.cell
def _(mu, plt, random, sigma):
    n_samples_ = 500000
    _key = random.key(33)
    x_init = sigma * random.normal(_key, (2, n_samples_ // 2)) + mu  # assumes equal 'norm', use bernoulli to sample from the two gaussians if not
    _ax = plt.subplot()
    _ax.hist(x_init.flatten(), bins=100, density=True, histtype='step', label='Initial samples')
    _ax.set_xlabel('$x_i$')
    _ax.set_ylabel('$p(x_i)$')
    return (x_init,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A utility function borrowed from the blog post we will need later:
    """)
    return


@app.cell
def _(jax, jnp, partial, plt, random):
    default_hist = partial(jnp.histogram, bins=100, range=(-3, 3), density=True)

    batched_histogram = jax.vmap(default_hist)

    def plot_trace(trace, n_traces=5, ax=None, x_min=-3, x_max=3):
        """Plot distribution at multiple points in time as trace"""
        hist_values, _ = batched_histogram(trace)    

        n_iter, n_samples = trace.shape

        ax = plt.subplot() or ax
        ax.imshow(hist_values.T[:, :], extent=[0, n_iter, x_min, x_max], aspect="auto", origin="lower")

        # plot some example traces
        key = random.PRNGKey(9823)
        for idx in random.randint(key, (n_traces,), 0, n_samples):
            ax.plot(trace[:, idx])

        ax.set_ylim(x_min, x_max)
        ax.set_xlabel("# Iteration")
        ax.set_ylabel("x")
        return ax

    return (plot_trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##  Langevin dynamics

    The SDE
    $$
    dx = -D \nabla_x U(x) \, dt + \sqrt{2D} \, dW
    $$

    has a stationary distribution $p(x) \propto e^{-U(x)}$.

    So one way that we can draw samples from a distribution is to run Langevin dynamics with $U(x) = -\log p(x)$:

    $$
    dx = D \nabla_x \log p(x) \, dt + \sqrt{2D} \, dW
    $$

    The vector function $\nabla_x \log p(x)$ is called the score function, and it points in the direction of increasing probability.  So this process drifts toward higher probability regions, while also adding noise to ensure that we explore the space.  Note that the score here is **fixed** — it's the score of the target distribution $p(x)$, not of some time-varying marginal.  We simply run the dynamics long enough to reach equilibrium.

    Note that we need to know the score function to run this process, which is in general not known.  However, in some cases it is: for example we often know a posterior up to a normalizing constant and can compute the score function (which doesn't depend on the normalizing constant). This leads to [Bayesian Learning via Stochastic Gradient Langevin Dynamics](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) which can compete with HMC in certain cases, for example with large data sets since this is compatible with 'batched' learning.

    The discrete-time Langevin update is:
    $$
    x_{t+1} = x_t + \frac{\alpha}{2} \nabla_x \log p(x) + \sqrt{\alpha} \cdot z, \quad z \sim \mathcal{N}(0, I)
    $$

    $\alpha$ is the step size, which can be thought of as a learning rate, which we can decrease toward zero to ensure convergence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this case we **do** know the score function as we can just compute the derivative using jax.
    """)
    return


@app.cell
def _(gmm, jax, jnp, mu, norm, partial, plt, sigma):
    def log_gmm(x, norm, mu, sigma):
        """Log of the GMM"""
        return jnp.log(gmm(x, norm, mu, sigma))
    gmm_log_part = partial(log_gmm, norm=norm, mu=mu, sigma=sigma)
    score_fun = jax.vmap(jax.grad(gmm_log_part))  # the partial just simplifies not parsing the parameters later...
    _ax = plt.subplot()
    _x_plot = jnp.linspace(-2, 2, 1000)
    _ax.plot(_x_plot, score_fun(_x_plot))
    _ax.set_xlabel('x')
    _ax.set_ylabel('d/dx log p(x)')
    return (score_fun,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So we can kind of see the intuition here.. The score function is pushing the samples towards the modes of the distribution, while the noise is adding some randomness to prevent getting stuck in local minima.  As we iterate this process, we should see the samples converging to the true distribution.

    PS: Note the functional style that is used here.  scan works a lot like foldl in Haskell.
    """)
    return


@app.cell
def _(jax, jnp, namedtuple, partial, plot_trace, random, score_fun):
    # JAX can natively handle "structs of arrays" or "PyTrees"
    SampleArgs = namedtuple('Args', ['key', 'idx', 'x', 'alpha_0', 'p_0'])

    def sample(score, args, _):
        alpha = args.alpha_0 * (args.p_0 ** args.idx) ** 2  # compute the "learning rate" depending on the iteration
        key, subkey = random.split(args.key)
        dx = random.normal(subkey, args.x.shape)
        x = args.x + 0.5 * alpha * score(args.x) + jnp.sqrt(alpha) * dx  # sample stochastic update
        return (SampleArgs(key, args.idx + 1, x, args.alpha_0, args.p_0), x)
    n_samples = 100000
    n_iter = 500
    _key = random.PRNGKey(42)  # combine the gradient and the stochastic update
    _key, _subkey = random.split(_key)
    init = SampleArgs(key=_key, idx=0, x=random.normal(_subkey, (n_samples,)), alpha_0=0.002, p_0=1.0)
    result, sample_trace = jax.lax.scan(partial(sample, score_fun), init, length=n_iter)
    plot_trace(sample_trace)
    return SampleArgs, n_iter, n_samples, sample, sample_trace


@app.cell
def _(gmm, jnp, mu, norm, plt, sample_trace, sigma):
    _ax = plt.subplot()
    _ax.hist(sample_trace[-1], density=True, bins=70, label='Samples')
    _x_plot = jnp.linspace(-2, 2, 100)
    _y = gmm(_x_plot, norm, mu, sigma)
    _ax.plot(_x_plot, _y, label='Target distribution')
    _ax.set_xlim()
    _ax.set_xlabel('x')
    _ax.set_ylabel('PDF')
    _ax.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So this works, if we know the score we can use Langevin dynamics to sample from the score function. But for SBI we generally dont know the score function, all we have is draws from the distribution.  So we need to learn the score function from data, which is what score matching is for.  We will get to that in the next section.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Score matching
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In general we dont have access to the score function, we only have access to the data distribution.  But we can still estimate the score function using a technique called score matching, which is a way to train a model to approximate the score function.  The idea is to minimize the following objective:
    $$
    J(\theta) = \mathbb{E}_{p(x)} \left[  \| s_\theta(x) - \nabla_x \log p_\text{data}(x) \|^2 \right]
    $$

    Although we dont have access to the score function, we can still compute the gradient of the log density using the data distribution.  However by integrating by parts (see references) we can show that this objective is equivalent to:
    $$
    J(\theta) = \mathbb{E}_{p(x)} \left[  \|s_\theta(x) \|^2 + 2 tr(\nabla_x  s_\theta(x) )\right] + \text{const}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here the blog post this is based on switches to a different problem, the 'Jelly Roll' dataset, which is a 2D dataset that looks like a jelly roll.  The idea is to train a neural network to approximate the score function for this dataset, and then use it to sample from the reverse process.

    BBut i would rather stick to the original case and see if we can get it to work!
    """)
    return


@app.cell
def _(jnp, random):
    def init_mlp(key, layer_sizes):
        """Initialize MLP parameters with He initialization."""
        params = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            key, subkey = random.split(key)
            w = random.normal(subkey, (in_size, out_size)) * jnp.sqrt(2.0 / in_size)
            b = jnp.zeros(out_size)
            params.append((w, b))
        return params

    def mlp(params, x):
        """MLP forward pass: scalar -> scalar."""
        h = jnp.atleast_1d(x)
        for w, b in params[:-1]:
            h = jnp.tanh(h @ w + b)
        w, b = params[-1]
        return (h @ w + b).squeeze()

    return init_mlp, mlp


@app.cell
def _(jax, mlp):
    def score_matching_loss(params, samples):
        """Implicit score matching loss (Hyvärinen 2005).

        In 1D the general objective simplifies to:
          J = E[ s(x)^2 / 2 + ds/dx ]
        where s(x) is the model's score estimate and ds/dx is its derivative w.r.t. x.
        """
        model = lambda x: mlp(params, x)
        scores = jax.vmap(model)(samples)
        norm_loss = scores ** 2 / 2.0
        score_deriv = jax.vmap(jax.grad(model))(samples)
        return (norm_loss + score_deriv).mean()

    return (score_matching_loss,)


@app.cell
def _(init_mlp, jax, jnp, random, score_matching_loss, x_init):
    import optax
    _key = random.PRNGKey(0)
    params = init_mlp(_key, [1, 128, 128, 1])
    optimizer = optax.adam(0.0003)
    opt_state = optimizer.init(params)
    train_data = x_init.flatten()
    batch_size = 512

    @jax.jit
    def train_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(score_matching_loss)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, loss)
    n_epochs = 50
    losses = []
    for epoch in range(n_epochs):
        _key, _subkey = random.split(_key)
        perm = random.permutation(_subkey, train_data.shape[0])
        shuffled = train_data[perm]
        epoch_losses = []
        for i in range(0, len(shuffled) - batch_size, batch_size):
            batch = shuffled[i:i + batch_size]
            params, opt_state, loss = train_step(params, opt_state, batch)
            epoch_losses.append(loss)
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}, loss: {avg_loss:.4f}')
    print(f'Epoch {n_epochs - 1:3d}, loss: {losses[-1]:.4f}')
    return losses, params


@app.cell
def _(jax, jnp, losses, mlp, params, plt, score_fun):
    learned_score = jax.vmap(lambda x: mlp(params, x))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _x_plot = jnp.linspace(-2.5, 2.5, 500)
    axes[0].plot(_x_plot, score_fun(_x_plot), label='True score')
    axes[0].plot(_x_plot, learned_score(_x_plot), '--', label='Learned score')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('$\\nabla_x \\log p(x)$')
    axes[0].legend()
    axes[1].plot(losses)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    plt.tight_layout()
    fig
    return (learned_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now plug the learned score into the Langevin sampler we already have — starting from pure noise, we should recover the GMM using only the learned score (no access to the true density).
    """)
    return


@app.cell
def _(
    SampleArgs,
    gmm,
    jax,
    jnp,
    learned_score,
    mu,
    n_iter,
    n_samples,
    norm,
    partial,
    plt,
    random,
    sample,
    sigma,
):
    _key = random.PRNGKey(99)
    _key, _subkey = random.split(_key)
    init_learned = SampleArgs(key=_key, idx=0, x=random.normal(_subkey, (n_samples,)), alpha_0=0.002, p_0=1.0)
    _, sample_trace_learned = jax.lax.scan(partial(sample, learned_score), init_learned, length=n_iter)
    _ax = plt.subplot()
    _ax.hist(sample_trace_learned[-1], density=True, bins=70, label='Samples (learned score)')
    _x_plot = jnp.linspace(-2, 2, 100)
    _ax.plot(_x_plot, gmm(_x_plot, norm, mu, sigma), label='Target distribution')
    _ax.set_xlabel('x')
    _ax.set_ylabel('PDF')
    _ax.legend()
    return


if __name__ == "__main__":
    app.run()
