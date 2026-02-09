# Score Matching and Langevin Dynamics
A simple 1D example demonstrating how to sample from an unknown distribution using score matching and Langevin dynamics, implemented in JAX. Based in part on [A Beginner's Friendly Introduction to Diffusion Models in JAX](https://axeldonath.com/jax-diffusion-models-pydata-boston-2025/) by Axel Donath.

The notebook `langevin_demo.ipynb` covers:
1. **Langevin dynamics** — sampling from a distribution using only its score function $\nabla_x \log p(x)$
2. **Score matching** — learning the score function from data (no access to the density), using the implicit score matching loss (Hyvärinen 2005)
3. **Putting it together** — plugging the learned score into Langevin sampling to recover the target distribution from pure noise

The target distribution is a simple 1D Gaussian mixture model. Everything runs on CPU.

## Future ? 

[ ] Add section on Denoising Diffusion Probabilistic Models (DDPMs) and how they relate to this framework.  This is the basis for modern diffusion models like Stable Diffusion, and is a more general framework that includes a time-varying noise schedule and a parameterized score function (the "denoiser") that is trained to match the score at each noise level.

## Resources

- [Original Tutorial](https://axeldonath.com/jax-diffusion-models-pydata-boston-2025/)
- [JAX Documentation](https://docs.jax.dev/en/latest/)
- [Bayesian Learning via Stochastic Gradient Langevin Dynamics](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)
- [Yang Song's Blog Post on Score-Based Generative Modeling](https://yang-song.net/blog/2021/score/)
- [Recommended Prep Videos by Deepia](https://www.youtube.com/@Deepia-ls2fo)
