import jax
import jax.numpy as jnp

class Diffusion:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.beta = jnp.exp(jnp.linspace(jnp.log(0.0001), jnp.log(0.02), timesteps))
        self.alpha = 1.0 - self.beta
        self.alpha_bar = jnp.cumprod(self.alpha)

    def forward(self, x0, t, key):
        """
        Forward diffusion process: q(x_t | x_0)
        Add noise to the input image according to the noise schedule
        """
        noise = jax.random.normal(key, x0.shape)
        
        # Reshape t to have proper dimensions for broadcasting with x0
        t_reshaped = t.reshape(-1, *([1] * (len(x0.shape) - 1)))
        
        # Now properly broadcast
        alpha_bar_t = self.alpha_bar[t_reshaped]
        
        # Apply the forward diffusion formula
        return jnp.sqrt(alpha_bar_t) * x0 + jnp.sqrt(1 - alpha_bar_t) * noise, noise

    def reverse(self, xt, t, model, params, key):
        pred_noise = model.apply(params, xt, t)
        mean = (xt - (1 - self.alpha[t]) / jnp.sqrt(1 - self.alpha_bar[t]) * pred_noise) / jnp.sqrt(self.alpha[t])
        return mean + jnp.sqrt(self.beta[t]) * jax.random.normal(key, xt.shape)
