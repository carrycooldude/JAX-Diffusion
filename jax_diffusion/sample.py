# sample.py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
from unet import UNet
from diffusion import Diffusion

# Simple TrainState class matching the one in train.py
class TrainState:
    def __init__(self, params, model, diffusion, optimizer, opt_state):
        self.params = params
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.opt_state = opt_state

    def replace(self, **kwargs):
        return TrainState(**{**self.__dict__, **kwargs})

def sample(state, num_samples=10):
    """Generates images from pure noise using the reverse diffusion process."""
    key = jax.random.PRNGKey(42)
    # Start from pure noise: CIFAR-10 images are 32x32 with 3 channels
    xt = jax.random.normal(key, (num_samples, 32, 32, 3))
    # Iterate backwards from T to 1
    for t in range(state.diffusion.timesteps - 1, 0, -1):
        key, subkey = jax.random.split(key)
        xt = state.diffusion.reverse(xt, t, state.model, state.params, subkey)
    return xt

def main():
    # Attempt to load the checkpoint saved by the training script
    try:
        with open("checkpoint.pkl", "rb") as f:
            state = pickle.load(f)
    except FileNotFoundError:
        print("No checkpoint found. Please train the model first and save a checkpoint.")
        return

    # Generate samples
    samples = sample(state, num_samples=10)

    # Display generated images
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i, ax in enumerate(axes):
        # Convert images from [-1, 1] to [0, 1] for display
        ax.imshow((samples[i] + 1) / 2)
        ax.axis("off")
    plt.show()

if __name__ == "__main__":
    main()