import tensorflow_datasets as tfds
import numpy as np
import jax.numpy as jnp

def load_cifar10():
    ds = tfds.load("cifar10", split="train", as_supervised=True)
    def preprocess(image, _):
        image = tfds.as_numpy(image).astype(jnp.float32) / 127.5 - 1.0
        return image
    return np.array([preprocess(img, lbl) for img, lbl in ds])