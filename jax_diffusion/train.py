import jax
import jax.numpy as jnp
import optax
import pickle
from unet import UNet
from diffusion import Diffusion
from dataset import load_cifar10
import flax.struct

@flax.struct.dataclass
class TrainState:
    params: any
    opt_state: any

def train(epochs=5000, lr=2e-4):
    dataset = load_cifar10()
    model = UNet()
    diffusion = Diffusion()
    
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3)), 0)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # Create initial state - only include JAX-compatible arrays
    state = TrainState(params=params, opt_state=opt_state)
    
    # Replicate the state for pmap
    devices = jax.local_devices()
    num_devices = len(devices)
    state = jax.device_put_replicated(state, devices)

    # Define the loss function
    def loss_fn(params, x0, key):
        # Sample random timesteps for each sample in the batch
        t = jax.random.randint(key, (x0.shape[0],), 0, diffusion.timesteps)
        xt, noise = diffusion.forward(x0, t, key)
        pred_noise = model.apply(params, xt, t)
        return jnp.mean((pred_noise - noise) ** 2)

    # Define the pmapped training step
    @jax.pmap
    def train_step(state, batch, key):
        # Compute loss and gradients, then update parameters
        loss, grads = jax.value_and_grad(loss_fn)(state.params, batch, key)
        updates, new_opt_state = optimizer.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        return TrainState(params=new_params, opt_state=new_opt_state), loss

    for epoch in range(epochs):
        # Use a single key for randint
        single_key = jax.random.PRNGKey(epoch)
        # Adjust batch size to be divisible by number of devices
        batch_size = 32 * num_devices
        indices = jax.random.randint(single_key, (batch_size,), 0, dataset.shape[0])
        full_batch = dataset[indices]
        
        # Reshape batch for pmap: (num_devices, per_device_batch, ...)
        batch = full_batch.reshape(num_devices, -1, *full_batch.shape[1:])
        
        # Create a separate key for the pmapped train_step
        pmap_key = jax.random.split(jax.random.PRNGKey(epoch + 1000), num_devices)
        state, loss = train_step(state, batch, pmap_key)

        if epoch % 500 == 0:
            # loss is replicated, gather from first device
            print(f"Epoch {epoch}, Loss: {loss[0]:.4f}")

    # Save checkpoint after training is complete
    # Note: You may want to gather state from devices before saving
    state_for_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
    with open("checkpoint.pkl", "wb") as f:
        pickle.dump(state_for_save, f)

    return state

if __name__ == "__main__":
    train()
