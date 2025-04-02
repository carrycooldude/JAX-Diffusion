import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange

class SelfAttention(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        qkv = nn.Dense(self.dim * 3)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = rearrange(q, "b h w c -> b (h w) c")
        k = rearrange(k, "b h w c -> b (h w) c")
        v = rearrange(v, "b h w c -> b (h w) c")

        attn = jnp.einsum("b i c, b j c -> b i j", q, k) / jnp.sqrt(C)
        attn = nn.softmax(attn, axis=-1)
        out = jnp.einsum("b i j, b j c -> b i c", attn, v)
        return rearrange(out, "b (h w) c -> b h w c", h=H, w=W)

class UNet(nn.Module):
    @nn.compact
    def __call__(self, x, t):
        # Check if t is a scalar (during init) or batched (during training)
        # and handle accordingly
        
        # First, embed the time using sinusoidal embeddings
        t_emb = jnp.sin(t * 10.0)
        
        # Check t's rank and handle accordingly
        rank = len(jnp.shape(t_emb))
        if rank == 0:  # Scalar case (during initialization)
            # Just add the needed dimensions directly
            t_emb = t_emb[None, None, None]
        else:  # Batched case (during training)
            # Add extra dimensions for broadcasting with spatial dimensions
            t_emb = t_emb[:, None, None, None]
        
        # Create a broadcast-compatible time projection
        # Use a constant tensor with the shape of x but with only 1 channel
        # and multiply it by the time embedding
        dummy_spatial = jnp.ones((*x.shape[:-1], 1))
        time_input = dummy_spatial * t_emb
        
        # Project to the right number of channels (64) using a 1x1 conv
        time_proj = nn.Conv(64, kernel_size=(1, 1))(time_input)
        
        # Main UNet architecture
        x1 = nn.Conv(64, (3, 3), padding="SAME")(x)
        x1 = nn.relu(x1 + time_proj)  # Add time information
        
        x2 = nn.Conv(128, (3, 3), strides=(2, 2), padding="SAME")(x1)
        x3 = nn.Conv(256, (3, 3), padding="SAME")(x2)
        x3 = SelfAttention(dim=256)(x3)
        x4 = nn.ConvTranspose(128, (3, 3), strides=(2, 2), padding="SAME")(x3)
        x5 = nn.Conv(64, (3, 3), padding="SAME")(x4)
        return nn.Conv(3, (3, 3), padding="SAME")(x5)