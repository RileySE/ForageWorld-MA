import jax
import jax.numpy as jnp
import numpy as np
import hashlib


def load_rendering_resources(env_name, pixel_size_preference=16):
    if "Coop" in env_name:
        from craftax.craftax_coop.constants import TEXTURES, load_player_specific_textures
    else:
        from craftax.craftax_ma.constants import TEXTURES, load_player_specific_textures
    
    # Select pixel size
    available_sizes = list(TEXTURES.keys())
    if pixel_size_preference in available_sizes:
        pixel_size = pixel_size_preference
    elif 16 in available_sizes:
        pixel_size = 16
    elif 64 in available_sizes:
        pixel_size = 64
    else:
        pixel_size = available_sizes[0]
    
    print(f"Pixel size: {pixel_size}")
    
    return {
        "TEXTURES": TEXTURES,
        "pixel_size": pixel_size,
        "load_player_specific_textures": load_player_specific_textures
    }


def create_rng_fixed_state(state, fixed_key_value=None):
    if fixed_key_value is None:
        fixed_key_value = int(hashlib.md5(b"fixed_render_key").hexdigest()[:8], 16)
    
    return state.replace(state_rng=jax.random.PRNGKey(fixed_key_value))


def convert_pixels_to_uint8(pixels):
    pixels_np = np.array(pixels)
    # Pixels are already in [0, 255] range, just convert dtype
    return pixels_np.astype(np.uint8)


def print_rendering_info(player_specific_textures, num_agents, env_name):
    print(f"Environment: {env_name}")
    print(f"Agents: {num_agents}")
