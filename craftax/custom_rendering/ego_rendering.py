import jax
import jax.numpy as jnp
import hashlib


def create_rng_fixed_wrapper(render_fn, fixed_key_value=None):
    if fixed_key_value is None:
        fixed_key_value = int(hashlib.md5(b"fixed_render_key").hexdigest()[:8], 16)
    
    def render_craftax_pixels_no_flicker(state, pixel_size, static_params, player_textures):
        state_fixed = state.replace(state_rng=jax.random.PRNGKey(fixed_key_value))
        return render_fn(state_fixed, pixel_size, static_params, player_textures)
    
    return render_craftax_pixels_no_flicker


def render_ego_perspective(state, pixel_size, static_params, player_textures, env_name):
    # Import the appropriate renderer based on environment type
    if "Coop" in env_name:
        from craftax.craftax_coop.renderer.renderer_pixels import render_craftax_pixels
    else:
        from craftax.craftax_ma.renderer.renderer_pixels import render_craftax_pixels
    
    # Create RNG-fixed wrapper
    render_fn = create_rng_fixed_wrapper(render_craftax_pixels)
    
    # Render with fixed RNG to prevent flickering
    return render_fn(state, pixel_size, static_params, player_textures)
