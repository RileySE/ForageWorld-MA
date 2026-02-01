import jax
import jax.numpy as jnp
import numpy as np
import time
import datetime
from typing import Callable, Dict, Any, Optional

from .base_rendering import load_rendering_resources
from .ego_rendering import render_ego_perspective
from .full_map_rendering import render_full_map
from .episode_recorder import VideoRecorder, save_videos


def render_policy_episode(
    env,
    policy_fn: Callable,
    config: Dict[str, Any],
    update_num: Optional[int] = None,
    verbose: bool = True
):
    if verbose:
        print("\nRendering evaluation episode...")
    
    render_start = time.time()
    
    # Load rendering resources
    env_name = config.get("ENV_NAME", "")
    rendering_resources = load_rendering_resources(env_name, pixel_size_preference=16)
    pixel_size = rendering_resources["pixel_size"]
    TEXTURES = rendering_resources["TEXTURES"]
    player_specific_textures = rendering_resources["load_player_specific_textures"](
        TEXTURES[pixel_size],
        env.num_agents
    )
    
    # Create rendering functions
    def render_ego_fn(state):
        return render_ego_perspective(
            state, pixel_size, env.static_env_params,
            player_specific_textures, env_name
        )
    
    def render_fullmap_fn(state):
        return render_full_map(
            state, env.static_env_params,
            TEXTURES[pixel_size], player_specific_textures, pixel_size
        )
    
    # Initialize video recorder
    max_steps = config.get("MAX_EPISODE_STEPS", 1000)
    recorder = VideoRecorder(
        env.num_agents,
        render_ego=config.get("RENDER_EGO_PERSPECTIVE", True),
        render_full_map=config.get("RENDER_FULL_MAP", True)
    )
    
    # Reset environment
    seed_offset = (update_num or 0) + 999
    test_rng = jax.random.PRNGKey(config.get("SEED", 0) + seed_offset)
    test_rng, reset_rng = jax.random.split(test_rng)
    obs, state = env.reset(reset_rng)
    
    # Initialize episode termination tracking
    done = {agent: False for agent in env.agents}
    done["__all__"] = False
    
    # Execute episode rollout
    for step in range(max_steps):
        # Render current frame for ego-perspective view
        if config.get("RENDER_EGO_PERSPECTIVE", True):
            ego_pixels = render_ego_fn(state)
            ego_pixels_np = np.array(ego_pixels).astype(np.uint8)
            for agent_idx in range(env.num_agents):
                recorder.add_ego_frame(agent_idx, ego_pixels_np[agent_idx])
        
        # Render current frame for full map view
        if config.get("RENDER_FULL_MAP", True):
            fullmap_pixels = render_fullmap_fn(state)
            fullmap_np = np.array(fullmap_pixels).astype(np.uint8)
            recorder.add_full_map_frame(fullmap_np)
        
        # Check episode termination
        if done["__all__"]:
            break
        
        # Query policy for action
        # Note: Policy function manages its own internal state (e.g., RNN hidden states)
        test_rng, action_rng = jax.random.split(test_rng)
        action_dict = policy_fn(action_rng, obs, state, done)
        
        # Execute environment step
        test_rng, step_rng = jax.random.split(test_rng)
        obs, state, reward, done, info = env.step(step_rng, state, action_dict)
    
    # Determine output directory
    run_id = config.get("RUN_ID", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if update_num is not None:
        env_step = update_num * config.get('NUM_STEPS', 1) * config.get('NUM_ENVS', 1)
        output_dir = f"videos/{run_id}/step_{env_step}"
    else:
        output_dir = f"videos/{run_id}/final"
    
    # Save rendered videos to disk
    save_videos(recorder, output_dir=output_dir, fps=10, video_format="mp4")
    
    render_time = time.time() - render_start
    
    if verbose:
        print(f"Rendering completed in {render_time:.2f}s")
        if update_num is not None:
            env_step = update_num * config.get('NUM_STEPS', 1) * config.get('NUM_ENVS', 1)
            print(f"Saved to {output_dir} (environment step {env_step})\n")
        else:
            print(f"Saved to {output_dir}\n")
