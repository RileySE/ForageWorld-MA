import os
import jax
import jax.numpy as jnp
import numpy as np
import imageio
from typing import Callable, Dict, Any, Optional, List, Tuple


class VideoRecorder:
    
    def __init__(self, num_agents: int, render_ego: bool = True, render_full_map: bool = False):
        self.num_agents = num_agents
        self.render_ego = render_ego
        self.render_full_map = render_full_map
        
        # Storage for frames
        self.frames_per_agent = {i: [] for i in range(num_agents)} if render_ego else {}
        self.frames_full_map = [] if render_full_map else None
        
    def add_ego_frame(self, agent_idx: int, frame: np.ndarray):
        if self.render_ego:
            self.frames_per_agent[agent_idx].append(frame)
    
    def add_full_map_frame(self, frame: np.ndarray):
        if self.render_full_map:
            self.frames_full_map.append(frame)
    
    def get_frame_counts(self) -> Dict[str, int]:
        counts = {}
        if self.render_ego:
            for agent_idx in range(self.num_agents):
                counts[f"agent_{agent_idx}"] = len(self.frames_per_agent[agent_idx])
        if self.render_full_map:
            counts["full_map"] = len(self.frames_full_map)
        return counts


def save_videos(
    recorder: VideoRecorder,
    output_dir: str = "videos",
    fps: int = 10,
    video_format: str = "mp4"
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    print("Saving videos...")
    
    # Save ego-perspective videos
    if recorder.render_ego:
        
        for agent_idx in range(recorder.num_agents):
            frames = recorder.frames_per_agent[agent_idx]
            if len(frames) == 0:
                continue
            
            # Try MP4 first, fallback to GIF if needed
            try:
                if video_format == "mp4":
                    video_path = os.path.join(output_dir, f"agent_{agent_idx}_episode.mp4")
                    imageio.mimsave(video_path, frames, fps=fps, codec='libx264')
                else:
                    video_path = os.path.join(output_dir, f"agent_{agent_idx}_episode.gif")
                    imageio.mimsave(video_path, frames, fps=fps, loop=0)
                
                print(f"  {video_path}")
                saved_paths.append(video_path)
            except Exception as e:
                video_path = os.path.join(output_dir, f"agent_{agent_idx}_episode.gif")
                imageio.mimsave(video_path, frames, fps=fps, loop=0)
                print(f"  {video_path} (fallback to GIF)")
                saved_paths.append(video_path)
    
    # Save full map video
    if recorder.render_full_map:
        frames = recorder.frames_full_map
        
        if len(frames) > 0:
            try:
                if video_format == "mp4":
                    video_path = os.path.join(output_dir, "full_map_episode.mp4")
                    imageio.mimsave(video_path, frames, fps=fps, codec='libx264')
                else:
                    video_path = os.path.join(output_dir, "full_map_episode.gif")
                    imageio.mimsave(video_path, frames, fps=fps, loop=0)
                
                print(f"  {video_path}")
                saved_paths.append(video_path)
            except Exception as e:
                video_path = os.path.join(output_dir, "full_map_episode.gif")
                imageio.mimsave(video_path, frames, fps=fps, loop=0)
                print(f"  {video_path} (fallback to GIF)")
                saved_paths.append(video_path)
    
    return saved_paths


def record_episode(
    env,
    initial_state,
    action_fn: Callable,
    render_ego_fn: Optional[Callable] = None,
    render_full_map_fn: Optional[Callable] = None,
    max_steps: int = 1000,
    rng_key = None,
    render_ego: bool = True,
    render_full_map: bool = False,
    verbose: bool = True
) -> VideoRecorder:
    if verbose:
        mode_str = []
        if render_ego:
            mode_str.append("ego-view")
        if render_full_map:
            mode_str.append("full-map")
        print(f"Recording episode ({', '.join(mode_str)}, max {max_steps} steps)...")
    
    # Initialize recorder
    recorder = VideoRecorder(env.num_agents, render_ego, render_full_map)
    
    # Initialize state
    state = initial_state
    done = {agent: False for agent in env.agents}
    done["__all__"] = False
    
    # Initialize RNG if needed
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    for step in range(max_steps):
        # Render ego-perspective
        if render_ego and render_ego_fn is not None:
            pixels = render_ego_fn(state)
            pixels_np = np.array(pixels).astype(np.uint8)
            
            for agent_idx in range(env.num_agents):
                frame = pixels_np[agent_idx]
                recorder.add_ego_frame(agent_idx, frame)
        
        # Render full map
        if render_full_map and render_full_map_fn is not None:
            full_map_pixels = render_full_map_fn(state)
            full_map_np = np.array(full_map_pixels).astype(np.uint8)
            recorder.add_full_map_frame(full_map_np)
        
        # Get actions from action function
        rng_key, action_rng = jax.random.split(rng_key)
        
        # Get current observation (needed for policy-based actions)
        obs = env.get_obs(state)
        action_dict = action_fn(action_rng, state, obs)
        
        # Step environment
        rng_key, step_rng = jax.random.split(rng_key)
        obs, state, reward, done, info = env.step(step_rng, state, action_dict)
        
        if done["__all__"]:
            break
    
    if verbose:
        print(f"Recorded {step + 1} steps")
    
    return recorder


def create_random_action_fn(env, num_agents: int) -> Callable:
    def random_actions(rng, state, obs):
        action_keys = jax.random.split(rng, num_agents)
        action_dict = {}
        
        for i, agent in enumerate(env.agents):
            action_dict[agent] = jax.random.randint(
                action_keys[i],
                shape=(),
                minval=0,
                maxval=env.action_space(agent).n
            ).item()
        
        return action_dict
    
    return random_actions
