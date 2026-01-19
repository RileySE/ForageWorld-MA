"""
Code is adapted from the IPPO RNN implementation of JaxMARL (https://github.com/FLAIROx/JaxMARL/tree/main) 
Credit goes to the original authors: Rutherford et al.

Modified to use SEPARATE network parameters per agent (no parameter sharing).
Each agent has its own ActorCriticRNN with independent parameters.
Gradient clipping is done PER-AGENT to avoid coupling through global norm computation.
"""

# ===========================
# Imports and Configuration
# ===========================
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import functools
import yaml
from typing import Sequence, NamedTuple, Dict

import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

import optax
import distrax

import wandb

from jaxmarl.wrappers.baselines import LogWrapper
from craftax.craftax_env import make_craftax_env_from_name

# ===========================
# Model Definitions
# ===========================
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

# ===========================
# Data Structures and Utilities
# ===========================
class Transition(NamedTuple):
    """Full transition including info for logging."""
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

class TrainBatch(NamedTuple):
    """Batch for PPO update (without info to avoid minibatch issues)."""
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray

def batchify(x: dict, agent_list):
    """Stack agent observations, preserving agent dimension.
    
    Returns shape: (num_agents, num_envs, obs_dim)
    """
    return jnp.stack([x[a] for a in agent_list], axis=0)

def unbatchify(x: jnp.ndarray, agent_list):
    """Convert stacked array back to agent dict.
    
    Input shape: (num_agents, num_envs, ...) or (num_agents, num_envs)
    """
    return {a: x[i] for i, a in enumerate(agent_list)}

# ===========================
# Training Function
# ===========================
def make_train(config, env):
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # Per-agent gradient clipping to avoid coupling agents through global norm
    def per_agent_clip_by_global_norm(max_norm):
        """Clip gradients per agent independently, not across all agents."""
        def init_fn(params):
            del params
            return optax.EmptyState()
        
        def update_fn(updates, state, params=None):
            del params
            # updates has shape (num_agents, ...) for each leaf
            # We need to clip each agent's gradients independently
            
            def clip_single_agent(agent_grads):
                # Compute norm for this agent only
                leaves = jax.tree_util.tree_leaves(agent_grads)
                sum_of_squares = sum(jnp.sum(jnp.square(x)) for x in leaves)
                norm = jnp.sqrt(sum_of_squares)
                # Clip
                scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
                return jax.tree_util.tree_map(lambda x: x * scale, agent_grads)
            
            # Vmap over the agent dimension (axis 0 of each leaf)
            clipped_updates = jax.vmap(clip_single_agent)(updates)
            return clipped_updates, state
        
        return optax.GradientTransformation(init_fn, update_fn)

    def train(rng):
        # INIT NETWORK - separate params per agent
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)
        
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        
        # Initialize separate params for each agent using vmap
        agent_rngs = jax.random.split(_rng, env.num_agents)
        
        def init_single_agent(agent_rng):
            return network.init(agent_rng, init_hstate, init_x)
        
        # Stacked network variables: leading dim is num_agents
        stacked_network_variables = jax.vmap(init_single_agent)(agent_rngs)
        # Extract only params (network.init returns {"params": ...})
        stacked_network_params = stacked_network_variables["params"]
        
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                per_agent_clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                per_agent_clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=stacked_network_params,  # (num_agents, ...) - only params, not full variables
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        # Hidden state shape: (num_agents, num_envs, hidden_dim)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        init_hstate = jnp.tile(init_hstate[np.newaxis, :, :], (env.num_agents, 1, 1))
        # Initial done flags as dict (will be converted to array in _env_step)
        init_done = {a: jnp.zeros((config["NUM_ENVS"],), dtype=bool) for a in env.agents}

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                # obs_batch shape: (num_agents, num_envs, obs_dim)
                obs_batch = batchify(last_obs, env.agents)
                # done_batch shape: (num_agents, num_envs)
                # last_done is a dict from env, convert to array
                done_batch_in = batchify(last_done, env.agents)
                
                # Forward pass for each agent with their own params
                # ac_in: (1, num_envs, obs_dim), (1, num_envs)
                # hstate: (num_agents, num_envs, hidden_dim)
                def forward_single_agent(params, hs, obs, done):
                    ac_in = (obs[np.newaxis, :], done[np.newaxis, :])
                    return network.apply({"params": params}, hs, ac_in)
                
                hstate, pi, value = jax.vmap(forward_single_agent)(
                    train_state.params,  # (num_agents, ...)
                    hstate,              # (num_agents, num_envs, hidden_dim)
                    obs_batch,           # (num_agents, num_envs, obs_dim)
                    done_batch_in,       # (num_agents, num_envs)
                )
                # pi.logits shape: (num_agents, 1, num_envs, action_dim)
                # value shape: (num_agents, 1, num_envs)
                
                # Sample actions - distrax is batch-aware, sample directly
                # pi.logits: (num_agents, 1, num_envs, action_dim)
                action = pi.sample(seed=_rng)  # (num_agents, 1, num_envs)
                log_prob = pi.log_prob(action)  # (num_agents, 1, num_envs)
                
                action = action.squeeze(axis=1)      # (num_agents, num_envs)
                log_prob = log_prob.squeeze(axis=1)  # (num_agents, num_envs)
                value = value.squeeze(axis=1)        # (num_agents, num_envs)
                
                env_act = unbatchify(action, env.agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                
                done_batch = batchify(done, env.agents)  # (num_agents, num_envs)
                reward_batch = batchify(reward, env.agents)  # (num_agents, num_envs)
                
                transition = Transition(
                    jnp.tile(done["__all__"][np.newaxis, :], (env.num_agents, 1)),  # (num_agents, num_envs)
                    done_batch_in,   # (num_agents, num_envs)
                    action,          # (num_agents, num_envs)
                    value,           # (num_agents, num_envs)
                    reward_batch,    # (num_agents, num_envs)
                    log_prob,        # (num_agents, num_envs)
                    obs_batch,       # (num_agents, num_envs, obs_dim)
                    info,
                )
                # Keep done as dict for next iteration (env returns dict)
                runner_state = (train_state, env_state, obsv, done, hstate, rng)
                return runner_state, transition

            # Save initial hidden state BEFORE rollout for PPO rerun
            initial_hstate = runner_state[4]  # hstate before rollout
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents)
            last_done_batch = batchify(last_done, env.agents)  # last_done is dict from env
            
            def forward_single_agent(params, hs, obs, done):
                ac_in = (obs[np.newaxis, :], done[np.newaxis, :])
                return network.apply({"params": params}, hs, ac_in)
            
            _, _, last_val = jax.vmap(forward_single_agent)(
                train_state.params, hstate, last_obs_batch, last_done_batch
            )
            last_val = last_val.squeeze()  # (num_agents, num_envs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)
            
            # Extract TrainBatch without info for minibatching
            train_batch = TrainBatch(
                global_done=traj_batch.global_done,
                done=traj_batch.done,
                action=traj_batch.action,
                value=traj_batch.value,
                reward=traj_batch.reward,
                log_prob=traj_batch.log_prob,
                obs=traj_batch.obs,
            )
            # Keep info separate for logging
            traj_info = traj_batch.info

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, train_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, train_batch, gae, targets):
                        # RERUN NETWORK for each agent
                        # init_hstate: (num_agents, num_envs_minibatch, hidden_dim)
                        # traj_batch.obs: (num_steps, num_agents, num_envs_minibatch, obs_dim)
                        
                        def forward_single_agent(p, hs, obs, done):
                            # obs: (num_steps, num_envs_minibatch, obs_dim)
                            # done: (num_steps, num_envs_minibatch)
                            return network.apply({"params": p}, hs, (obs, done))
                        
                        # Transpose train_batch for per-agent processing
                        obs_per_agent = jnp.transpose(train_batch.obs, (1, 0, 2, 3))  # (num_agents, num_steps, num_envs, obs_dim)
                        done_per_agent = jnp.transpose(train_batch.done, (1, 0, 2))   # (num_agents, num_steps, num_envs)
                        action_per_agent = jnp.transpose(train_batch.action, (1, 0, 2))  # (num_agents, num_steps, num_envs)
                        
                        _, pi, value = jax.vmap(forward_single_agent)(
                            params,          # (num_agents, ...)
                            init_hstate,     # (num_agents, num_envs_minibatch, hidden_dim)
                            obs_per_agent,   # (num_agents, num_steps, num_envs_minibatch, obs_dim)
                            done_per_agent,  # (num_agents, num_steps, num_envs_minibatch)
                        )
                        # pi.logits: (num_agents, num_steps, num_envs_minibatch, action_dim)
                        # value: (num_agents, num_steps, num_envs_minibatch)
                        
                        # Use distrax batch operations directly (no vmap over distribution objects)
                        log_prob = pi.log_prob(action_per_agent)
                        # log_prob: (num_agents, num_steps, num_envs_minibatch)
                        
                        # Transpose back to (num_steps, num_agents, num_envs_minibatch)
                        log_prob = jnp.transpose(log_prob, (1, 0, 2))
                        value = jnp.transpose(value, (1, 0, 2))
                        
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = train_batch.value + (
                            value - train_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - train_batch.log_prob
                        ratio = jnp.exp(logratio)
                        # Normalize advantages PER AGENT (no coupling between agents)
                        # gae shape: (num_steps, num_agents, num_envs_minibatch)
                        # Normalize over time (axis 0) and envs (axis 2), independently for each agent
                        gae_mean = gae.mean(axis=(0, 2), keepdims=True)
                        gae_std = gae.std(axis=(0, 2), keepdims=True)
                        gae = (gae - gae_mean) / (gae_std + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        
                        # Entropy: use distrax directly (batch-aware)
                        # pi.entropy() returns (num_agents, num_steps, num_envs_minibatch)
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, train_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    train_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                # Prepare batch for minibatching
                # init_hstate: (num_agents, num_envs, hidden_dim)
                # train_batch shapes: (num_steps, num_agents, num_envs, ...)
                # advantages/targets: (num_steps, num_agents, num_envs)
                
                # Permute over num_envs dimension
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

                # Shuffle init_hstate: (num_agents, num_envs, hidden_dim) -> axis 1
                init_hstate_shuffled = jnp.take(init_hstate, permutation, axis=1)
                
                # Shuffle train_batch components: (num_steps, num_agents, num_envs, ...) -> axis 2
                def shuffle_batch(x):
                    return jnp.take(x, permutation, axis=2)
                
                train_batch_shuffled = TrainBatch(
                    global_done=shuffle_batch(train_batch.global_done),
                    done=shuffle_batch(train_batch.done),
                    action=shuffle_batch(train_batch.action),
                    value=shuffle_batch(train_batch.value),
                    reward=shuffle_batch(train_batch.reward),
                    log_prob=shuffle_batch(train_batch.log_prob),
                    obs=shuffle_batch(train_batch.obs),
                )
                
                # Shuffle advantages/targets: (num_steps, num_agents, num_envs) -> axis 2
                advantages_shuffled = jnp.take(advantages, permutation, axis=2)
                targets_shuffled = jnp.take(targets, permutation, axis=2)
                
                # Create minibatches
                def minibatch_hstate(x):
                    # x: (num_agents, num_envs, hidden_dim)
                    # -> (num_minibatches, num_agents, minibatch_size, hidden_dim)
                    num_agents, num_envs, hidden_dim = x.shape
                    minibatch_size = num_envs // config["NUM_MINIBATCHES"]
                    return x.reshape(num_agents, config["NUM_MINIBATCHES"], minibatch_size, hidden_dim).swapaxes(0, 1)
                
                def minibatch_array(x):
                    # x: (num_steps, num_agents, num_envs, ...) 
                    # -> (num_minibatches, num_steps, num_agents, minibatch_size, ...)
                    shape = list(x.shape)
                    num_steps, num_agents, num_envs = shape[:3]
                    rest = shape[3:]
                    minibatch_size = num_envs // config["NUM_MINIBATCHES"]
                    new_shape = [num_steps, num_agents, config["NUM_MINIBATCHES"], minibatch_size] + rest
                    reshaped = x.reshape(new_shape)
                    # Move minibatch axis to front
                    return jnp.moveaxis(reshaped, 2, 0)
                
                init_hstate_mb = minibatch_hstate(init_hstate_shuffled)
                
                train_batch_mb = TrainBatch(
                    global_done=minibatch_array(train_batch_shuffled.global_done),
                    done=minibatch_array(train_batch_shuffled.done),
                    action=minibatch_array(train_batch_shuffled.action),
                    value=minibatch_array(train_batch_shuffled.value),
                    reward=minibatch_array(train_batch_shuffled.reward),
                    log_prob=minibatch_array(train_batch_shuffled.log_prob),
                    obs=minibatch_array(train_batch_shuffled.obs),
                )
                
                advantages_mb = minibatch_array(advantages_shuffled)
                targets_mb = minibatch_array(targets_shuffled)
                
                minibatches = (init_hstate_mb, train_batch_mb, advantages_mb, targets_mb)

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    train_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                train_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            
            # traj_info is a dict from LogWrapper - use it directly like in original ippo_rnn.py
            # (Don't spread with {**traj_info, ...} as that can fail with FrozenDict in JIT)
            metric = traj_info

            # ratio_0: get before mean reduction (like original)
            ratio_0 = loss_info[1][3].at[0, 0].get().mean()
            
            # Per-agent losses before mean reduction
            # loss_info shape: (num_epochs, num_minibatches, ...)
            # After mean over epochs and minibatches, we get per-agent values
            loss_per_agent = jax.tree.map(lambda x: x.mean(axis=(0, 1)), loss_info)
            # loss_per_agent[0] is total_loss per agent: (num_agents,)
            # loss_per_agent[1][0] is value_loss per agent: (num_agents,)
            # etc.
            
            # Global mean for backward compatibility
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            
            # Add keys directly to metric dict (mutation works on LogWrapper's dict)
            metric["update_steps"] = update_steps
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }
            # Per-agent losses for monitoring individual agent training
            metric["loss_per_agent"] = {
                "total_loss": loss_per_agent[0],      # (num_agents,)
                "value_loss": loss_per_agent[1][0],   # (num_agents,)
                "actor_loss": loss_per_agent[1][1],   # (num_agents,)
                "entropy": loss_per_agent[1][2],      # (num_agents,)
            }

            rng = update_state[-1]

            def callback(metrics, actor_state: TrainState, step):
                env_step = (
                    metrics["update_steps"]
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"]
                )
                to_log = {
                    "env_step": env_step,
                    **metrics["loss"],
                }
                
                # Log per-agent losses
                num_agents = metrics["loss_per_agent"]["total_loss"].shape[0]
                for i in range(num_agents):
                    to_log[f"agent_{i}/total_loss"] = metrics["loss_per_agent"]["total_loss"][i]
                    to_log[f"agent_{i}/value_loss"] = metrics["loss_per_agent"]["value_loss"][i]
                    to_log[f"agent_{i}/actor_loss"] = metrics["loss_per_agent"]["actor_loss"][i]
                    to_log[f"agent_{i}/entropy"] = metrics["loss_per_agent"]["entropy"][i]
                
                if metrics["returned_episode"].any():
                    to_log.update(jax.tree.map(
                        lambda x: x[metrics["returned_episode"]].mean(),
                        metrics["user_info"]
                    ))
                    to_log["episode_lengths"] = metrics["returned_episode_lengths"][:, :, 0][
                        metrics["returned_episode"][:, :, 0]
                    ].mean()
                    to_log["episode_returns"] = metrics["returned_episode_returns"][:, :, 0][
                        metrics["returned_episode"][:, :, 0]
                    ].mean()
                print(to_log)
                wandb.log(to_log, step=metrics["update_steps"])

            jax.experimental.io_callback(callback, None, metric, train_state, update_steps)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            init_done,  # Keep as dict for consistency with env.step output
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train

# ===========================
# Main Run Function
# ===========================
def single_run(config):
    alg_name = config.get("ALG_NAME", "seperate-ippo-rnn")
    env_name = config.get("ENV_NAME", "Craftax-Coop-Symbolic")
    env = make_craftax_env_from_name(env_name)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=config["RUN_NAME"],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Name of the config YAML file (in baselines/config/)")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "config", args.config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    single_run(config)


if __name__ == "__main__":
    main()
