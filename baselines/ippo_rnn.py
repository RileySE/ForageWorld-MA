"""
Code is adapted from the IPPO RNN implementation of JaxMARL (https://github.com/FLAIROx/JaxMARL/tree/main) 
Credit goes to the original authors: Rutherford et al.
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
from craftax.environment_base.wrappers import VideoPlotWrapper


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
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
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
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env)
    env = VideoPlotWrapper(env, './output/', 256, False)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            ac_in = (
                obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            env_act = unbatchify(
                action, env.agents, config["NUM_ENVS"], env.num_agents
            )
            env_act = {k: v.squeeze() for k, v in env_act.items()}

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(rng_step, env_state, env_act)
            done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
            transition = Transition(
                jnp.tile(done["__all__"], env.num_agents),
                last_done,
                action.squeeze(),
                value.squeeze(),
                batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                log_prob.squeeze(),
                obs_batch,
                info,
            )

            # Compute distance to origin for aux loss
            # starting_pos = env_state.env_state.player_starting_position[env_state.env_state.player_level]
            # deltas_to_start = env_state.env_state.player_position - starting_pos

            # Add hstate and other non-env metrics to info so they can be logged
            info['action'] = action.squeeze()
            info['done'] = done_batch
            info['value'] = value
            info['hidden_state'] = hstate
            # info['pred_delta'] = aux
            # info['delta'] = deltas_to_start
            # TODO is this all the agent distributions rolled into one?
            info['entropy'] = pi.entropy().squeeze(0)
            info['log_prob'] = log_prob

            runner_state = (train_state, env_state, obsv, done_batch, hstate, rng)
            return runner_state, transition

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

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

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
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
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                # adding an additional "fake" dimensionality to perform minibatching correctly
                init_hstate = jnp.reshape(
                    init_hstate, (1, config["NUM_ACTORS"], -1)
                )
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate.squeeze(),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info

            ratio_0 = loss_info[1][3].at[0,0].get().mean()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
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


        # Do one "step" of logging, writing the result to a file.
        # Several steps can be run in series using --logging_steps_per_viz to do long rollouts without hitting memory limits
        def _logging_step(runner_state, unused, logging_threads, update_step):
            # Visualization rollouts
            #update_step = runner_state[-1]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["LOGGING_STEPS_PER_CALL"],
            )

            # Finally, log data associated with the visualization runs

            hidden_states = traj_batch.info['hidden_state']
            # Reshape from (T, NUM_ACTORS, hidden_dim) to (T, num_agents, NUM_ENVS, hidden_dim)
            # Agent is the outer dimension because batchify stacks: [agent0, agent1, ...] then flattens
            hidden_states = hidden_states.reshape(
                hidden_states.shape[0],   # T
                env.num_agents,           # num_agents
                config["NUM_ENVS"],       # NUM_ENVS  
                config["GRU_HIDDEN_DIM"]  # hidden_dim
            )
            # Null this for memory savings
            traj_batch.info['hidden_state'] = None

            # Add new logging fields here
            fields_to_log = ['health', 'food', 'drink', 'energy', 'done', 'is_sleeping', 'is_resting',
                             'player_position_x',
                             'player_position_y', 'recover', 'hunger', 'thirst', 'fatigue', 'light_level',
                             #'dist_to_melee_l1',
                             #'melee_on_screen', 'dist_to_passive_l1', 'passive_on_screen', 'dist_to_ranged_l1',
                             #'ranged_on_screen', 'num_melee_nearby', 'num_passives_nearby', 'num_ranged_nearby',
                             #'delta', 'pred_delta',
                             'num_monsters_killed',
                             'has_sword', 'has_pick', 'held_iron', 'value',
                             'entropy', 'log_prob', #'episode_id',
                            ]

            # Callback function for logging hidden states
            def write_rnn_hstate(hstate, scalars, increment=0, agent_n=0):

                header_field_names = ['health', 'food', 'drink', 'energy', 'done', 'is_sleeping', 'is_resting',
                                      'player_position_x',
                                      'player_position_y', 'recover', 'hunger', 'thirst', 'fatigue', 'light_level',
                                      #'dist_to_melee_l1',
                                      #'melee_on_screen', 'dist_to_passive_l1', 'passive_on_screen', 'dist_to_ranged_l1',
                                      #'ranged_on_screen', 'num_melee_nearby', 'num_passives_nearby',
                                      #'num_ranged_nearby', 'delta_x', 'delta_y', 'pred_delta_x', 'pred_delta_y',
                                      'num_monsters_killed',
                                      'has_sword',
                                      'has_pick', 'held_iron', 'value', 'entropy', 'log_prob', #'episode_id',
                                        ]

                run_out_path = os.path.join('./', wandb.run.id)
                os.makedirs(run_out_path, exist_ok=True)
                # Assemble header for the scalar file(s)
                scalar_file_header = 'action'
                for key in header_field_names:
                    scalar_file_header += ',' + key

                # We save to temp files and then append to the target file since numpy apparently cannot write files in append mode for some reason
                for i in range(logging_threads):
                    out_filename_hstates = os.path.join(run_out_path, 'hstates_{}_{}_{}.csv'.format(increment, agent_n, i))
                    temp_filename = os.path.join(run_out_path, 'temp.csv')
                    np.savetxt(temp_filename,
                               hstate[:, i, :], delimiter=',')
                    temp_file = open(temp_filename, 'r')
                    out_file_hstates = open(out_filename_hstates, 'a+')
                    out_file_hstates.write(temp_file.read())
                    out_file_hstates.close()
                    temp_file.close()
                    # Then do the same thing for the scalars
                    out_filename_scalars = os.path.join(run_out_path, 'scalars_{}_{}_{}.csv'.format(increment, agent_n, i))
                    np.savetxt(temp_filename,
                               scalars[:, i, :], delimiter=',', fmt='%f',
                               header=scalar_file_header
                               )
                    temp_file = open(temp_filename, 'r')
                    out_file_scalars = open(out_filename_scalars, 'a+')
                    out_file_scalars.write(temp_file.read())
                    temp_file.close()
                    out_file_scalars.close()
                    print('Writing log file', out_filename_hstates)

            # Add the specified field to the logging array
            # Network outputs (value, entropy, log_prob, done) have shape (T, num_agents, NUM_ENVS) - extract [:, agent, :]
            # Environment fields (health, food, etc.) have shape (T, NUM_ENVS, num_agents) - extract [:, :, agent]
            network_output_fields = {'value', 'entropy', 'log_prob', 'done'}
            
            def add_field_to_log_array(info_dict, log_array, field_key, agent_to_log):
                field_value = info_dict[field_key]
                # Select the current agent if this is a per-agent field
                if len(field_value.shape) == 3:
                    if field_key in network_output_fields:
                        # Network outputs: shape (T, num_agents, NUM_ENVS)
                        field_value = field_value[:, agent_to_log, :]
                    else:
                        # Environment fields: shape (T, NUM_ENVS, num_agents)
                        field_value = field_value[:, :, agent_to_log]
                new_shape = field_value.shape + (1,)
                field_value = field_value.reshape(new_shape)

                log_array = jnp.concatenate([log_array, field_value], axis=2)

                return log_array

            # Assemble logging variable array
            # Network outputs (action, done, value, log_prob, entropy) come from batchify with agent-major ordering:
            # [A0_E0, A0_E1, ..., A0_En, A1_E0, A1_E1, ...] so reshape must be (num_agents, NUM_ENVS)
            traj_batch.info['action'] = traj_batch.info['action'].reshape((traj_batch.info['action'].shape[0],) +
                                                              (env.num_agents, config['NUM_ENVS']))
            traj_batch.info['done'] = traj_batch.info['done'].reshape((traj_batch.info['done'].shape[0],) +
                                                              (env.num_agents, config['NUM_ENVS']))
            traj_batch.info['value'] = traj_batch.info['value'].reshape((traj_batch.info['value'].shape[0],) +
                                                              (env.num_agents, config['NUM_ENVS']))
            traj_batch.info['log_prob'] = traj_batch.info['log_prob'].reshape((traj_batch.info['log_prob'].shape[0],) +
                                                              (env.num_agents, config['NUM_ENVS']))
            traj_batch.info['entropy'] = traj_batch.info['entropy'].reshape((traj_batch.info['entropy'].shape[0],) +
                                                              (env.num_agents, config['NUM_ENVS']))
            for agent_n in range(env.num_agents):
                # Network outputs have shape (T, num_agents, NUM_ENVS) - extract agent_n -> (T, NUM_ENVS)
                log_array = traj_batch.info['action'][:, agent_n, :].reshape((traj_batch.info['action'].shape[0], config['NUM_ENVS'], 1))
                # Yes this is a for loop in the JAX code but this stuff was getting done in serial before anyway and it's cheap operations
                for field_to_log in fields_to_log:
                    log_array = add_field_to_log_array(traj_batch.info, log_array, field_to_log, agent_n)

                # Extract hidden states only for this agent: (T, NUM_ENVS, hidden_dim)
                agent_hidden_states = hidden_states[:, agent_n, :, :]
                jax.debug.callback(write_rnn_hstate, agent_hidden_states, log_array, update_step, agent_n)

            return runner_state, None

            # Func to interleave update steps and plotting

        def _update_plot(runner_state, unused):
            # First, update
            runner_state, metric = jax.lax.scan(
                _update_step, runner_state, None, config["LOGGING_UPDATES_INTERVAL"]
            )

            # Log model weights
            def save_weights_callback(weights, iter):
                weights_flat = jax.tree.flatten(weights)
                run_out_path = os.path.join('./', wandb.run.id)
                os.makedirs(run_out_path, exist_ok=True)
                weight_filename = os.path.join(run_out_path, 'weights_{}.csv'.format(iter))
                weight_file = open(weight_filename, 'w')
                weights_params = weights['params']

                def save_weight_dict(curr_value, key_string=''):
                    if type(curr_value) != dict:
                        np.savetxt(weight_file, np.transpose(curr_value), delimiter=',', fmt='%f', header=key_string)
                        return True
                    else:
                        for key in curr_value.keys():
                            save_weight_dict(curr_value[key], key_string + '/' + key)
                    return True

                save_weight_dict(weights_params)

                print('Saving weights in file', weight_filename)

            # TODO make weight saving work
            #jax.debug.callback(save_weights_callback, runner_state[0].params, runner_state[-1])

            # Can we save the environment state and resume training later?
            # runner_state_copy = runner_state

            # Then do iterations of logging
            state, update_steps = runner_state
            state, empty = jax.lax.scan(
                functools.partial(_logging_step, logging_threads=config["LOGGING_THREADS"], update_step=update_steps), state, None,
                config["LOGGING_NUM_CALLS"],
            )

            return (state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_plot, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train

# ===========================
# Main Run Function
# ===========================
def single_run(config):
    alg_name = config.get("ALG_NAME", "ippo-rnn")
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
