"""
Fully CENTRALIZED PPO runner (single policy + single critic).

Derived from baselines/seperate_ippo_rnn.py, but every trace of decentralized /
per-agent-network logic is removed.  The multi-agent problem is reduced to a
single-agent MDP over the joint observation and the joint action:

  * OBSERVATION  — the per-agent egocentric observations are concatenated in
    fixed agent order into one global vector of size num_agents * obs_dim.
    Both the actor and the critic consume this same global vector.
  * RECURRENCE   — ONE sequence model with ONE carry.  There is no per-agent
    hidden state.  POLICY_ARCH selects the model:
      - "gru" (default): one GRU, carry (num_envs, GRU_HIDDEN_DIM).
      - "transformer": one causal, fixed-window transformer, carry
        (num_envs, TF_MAX_HISTORY, TF_D_MODEL + 1) — a ring buffer of the last
        TF_MAX_HISTORY token embeddings plus a per-slot validity flag.  It
        attends over the last TF_MAX_HISTORY env steps and nothing older, and
        never across an episode boundary.  Both models emit a width-
        GRU_HIDDEN_DIM vector, so the actor / critic / aux heads are identical
        and the total parameter count stays within a few percent (with the
        defaults: 1.03x the GRU core).  See baselines/transformer_policy.py for
        the full hyperparameter table and the parameter-budget helpers.
  * ACTION       — the actor head emits num_agents * action_dim logits, viewed
    as (num_agents, action_dim).  The joint action is the concatenation of the
    per-agent actions, and the joint log-probability / entropy are the sums over
    agents.  The joint distribution is therefore factored across agents given
    the shared recurrent state (a non-factored distribution over action_dim ^
    num_agents joint actions is not tractable).  PPO's importance ratio is
    computed on the JOINT log-probability, i.e. one ratio per (step, env).
  * VALUE        — ONE critic head, V(global_state).  A single scalar per
    (step, env), a single GAE, a single advantage.  There are no per-agent
    values, per-agent advantages, or per-agent gradient clipping.
  * REWARD       — the per-agent rewards are aggregated into one team scalar via
    CENTRAL_REWARD_AGGREGATION ("mean", the default, or "sum").  With
    SHARED_REWARD: true all per-agent rewards are identical, so "mean" simply
    recovers the shared reward and keeps the reward scale identical to
    seperate_ippo_rnn.
  * TERMINATION  — the environment terminates globally (done["__all__"]), so the
    single episode / single value function is well defined.

Kept from seperate_ippo_rnn.py because they are environment constraints or
logging features rather than decentralization:
  * reduced action space (USE_REDUCED_ACTION_SPACE),
  * dead-agent NOOP masking (ACTION_MASK_WHILE_DEAD) — applied per action head,
  * dynamic episode caps (EARLY_EPISODE_CAP / GENERAL_EPISODE_CAP),
  * the auxiliary displacement-prediction head — now centralized: one head
    predicts EVERY agent's displacement from spawn, so AUX_OUTPUT_DIM =
    num_agents * 2.  USE_TEAMMATE_AUXILIARY_LOSS / AUX_TEAMMATE_WEIGHT are
    meaningless for a controller that already observes everything and are
    ignored (a warning is printed if they are set),
  * CSV scalar logging, video recording, and the wandb metric layout (including
    overview/sps).  The CSV column schema is unchanged so existing analysis
    tooling keeps working; note that the `value` column is now the single global
    V(s) repeated for every agent, while `entropy` / `log_prob` are that agent's
    action-head marginals,
  * orbax checkpointing / resume.  build_env() and single_run() are byte-identical
    to seperate_ippo_rnn.py apart from the ALG_NAME default and one error string,
    so the block loop, sidecar metadata, signal handling, buffer donation and
    MAX_BLOCKS_THIS_RUN behaviour match exactly.  The carry fingerprints differ
    between the two runners (different param/hidden-state structure), so orbax
    will refuse a cross-runner resume rather than silently mis-restore.

Code is adapted from the IPPO RNN implementation of JaxMARL
(https://github.com/FLAIROx/JaxMARL/tree/main).
Credit goes to the original authors: Rutherford et al.
"""

# ===========================
# Imports and Configuration
# ===========================
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import datetime
import time
import functools
import tempfile
import yaml
from typing import NamedTuple, Dict

import jax
if not hasattr(jax, 'tree_map'):
    jax.tree_map = jax.tree_util.tree_map
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

import optax
import distrax

import wandb

import imageio

from jaxmarl.wrappers.baselines import LogWrapper
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_coop.constants import reduced_action_ids
from craftax.environment_base.wrappers import VideoPlotWrapper
from craftax.custom_rendering.base_rendering import load_rendering_resources
from craftax.custom_rendering.ego_rendering import render_ego_perspective
from craftax.custom_rendering.full_map_rendering import render_full_map

import checkpoint_utils as ckpt
import transformer_policy as tfp

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


class CentralizedActorCriticRNN(nn.Module):
    """Single policy + single critic over the joint observation.

    Input  : global obs (T, B, num_agents * obs_dim), resets (T, B)
    Outputs: hidden      (B, hidden_dim)          -- GRU carry, or
                         (B, TF_MAX_HISTORY, TF_D_MODEL + 1) for the transformer
             pi          Categorical over logits (T, B, num_agents, action_dim)
             value       (T, B)                      -- ONE critic head
             aux         (T, B, num_agents * 2)

    The sequence model is selected by POLICY_ARCH ("gru" | "transformer").  Both
    consume (embedding, resets) and emit a width-GRU_HIDDEN_DIM vector, so the
    actor / critic / aux heads below are identical either way.
    """
    action_dim: int
    num_agents: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        if self.config.get("POLICY_ARCH", "gru") == "transformer":
            hidden, embedding = tfp.ScannedTransformer(
                **tfp.hparams_from_config(self.config)
            )(hidden, rnn_in)
        else:
            hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # ── Joint policy: one trunk, num_agents * action_dim logits ──
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.num_agents * self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)
        # (T, B, num_agents * action_dim) -> (T, B, num_agents, action_dim)
        action_logits = action_logits.reshape(
            action_logits.shape[:-1] + (self.num_agents, self.action_dim)
        )

        pi = distrax.Categorical(logits=action_logits)

        # ── Single centralized critic: V(global state) ──
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        # ── Auxiliary head: predicts every agent's displacement from spawn ──
        aux = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        aux = nn.relu(aux)
        aux = nn.Dense(self.config["AUX_OUTPUT_DIM"], kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            aux
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1), aux

# ===========================
# Data Structures and Utilities
# ===========================
class Transition(NamedTuple):
    """One centralized transition.  Every field is (num_envs, ...) per step."""
    global_done: jnp.ndarray   # (num_envs,)     episode termination
    done: jnp.ndarray          # (num_envs,)     previous done -> RNN reset
    alive: jnp.ndarray         # (num_envs, N)   per-agent alive flags (action masking)
    action: jnp.ndarray        # (num_envs, N)   joint action (policy indices)
    value: jnp.ndarray         # (num_envs,)     V(global state)
    reward: jnp.ndarray        # (num_envs,)     aggregated team reward
    log_prob: jnp.ndarray      # (num_envs,)     JOINT log-probability
    obs: jnp.ndarray           # (num_envs, N*obs_dim) global observation
    aux_targets: jnp.ndarray   # (num_envs, N*2) displacement-from-spawn targets
    info: jnp.ndarray

class TrainBatch(NamedTuple):
    """Same as Transition without info (info breaks minibatch reshaping)."""
    global_done: jnp.ndarray
    done: jnp.ndarray
    alive: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    aux_targets: jnp.ndarray

class LossAux(NamedTuple):
    """Auxiliary outputs from _loss_fn — passed through has_aux=True, only used for logging.
    Adding / reordering fields here won't silently misalign wandb metrics."""
    value_loss: jnp.ndarray
    loss_actor: jnp.ndarray
    entropy: jnp.ndarray
    ratio: jnp.ndarray
    approx_kl: jnp.ndarray
    clip_frac: jnp.ndarray
    aux_loss: jnp.ndarray
    entropy_per_agent: jnp.ndarray  # (num_agents,) per-action-head marginal entropy

# ===========================
# Training Function
# ===========================
def make_train(config, env):
    run_output_dir = None

    def sanitize_path_component(value: str, fallback: str = "run") -> str:
        cleaned = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_"
            for ch in str(value).strip()
        ).strip("._-")
        return cleaned or fallback

    if config["NUM_MINIBATCHES"] <= 0:
        raise ValueError(
            f"NUM_MINIBATCHES must be >= 1, got {config['NUM_MINIBATCHES']}."
        )
    if config["NUM_ENVS"] <= 0:
        raise ValueError(f"NUM_ENVS must be >= 1, got {config['NUM_ENVS']}.")
    if config["NUM_ENVS"] % config["NUM_MINIBATCHES"] != 0:
        raise ValueError(
            "NUM_ENVS must be divisible by NUM_MINIBATCHES for minibatch reshaping. "
            f"Got NUM_ENVS={config['NUM_ENVS']}, NUM_MINIBATCHES={config['NUM_MINIBATCHES']}."
        )

    logging_threads = int(config.get("LOGGING_THREADS", 1))
    if logging_threads <= 0:
        raise ValueError(f"LOGGING_THREADS must be >= 1, got {logging_threads}.")
    if logging_threads > config["NUM_ENVS"]:
        raise ValueError(
            "LOGGING_THREADS must be <= NUM_ENVS to avoid out-of-bounds logging access. "
            f"Got LOGGING_THREADS={logging_threads}, NUM_ENVS={config['NUM_ENVS']}."
        )

    reward_aggregation = str(config.get("CENTRAL_REWARD_AGGREGATION", "mean")).lower()
    if reward_aggregation not in {"mean", "sum"}:
        raise ValueError(
            "CENTRAL_REWARD_AGGREGATION must be 'mean' or 'sum', "
            f"got {reward_aggregation!r}."
        )

    config["NUM_AGENTS"] = env.num_agents
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_LOGGING_ITERS"] = config["NUM_UPDATES"] // config["LOGGING_UPDATES_INTERVAL"]
    config["REMAINING_UPDATES"] = config["NUM_UPDATES"] % config["LOGGING_UPDATES_INTERVAL"]
    # Centralized: one joint transition per (step, env), so minibatching is over
    # NUM_ENVS only.  Each minibatch is (num_steps, num_envs // NUM_MINIBATCHES, ...).
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] // config["NUM_MINIBATCHES"]

    # Load rendering resources BEFORE wrapping (need base env's static_env_params)
    _video_env_name = config.get("ENV_NAME", "Craftax-Coop-Symbolic")
    _video_pixel_size = config.get("VIDEO_PIXEL_SIZE", 16)
    _video_static_params = env.static_env_params
    _video_rendering_res = load_rendering_resources(_video_env_name, pixel_size_preference=_video_pixel_size)
    _video_textures = _video_rendering_res["TEXTURES"]
    _video_player_textures = _video_rendering_res["load_player_specific_textures"](
        _video_textures[_video_pixel_size], _video_static_params.player_count
    )
    _video_max_length = int(config.get("MAX_VIDEO_LENGTH", -1))

    def get_run_output_dir():
        nonlocal run_output_dir
        if run_output_dir is not None:
            return run_output_dir

        configured_output_dir = config.get("OUTPUT_DIR", "")
        output_root = os.path.expanduser(configured_output_dir) if configured_output_dir else "."
        run_group_dir = sanitize_path_component(config.get("RUN_NAME", "run"), fallback="run")
        run_id = getattr(wandb.run, "id", None) if wandb.run is not None else None
        run_instance_dir = "{}_{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            run_id if run_id is not None else "norun",
        )
        run_output_dir = os.path.join(output_root, run_group_dir, run_instance_dir)
        print(f"Run artifacts will be written to: {run_output_dir}")
        return run_output_dir

    # Two env references to avoid VideoPlotWrapper overhead during training:
    # env_train: LogWrapper only — used for training steps (no mob distance calculations)
    # env_log:   LogWrapper + VideoPlotWrapper — used for CSV logging steps (adds health, food, mob distances etc.)
    # Both share the same state structure (VideoPlotWrapper is a pass-through for state).
    env_train = LogWrapper(env)
    env_log = VideoPlotWrapper(env_train, os.path.join(get_run_output_dir(), 'debug_output'), 256, False)
    env = env_log  # default reference for property access (agents, num_agents, action_space, etc.)

    num_agents = env.num_agents
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    global_obs_dim = num_agents * obs_dim

    # ── Auxiliary loss configuration (centralized) ──
    # One head predicts every agent's displacement from spawn: (dx, dy) per agent.
    if config.get("USE_TEAMMATE_AUXILIARY_LOSS", False):
        print(
            "[central_ippo_rnn] USE_TEAMMATE_AUXILIARY_LOSS is ignored: the centralized "
            "controller already observes all agents, so there is no egocentric "
            "teammate-relative target.  The aux head predicts every agent's "
            "displacement from spawn."
        )
    config["AUX_OUTPUT_DIM"] = num_agents * 2
    _aux_output_dim = config["AUX_OUTPUT_DIM"]
    _aux_self_w = config.get("AUX_SELF_WEIGHT", 1.0)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    use_reduced_action_space = config.get("USE_REDUCED_ACTION_SPACE", False)
    action_mask_while_dead = config.get("ACTION_MASK_WHILE_DEAD", False)
    agents_per_team = len(config.get("TEAM_COMPOSITION", [1, 1, 2]))
    reduced_action_id_map = reduced_action_ids(agents_per_team)
    # Reduced mode keeps actions 0..GIVE and appends the team-local GIVE targets.
    REDUCED_ACTION_DIM = int(reduced_action_id_map.shape[0])
    full_action_dim = env.action_space(env.agents[0]).n
    action_dim = REDUCED_ACTION_DIM if use_reduced_action_space else full_action_dim

    def policy_action_to_env_action(action):
        if use_reduced_action_space:
            return reduced_action_id_map[action]
        return action

    noop_policy_action_idx = 0

    def global_obs(obs_dict):
        """Concatenate per-agent observations into the global vector.

        obs_dict[agent]: (num_envs, obs_dim) -> (num_envs, num_agents * obs_dim)
        """
        return jnp.concatenate([obs_dict[a] for a in env.agents], axis=-1)

    def compute_alive_mask(env_state):
        """player_alive is (num_envs, num_agents) after the env vmap."""
        return env_state.env_state.player_alive

    def apply_dead_action_mask(logits, alive_mask):
        """Force dead agents' action heads to NOOP by masking out all other actions.

        logits:     (..., num_agents, action_dim)
        alive_mask: (..., num_agents)
        """
        dead_only_noop_logits = jnp.full_like(logits, -1e9)
        dead_only_noop_logits = dead_only_noop_logits.at[..., noop_policy_action_idx].set(0.0)
        return jnp.where(alive_mask[..., None], logits, dead_only_noop_logits)

    # ── Sequence-model selection: GRU carry vs transformer window ──
    policy_arch = str(config.get("POLICY_ARCH", "gru")).lower()
    if policy_arch not in ("gru", "transformer"):
        raise ValueError(f'POLICY_ARCH must be "gru" or "transformer", got {policy_arch!r}')
    if policy_arch == "transformer":
        _tf_hp = tfp.hparams_from_config(config)
        # Write the resolved values back so the printed report, the checkpoint
        # sidecar and wandb all record what actually ran (matters for
        # TF_D_MODEL: "auto").
        config["TF_D_MODEL"] = _tf_hp["d_model"]
        config["TF_OUT_DIM"] = _tf_hp["out_dim"]
        print(tfp.describe(_tf_hp, int(config["FC_DIM_SIZE"]), int(config["GRU_HIDDEN_DIM"])))
    else:
        _tf_hp = None

    def zero_hstate(batch_size):
        """Zero carry for the active sequence model.

        gru         -> (batch_size, GRU_HIDDEN_DIM)
        transformer -> (batch_size, TF_MAX_HISTORY, TF_D_MODEL + 1)
        """
        if policy_arch == "transformer":
            return tfp.ScannedTransformer.initialize_carry(
                batch_size, _tf_hp["max_history"], _tf_hp["d_model"]
            )
        return ScannedRNN.initialize_carry(batch_size, config["GRU_HIDDEN_DIM"])

    def train(rng):
        # INIT NETWORK — a single parameter set, no agent dimension anywhere
        network = CentralizedActorCriticRNN(
            action_dim=action_dim, num_agents=num_agents, config=config
        )
        rng, _rng = jax.random.split(rng)

        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], global_obs_dim)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = zero_hstate(config["NUM_ENVS"])
        network_params = network.init(_rng, init_hstate, init_x)["params"]

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
        obsv, env_state = jax.vmap(env_train.reset, in_axes=(0,))(reset_rng)
        # ONE hidden state: (num_envs, hidden_dim) for the GRU, or one attention
        # window (num_envs, TF_MAX_HISTORY, TF_D_MODEL + 1) for the transformer
        init_hstate = zero_hstate(config["NUM_ENVS"])
        # ONE done flag per env (the environment terminates globally)
        init_done = jnp.zeros((config["NUM_ENVS"],), dtype=bool)

        # Override effective_max_timesteps in every vmapped env_state. Must be
        # re-applied after env.step, because world_gen resets the cap to its
        # default value whenever an episode terminates — so a one-shot patch
        # before the rollout scan only affects episodes already running at the
        # start of the rollout.
        def _patch_episode_cap(env_state, effective_cap):
            current_caps = env_state.env_state.effective_max_timesteps
            patched_caps = jnp.full_like(current_caps, effective_cap)
            patched_inner = env_state.env_state.replace(effective_max_timesteps=patched_caps)
            return env_state.replace(env_state=patched_inner)

        def forward(params, hstate, obs, done):
            """Single forward pass of the centralized network.

            obs: (T, num_envs, global_obs_dim), done: (T, num_envs)
            """
            return network.apply({"params": params}, hstate, (obs, done))

        # TRAIN LOOP
        # detailed_logging: when True, extra per-step fields (hidden_state, entropy,
        # log_prob, deltas, etc.) are added to info for CSV logging.  When False
        # (training path), these fields are omitted to save GPU memory that would
        # otherwise be accumulated by jax.lax.scan.
        # Use functools.partial to set the flag at compile time so JAX can
        # eliminate the dead code path entirely.

        def _env_step(runner_state, unused, detailed_logging=False, effective_cap=None):
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            obs_batch = global_obs(last_obs)          # (num_envs, global_obs_dim)
            alive_batch = compute_alive_mask(env_state)  # (num_envs, num_agents)

            hstate, pi, value, aux_pred = forward(
                train_state.params,
                hstate,                        # (num_envs, hidden_dim)
                obs_batch[np.newaxis, :],      # (1, num_envs, global_obs_dim)
                last_done[np.newaxis, :],      # (1, num_envs)
            )
            # pi.logits: (1, num_envs, num_agents, action_dim)
            # value:     (1, num_envs)
            # aux_pred:  (1, num_envs, AUX_OUTPUT_DIM)

            if action_mask_while_dead:
                masked_logits = apply_dead_action_mask(pi.logits, alive_batch[np.newaxis, :])
                pi = distrax.Categorical(logits=masked_logits)

            action = pi.sample(seed=_rng)             # (1, num_envs, num_agents)
            log_prob_per_agent = pi.log_prob(action)  # (1, num_envs, num_agents)
            entropy_per_agent = pi.entropy()          # (1, num_envs, num_agents)

            action = action.squeeze(axis=0)                          # (num_envs, num_agents)
            log_prob_per_agent = log_prob_per_agent.squeeze(axis=0)  # (num_envs, num_agents)
            entropy_per_agent = entropy_per_agent.squeeze(axis=0)    # (num_envs, num_agents)
            value = value.squeeze(axis=0)                            # (num_envs,)
            # JOINT log-probability of the concatenated action
            log_prob = log_prob_per_agent.sum(axis=-1)               # (num_envs,)

            env_action = policy_action_to_env_action(action)  # (num_envs, num_agents)
            # Index instead of squeeze so NUM_ENVS == 1 (video rollout) keeps its batch dim
            env_act = {a: env_action[:, i] for i, a in enumerate(env.agents)}

            # STEP ENV
            # Use env_log (with VideoPlotWrapper) only during logging to get CSV fields
            # (health, food, mob distances, etc.). During training, use env_train
            # (LogWrapper only) to skip expensive mob distance calculations.
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, obs_batch.shape[0])
            step_fn = env_log.step if detailed_logging else env_train.step
            obsv, env_state, reward, done, info = jax.vmap(
                step_fn, in_axes=(0, 0, 0)
            )(rng_step, env_state, env_act)

            # Re-apply the episode cap: env.step may have reset an episode and
            # restored effective_max_timesteps to the default.
            if effective_cap is not None:
                env_state = _patch_episode_cap(env_state, effective_cap)

            done_global = done["__all__"]  # (num_envs,)

            # Aggregate the per-agent rewards into ONE team reward
            reward_stacked = jnp.stack([reward[a] for a in env.agents], axis=-1)  # (num_envs, num_agents)
            if reward_aggregation == "sum":
                reward_central = reward_stacked.sum(axis=-1)
            else:
                reward_central = reward_stacked.mean(axis=-1)

            # Auxiliary targets (computed AFTER env.step so targets use position at t+1):
            # every agent's displacement from its spawn point.
            all_pos = env_state.env_state.player_position        # (num_envs, num_agents, 2)
            all_spawn = env_state.env_state.player_spawn_position
            self_delta = all_pos - all_spawn                     # (num_envs, num_agents, 2)
            aux_targets = self_delta.reshape(self_delta.shape[0], _aux_output_dim)

            transition = Transition(
                done_global,      # (num_envs,)
                last_done,        # (num_envs,)
                alive_batch,      # (num_envs, num_agents)
                action,           # (num_envs, num_agents)
                value,            # (num_envs,)
                reward_central,   # (num_envs,)
                log_prob,         # (num_envs,)
                obs_batch,        # (num_envs, global_obs_dim)
                aux_targets,      # (num_envs, num_agents * 2)
                info,
            )

            # Extra per-step fields for CSV logging — only computed in logging iterations.
            # CSV helpers expect per-agent fields as (num_agents, num_envs), so transpose.
            if detailed_logging:
                aux_pred_squeezed = aux_pred.squeeze(axis=0)  # (num_envs, AUX_OUTPUT_DIM)
                aux_pred_xy = aux_pred_squeezed.reshape(-1, num_agents, 2)  # (num_envs, N, 2)
                info['action'] = env_action.T                     # (num_agents, num_envs)
                # done is global; replicate per agent to keep the CSV schema unchanged
                info['done'] = jnp.broadcast_to(
                    done_global[np.newaxis, :], (num_agents, done_global.shape[0])
                )
                # ONE global value, repeated per agent (there are no per-agent critics)
                info['value'] = jnp.broadcast_to(
                    value[np.newaxis, :], (num_agents, value.shape[0])
                )
                info['hidden_state'] = hstate                    # (num_envs, hidden_dim) — shared
                info['entropy'] = entropy_per_agent.T            # (num_agents, num_envs)
                info['log_prob'] = log_prob_per_agent.T          # (num_agents, num_envs)
                info['joint_log_prob'] = jnp.broadcast_to(
                    log_prob[np.newaxis, :], (num_agents, log_prob.shape[0])
                )
                info['delta_x'] = self_delta[:, :, 0].T          # (num_agents, num_envs)
                info['delta_y'] = self_delta[:, :, 1].T
                info['pred_delta_x'] = aux_pred_xy[:, :, 0].T
                info['pred_delta_y'] = aux_pred_xy[:, :, 1].T

            runner_state = (train_state, env_state, obsv, done_global, hstate, rng)
            return runner_state, transition

        _early_episode_cap = config.get("EARLY_EPISODE_CAP", 0)
        _early_episode_cap_until = config.get("EARLY_EPISODE_CAP_UNTIL", 0)
        _general_episode_cap = config.get("GENERAL_EPISODE_CAP", 0)
        _default_max_timesteps = env.default_params.max_timesteps
        _post_early_cap = _general_episode_cap if _general_episode_cap > 0 else _default_max_timesteps

        def _get_effective_episode_cap(update_steps):
            if _early_episode_cap > 0 and _early_episode_cap_until > 0:
                return jax.lax.select(
                    update_steps < _early_episode_cap_until,
                    jnp.asarray(_early_episode_cap, dtype=jnp.float32),
                    jnp.asarray(_post_early_cap, dtype=jnp.float32),
                )
            if _general_episode_cap > 0:
                return jnp.asarray(_general_episode_cap, dtype=jnp.float32)
            return None

        def _apply_episode_cap_to_runner_state(runner_state, effective_cap):
            if effective_cap is None:
                return runner_state
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            env_state = _patch_episode_cap(env_state, effective_cap)
            return (train_state, env_state, last_obs, last_done, hstate, rng)

        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            # Dynamically cap episode length during early training. The cap must
            # be passed into _env_step and re-applied after every env.step —
            # see _patch_episode_cap for why a single pre-rollout patch is not
            # enough.
            effective_cap = _get_effective_episode_cap(update_steps)
            if effective_cap is not None:
                runner_state = _apply_episode_cap_to_runner_state(runner_state, effective_cap)
                scan_step_fn = functools.partial(_env_step, effective_cap=effective_cap)
            else:
                scan_step_fn = _env_step

            # Save initial hidden state BEFORE rollout for the PPO rerun
            initial_hstate = runner_state[4]
            runner_state, traj_batch = jax.lax.scan(
                scan_step_fn, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE — one bootstrap value per env
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            _, _, last_val, _ = forward(
                train_state.params,
                hstate,
                global_obs(last_obs)[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            last_val = last_val.squeeze(axis=0)  # (num_envs,)

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
                alive=traj_batch.alive,
                action=traj_batch.action,
                value=traj_batch.value,
                reward=traj_batch.reward,
                log_prob=traj_batch.log_prob,
                obs=traj_batch.obs,
                aux_targets=traj_batch.aux_targets,
            )
            # Keep info separate for logging
            traj_info = traj_batch.info

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, train_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, train_batch, gae, targets):
                        # RERUN NETWORK
                        # init_hstate:      (num_envs_mb, hidden_dim)
                        # train_batch.obs:  (num_steps, num_envs_mb, global_obs_dim)
                        _, pi, value, aux = forward(
                            params, init_hstate, train_batch.obs, train_batch.done
                        )
                        # pi.logits: (num_steps, num_envs_mb, num_agents, action_dim)
                        # value:     (num_steps, num_envs_mb)
                        # aux:       (num_steps, num_envs_mb, AUX_OUTPUT_DIM)

                        if action_mask_while_dead:
                            masked_logits = apply_dead_action_mask(pi.logits, train_batch.alive)
                            pi = distrax.Categorical(logits=masked_logits)

                        # JOINT log-probability of the concatenated action
                        log_prob = pi.log_prob(train_batch.action).sum(axis=-1)  # (T, mb)

                        # CALCULATE VALUE LOSS — single critic
                        value_pred_clipped = train_batch.value + (
                            value - train_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        # CALCULATE ACTOR LOSS — one ratio per (step, env)
                        logratio = log_prob - train_batch.log_prob
                        ratio = jnp.exp(logratio)
                        # Single controller -> normalize the advantage globally
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

                        # Entropy of the joint (factored) policy = sum over action heads
                        entropy_heads = pi.entropy()                      # (T, mb, num_agents)
                        entropy_per_agent = entropy_heads.mean(axis=(0, 1))  # (num_agents,) logging only
                        entropy = entropy_heads.sum(axis=-1).mean()

                        # Auxiliary loss: predict every agent's displacement from spawn
                        aux_loss = _aux_self_w * jnp.square(
                            aux - train_batch.aux_targets
                        ).mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                            + config["AUX_COEF"] * aux_loss
                        )
                        return total_loss, LossAux(
                            value_loss=value_loss,
                            loss_actor=loss_actor,
                            entropy=entropy,
                            ratio=ratio,
                            approx_kl=approx_kl,
                            clip_frac=clip_frac,
                            aux_loss=aux_loss,
                            entropy_per_agent=entropy_per_agent,
                        )

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

                # Every array is (num_steps, num_envs, ...) except init_hstate,
                # which is (num_envs, hidden_dim) -> shuffle/minibatch over envs.
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

                init_hstate_shuffled = jnp.take(init_hstate, permutation, axis=0)
                train_batch_shuffled = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), train_batch
                )
                advantages_shuffled = jnp.take(advantages, permutation, axis=1)
                targets_shuffled = jnp.take(targets, permutation, axis=1)

                def minibatch_hstate(x):
                    # Carry is env-major with no time axis:
                    #   gru         (num_envs, hidden_dim)
                    #   transformer (num_envs, max_history, d_model + 1)
                    # -> (num_minibatches, minibatch_size, *carry_dims)
                    num_envs = x.shape[0]
                    minibatch_size = num_envs // config["NUM_MINIBATCHES"]
                    return x.reshape(
                        (config["NUM_MINIBATCHES"], minibatch_size) + x.shape[1:]
                    )

                def minibatch_array(x):
                    # (num_steps, num_envs, ...) -> (num_minibatches, num_steps, minibatch_size, ...)
                    num_steps, num_envs = x.shape[:2]
                    rest = list(x.shape[2:])
                    minibatch_size = num_envs // config["NUM_MINIBATCHES"]
                    reshaped = x.reshape(
                        [num_steps, config["NUM_MINIBATCHES"], minibatch_size] + rest
                    )
                    return jnp.moveaxis(reshaped, 1, 0)

                minibatches = (
                    minibatch_hstate(init_hstate_shuffled),
                    jax.tree.map(minibatch_array, train_batch_shuffled),
                    minibatch_array(advantages_shuffled),
                    minibatch_array(targets_shuffled),
                )

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

            loss_aux = loss_info[1]  # LossAux with fields stacked (num_epochs, num_minibatches, ...)
            # ratio_0: get before mean reduction (like the original)
            ratio_0 = loss_aux.ratio.at[0, 0].get().mean()

            # Per-head entropy for diagnostics: (num_epochs, num_minibatches, num_agents) -> (num_agents,)
            entropy_per_agent = loss_aux.entropy_per_agent.mean(axis=(0, 1))

            loss_info_mean = jax.tree.map(lambda x: x.mean(), loss_info)
            loss_aux_mean = loss_info_mean[1]

            # Create new metric dict (don't mutate FrozenDict from LogWrapper)
            metric = {
                **dict(traj_info),  # Convert FrozenDict to regular dict
                "update_steps": update_steps,
                "central_reward": traj_batch.reward.mean(),
                "loss": {
                    "total_loss": loss_info_mean[0],
                    "value_loss": loss_aux_mean.value_loss,
                    "actor_loss": loss_aux_mean.loss_actor,
                    "entropy": loss_aux_mean.entropy,
                    "ratio": loss_aux_mean.ratio,
                    "ratio_0": ratio_0,
                    "approx_kl": loss_aux_mean.approx_kl,
                    "clip_frac": loss_aux_mean.clip_frac,
                    "aux_loss": loss_aux_mean.aux_loss,
                },
                "entropy_per_agent": entropy_per_agent,  # (num_agents,)
            }

            rng = update_state[-1]

            #State for sps computations
            _sps_state = {'last_time':None, 'last_env_step': None}

            def callback(metrics, step):
                env_step = (
                    metrics["update_steps"]
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"]
                )

                # Team config
                configured_comp = config.get("TEAM_COMPOSITION", [1, 1, 2])
                configured_num_teams = int(config.get("NUM_TEAMS", 2))
                configured_agents_per_team = max(1, len(configured_comp))

                # Derive agent count from runtime tensors (safer than config-only math).
                num_agents = int(np.asarray(metrics["entropy_per_agent"]).shape[0])

                # Team count is configuration-driven for stable logging layout.
                num_teams = max(1, configured_num_teams)

                # Keep team layout valid even if config/runtime diverge.
                num_teams = max(1, min(num_teams, num_agents))
                agents_per_team = configured_agents_per_team
                if agents_per_team * num_teams < num_agents:
                    agents_per_team = int(np.ceil(num_agents / num_teams))

                to_log = {}

                # Steps per second. Measures env-step throughput between logging calls
                # Note: SPS drops during logging phases when videos are being saved.
                # This is expected and not a training slowdown
                _now = time.perf_counter()
                _current_env_step = int(env_step)
                if _sps_state["last_time"] is not None:
                    _elapsed = _now - _sps_state["last_time"]
                    _steps_taken = _current_env_step - _sps_state["last_env_step"]
                    if _elapsed > 0:
                        to_log["overview/sps"] = _steps_taken/_elapsed

                _sps_state["last_time"] = _now
                _sps_state["last_env_step"] = _current_env_step

                # ── overview/ ──
                to_log["overview/env_step"] = env_step
                # ML global metrics — one policy, one critic, so these are THE losses
                for k in ["total_loss", "value_loss", "actor_loss", "entropy",
                           "ratio", "ratio_0", "approx_kl", "clip_frac", "aux_loss"]:
                    to_log[f"overview/{k}"] = metrics["loss"][k]
                # Mean per-step aggregated reward actually optimized by the critic
                to_log["overview/central_step_reward"] = np.asarray(metrics["central_reward"]).item()

                # ── agent_{i}/ per-action-head entropy (diagnostic only) ──
                for i in range(num_agents):
                    to_log[f"agent_{i}/entropy"] = np.asarray(metrics["entropy_per_agent"][i]).item()

                # ── Episode-level metrics (only when episodes returned) ──
                if metrics["returned_episode"].any():
                    info = metrics["user_info"]
                    ep_mask = metrics["returned_episode"]  # (num_steps, num_envs, num_agents)

                    def _team_agent_indices(team_idx):
                        start = team_idx * agents_per_team
                        end = min(start + agents_per_team, num_agents)
                        return range(start, end)

                    def _agent_mean(key, agent_idx):
                        """Mean of metric for agent over returned episodes."""
                        mask = ep_mask[:, :, agent_idx]
                        if not mask.any():
                            return None
                        return np.asarray(info[key][:, :, agent_idx][mask].mean()).item()

                    def _team_mean(key, team_idx):
                        """Mean of metric across team members over returned episodes."""
                        vals = []
                        for agent_idx in _team_agent_indices(team_idx):
                            v = _agent_mean(key, agent_idx)
                            if v is not None:
                                vals.append(v)
                        return np.mean(vals) if vals else None

                    def _team_sum(key, team_idx):
                        """Sum per-agent episode counters across team members."""
                        vals = []
                        for agent_idx in _team_agent_indices(team_idx):
                            v = _agent_mean(key, agent_idx)
                            if v is not None:
                                vals.append(v)
                        return np.sum(vals) if vals else None

                    def _global_mean(key):
                        """Mean of metric over all returned episodes (agent 0, broadcast metric)."""
                        mask = ep_mask[:, :, 0]
                        if not mask.any():
                            return None
                        return np.asarray(info[key][:, :, 0][mask].mean()).item()

                    # overview/ episode metrics
                    ep_lengths = metrics["returned_episode_lengths"]
                    ep_returns = metrics["returned_episode_returns"]
                    mask0 = ep_mask[:, :, 0]
                    if mask0.any():
                        to_log["overview/episode_length"] = np.asarray(ep_lengths[:, :, 0][mask0].mean()).item()

                    # Overview shared/team reward as mean across all agent returns.
                    # When SHARED_REWARD=True, all per-agent returns are identical and
                    # this exactly matches the shared reward. When SHARED_REWARD=False,
                    # this stays a useful team-level overview metric.
                    all_agent_returns = []
                    for ai in range(num_agents):
                        mask_ai = ep_mask[:, :, ai]
                        if mask_ai.any():
                            agent_return = np.asarray(ep_returns[:, :, ai][mask_ai].mean()).item()
                            all_agent_returns.append(agent_return)
                            to_log[f"agent_{ai}/episode_return"] = agent_return
                            individual_reward = _agent_mean("Reward/individual_reward", ai)
                            to_log[f"agent_{ai}/individual_reward"] = (
                                individual_reward if individual_reward is not None else agent_return
                            )
                    if all_agent_returns:
                        mean_return = float(np.mean(all_agent_returns))
                        to_log["overview/shared_reward"] = mean_return
                        # Backward-compatible alias for older dashboards.
                        to_log["overview/avg_reward"] = mean_return
                        # The quantity the centralized critic actually predicts.
                        to_log["overview/central_episode_return"] = (
                            float(np.sum(all_agent_returns))
                            if reward_aggregation == "sum"
                            else mean_return
                        )

                    # Per-agent and overview movement (mean walking distance over returned episodes)
                    all_walk = []
                    for ai in range(num_agents):
                        v = _agent_mean("Movement/walking_distance", ai)
                        if v is not None:
                            to_log[f"agent_{ai}/walking_distance"] = v
                            to_log[f"agent_{ai}/movement"] = v
                            all_walk.append(v)
                    if all_walk:
                        to_log["overview/walking_distance"] = np.mean(all_walk)
                        to_log["overview/movement"] = np.mean(all_walk)

                    # Per-agent and overview distance to spawn (mean over returned episodes)
                    all_spawn_dists = []
                    for ai in range(num_agents):
                        v = _agent_mean("Movement/distance_to_spawn", ai)
                        if v is not None:
                            to_log[f"agent_{ai}/distance_to_spawn"] = v
                            all_spawn_dists.append(v)
                    if all_spawn_dists:
                        to_log["overview/distance_to_spawn"] = np.mean(all_spawn_dists)

                    combat_damage_keys = [
                        "damage_taken_melee",
                        "damage_taken_health_food",
                        "damage_taken_health_drink",
                        "damage_taken_health_energy",
                        "damage_taken_health_other",
                        "damage_taken_ff",
                    ]
                    for ai in range(num_agents):
                        for dk in combat_damage_keys:
                            v = _agent_mean(f"Combat/{dk}", ai)
                            if v is not None:
                                to_log[f"agent_{ai}/{dk}"] = v

                    # overview/ trades (broadcast scalars, take from agent 0)
                    for trade_key in ["total_trades", "food_trades", "drink_trades"]:
                        v = _global_mean(f"Trade/{trade_key}")
                        if v is not None:
                            to_log[f"overview/{trade_key}"] = v

                    v = _global_mean("Revive/revives")
                    if v is not None:
                        to_log["overview/revives"] = v

                    per_agent_event_keys = [
                        ("Trade/trades_given", "trades_given"),
                        ("Trade/trades_received", "trades_received"),
                        ("Revive/revives_given", "revives_given"),
                        ("Revive/revives_received", "revives_received"),
                    ]
                    for ai in range(num_agents):
                        for info_key, log_key in per_agent_event_keys:
                            v = _agent_mean(info_key, ai)
                            if v is not None:
                                to_log[f"agent_{ai}/{log_key}"] = v

                    # ── team_{t}/ metrics ──
                    for ti in range(num_teams):
                        tp = f"team_{ti}"

                        # shared_reward (= episode_returns averaged over team members)
                        team_ret = []
                        for idx in _team_agent_indices(ti):
                            mask_ai = ep_mask[:, :, idx]
                            if mask_ai.any():
                                team_ret.append(np.asarray(ep_returns[:, :, idx][mask_ai].mean()).item())
                        if team_ret:
                            to_log[f"{tp}/shared_reward"] = np.mean(team_ret)

                        # walking_distance per team
                        v = _team_mean("Movement/walking_distance", ti)
                        if v is not None:
                            to_log[f"{tp}/walking_distance"] = v

                        # distance_to_spawn per team
                        v = _team_mean("Movement/distance_to_spawn", ti)
                        if v is not None:
                            to_log[f"{tp}/distance_to_spawn"] = v

                        # combat: damage taken (aggregated over team members)
                        for dk in combat_damage_keys:
                            v = _team_mean(f"Combat/{dk}", ti)
                            if v is not None:
                                to_log[f"{tp}/{dk}"] = v

                        # event counters: sum over team members
                        for info_key, log_key in per_agent_event_keys:
                            v = _team_sum(info_key, ti)
                            if v is not None:
                                to_log[f"{tp}/{log_key}"] = v

                        # combat: damage dealt to other team + kills (broadcast scalars)
                        v = _global_mean(f"Combat/team_{ti}_damage_dealt")
                        if v is not None:
                            to_log[f"{tp}/damage_to_other_team"] = v
                        v = _global_mean(f"Combat/team_{ti}_kills")
                        if v is not None:
                            to_log[f"{tp}/kills_against_other_team"] = v

                    # ── team_achievements/team_{t}/ ──
                    for ti in range(num_teams):
                        tp = f"team_achievements/team_{ti}"
                        for achievement_key in [k for k in info.keys() if k.startswith("Achievements/")]:
                            short_name = achievement_key.split("/", 1)[1]
                            v = _team_mean(achievement_key, ti)
                            if v is not None:
                                to_log[f"{tp}/{short_name}"] = v

                wandb.log(to_log, step=metrics["update_steps"])

            jax.experimental.io_callback(callback, None, metric, update_steps, ordered=True)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), metric


        # Do one "step" of logging, writing the result to a file.
        # Several steps can be run in series using LOGGING_NUM_CALLS to do long
        # rollouts without hitting memory limits.
        def _logging_step(carry, unused, logging_threads, update_step, effective_cap=None):
            runner_state, episode_count = carry
            runner_state = _apply_episode_cap_to_runner_state(runner_state, effective_cap)
            # Visualization rollouts (with detailed logging for CSV)
            runner_state, traj_batch = jax.lax.scan(
                functools.partial(_env_step, detailed_logging=True, effective_cap=effective_cap),
                runner_state,
                None,
                config["LOGGING_STEPS_PER_CALL"],
            )

            # Finally, log data associated with the visualization runs

            save_hstates = config.get("SAVE_HIDDEN_STATES", False)
            if save_hstates:
                # ONE shared hidden state: (T, NUM_ENVS, hidden_dim)
                hidden_states = traj_batch.info['hidden_state']
            # Null this for memory savings
            traj_batch.info['hidden_state'] = None

            # Compute a pseudo episode_id from cumulative done flags
            # done shape: (T, num_agents, NUM_ENVS)  (replicated global done)
            # episode_count shape: (num_agents, NUM_ENVS) — carried across logging steps
            # Shift by 1 so the done step itself still belongs to the old episode
            done_shifted = jnp.concatenate([
                jnp.zeros((1,) + traj_batch.info['done'].shape[1:]),
                traj_batch.info['done'][:-1]
            ], axis=0)
            local_episode_id = jnp.cumsum(done_shifted, axis=0)  # (T, num_agents, NUM_ENVS)
            traj_batch.info['episode_id'] = (episode_count[None, :, :] + local_episode_id).astype(jnp.float32)
            # Update episode_count for next logging step: add total dones in this chunk
            episode_count = episode_count + traj_batch.info['done'].sum(axis=0).astype(episode_count.dtype)

            # Add new logging fields here.
            # NOTE: `value` is the single global V(s) repeated per agent, and
            # `log_prob` / `entropy` are that agent's action-head marginals;
            # `joint_log_prob` is the log-probability of the full joint action.
            fields_to_log = ['health', 'food', 'drink', 'energy', 'done', 'is_sleeping', 'is_resting',
                             'player_position_x',
                             'player_position_y', 'recover', 'hunger', 'thirst', 'fatigue', 'light_level',
                             'dist_to_melee_l1',
                             'melee_on_screen', 'dist_to_passive_l1', 'passive_on_screen', 'dist_to_ranged_l1',
                             'ranged_on_screen', 'num_melee_nearby', 'num_passives_nearby', 'num_ranged_nearby',
                             'trade_give', 'trade_receive', 'trade_give_material_id', 'trade_receive_material_id',
                             'trade_give_partner_id', 'trade_receive_partner_id',
                             'revive_as_reviver', 'revive_as_revived', 'revive_partner_id', 'auto_respawned',
                             'delta_x', 'delta_y', 'pred_delta_x', 'pred_delta_y',
                             'num_monsters_killed',
                             'has_sword', 'has_pick', 'bow', 'arrows', 'held_iron', 'value',
                             'entropy', 'log_prob', 'joint_log_prob', 'episode_id',
                            ]

            header_field_names = list(fields_to_log)

            def write_scalars(scalars, increment=0, agent_n=0):
                run_out_path = get_run_output_dir()
                os.makedirs(run_out_path, exist_ok=True)
                temp_dir = os.path.join(run_out_path, '.tmp')
                os.makedirs(temp_dir, exist_ok=True)
                # Assemble header for the scalar file(s)
                scalar_file_header = 'action'
                for key in header_field_names:
                    scalar_file_header += ',' + key

                # Keep temp files local to this run so different runs can share an
                # OUTPUT_DIR root without colliding.
                for i in range(logging_threads):
                    out_filename_scalars = os.path.join(run_out_path, 'scalars_{}_{}_{}.csv'.format(increment, agent_n, i))
                    with tempfile.NamedTemporaryFile(mode='w+', dir=temp_dir, suffix='.csv', delete=False) as temp_handle:
                        temp_filename = temp_handle.name
                    try:
                        np.savetxt(temp_filename,
                                   scalars[:, i, :], delimiter=',', fmt='%f',
                                   header=scalar_file_header
                                   )
                        with open(temp_filename, 'r', encoding='utf-8') as temp_file, open(out_filename_scalars, 'a+', encoding='utf-8') as out_file_scalars:
                            out_file_scalars.write(temp_file.read())
                    finally:
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)
                    print('Writing log file', out_filename_scalars)

            def write_hstates(hstate, increment=0):
                """The recurrent state is shared by the whole team, so it is written
                once per logged env rather than once per agent.

                GRU carry is (T, num_envs, hidden_dim) -> one row per step.  The
                transformer carry is (T, num_envs, max_history, d_model + 1), so
                the window is flattened row-major into a single wide row per step:
                slot-major, oldest slot first, each slot's last column being its
                validity flag.
                """
                hstate = np.asarray(hstate)
                if hstate.ndim > 3:
                    hstate = hstate.reshape(hstate.shape[0], hstate.shape[1], -1)
                run_out_path = get_run_output_dir()
                os.makedirs(run_out_path, exist_ok=True)
                temp_dir = os.path.join(run_out_path, '.tmp')
                os.makedirs(temp_dir, exist_ok=True)
                for i in range(logging_threads):
                    out_filename_hstates = os.path.join(run_out_path, 'hstates_{}_central_{}.csv'.format(increment, i))
                    with tempfile.NamedTemporaryFile(mode='w+', dir=temp_dir, suffix='.csv', delete=False) as temp_handle:
                        temp_filename = temp_handle.name
                    try:
                        np.savetxt(temp_filename, hstate[:, i, :], delimiter=',')
                        with open(temp_filename, 'r', encoding='utf-8') as temp_file, open(out_filename_hstates, 'a+', encoding='utf-8') as out_file_hstates:
                            out_file_hstates.write(temp_file.read())
                    finally:
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)
                    print('Writing log file', out_filename_hstates)

            # Add the specified field to the logging array.
            # Network outputs (action, done, value, entropy, log_prob, ...) have
            # shape (T, num_agents, NUM_ENVS); environment fields coming from the
            # wrappers have shape (T, NUM_ENVS, num_agents).
            network_output_fields = {'value', 'entropy', 'log_prob', 'joint_log_prob', 'done',
                                     'action', 'episode_id',
                                     'delta_x', 'delta_y', 'pred_delta_x', 'pred_delta_y'}

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

            # Assemble logging variable array — one CSV per agent, as before
            for agent_n in range(env.num_agents):
                # Network outputs have shape (T, num_agents, NUM_ENVS) - extract agent_n -> (T, NUM_ENVS)
                log_array = traj_batch.info['action'][:, agent_n, :].reshape((traj_batch.info['action'].shape[0], config['NUM_ENVS'], 1))
                # Yes this is a for loop in the JAX code but this stuff was getting done in serial before anyway and it's cheap operations
                for field_to_log in fields_to_log:
                    log_array = add_field_to_log_array(traj_batch.info, log_array, field_to_log, agent_n)

                jax.experimental.io_callback(
                    write_scalars, None, log_array, update_step, agent_n, ordered=True
                )

            if save_hstates:
                jax.experimental.io_callback(
                    write_hstates, None, hidden_states, update_step, ordered=True
                )

            return (runner_state, episode_count), None

            # Func to interleave update steps and plotting

        # ===========================
        # Video frame buffer — frames are streamed to host via callback
        # instead of being accumulated in GPU memory by jax.lax.scan.
        # ===========================
        _video_frame_buffer = {'ego': [], 'map': []}

        def _collect_video_frame(ego_frame, map_frame):
            """Host-side callback: appends one rendered frame (uint8) to the buffer."""
            if _video_max_length > 0 and len(_video_frame_buffer['ego']) >= _video_max_length:
                return
            _video_frame_buffer['ego'].append(np.asarray(ego_frame).astype(np.uint8))
            _video_frame_buffer['map'].append(np.asarray(map_frame).astype(np.uint8))

        def _clear_video_buffer():
            _video_frame_buffer['ego'].clear()
            _video_frame_buffer['map'].clear()

        # ===========================
        # Video Rollout Step (single env — memory-efficient)
        # ===========================
        def _video_step_1env(runner_state, unused, effective_cap=None):
            """One env step for a single environment for video recording.

            Uses only 1 env instead of NUM_ENVS.  Rendered frames are streamed
            to the host via io_callback so that jax.lax.scan does NOT
            accumulate them in GPU memory (saves several GB).
            """
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            obs_batch = global_obs(last_obs)  # (1, global_obs_dim)

            hstate, pi, value, aux_pred = forward(
                train_state.params,
                hstate,                    # (1, hidden_dim)
                obs_batch[np.newaxis, :],  # (1, 1, global_obs_dim)
                last_done[np.newaxis, :],  # (1, 1)
            )

            if action_mask_while_dead:
                alive_batch = compute_alive_mask(env_state)  # (1, num_agents)
                masked_logits = apply_dead_action_mask(pi.logits, alive_batch[np.newaxis, :])
                pi = distrax.Categorical(logits=masked_logits)

            action = pi.sample(seed=_rng)    # (1, 1, num_agents)
            action = action.squeeze(axis=0)  # (1, num_agents)

            # Index instead of squeeze — keeps the (1,) batch dim for vmap
            env_action = policy_action_to_env_action(action)
            env_act = {a: env_action[:, i] for i, a in enumerate(env.agents)}

            # STEP 1 env (use env_train — video doesn't need CSV fields)
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, 1)
            obsv, env_state, reward, done, info = jax.vmap(
                env_train.step, in_axes=(0, 0, 0)
            )(rng_step, env_state, env_act)
            if effective_cap is not None:
                env_state = _patch_episode_cap(env_state, effective_cap)

            # Render the single env
            craftax_state = jax.tree_util.tree_map(lambda x: x[0], env_state.env_state)

            # Render ego-perspective: (num_agents, H, W, 3) float32 [0, 255]
            ego_frame = render_ego_perspective(
                craftax_state, _video_pixel_size, _video_static_params,
                _video_player_textures, _video_env_name
            )

            # Render full map: (map_H, map_W, 3) float32 [0, 255]
            full_map_frame = render_full_map(
                craftax_state, _video_static_params,
                _video_textures[_video_pixel_size], _video_player_textures,
                _video_pixel_size, env_name=_video_env_name
            )
            ego_frame = ego_frame.astype(jnp.uint8)
            full_map_frame = full_map_frame.astype(jnp.uint8)

            # Stream frame to host immediately — NOT accumulated by scan
            # io_callback guarantees execution order and is not optimised away
            jax.experimental.io_callback(_collect_video_frame, None, ego_frame, full_map_frame, ordered=True)

            runner_state = (train_state, env_state, obsv, done["__all__"], hstate, rng)
            return runner_state, None

        def _update_plot(runner_state, unused):
            # First, do iterations of logging
            state, update_steps = runner_state
            effective_cap = _get_effective_episode_cap(update_steps)
            state = _apply_episode_cap_to_runner_state(state, effective_cap)
            # episode_count tracks cumulative episode IDs across logging steps: (num_agents, NUM_ENVS)
            episode_count = jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=jnp.int32)
            (state, episode_count), empty = jax.lax.scan(
                functools.partial(
                    _logging_step,
                    logging_threads=config["LOGGING_THREADS"],
                    update_step=update_steps,
                    effective_cap=effective_cap,
                ),
                (state, episode_count), None,
                config["LOGGING_NUM_CALLS"],
            )

            # ===========================
            # Video Rollout (fresh episode so we see the overworld spawn)
            # ===========================
            if config.get("SAVE_VIDEO", False):
                video_length = int(config.get("VIDEO_LENGTH", 200))
                if _video_max_length > 0:
                    video_length = min(video_length, _video_max_length)

                # Unpack training state
                train_state_v, env_state_v, obsv_v, done_v, hstate_v, rng_v = state

                # Fork RNG: one for video, one to continue training
                rng_video, rng_continue = jax.random.split(rng_v)

                # Reset only 1 env for video (saves ~NUM_ENVS × video_length env-state memory)
                video_reset_rngs = jax.random.split(rng_video, 1)
                video_obsv, video_env_state = jax.vmap(env_train.reset, in_axes=(0,))(video_reset_rngs)
                if effective_cap is not None:
                    video_env_state = _patch_episode_cap(video_env_state, effective_cap)

                # Fresh hidden state / attention window (1 env) and done flag
                video_hstate = zero_hstate(1)
                video_done = jnp.zeros((1,), dtype=bool)

                # Build video runner state (uses current policy weights)
                rng_video2, _ = jax.random.split(rng_video)
                video_runner = (train_state_v, video_env_state, video_obsv, video_done, video_hstate, rng_video2)

                # Clear host-side frame buffer before video rollout
                jax.experimental.io_callback(_clear_video_buffer, None, ordered=True)

                # Run video rollout — frames are streamed to host via callback,
                # scan output is None (no GPU memory accumulation)
                _, _ = jax.lax.scan(
                    functools.partial(_video_step_1env, effective_cap=effective_cap),
                    video_runner,
                    None,
                    video_length,
                )

                # Restore training state with updated RNG (video state discarded)
                state = (train_state_v, env_state_v, obsv_v, done_v, hstate_v, rng_continue)

                def save_video_from_buffer(step):
                    """Assemble streamed frames from buffer and save as video files."""
                    step_int = int(np.asarray(step).flat[0])
                    if not _video_frame_buffer['ego']:
                        print(f'Warning: No video frames collected at step {step_int}, skipping video save.')
                        return
                    run_out_path = os.path.join(get_run_output_dir(), 'videos')
                    os.makedirs(run_out_path, exist_ok=True)

                    ego_frames = np.stack(_video_frame_buffer['ego'])    # (T, num_agents, H, W, 3) uint8
                    full_map_frames = np.stack(_video_frame_buffer['map'])  # (T, map_H, map_W, 3) uint8

                    # Save per-agent ego videos
                    num_agents = ego_frames.shape[1]
                    wandb_videos = {}
                    for agent_idx in range(num_agents):
                        agent_frames = ego_frames[:, agent_idx]  # already uint8
                        video_path = os.path.join(run_out_path, f'ego_agent_{agent_idx}_{step_int}.mp4')
                        try:
                            imageio.mimsave(video_path, agent_frames, fps=15, macro_block_size=1)
                            wandb_videos[f"video/ego_agent_{agent_idx}"] = wandb.Video(video_path, fps=15, format="mp4")
                            print(f'Saved ego video: {video_path}')
                        except Exception as e:
                            print(f'Failed to save ego video for agent {agent_idx}: {e}')

                    # Save full map video
                    fullmap_path = os.path.join(run_out_path, f'full_map_{step_int}.mp4')
                    try:
                        imageio.mimsave(fullmap_path, full_map_frames, fps=15, macro_block_size=1)
                        wandb_videos["video/full_map"] = wandb.Video(fullmap_path, fps=15, format="mp4")
                        print(f'Saved full map video: {fullmap_path}')
                    except Exception as e:
                        print(f'Failed to save full map video: {e}')

                    # Log all videos to wandb in one call
                    if wandb_videos:
                        wandb.log(wandb_videos, step=step_int)
                    _clear_video_buffer()

                jax.experimental.io_callback(save_video_from_buffer, None, update_steps, ordered=True)

            runner_state = (state, update_steps)

            # Then, update (training)
            runner_state, metric = jax.lax.scan(
                _update_step, runner_state, None, config["LOGGING_UPDATES_INTERVAL"]
            )

            return runner_state, None

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            init_done,
            init_hstate,
            _rng,
        )
        init_carry = (runner_state, jnp.asarray(0, dtype=jnp.int32))
        return init_carry, _update_plot, _update_step

    return train

# ===========================
# Main Run Function
# ===========================
def build_env(config):
    """Construct the wrapped Craftax environment described by the config."""
    env_name = config.get("ENV_NAME", "Craftax-Coop-Symbolic")
    num_teams = config.get("NUM_TEAMS", 2)
    team_composition = tuple(config.get("TEAM_COMPOSITION", [1, 1, 2]))
    max_melee_mobs = int(config.get("MAX_MELEE_MOBS", 40))
    max_passive_mobs = int(config.get("MAX_PASSIVE_MOBS", 105))
    max_ranged_mobs = int(config.get("MAX_RANGED_MOBS", 0))
    disable_revive = config.get("DISABLE_REVIVE", False)
    terminate_on_any_death = config.get("TERMINATE_ON_ANY_DEATH", False)
    terminate_on_any_death_offset = config.get("TERMINATE_ON_ANY_DEATH_OFFSET", 200)
    reviving_cooldown_steps = config.get("REVIVING_COOLDOWN_STEPS", 0)
    action_mask_while_dead = config.get("ACTION_MASK_WHILE_DEAD", False)
    teammate_alive_bonus = config.get("TEAMMATE_ALIVE_BONUS", 0.0)
    all_team_alive_bonus = config.get("ALL_TEAM_ALIVE_BONUS", 0.0)
    dead_self_penalty_weight = config.get("DEAD_SELF_PENALTY_WEIGHT", 0.0)
    one_time_death_penalty_shared = config.get("ONE_TIME_DEATH_PENALTY_SHARED", 0.0)
    one_time_death_penalty_individual = config.get("ONE_TIME_DEATH_PENALTY_INDIVIDUAL", 0.0)
    warrior_melee_kill_reward = config.get("WARRIOR_MELEE_KILL_REWARD", 0.0)
    forager_melee_kill_reward = config.get("FORAGER_MELEE_KILL_REWARD", 0.0)
    warrior_passive_food_gain = int(config.get("WARRIOR_PASSIVE_FOOD_GAIN", 1))
    forager_passive_food_gain = int(config.get("FORAGER_PASSIVE_FOOD_GAIN", 3))
    forager_food_capacity = int(config.get("FORAGER_FOOD_CAPACITY", 27))
    forager_predator_damage_multiplier = float(config.get("FORAGER_PREDATOR_DAMAGE_MULTIPLIER", 1.0))
    forager_to_warrior_food_trade_reward = config.get("FORAGER_TO_WARRIOR_FOOD_TRADE_REWARD", 0.0)
    forager_to_warrior_drink_trade_reward = config.get("FORAGER_TO_WARRIOR_DRINK_TRADE_REWARD", 0.0)
    warrior_to_warrior_food_trade_reward = config.get("WARRIOR_TO_WARRIOR_FOOD_TRADE_REWARD", 0.0)
    warrior_to_warrior_drink_trade_reward = config.get("WARRIOR_TO_WARRIOR_DRINK_TRADE_REWARD", 0.0)
    trade_reward_requires_both_outside_starter_room = config.get("TRADE_REWARD_REQUIRES_BOTH_OUTSIDE_STARTER_ROOM", False)
    melee_mobs_despawn_when_far = config.get("MELEE_MOBS_DESPAWN_WHEN_FAR", False)
    melee_mob_despawn_distance = int(config.get("MELEE_MOB_DESPAWN_DISTANCE", 14))
    passive_mobs_static = config.get("PASSIVE_MOBS_STATIC", False)
    enable_auto_respawning = config.get("ENABLE_AUTO_RESPAWNING", False)
    enable_warrior_to_warrior_trading = config.get("ENABLE_WARRIOR_TO_WARRIOR_TRADING", False)
    auto_respawn_steps = int(config.get("AUTO_RESPAWN_STEPS", 50))
    restrict_auto_respawning_to_spawn_room = config.get("RESTRICT_AUTO_RESPAWNING_TO_SPAWN_ROOM", True)
    initial_predators_spawn_in_warrior_rooms_only = config.get("INITIAL_PREDATORS_SPAWN_IN_WARRIOR_ROOMS_ONLY", False)
    non_forager_always_in_lone_room = config.get("NON_FORAGER_ALWAYS_IN_LONE_ROOM", False)
    spread_non_foragers_across_rooms = config.get("SPREAD_NON_FORAGERS_ACROSS_ROOMS", False)
    trade_radius = int(config.get("TRADE_RADIUS", 18))
    trade_radius_shape = str(config.get("TRADE_RADIUS_SHAPE", "square")).lower()
    shared_reward = config.get("SHARED_REWARD", True)
    if trade_radius < 0:
        raise ValueError(f"TRADE_RADIUS must be >= 0, got {trade_radius}.")
    if trade_radius_shape not in {"square", "circle"}:
        raise ValueError(
            f"TRADE_RADIUS_SHAPE must be 'square' or 'circle', got {trade_radius_shape!r}."
        )
    if warrior_passive_food_gain < 0:
        raise ValueError(
            f"WARRIOR_PASSIVE_FOOD_GAIN must be >= 0, got {warrior_passive_food_gain}."
        )
    if forager_melee_kill_reward < 0:
        raise ValueError(
            f"FORAGER_MELEE_KILL_REWARD must be >= 0, got {forager_melee_kill_reward}."
        )
    if forager_passive_food_gain < 0:
        raise ValueError(
            f"FORAGER_PASSIVE_FOOD_GAIN must be >= 0, got {forager_passive_food_gain}."
        )
    if forager_food_capacity < 1:
        raise ValueError(
            f"FORAGER_FOOD_CAPACITY must be >= 1, got {forager_food_capacity}."
        )
    if forager_predator_damage_multiplier < 0:
        raise ValueError(
            "FORAGER_PREDATOR_DAMAGE_MULTIPLIER must be >= 0, "
            f"got {forager_predator_damage_multiplier}."
        )
    if max_melee_mobs < 0:
        raise ValueError(f"MAX_MELEE_MOBS must be >= 0, got {max_melee_mobs}.")
    if max_passive_mobs < 1:
        raise ValueError(f"MAX_PASSIVE_MOBS must be >= 1, got {max_passive_mobs}.")
    if max_ranged_mobs < 0:
        raise ValueError(f"MAX_RANGED_MOBS must be >= 0, got {max_ranged_mobs}.")
    static_env_params_kwargs = {
        "max_melee_mobs": max_melee_mobs,
        "max_passive_mobs": max_passive_mobs,
        "max_ranged_mobs": max_ranged_mobs,
    }
    env_params_kwargs = {
        "disable_revive": disable_revive,
        "terminate_on_any_death": terminate_on_any_death,
        "terminate_on_any_death_offset": terminate_on_any_death_offset,
        "reviving_cooldown_steps": reviving_cooldown_steps,
        "teammate_alive_bonus": teammate_alive_bonus,
        "all_team_alive_bonus": all_team_alive_bonus,
        "dead_self_penalty_weight": dead_self_penalty_weight,
        "one_time_death_penalty_shared": one_time_death_penalty_shared,
        "one_time_death_penalty_individual": one_time_death_penalty_individual,
        "warrior_melee_kill_reward": warrior_melee_kill_reward,
        "forager_melee_kill_reward": forager_melee_kill_reward,
        "warrior_passive_food_gain": warrior_passive_food_gain,
        "forager_passive_food_gain": forager_passive_food_gain,
        "forager_food_capacity": forager_food_capacity,
        "forager_predator_damage_multiplier": forager_predator_damage_multiplier,
        "forager_to_warrior_food_trade_reward": forager_to_warrior_food_trade_reward,
        "forager_to_warrior_drink_trade_reward": forager_to_warrior_drink_trade_reward,
        "warrior_to_warrior_food_trade_reward": warrior_to_warrior_food_trade_reward,
        "warrior_to_warrior_drink_trade_reward": warrior_to_warrior_drink_trade_reward,
        "enable_warrior_to_warrior_trading": enable_warrior_to_warrior_trading,
        "trade_reward_requires_both_outside_starter_room": trade_reward_requires_both_outside_starter_room,
        "enable_auto_respawning": enable_auto_respawning,
        "auto_respawn_steps": auto_respawn_steps,
        "melee_mobs_despawn_when_far": melee_mobs_despawn_when_far,
        "melee_mob_despawn_distance": melee_mob_despawn_distance,
        "passive_mobs_static": passive_mobs_static,
        "restrict_auto_respawning_to_spawn_room": restrict_auto_respawning_to_spawn_room,
        "initial_predators_spawn_in_warrior_rooms_only": initial_predators_spawn_in_warrior_rooms_only,
        "non_forager_always_in_lone_room": non_forager_always_in_lone_room,
        "spread_non_foragers_across_rooms": spread_non_foragers_across_rooms,
        "trade_radius": trade_radius,
        "trade_radius_shape": trade_radius_shape,
        "shared_reward": shared_reward,
    }
    config["ACTION_MASK_WHILE_DEAD"] = action_mask_while_dead
    env = make_craftax_env_from_name(
        env_name,
        num_teams=num_teams,
        team_composition=team_composition,
        env_params_kwargs=env_params_kwargs,
        static_env_params_kwargs=static_env_params_kwargs,
    )
    return env


def single_run(config):
    alg_name = config.get("ALG_NAME", "central-ippo-rnn")
    env_name = config.get("ENV_NAME", "Craftax-Coop-Symbolic")
    env = build_env(config)

    if config["NUM_SEEDS"] != 1:
        raise ValueError(
            "central_ippo_rnn currently supports NUM_SEEDS == 1 for reliable logging/video callbacks."
        )

    checkpointing = config.get("CHECKPOINTING", True)
    ckpt_dir = ckpt.default_checkpoint_dir(config)
    resume_mode = ckpt.normalize_resume_mode(config.get("RESUME", "auto"))

    mngr = None
    meta = None
    resuming = False
    if checkpointing:
        mngr = ckpt.make_manager(ckpt_dir, config)
        meta = ckpt.read_sidecar(ckpt_dir)
        have_ckpt = mngr.latest_step() is not None and meta is not None
        if resume_mode == "true" and not have_ckpt:
            raise FileNotFoundError(
                f"RESUME is true but no complete checkpoint was found in {ckpt_dir}."
            )
        resuming = have_ckpt and resume_mode != "false"
        if have_ckpt and not resuming:
            print(
                f"[warning] existing checkpoints in {ckpt_dir} are being ignored "
                "because RESUME is false; new checkpoints may mix with old steps."
            )

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=config["RUN_NAME"],
        id=(meta["wandb_run_id"] if resuming else None),
        resume=("allow" if resuming else None),
        config=config,
        mode=config["WANDB_MODE"],
    )
    wandb_run_id = getattr(wandb.run, "id", None) if wandb.run is not None else None

    rng = jax.random.PRNGKey(config["SEED"])
    train_fn = make_train(config, env)
    init_carry, update_plot_fn, update_step_fn = train_fn(rng)
    # Donate the carry so each per-block call reuses the input buffers for its
    # output instead of transiently holding two full copies of the training
    # state. Safe alongside async checkpointing: orbax completes the
    # device-to-host copy synchronously inside save(), so donated buffers are
    # never still in flight when the next block runs.
    jit_plot = jax.jit(update_plot_fn, donate_argnums=0)
    fingerprint = ckpt.structural_fingerprint(init_carry)

    if resuming:
        ckpt.check_compatibility(meta, fingerprint, config)
        carry = ckpt.restore_carry(mngr, init_carry)
        start_block = int(carry[1]) // config["LOGGING_UPDATES_INTERVAL"]
        print(f"[resume] update_steps={int(carry[1])} start_block={start_block}")
    else:
        carry = init_carry
        start_block = 0
        if checkpointing:
            ckpt.write_sidecar(
                ckpt_dir,
                {
                    "wandb_run_id": wandb_run_id,
                    "fingerprint": fingerprint,
                    "blocks_done": 0,
                    "update_steps": 0,
                    "logging_updates_interval": config["LOGGING_UPDATES_INTERVAL"],
                    "num_updates": config["NUM_UPDATES"],
                    "final": False,
                },
            )

    # Drop the init_carry name so the initial training-state buffers can be
    # freed (fresh runs: donated to jit_plot on the first call; resumed runs:
    # the restore template is no longer needed once carry is restored).
    del init_carry

    num_logging_iters = config["NUM_LOGGING_ITERS"]
    interval = max(1, int(config.get("CHECKPOINT_INTERVAL_BLOCKS", 1)))
    max_blocks = int(config.get("MAX_BLOCKS_THIS_RUN", 0))
    ckpt.install_signal_handler()

    blocks_this_run = 0
    for block in range(start_block, num_logging_iters):
        carry, _ = jit_plot(carry, None)
        carry = jax.block_until_ready(carry)
        blocks_this_run += 1
        done_blocks = block + 1
        stop = ckpt.stop_requested() or (max_blocks and blocks_this_run >= max_blocks)
        periodic = done_blocks % interval == 0 or done_blocks == num_logging_iters
        if checkpointing and (periodic or stop):
            ckpt.save_checkpoint(mngr, carry, done_blocks, ckpt_dir, wandb_run_id, fingerprint, config)
        if stop:
            if checkpointing:
                mngr.wait_until_finished()
            print(f"[stop] checkpointed at block {done_blocks}; exiting for resume")
            wandb.finish()
            return

    if config["REMAINING_UPDATES"] > 0 and int(carry[1]) < config["NUM_UPDATES"]:
        # Drop the per-update metric inside the jit so XLA can dead-code-eliminate
        # the stacked (REMAINING_UPDATES, ...) metric buffers instead of
        # materializing them as live outputs. Wandb logging is unaffected: it runs
        # via the io_callback inside update_step_fn.
        scan_tail = jax.jit(
            lambda c: jax.lax.scan(
                lambda inner_c, unused: (update_step_fn(inner_c, unused)[0], None),
                c,
                None,
                config["REMAINING_UPDATES"],
            ),
            donate_argnums=0,
        )
        carry, _ = scan_tail(carry)
        carry = jax.block_until_ready(carry)

    if checkpointing:
        ckpt.save_checkpoint(
            mngr, carry, num_logging_iters, ckpt_dir, wandb_run_id, fingerprint, config, final=True
        )
        mngr.wait_until_finished()
    wandb.finish()


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
