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
import gzip
import io
import yaml
from typing import Callable, NamedTuple, Dict

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

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_coop.constants import reduced_action_ids
from craftax.environment_base.wrappers import SelectiveResetVecEnvWrapper, add_logging_fields
from craftax.custom_rendering.base_rendering import load_rendering_resources
from craftax.custom_rendering.ego_rendering import render_ego_perspective
from craftax.custom_rendering.full_map_rendering import render_full_map

import checkpoint_utils as ckpt
import pruning_utils as pruning
import transformer_policy as tfp

# ===========================
# Model Definitions
# ===========================
class _DenseParams(nn.Module):
    """The parameters nn.Dense would create (same names, order, shapes and initializers)."""
    in_features: int
    features: int
    kernel_init: Callable
    use_bias: bool = True

    @nn.compact
    def __call__(self):
        kernel = self.param("kernel", self.kernel_init, (self.in_features, self.features), jnp.float32)
        if not self.use_bias:
            return kernel, None
        return kernel, self.param("bias", nn.initializers.zeros_init(), (self.features,), jnp.float32)


class _GRUCellParams(nn.Module):
    """The parameters of nn.GRUCell (same names and initial values), for ScannedRNN to apply."""
    in_features: int
    hidden_features: int

    @nn.compact
    def __call__(self):
        input_dense = functools.partial(
            _DenseParams, self.in_features, self.hidden_features, nn.initializers.lecun_normal()
        )
        hidden_dense = functools.partial(
            _DenseParams, self.hidden_features, self.hidden_features, nn.initializers.orthogonal()
        )
        return {
            "ir": input_dense(name="ir")(),
            "iz": input_dense(name="iz")(),
            "in": input_dense(name="in")(),
            "hr": hidden_dense(use_bias=False, name="hr")(),
            "hz": hidden_dense(use_bias=False, name="hz")(),
            "hn": hidden_dense(name="hn")(),
        }


class ScannedRNN(nn.Module):
    """nn.GRUCell run over a sequence, computed faster.

    The input projections of all time steps are one matmul before the time loop, the three recurrent
    projections are one matmul per step, and the loop is unrolled. Parameters (names and initial
    values) are those of the nn.GRUCell this replaced; results differ only in float rounding.
    """

    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x  # (time, batch, features), (time, batch)
        hidden = carry.shape[-1]
        # Named like the nn.GRUCell this replaced, so the parameters are unchanged.
        w = _GRUCellParams(ins.shape[-1], hidden, name="GRUCell_1")()
        input_kernel = jnp.concatenate([w["ir"][0], w["iz"][0], w["in"][0]], axis=1)
        input_bias = jnp.concatenate([w["ir"][1], w["iz"][1], w["in"][1]])
        hidden_kernel = jnp.concatenate([w["hr"][0], w["hz"][0], w["hn"][0]], axis=1)
        hidden_n_bias = w["hn"][1]
        input_proj = ins @ input_kernel + input_bias

        def step(h, inputs):
            input_proj_t, reset = inputs
            h = jnp.where(reset[:, np.newaxis], jnp.zeros_like(h), h)
            hidden_proj = h @ hidden_kernel
            r = nn.sigmoid(input_proj_t[:, :hidden] + hidden_proj[:, :hidden])
            z = nn.sigmoid(input_proj_t[:, hidden:2 * hidden] + hidden_proj[:, hidden:2 * hidden])
            n = jnp.tanh(input_proj_t[:, 2 * hidden:] + r * (hidden_proj[:, 2 * hidden:] + hidden_n_bias))
            h = (1.0 - z) * n + z * h
            return h, h

        return jax.lax.scan(step, carry, (input_proj, resets), unroll=8)

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)


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

def append_gzipped_csv(path, array, **savetxt_kwargs):
    """Append array to a gzipped CSV as one new gzip member.

    Concatenated gzip members form a valid .gz file, so gzip.open, zcat,
    np.loadtxt and pandas.read_csv read every appended chunk in order.
    """
    buffer = io.BytesIO()
    np.savetxt(buffer, array, delimiter=',', **savetxt_kwargs)
    # Level 6 gives ~99% of level 9's size reduction on hstate logs in half the time.
    compressed = gzip.compress(buffer.getvalue(), compresslevel=6)
    with open(path, 'ab') as out_file:
        out_file.write(compressed)

# ===========================
# Training Function
# ===========================
def make_train(config, env, wandb_start_step=-1):
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

    # Envs whose episode ended take start states from a pool generated in batches of this size
    # (see SelectiveResetVecEnvWrapper.step_with_reset_pool); 0 generates each one from the env's
    # own reset key instead.
    reset_pool_size = int(config.get("RESET_POOL_SIZE", config["NUM_ENVS"]))
    if reset_pool_size != 0 and reset_pool_size < config["NUM_ENVS"]:
        raise ValueError(
            f"RESET_POOL_SIZE must be 0 or >= NUM_ENVS, got {reset_pool_size} (NUM_ENVS={config['NUM_ENVS']})."
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
    if pruning.pruning_enabled(config):
        # pruning_utils prunes agent-stacked params one agent at a time; the centralized
        # runner has a single unstacked parameter set.
        raise ValueError(
            f"SPARSE_ALG={config['SPARSE_ALG']!r}: pruning is not supported by the centralized runner."
        )

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

    # vec_env steps all NUM_ENVS envs at once and only regenerates worlds for envs whose episode
    # ended (bit-identical to vmapping jaxmarl's LogWrapper.step, which regenerates every env's
    # world every step). Logging iterations add the extra CSV fields with add_logging_fields.
    vec_env = SelectiveResetVecEnvWrapper(env)
    env = vec_env  # default reference for property access (agents, num_agents, action_space, etc.)
    get_run_output_dir()  # pick (and print) the run output directory at startup

    num_agents = env.num_agents
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    global_obs_dim = num_agents * obs_dim

    # CSV columns: running episode-average distance to spawn, averaged over each team's foragers / warriors.
    # Roles are TEAM_COMPOSITION tiled per team (1 = forager, 2 = warrior); agent i is on team i // agents_per_team.
    _agents_per_team = len(config.get("TEAM_COMPOSITION", [1, 1, 2]))
    _agent_roles = np.tile(np.array(config.get("TEAM_COMPOSITION", [1, 1, 2])), num_agents // _agents_per_team)
    _agent_teams = np.arange(num_agents) // _agents_per_team
    # Every team gets both columns, so the CSV layout never depends on TEAM_COMPOSITION; a team with
    # no agents of a role writes -1 in that column.
    _role_dist_masks = {}  # column name -> (num_agents,) bool mask of the agents it averages over
    for _t in range(num_agents // _agents_per_team):
        for _role_id, _role_name in ((1, "foragers"), (2, "warriors")):
            _mask = (_agent_teams == _t) & (_agent_roles == _role_id)
            _role_dist_masks[f"{_role_name}_team{_t}_distance_to_origin"] = _mask
    _role_dist_fields = list(_role_dist_masks)

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

    _lr_anneal_updates = config.get("LR_ANNEAL_UPDATES", config["NUM_UPDATES"])
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / _lr_anneal_updates
        )
        frac = jnp.maximum(frac, 0.0)
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
        obsv, env_state = vec_env.reset(reset_rng)
        reset_pool = (
            vec_env.init_reset_pool(jax.random.fold_in(_rng, 1), reset_pool_size)
            if reset_pool_size > 0
            else None
        )
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
            train_state, env_state, last_obs, last_done, hstate, rng, reset_pool = runner_state

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

            if detailed_logging:
                # Read before the step, which may reset into the next episode, so the logged
                # origin is the same on every row of an episode_id, its done row included.
                spawn_before_step = env_state.env_state.player_spawn_position  # (num_envs, num_agents, 2)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, obs_batch.shape[0])
            if reset_pool is None:
                obsv, env_state, reward, done, info = vec_env.step(rng_step, env_state, env_act)
            else:
                obsv, env_state, reward, done, info, reset_pool = vec_env.step_with_reset_pool(
                    rng_step, env_state, env_act, reset_pool
                )
            # CSV fields (health, food, mob distances, etc.) are only needed during logging.
            if detailed_logging:
                info = jax.vmap(add_logging_fields)(info, env_state.env_state)
                info['spawn_position_x'] = spawn_before_step[..., 0]  # (num_envs, num_agents)
                info['spawn_position_y'] = spawn_before_step[..., 1]
                # Running mean over the current episode of each agent's Manhattan distance to its spawn
                _ep_steps = jnp.maximum(env_state.env_state.timestep.astype(jnp.float32), 1.0)  # (num_envs,)
                _run_dist = env_state.env_state.sum_distance_to_spawn / _ep_steps[:, None]  # (num_envs, num_agents)
                for _key, _mask in _role_dist_masks.items():
                    if _mask.any():
                        _v = _run_dist[:, _mask].mean(axis=1)  # (num_envs,)
                    else:
                        _v = jnp.full(_run_dist.shape[:1], -1.0)  # no agents of this role on this team
                    info[_key] = jnp.broadcast_to(_v[:, None], _run_dist.shape)

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

            runner_state = (train_state, env_state, obsv, done_global, hstate, rng, reset_pool)
            return runner_state, transition

        _early_episode_cap = config.get("EARLY_EPISODE_CAP", 0)
        _early_episode_cap_until = config.get("EARLY_EPISODE_CAP_UNTIL", 0)
        _general_episode_cap = config.get("GENERAL_EPISODE_CAP", 0)
        _default_max_timesteps = env.default_params.max_timesteps
        _post_early_cap = _general_episode_cap if _general_episode_cap > 0 else _default_max_timesteps

        # Progressive multi-phase cap schedule
        # Format: list of [start_update, cap]
        # pairs, sorted, first entry must start at 0.
        _cap_schedule = config.get("EPISODE_CAP_SCHEDULE", None)
        if _cap_schedule is not None:
            assert len(_cap_schedule) > 0, "EPISODE_CAP_SCHEDULE is empty"
            _starts = [int(s) for s, c in _cap_schedule]
            _caps = [float(c) for s, c in _cap_schedule]
            assert _starts[0] == 0, "EPISODE_CAP_SCHEDULE must start at step 0"
            assert _starts == sorted(_starts), "EPISODE_CAP_SCHEDULE must be sorted by start step"
            _cap_starts_arr = jnp.asarray(_starts, dtype=jnp.int32)
            _cap_values_arr = jnp.asarray(_caps, dtype=jnp.float32)
            _n_cap_phases = len(_cap_schedule)

        def _get_effective_episode_cap(update_steps):
            if _cap_schedule is not None:
                idx = jnp.sum(update_steps >= _cap_starts_arr) - 1
                idx = jnp.clip(idx, 0, _n_cap_phases - 1)
                return _cap_values_arr[idx]
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
            train_state, env_state, last_obs, last_done, hstate, rng, reset_pool = runner_state
            env_state = _patch_episode_cap(env_state, effective_cap)
            return (train_state, env_state, last_obs, last_done, hstate, rng, reset_pool)

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
            train_state, env_state, last_obs, last_done, hstate, rng, reset_pool = runner_state
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

                # Every array is (num_steps, num_envs, ...) except init_hstate, which is
                # (num_envs, *carry_dims). Permute over envs; minibatch m uses envs
                # permutation[m * MINIBATCH_SIZE:(m + 1) * MINIBATCH_SIZE]. Each minibatch is
                # gathered when it is used rather than copying the whole shuffled batch first.
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                minibatch_envs = permutation.reshape(config["NUM_MINIBATCHES"], config["MINIBATCH_SIZE"])

                def _gather_and_update_minbatch(train_state, envs):
                    batch_info = (
                        jnp.take(init_hstate, envs, axis=0),
                        jax.tree.map(lambda x: jnp.take(x, envs, axis=1), train_batch),
                        jnp.take(advantages, envs, axis=1),
                        jnp.take(targets, envs, axis=1),
                    )
                    return _update_minbatch(train_state, batch_info)

                train_state, total_loss = jax.lax.scan(
                    _gather_and_update_minbatch, train_state, minibatch_envs
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

            # Episode metrics are reduced on device to per-agent means over the episodes that
            # finished during this rollout, so the per-update host callback receives a few
            # hundred scalars instead of every (step, env, agent) value of every info field.
            finished = traj_info["returned_episode"]  # (num_steps, num_envs, num_agents)
            finished_count = finished.sum(axis=(0, 1))  # (num_agents,)

            def _finished_episode_mean(x):
                return jnp.where(finished, x, 0.0).sum(axis=(0, 1)) / jnp.maximum(finished_count, 1)

            metric = {
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
                "finished_episodes": finished_count,  # (num_agents,)
                "episode_means": jax.tree.map(  # every leaf: (num_agents,)
                    _finished_episode_mean,
                    {
                        "user_info": traj_info["user_info"],
                        "returned_episode_returns": traj_info["returned_episode_returns"],
                        "returned_episode_lengths": traj_info["returned_episode_lengths"],
                    },
                ),
            }

            rng = update_state[-1]

            #State for sps computations
            _sps_state = {'last_time':None, 'last_env_step': None}

            def callback(metrics, step):
                # Python int: update_steps arrives as an int32 array, and the product
                # overflows int32 once it passes 2**31 env steps (update 65536 in the
                # canonical config).
                env_step = (
                    int(metrics["update_steps"])
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
                if metrics["finished_episodes"].any():
                    finished_count = np.asarray(metrics["finished_episodes"])  # (num_agents,)
                    episode_means = metrics["episode_means"]
                    info = episode_means["user_info"]

                    def _team_agent_indices(team_idx):
                        start = team_idx * agents_per_team
                        end = min(start + agents_per_team, num_agents)
                        return range(start, end)

                    def _finished_mean(per_agent_means, agent_idx):
                        """Mean over this agent's returned episodes, or None if none returned."""
                        if finished_count[agent_idx] == 0:
                            return None
                        return np.asarray(per_agent_means[agent_idx]).item()

                    def _agent_mean(key, agent_idx):
                        """Mean of metric for agent over returned episodes."""
                        return _finished_mean(info[key], agent_idx)

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
                        return _agent_mean(key, 0)

                    # overview/ episode metrics
                    ep_returns = episode_means["returned_episode_returns"]
                    episode_length = _finished_mean(episode_means["returned_episode_lengths"], 0)
                    if episode_length is not None:
                        to_log["overview/episode_length"] = episode_length

                    # Overview shared/team reward as mean across all agent returns.
                    # When SHARED_REWARD=True, all per-agent returns are identical and
                    # this exactly matches the shared reward. When SHARED_REWARD=False,
                    # this stays a useful team-level overview metric.
                    all_agent_returns = []
                    for ai in range(num_agents):
                        agent_return = _finished_mean(ep_returns, ai)
                        if agent_return is not None:
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

                    # Distance to spawn averaged over all foragers / all warriors (roles from TEAM_COMPOSITION, tiled per team)
                    for role_id, role_name in ((1, "foragers"), (2, "warriors")):
                        role_dists = []
                        for ai in range(num_agents):
                            if configured_comp[ai % len(configured_comp)] != role_id:
                                continue
                            v = _agent_mean("Movement/distance_to_spawn", ai)
                            if v is not None:
                                role_dists.append(v)
                        if role_dists:
                            to_log[f"overview/distance_to_spawn_{role_name}"] = np.mean(role_dists)

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
                        v = _agent_mean("killed_another_agent", ai)
                        if v is not None:
                            to_log[f"agent_{ai}/kills_per_1000_steps"] = v * 1000

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
                            v = _finished_mean(ep_returns, idx)
                            if v is not None:
                                team_ret.append(v)
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

                        v = _global_mean(f"Combat/team_{ti}_kills")
                        if v is not None:
                            to_log[f"{tp}/kills_against_other_team"] = v
                        v = _team_sum("killed_another_agent", ti)
                        if v is not None:
                            to_log[f"{tp}/kills_per_1000_steps"] = v * 1000

                    # ── team_achievements/team_{t}/ ──
                    for ti in range(num_teams):
                        tp = f"team_achievements/team_{ti}"
                        for achievement_key in [k for k in info.keys() if k.startswith("Achievements/")]:
                            short_name = achievement_key.split("/", 1)[1]
                            v = _team_mean(achievement_key, ti)
                            if v is not None:
                                to_log[f"{tp}/{short_name}"] = v

                if metrics["update_steps"] > wandb_start_step:
                    to_log["overview/lr"] = config["LR"] * max(0.0, 1.0 - metrics["update_steps"] / _lr_anneal_updates)
                    wandb.log(to_log, step=metrics["update_steps"])

            jax.experimental.io_callback(callback, None, metric, update_steps, ordered=True)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng, reset_pool)
            return (runner_state, update_steps), metric


        # Do one "step" of logging, writing the result to a file.
        # Several steps can be run in series using LOGGING_NUM_CALLS to do long
        # rollouts without hitting memory limits.
        def _logging_step(carry, unused, logging_threads, update_step, effective_cap=None):
            runner_state, episode_count, ep_time_last, episode_id_last = carry
            runner_state = _apply_episode_cap_to_runner_state(runner_state, effective_cap)

            def save_arena_frame(frame, increment, env_index, episode_id):
                arena_dir = os.path.join(get_run_output_dir(), 'arenas')
                os.makedirs(arena_dir, exist_ok=True)
                path = os.path.join(
                    arena_dir, f'arena_{int(increment)}_{int(env_index)}_{int(episode_id)}.png'
                )
                # An episode can span multiple logging chunks; keep its first frame.
                if not os.path.exists(path):
                    imageio.imwrite(path, np.asarray(frame))

            def logging_env_step(carry, step_index):
                state, frame_episode_ids = carry
                state, transition = _env_step(
                    state, None, detailed_logging=True, effective_cap=effective_cap
                )
                # Names the arena frames only; the CSV episode_id is computed below. The env
                # resets inside the step, so the done row's state is already the next episode,
                # whose CSV rows (from the row after done) carry this advanced id.
                frame_episode_ids = frame_episode_ids + transition.info['done'].astype(frame_episode_ids.dtype)
                if config.get("SAVE_ARENA_FRAMES", True):
                    for env_index in range(logging_threads):
                        def capture_frame(_):
                            arena_state = jax.tree_util.tree_map(
                                lambda x: x[env_index], state[1].env_state
                            )
                            frame = render_full_map(
                                arena_state, _video_static_params,
                                _video_textures[_video_pixel_size], _video_player_textures,
                                _video_pixel_size, env_name=_video_env_name, terrain_only=True,
                            )
                            jax.experimental.io_callback(
                                save_arena_frame, None, frame.astype(jnp.uint8),
                                update_step, env_index, frame_episode_ids[0, env_index], ordered=True,
                            )

                        jax.lax.cond(
                            (step_index == 0) | transition.global_done[env_index],
                            capture_frame, lambda _: None, operand=None,
                        )
                return (state, frame_episode_ids), transition

            # Visualization rollouts (with detailed logging for CSV)
            (runner_state, _), traj_batch = jax.lax.scan(
                logging_env_step, (runner_state, episode_count),
                jnp.arange(config["LOGGING_STEPS_PER_CALL"]),
            )

            # Finally, log data associated with the visualization runs

            save_hstates = config.get("SAVE_HIDDEN_STATES", False)
            if save_hstates:
                # ONE shared hidden state: (T, NUM_ENVS, hidden_dim)
                hidden_states = traj_batch.info['hidden_state']
            # Null this for memory savings
            traj_batch.info['hidden_state'] = None

            # Compute a pseudo episode_id from cumulative done flags
            # done shape: (T, num_agents, logging envs)  (replicated global done)
            # episode_count shape: (num_agents, logging envs) — carried across logging steps
            # Shift by 1 so the done step itself still belongs to the old episode
            done_shifted = jnp.concatenate([
                jnp.zeros((1,) + traj_batch.info['done'].shape[1:]),
                traj_batch.info['done'][:-1]
            ], axis=0)
            local_episode_id = jnp.cumsum(done_shifted, axis=0)  # (T, num_agents, NUM_ENVS)
            traj_batch.info['episode_id'] = (episode_count[None, :, :] + local_episode_id).astype(jnp.float32)

            # ep_time: steps since the current episode started, one value per row.
            # Rows of the episode already running when this logging window opened
            # (episode_id == 0) get -1, because its true start is unknown. Otherwise
            # it is 0 on an episode's first row (the row after the previous episode's
            # done row, same boundary as episode_id) and counts up by 1 per row.
            # ep_time_last / episode_id_last hold the previous chunk's last row so the
            # count continues across logging steps.
            episode_id_int = (episode_count[None, :, :] + local_episode_id).astype(jnp.int32)  # (T, num_agents, NUM_ENVS)
            prev_episode_id = jnp.concatenate([episode_id_last[None, :, :], episode_id_int[:-1]], axis=0)
            episode_start = episode_id_int != prev_episode_id  # True on each episode's first row
            row_idx = jnp.arange(episode_id_int.shape[0], dtype=jnp.int32)[:, None, None]
            last_start_row = jax.lax.cummax(jnp.where(episode_start, row_idx, -1), axis=0)
            ep_time = jnp.where(
                last_start_row >= 0,
                row_idx - last_start_row,                 # episode started inside this chunk
                ep_time_last[None, :, :] + 1 + row_idx,   # episode continues from the previous chunk
            )
            ep_time = jnp.where(episode_id_int == 0, -1, ep_time)
            traj_batch.info['ep_time'] = ep_time.astype(jnp.float32)
            ep_time_last = ep_time[-1]
            episode_id_last = episode_id_int[-1]

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
                             'entropy', 'log_prob', 'joint_log_prob', 'episode_id', 'ep_time', *_role_dist_fields,
                             'spawn_position_x', 'spawn_position_y',
                            ]

            def write_scalars(scalars, increment=0, agent_n=0):
                run_out_path = get_run_output_dir()
                os.makedirs(run_out_path, exist_ok=True)
                # Columns are action, then fields_to_log in the order they are added below
                scalar_file_header = ','.join(['action', *fields_to_log])

                for i in range(logging_threads):
                    out_filename_scalars = os.path.join(run_out_path, 'scalars_{}_{}_{}.csv.gz'.format(increment, agent_n, i))
                    append_gzipped_csv(out_filename_scalars, scalars[:, i, :], fmt='%f', header=scalar_file_header)
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
                for i in range(logging_threads):
                    out_filename_hstates = os.path.join(run_out_path, 'hstates_{}_central_{}.csv.gz'.format(increment, i))
                    # 9 significant digits round-trip float32 exactly; numpy's default %.18e only adds noise digits.
                    append_gzipped_csv(out_filename_hstates, hstate[:, i, :], fmt='%.9g')
                    print('Writing log file', out_filename_hstates)

            # Add the specified field to the logging array.
            # Network outputs (action, done, value, entropy, log_prob, ...) have
            # shape (T, num_agents, NUM_ENVS); environment fields coming from the
            # wrappers have shape (T, NUM_ENVS, num_agents).
            network_output_fields = {'value', 'entropy', 'log_prob', 'joint_log_prob', 'done',
                                     'action', 'episode_id', 'ep_time',
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
                log_array = traj_batch.info['action'][:, agent_n, :, None]  # (T, logging envs, 1)
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

            return (runner_state, episode_count, ep_time_last, episode_id_last), None

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

            # STEP 1 env (video doesn't need CSV fields)
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, 1)
            obsv, env_state, reward, done, info = vec_env.step(rng_step, env_state, env_act)
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

            # The logging rollout runs on a copy of the first LOGGING_THREADS envs (the ones
            # written to the CSVs). The training envs do not advance; their RNG is split once.
            train_state_l, env_state_l, obsv_l, done_l, hstate_l, rng_l, reset_pool_l = state
            rng_l, rng_logging = jax.random.split(rng_l)
            logging_envs = lambda x: x[:config["LOGGING_THREADS"]]
            logging_runner = (
                train_state_l,
                jax.tree.map(logging_envs, env_state_l),
                jax.tree.map(logging_envs, obsv_l),
                jax.tree.map(logging_envs, done_l),
                hstate_l[:config["LOGGING_THREADS"]],  # (num_envs, hidden_dim) -> first LOGGING_THREADS envs
                rng_logging,
                None,  # finished logging envs generate their own new world
            )
            state = (train_state_l, env_state_l, obsv_l, done_l, hstate_l, rng_l, reset_pool_l)
            # episode_count tracks cumulative episode IDs across logging steps: (num_agents, LOGGING_THREADS)
            episode_count = jnp.zeros((env.num_agents, config["LOGGING_THREADS"]), dtype=jnp.int32)
            # ep_time / episode_id of the last logged row, carried across logging steps.
            # Starting at -1 / 0 marks the window's first episode as having an unknown start.
            ep_time_last = jnp.full((env.num_agents, config["LOGGING_THREADS"]), -1, dtype=jnp.int32)
            episode_id_last = jnp.zeros((env.num_agents, config["LOGGING_THREADS"]), dtype=jnp.int32)
            jax.lax.scan(
                functools.partial(
                    _logging_step,
                    logging_threads=config["LOGGING_THREADS"],
                    update_step=update_steps,
                    effective_cap=effective_cap,
                ),
                (logging_runner, episode_count, ep_time_last, episode_id_last), None,
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
                train_state_v, env_state_v, obsv_v, done_v, hstate_v, rng_v, reset_pool_v = state

                # Fork RNG: one for video, one to continue training
                rng_video, rng_continue = jax.random.split(rng_v)

                # Reset only 1 env for video (saves ~NUM_ENVS × video_length env-state memory)
                video_reset_rngs = jax.random.split(rng_video, 1)
                video_obsv, video_env_state = vec_env.reset(video_reset_rngs)
                if effective_cap is not None:
                    video_env_state = _patch_episode_cap(video_env_state, effective_cap)

                # Fresh hidden state (1 env) and done flag
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
                state = (train_state_v, env_state_v, obsv_v, done_v, hstate_v, rng_continue, reset_pool_v)

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
                        if step_int > wandb_start_step:
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
            reset_pool,
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
    warrior_inter_team_kill_reward = config.get("WARRIOR_INTER_TEAM_KILL_REWARD", 0.0)
    forager_inter_team_kill_reward = config.get("FORAGER_INTER_TEAM_KILL_REWARD", 0.0)
    warrior_passive_food_gain = int(config.get("WARRIOR_PASSIVE_FOOD_GAIN", 1))
    forager_passive_food_gain = int(config.get("FORAGER_PASSIVE_FOOD_GAIN", 3))
    forager_food_capacity = int(config.get("FORAGER_FOOD_CAPACITY", 27))
    warrior_food_capacity = int(config.get("WARRIOR_FOOD_CAPACITY", 0))  # 0 = default 7 + 2 * dexterity (= 9)
    forager_predator_damage_multiplier = float(config.get("FORAGER_PREDATOR_DAMAGE_MULTIPLIER", 1.0))
    melee_mob_health_override = float(config.get("MELEE_MOB_HEALTH", 0.0))  # 0 = per-mob-type default
    melee_mob_damage_override = float(config.get("MELEE_MOB_DAMAGE", 0.0))  # 0 = per-mob-type default
    forager_to_warrior_food_trade_reward = config.get("FORAGER_TO_WARRIOR_FOOD_TRADE_REWARD", 0.0)
    forager_to_warrior_drink_trade_reward = config.get("FORAGER_TO_WARRIOR_DRINK_TRADE_REWARD", 0.0)
    warrior_to_warrior_food_trade_reward = config.get("WARRIOR_TO_WARRIOR_FOOD_TRADE_REWARD", 0.0)
    warrior_to_warrior_drink_trade_reward = config.get("WARRIOR_TO_WARRIOR_DRINK_TRADE_REWARD", 0.0)
    trade_reward_requires_both_outside_starter_room = config.get("TRADE_REWARD_REQUIRES_BOTH_OUTSIDE_STARTER_ROOM", False)
    use_teammate_direction = config.get("USE_TEAMMATE_DIRECTION", True)
    teammate_in_fov_reward = config.get("TEAMMATE_IN_FOV_REWARD", 0.0)
    teammate_fov_reward_count = config.get("TEAMMATE_FOV_REWARD_COUNT", False)
    melee_mobs_despawn_when_far = config.get("MELEE_MOBS_DESPAWN_WHEN_FAR", False)
    melee_mob_despawn_distance = int(config.get("MELEE_MOB_DESPAWN_DISTANCE", 14))
    passive_mobs_static = config.get("PASSIVE_MOBS_STATIC", False)
    enable_auto_respawning = config.get("ENABLE_AUTO_RESPAWNING", False)
    enable_warrior_to_warrior_trading = config.get("ENABLE_WARRIOR_TO_WARRIOR_TRADING", False)
    auto_respawn_steps = int(config.get("AUTO_RESPAWN_STEPS", 50))
    restrict_auto_respawning_to_spawn_room = config.get("RESTRICT_AUTO_RESPAWNING_TO_SPAWN_ROOM", True)
    auto_respawn_team_penalty = config.get("AUTO_RESPAWN_TEAM_PENALTY", 0.0)
    num_rooms = config.get("NUM_ROOMS", 24)
    min_room_size = config.get("MIN_ROOM_SIZE", 5)
    max_room_size = config.get("MAX_ROOM_SIZE", 10)
    initial_predators_spawn_in_warrior_rooms_only = config.get("INITIAL_PREDATORS_SPAWN_IN_WARRIOR_ROOMS_ONLY", False)
    non_forager_always_in_lone_room = config.get("NON_FORAGER_ALWAYS_IN_LONE_ROOM", False)
    spread_non_foragers_across_rooms = config.get("SPREAD_NON_FORAGERS_ACROSS_ROOMS", False)
    trade_radius = int(config.get("TRADE_RADIUS", 18))
    trade_radius_shape = str(config.get("TRADE_RADIUS_SHAPE", "square")).lower()
    shared_reward = config.get("SHARED_REWARD", True)
    _max_rooms = (96 // 16) * (96 // 16)  # 36 chunks at 96x96 map
    assert num_rooms <= _max_rooms, f"NUM_ROOMS ({num_rooms}) exceeds available chunks ({_max_rooms})"
    assert min_room_size < max_room_size, f"MIN_ROOM_SIZE ({min_room_size}) must be < MAX_ROOM_SIZE ({max_room_size})"

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
    if warrior_food_capacity < 0:
        raise ValueError(
            f"WARRIOR_FOOD_CAPACITY must be >= 0 (0 = default), got {warrior_food_capacity}."
        )
    if forager_predator_damage_multiplier < 0:
        raise ValueError(
            "FORAGER_PREDATOR_DAMAGE_MULTIPLIER must be >= 0, "
            f"got {forager_predator_damage_multiplier}."
        )
    if melee_mob_health_override < 0:
        raise ValueError(f"MELEE_MOB_HEALTH must be >= 0 (0 = default), got {melee_mob_health_override}.")
    if melee_mob_damage_override < 0:
        raise ValueError(f"MELEE_MOB_DAMAGE must be >= 0 (0 = default), got {melee_mob_damage_override}.")
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
        "num_rooms": num_rooms,
        "min_room_size": min_room_size,
        "max_room_size": max_room_size,
        "use_teammate_direction": use_teammate_direction,
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
        "warrior_inter_team_kill_reward": warrior_inter_team_kill_reward,
        "forager_inter_team_kill_reward": forager_inter_team_kill_reward,
        "warrior_passive_food_gain": warrior_passive_food_gain,
        "forager_passive_food_gain": forager_passive_food_gain,
        "forager_food_capacity": forager_food_capacity,
        "warrior_food_capacity": warrior_food_capacity,
        "forager_predator_damage_multiplier": forager_predator_damage_multiplier,
        "melee_mob_health_override": melee_mob_health_override,
        "melee_mob_damage_override": melee_mob_damage_override,
        "forager_to_warrior_food_trade_reward": forager_to_warrior_food_trade_reward,
        "forager_to_warrior_drink_trade_reward": forager_to_warrior_drink_trade_reward,
        "warrior_to_warrior_food_trade_reward": warrior_to_warrior_food_trade_reward,
        "warrior_to_warrior_drink_trade_reward": warrior_to_warrior_drink_trade_reward,
        "enable_warrior_to_warrior_trading": enable_warrior_to_warrior_trading,
        "trade_reward_requires_both_outside_starter_room": trade_reward_requires_both_outside_starter_room,
        "teammate_in_fov_reward": teammate_in_fov_reward,
        "teammate_fov_reward_count": teammate_fov_reward_count,
        "enable_auto_respawning": enable_auto_respawning,
        "auto_respawn_steps": auto_respawn_steps,
        "auto_respawn_team_penalty": auto_respawn_team_penalty,
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
    wandb_start_step = wandb.run.step if (resuming and wandb.run is not None) else -1
    print(f"[wandb] resume floor step = {wandb_start_step}")

    rng = jax.random.PRNGKey(config["SEED"])
    train_fn = make_train(config, env, wandb_start_step=wandb_start_step)
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
                    "pruning": pruning.pruning_settings(config),
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
