from typing import List

import jax.numpy as jnp
from craftax_coop.craftax_state import EnvState, StaticEnvParams
from craftax_coop.constants import *

def compute_score(state: EnvState, done: bool, static_params: StaticEnvParams):
    achievements = state.achievements * done * 100.0
    info = {}
    for achievement in Achievement:
        achievement_name = f"Achievements/{achievement.name.lower()}"
        info[achievement_name] = achievements[:, achievement.value]

    # Log interactions between agents
    interaction_info = log_interactions(state, done, static_params)
    info.update(interaction_info)

    return info


def log_interactions(state: EnvState, done: bool, static_params: StaticEnvParams):
    """
    Log aggregated interactions between agents.
    Sums interactions received by each agent across all interaction types.
    Returns per-agent interaction counts to match the shape of other metrics.
    """
    interactions = state.interactions * done * 1.0
    info = {}
    
    # Aggregate interactions per agent: sum over actor and interaction type
    # Result shape: (player_count,) - total interactions received by each agent
    for interaction in Interaction:
        interaction_name = interaction.name.lower()
        # Sum over all actors for each receiver
        per_agent = jnp.sum(interactions[:, :, interaction.value], axis=0)
        key = f"Interactions/{interaction_name}"
        info[key] = per_agent
    
    return info
