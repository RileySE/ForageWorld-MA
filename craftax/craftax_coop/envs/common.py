from typing import List

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
    Log all interactions between agents including actor and receiver.
    Interactions are 3D arrays with shape (actor, receiver, interaction_type).
    Multiplies by done flag but uses multiplier of 1.0.
    """
    interactions = state.interactions * done * 1.0
    info = {}
    
    for interaction in Interaction:
        interaction_name = interaction.name.lower()
        for actor in range(static_params.player_count):
            for receiver in range(static_params.player_count):
                key = f"Interactions/{interaction_name}/actor_{actor}/receiver_{receiver}"
                info[key] = interactions[actor, receiver, interaction.value]
    
    return info
