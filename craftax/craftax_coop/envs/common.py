from typing import List

from craftax_coop.craftax_state import EnvState, StaticEnvParams
from craftax_coop.constants import *

def compute_score(state: EnvState, done: bool, static_params: StaticEnvParams):
    achievements = state.achievements * done * 100.0
    interactions = state.interactions * done * 100.0
    info = {}
    for achievement in Achievement:
        achievement_name = f"Achievements/{achievement.name.lower()}"
        info[achievement_name] = achievements[:, achievement.value]
    for i in Interaction:
        interaction_name = f"Interactions/{interaction.name.lower()}"
        info[interaction_name] = interactions[:, interaction.value]
    return info
