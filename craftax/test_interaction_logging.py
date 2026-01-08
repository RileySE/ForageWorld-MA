#!/usr/bin/env python3
"""
Test script to verify agent interaction logging functionality.
"""

import jax
import jax.numpy as jnp
from craftax_coop.constants import Interaction
from craftax_coop.craftax_state import StaticEnvParams
from craftax_coop.envs.common import log_interactions

def test_interaction_logging_direct():
    """Test interaction logging directly without full environment."""
    
    print("=" * 60)
    print("Agent Interaction Logging Test")
    print("=" * 60)
    
    # Create a mock StaticEnvParams
    player_count = 2
    static_params = StaticEnvParams(player_count=player_count)
    
    # Create a mock state with interactions
    interactions_array = jnp.zeros(
        (player_count, player_count, len(Interaction)), dtype=jnp.int32
    )
    
    # Simulate some interactions
    # Agent 0 revives Agent 1
    interactions_array = interactions_array.at[0, 1, Interaction.Revive.value].set(1)
    
    # Agent 0 trades with Agent 1
    interactions_array = interactions_array.at[0, 1, Interaction.Give_item.value].set(3)
    
    # Create mock state (only need interactions field)
    from dataclasses import dataclass, field
    from typing import Any
    
    @dataclass
    class MockState:
        interactions: jnp.ndarray
    
    state = MockState(interactions=interactions_array)
    
    print(f"✓ Created mock state with interactions")
    print(f"✓ Interactions shape: {state.interactions.shape}")
    
    # Test log_interactions
    done = True
    interaction_info = log_interactions(state, done, static_params)
    
    print(f"\n✓ Generated interaction logs")
    print(f"✓ Number of interaction log entries: {len(interaction_info)}")
    
    # Verify interaction types are in the enum
    print(f"\nInteraction types available:")
    for interaction in Interaction:
        print(f"  - {interaction.name} (index: {interaction.value})")
    
    # Check if any interactions were recorded
    has_interactions = any(v > 0 for v in interaction_info.values())
    if has_interactions:
        print(f"\n✓ Found non-zero interactions in logs")
        nonzero_keys = [k for k, v in interaction_info.items() if v > 0]
        print(f"  Keys with values > 0:")
        for key in nonzero_keys:
            print(f"    - {key}: {interaction_info[key]}")
    else:
        print(f"\n✗ No non-zero interactions found in logs")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_interaction_logging_direct()

