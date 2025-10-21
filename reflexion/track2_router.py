"""
Track 2 Agent Router - Mind Games Challenge

This module provides a factory function for creating appropriate agents for Track 2 games:
- Codenames
- Colonel Blotto
- Three Player IPD
"""

from typing import Optional
from envs.agent import Agent

# Import specialized agents (will be implemented next)
# Using forward references to avoid circular imports
CodenamesAgent = None
BlottoAgent = None
IPDAgent = None


def detect_game_from_observation(observation: str) -> str:
    """
    Detect which game is being played based on the initial observation

    Args:
        observation: Initial observation string from the environment

    Returns:
        Game name: "Codenames", "ColonelBlotto", or "ThreePlayerIPD"
    """
    observation_lower = observation.lower()

    # Detection patterns
    if "codenames" in observation_lower or "spymaster" in observation_lower:
        return "Codenames"
    elif "colonel blotto" in observation_lower or "commander" in observation_lower:
        return "ColonelBlotto"
    elif "prisoner's dilemma" in observation_lower or "cooperate" in observation_lower:
        return "ThreePlayerIPD"

    # Default fallback
    return "Unknown"


def create_track2_agent(
    model_name: str,
    api_model_spec: str,
    env_name: Optional[str] = None,
    enable_logging: bool = True
) -> Agent:
    """
    Factory function to create appropriate agent based on environment

    Args:
        model_name: Model name to use for agent identification
        api_model_spec: API model specification for the LLM backend
        env_name: Environment name (if known)
        enable_logging: Whether to enable logging

    Returns:
        Appropriate specialized agent for the game

    Raises:
        ImportError: If required agent module isn't available
        ValueError: If environment can't be determined
    """
    # Import specialized agents (lazy import)
    # This prevents circular imports and allows importing only what's needed
    # Known environment, create specific agent
    if "Codenames" in env_name:
        from reflexion.codename_runs.codenames_agent import CodenamesAgent
        return CodenamesAgent(model_name, api_model_spec, enable_logging=enable_logging)
    elif "ColonelBlotto" in env_name:
        from reflexion.blotto_runs.blotto_agent import BlottoAgent
        return BlottoAgent(model_name, api_model_spec, enable_logging=enable_logging)
    elif "ThreePlayerIPD" in env_name:
        if model_name.startswith('bsl'):
            from reflexion.ipd_runs.ipd_agent_baseline import IPDAgent
            return IPDAgent(model_name, api_model_spec, enable_logging=enable_logging)
        else:
            from reflexion.ipd_runs.ipd_agent import IPDAgent
            return IPDAgent(model_name, api_model_spec, enable_logging=enable_logging)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


class TrackTwoAutoAgent(Agent):
    """
    Auto-detecting agent for Track 2 games.
    This is a wrapper that creates the appropriate specialized agent on first observation.
    """

    def __init__(self, model_name: str, api_model_spec: str, enable_logging: bool = True):
        self.model_name = model_name
        self.api_model_spec = api_model_spec
        self.enable_logging = enable_logging
        self.specialized_agent = None

    def __call__(self, observation: str) -> str:
        # If first observation, detect game type and create specialized agent
        if self.specialized_agent is None:
            game_type = detect_game_from_observation(observation)
            print(f"Detected game type: {game_type}")

            self.specialized_agent = create_track2_agent(
                model_name=self.model_name,
                api_model_spec=self.api_model_spec,
                env_name=game_type,
                enable_logging=self.enable_logging
            )

        # Delegate to specialized agent
        return self.specialized_agent(observation)