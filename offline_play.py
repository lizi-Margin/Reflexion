"""
This script allows you to play against a fixed model.
You (human player) will be player 0, and the AI model will be player 1.
"""

import textarena as ta
from src.agent import LLMAgent
from corleone.family import Vito, Michael
from corleone.agents.track2.track2_router import create_track2_agent
from corleone.agents.track2.codenames_agent import CodenamesAgent
import sys
import io
import os

try:
  from uhtk.print_pack import print_bold_green as print
except: pass


def create_agents(env_id, npc_num, model_name='test', api_model_spec='qwen3-8b', enable_logging=False):
    """
    Create a dictionary of agents for the specified environment.

    Args:
        env_id: Environment ID (e.g., "Codenames-v0-train", "ThreePlayerIPD-v0-train")
        npc_num: Number of NPC agents to create
        model_name: Name for the main agent (player 0)
        api_model_spec: API model specification (e.g., 'qwen3-8b')
        enable_logging: Whether to enable game logging

    Returns:
        Dictionary mapping player_id -> agent
    """
    agents = {
        0: create_track2_agent(
            model_name=model_name,
            api_model_spec=api_model_spec,
            env_name=env_id,
            enable_logging=enable_logging
        ),
    }
    for i in range(1, npc_num + 1):
        agents[i] = create_track2_agent(
            model_name=f'bsl{i}',
            api_model_spec=api_model_spec,
            env_name=env_id,
            enable_logging=enable_logging
        )
    return agents


def play_game(env_id, agents, verbose=True):
    """
    Play a single game with the provided agents.

    Args:
        env_id: Environment ID
        agents: Dictionary mapping player_id -> agent
        verbose: Whether to print step-by-step information

    Returns:
        Tuple of (rewards, game_info)
    """
    if verbose:
        print(f"Playing {env_id} with {len(agents)} agents")
        print("start make env, env_id:", env_id)

    env = ta.make(env_id=env_id)
    env.reset(num_players=len(agents))

    # main game loop
    done = False
    step = 0  # it is not real step, just one round chat for one player
    while not done:
        step += 1
        if verbose:
            print(f"Step {step} start! ------------------------------------------------>")
        player_id, observation = env.get_observation()
        if verbose:
            print(f"Player {player_id} observation: {observation}")
        action = agents[player_id](observation)
        if verbose:
            print(f"Player {player_id} action: {action}")
        done, step_info = env.step(action=action)
        if verbose:
            print(f"Step {step} end! ------------------------------------------------>")

    rewards, game_info = env.close()

    if verbose:
        print(f"Rewards: {rewards}")
        print(f"Game Info: {game_info}")

    return rewards, game_info


def main():
    """Main function for single game play"""
    # Configure environment
    # env_id = "Codenames-v0-train"; npc_num = 3  # 2v2
    env_id = "ColonelBlotto-v0-train"; npc_num = 1  # 1v1
    # env_id = "ThreePlayerIPD-v0-train"; npc_num = 2  # 3 players

    # Create agents
    agents = create_agents(
        env_id=env_id,
        npc_num=npc_num,
        model_name='test',
        api_model_spec='qwen3-8b',
        enable_logging=False
    )

    # Play game
    rewards, game_info = play_game(env_id=env_id, agents=agents, verbose=True)

    return rewards, game_info


if __name__ == "__main__":
    main()