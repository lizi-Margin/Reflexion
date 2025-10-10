"""
This script allows you to play against a fixed model.
You (human player) will be player 0, and the AI model will be player 1.
"""

import textarena as ta
from src.agent import LLMAgent
from corleone.family import Vito, Michael
from corleone.agents.track2.track2_router import create_track2_agent
import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')



# env_ids = ["SecretMafia-v0-train"]
# env_ids = ["Codenames-v0-train", "ColonelBlotto-v0-train", "ThreePlayerIPD-v0-train"]
# initialize the environment
env_id = "Codenames-v0-train"
agents = {
    0: create_track2_agent(model_name='test', api_model_spec='qwen3-8b', env_name=env_id, enable_logging=False),
    1: create_track2_agent(model_name='bsl1', api_model_spec='qwen3-8b', env_name=env_id, enable_logging=False),
    2: create_track2_agent(model_name='bsl2', api_model_spec='qwen3-8b', env_name=env_id, enable_logging=False),
    3: create_track2_agent(model_name='bsl3', api_model_spec='qwen3-8b', env_name=env_id, enable_logging=False),
    4: create_track2_agent(model_name='bsl4', api_model_spec='qwen3-8b', env_name=env_id, enable_logging=False),
    5: create_track2_agent(model_name='bsl5', api_model_spec='qwen3-8b', env_name=env_id, enable_logging=False),
}


env = ta.make(env_id=env_id)
env.reset(num_players=len(agents))

# main game loop
done = False 
while not done:
  player_id, observation = env.get_observation()
  action = agents[player_id](observation)
  done, step_info = env.step(action=action)
rewards, game_info = env.close()

print(f"Rewards: {rewards}")
print(f"Game Info: {game_info}")