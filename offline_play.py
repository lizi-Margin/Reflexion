"""
This script allows you to play against a fixed model.
You (human player) will be player 0, and the AI model will be player 1.
"""

import textarena as ta
from src.agent import LLMAgent
from corleone.family import Vito, Michael
import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
MODEL_NAME = "Vito-1.3" # Replace with your model name

# initialize the agents
agents = {
    0: Michael(model_name="qwen3-8b"),
    1: Michael(model_name="qwen3-8b"),
    2: Michael(model_name="qwen3-8b"),
    3: Michael(model_name="deepseek-r1"),
    4: Michael(model_name="deepseek-r1"),
    5: Michael(model_name="deepseek-r1"),
}
# change model_name to change the model used; if you add new model, please change the function api() in family - Class Michael accordingly.

# initialize the environment
env = ta.make(env_id="SecretMafia-v0")
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