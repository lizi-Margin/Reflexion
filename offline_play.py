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
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
  from uhtk.print_pack import print_bold_green as print
except: pass


# env_ids = ["SecretMafia-v0-train"]
# env_ids = ["Codenames-v0-train", "ColonelBlotto-v0-train", "ThreePlayerIPD-v0-train"]
# initialize the environment
# env_id = "Codenames-v0-train"; npc_num = 3  # 2v2 
env_id = "ColonelBlotto-v0-train"; npc_num = 1  # 1v1
# env_id = "ThreePlayerIPD-v0-train"; npc_num = 2  # 3 players
agents = {
    0: create_track2_agent(model_name='test', api_model_spec='qwen3-8b', env_name=env_id, enable_logging=False),
}
for i in range(1, npc_num+1):
    agents[i] = create_track2_agent(model_name=f'bsl{i}', api_model_spec='qwen3-8b', env_name=env_id, enable_logging=False)

print(f"Playing {env_id} with {npc_num} NPCs")
print("start make env, env_id:", env_id)
env = ta.make(env_id=env_id)
env.reset(num_players=len(agents))

# main game loop
done = False 
step = 0  # it is not real step, just one round chat for one player
while not done:
  step += 1
  print(f"Step {step} start! ------------------------------------------------>")
  player_id, observation = env.get_observation()
  print(f"Player {player_id} observation: {observation}")
  action = agents[player_id](observation)
  print(f"Player {player_id} action: {action}")
  done, step_info = env.step(action=action)
  print(f"Step {step} end! ------------------------------------------------>")
rewards, game_info = env.close()

print(f"Rewards: {rewards}")
print(f"Game Info: {game_info}")