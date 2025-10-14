"""
Track 1: Social Detection Track
This script connects your agent to the online competition for the social detection track.
Environment: SecretMafia-v0
"""

import textarena as ta
from src.agent import LLMAgent
from xushuhang.agents.family import Vito, Michael
import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


MODEL_NAME = "Vito-1.0" # Replace with your model name
# The name is used to identify your agent in the online arena and leaderboard.
# It should be unique and descriptive.
# For different versions of your agent, you should use different names.
MODEL_DESCRIPTION = "This agent is for Track 1 - Social Detection (SecretMafia-v0)."
team_hash = "MG25-F5C82328D3" # Replace with your team hash 

# Initialize your agent
# agent = LLMAgent(model_name="Qwen/Qwen3-4B")
agent = Michael(model_name="qwen3-8b")

env = ta.make_mgc_online(
    track="Social Detection", 
    model_name=MODEL_NAME,
    model_description=MODEL_DESCRIPTION,
    team_hash=team_hash,
    agent=agent,
    small_category=True  # Set to True to participate in the efficient division
)
env.reset(num_players=1) # always set to 1 when playing online, even when playing multiplayer games.

done = False
while not done:
    player_id, observation = env.get_observation()
    action = agent(observation)
    done, step_info = env.step(action=action)

rewards, game_info = env.close()