"""
This script allows you to play against a fixed model with game logging enabled.
All game sessions will be recorded for training and analysis purposes.
"""

import textarena as ta
from src.agent import LLMAgent, HumanAgent
from src.game_logger import GameLogger, LoggedAgent
from xushuhang.agents.family import Vito, Michael
import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Game configuration
MODEL_NAME = "Vito-1.3"  # Replace with your model name
ENV_ID = "SecretMafia-v0"
ENABLE_LOGGING = True  # Set to False to disable logging
LOG_DIR = "game_logs"

# Initialize game logger
logger = GameLogger(log_dir=LOG_DIR, enabled=ENABLE_LOGGING)

# Initialize the agents
raw_agents = {
    0: Michael(model_name="qwen3-8b"),
    1: Michael(model_name="qwen3-8b"),
    2: Michael(model_name="qwen3-8b"),
    3: Michael(model_name="deepseek-r1"),
    4: Michael(model_name="deepseek-r1"),
    5: Michael(model_name="deepseek-r1"),
}
# change model_name to change the model used; if you add new model, please change the function api() in family - Class Michael accordingly.

# Wrap agents with logging if enabled
if ENABLE_LOGGING:
    agents = {}
    for player_id, agent in raw_agents.items():
        agents[player_id] = LoggedAgent(agent, logger, player_id)
else:
    agents = raw_agents

# Initialize the environment
env = ta.make(env_id=ENV_ID)

# Start logging session
session_id = logger.start_session(
    env_id=ENV_ID,
    agents=raw_agents,
    num_players=len(agents),
    additional_info={
        "script": "offline_play.py",
        "model_name": MODEL_NAME,
        "description": "Offline testing with multiple agents"
    }
)

# Main game loop
done = False
turn_count = 0
while not done:
    turn_count += 1
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, step_info = env.step(action=action)

    if ENABLE_LOGGING:
        print(f"Turn {turn_count}: Player {player_id} acted")
    else:
        print(f"Turn {turn_count}: Player {player_id}: {action}")

# End game and log results
rewards, game_info = env.close()

if ENABLE_LOGGING:
    logger.log_results(rewards, game_info)
    logger.end_session()
    print(f"\nGame session logged: {session_id}")
    print(f"Log directory: {LOG_DIR}")

print(f"\nFinal Results:")
print(f"Rewards: {rewards}")
print(f"Game Info: {game_info}")

# Display logging statistics if enabled
if ENABLE_LOGGING:
    stats = logger.get_session_stats()
    print(f"\nLogging Statistics:")
    print(f"Total sessions: {stats['total_sessions']}")
    print(f"Sessions by environment: {stats['sessions_by_env']}")