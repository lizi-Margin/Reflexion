"""
Track 2: Generalization Track (Advanced Version)
This script connects your specialized agents to the online competition for the generalization track.
Environments: Codenames-v0, ColonelBlotto-v0, ThreePlayerIPD-v0

This advanced version uses specialized agents for each game type:
- CodenamesAgent: Sophisticated 4-phase reasoning for Codenames
- BlottoAgent: Adaptive resource allocation for Colonel Blotto
- IPDAgent: Strategic cooperation/defection for Three Player IPD

These specialized agents analyze game state more deeply and use multi-phase reasoning
for better performance compared to the generic ApiAgent.
"""

import textarena as ta
from xushuhang.agents.track2_router import create_track2_agent

try: 
    from uhtk.print_pack import print_bold_green as print
except: pass

# Model configuration
MODEL_NAME = "VitoTrack2-Adv-ds_v31"  # Replace with your model name
API_MODEL_SPEC = 'deepseek-v3-1-250821'

# Description for the competition
MODEL_DESCRIPTION = "Advanced multi-phase specialized agents for Track 2 - Generalization"
TEAM_HASH = "MG25-F5C82328D3"  # Replace with your team hash

# Initialize specialized agent router (auto-detects game type)
agent = create_track2_agent(
    model_name=MODEL_NAME,
    api_model_spec=API_MODEL_SPEC,
    env_name="auto-detect",  # Will determine game type from first observation
    enable_logging=True
)

print(f"Initialized specialized Track 2 agent: {MODEL_NAME}")
print(f"Using API model: {API_MODEL_SPEC}")
print("Game type will be auto-detected from the first observation")

# Connect to online competition
env = ta.make_mgc_online(
    track="Generalization",
    model_name=MODEL_NAME,
    model_description=MODEL_DESCRIPTION,
    team_hash=TEAM_HASH,
    agent=agent,
    small_category=False  # Set to True to participate in the efficient division
)

# Reset environment (always use num_players=1 for online competition)
env.reset(num_players=1)

# Main game loop
done = False
print("Waiting for game to start...")
while not done:
    player_id, observation = env.get_observation()
    print(f"Received observation for Player {player_id}, length: {len(observation)} chars")

    action = agent(observation)
    print(f"Generated action: {action}")

    done, step_info = env.step(action=action)
    if step_info:
        print(f"Step info: {step_info}")

# Game complete
rewards, game_info = env.close()
print("Game complete!")
print(f"Rewards: {rewards}")
print(f"Game Info: {game_info}")