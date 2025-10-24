"""
Competition Entry Point for Mind Games Challenge - Track 2
This script provides a single entry point for running competition matches across all three games.
"""

import argparse
import textarena as ta
from reflexion.track2_router import create_track2_agent

# Default configuration
DEFAULT_MODEL = "qwen3-8b"
DEFAULT_TEAM_HASH = "MG25-F5C82328D3"

# DEFAULT_MODEL_NAME = "VitoTrack2-Adv-ds_v31"
# DEFAULT_MODEL_DESCRIPTION = "Advanced multi-phase specialized agents for Track 2 - Generalization"

DEFAULT_MODEL_NAME = "Default_k2"
DEFAULT_MODEL_DESCRIPTION = "This agent is for Track 2 - Generalization (Multiple environments)."

small_category = True

# Game environments for Track 2
TRACK2_GAMES = [
    "Codenames-v0-train",
    "ColonelBlotto-v0-train",
    "ThreePlayerIPD-v0-train"
]

# Number of NPCs for each game
GAME_NPC_CONFIG = {
    "Codenames-v0-train": 3,      # 2v2
    "ColonelBlotto-v0-train": 1,  # 1v1
    "ThreePlayerIPD-v0-train": 2  # 3 players
}

def run_single_game_offline(
    model_name: str,
    model_description: str,
    team_hash: str,
    api_model_spec: str,
    small_category: bool = False
):
    """
    Run a single offline self-play game with random environment selection

    Args:
        model_name: Unique model identifier
        api_model_spec: API model specification
        enable_logging: Whether to enable logging
    """
    import random

    # Randomly select a game environment
    env_id = random.choice(TRACK2_GAMES)
    npc_num = GAME_NPC_CONFIG[env_id]

    print(f"\n{'='*60}")
    print(f"Offline Self-Play Test")
    print(f"Environment: {env_id}")
    print(f"Model: {api_model_spec}")
    print(f"Total Players: {npc_num + 1}")
    print(f"{'='*60}\n")

    # Create agents in the same way as online
    agents = {
        0: create_track2_agent(
            model_name=model_name,
            api_model_spec=api_model_spec,
            env_name='auto-detect',
            enable_logging=True,
            memory=True,
        ),
    }

    # Create NPC agents
    for i in range(1, npc_num + 1):
        agents[i] = create_track2_agent(
            model_name=f'bsl{i}',
            api_model_spec=api_model_spec,
            env_name='auto-detect',
            enable_logging=False,
            memory=False,
        )

    # Create offline environment
    env = ta.make(env_id=env_id)
    env.reset(num_players=len(agents))

    # Game loop
    done = False
    turn_count = 0
    while not done:
        player_id, observation = env.get_observation()
        if hasattr(agents[player_id], '__IS_ROUTER'):
            REAL = agents[player_id].get(observation)
            if REAL is not None:
                agents[player_id] = REAL
            
        action = agents[player_id](observation)
        done, step_info = env.step(action=action)
        turn_count += 1

    # Close environment and get results
    rewards, game_info = env.close()
    # Update agent memory with rewards and game_info
    for agent_id, agent_instance in agents.items():
        if hasattr(agent_instance, 'finalize_game'):
            agent_instance.finalize_game(rewards=rewards, game_info=game_info)
        else:
            assert False, f"{agent_instance.__class__.__name__} does not have finalize_game method."

    stats = agents[0].memory.get_statistics()
    game_info['stats'] = stats

    print(f"\n{'='*60}")
    print(f"Game completed: {env_id}")
    print(f"Total turns: {turn_count}")
    print(f"Rewards: {rewards}")
    print(f"{'='*60}\n")

    return rewards, game_info


def main():
    """Main entry point for competition"""
    parser = argparse.ArgumentParser(
        description="Mind Games Challenge - Track 2 Competition Entry Point"
    )

    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Number of games to play per environment (default: 1)"
    )


    args = parser.parse_args()

    print("\n" + "="*60)
    print("Mind Games Challenge - Track 2 Competition")
    print("="*60)
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Model Name: {DEFAULT_MODEL_NAME}")
    print(f"Team Hash: {DEFAULT_TEAM_HASH}")
    print(f"Games per environment: {args.games}")
    print("="*60 + "\n")

    from uhtk.mcv_log_manager import LogManager
    from uhtk.VISUALIZE.mcom import mcom
    lm = LogManager(mcv=mcom(path='./VISUALIZE_logdir/', logdir='./VISUALIZE_logdir/'), who='blotto_reflexion')

    # Run games
    all_results = {}

    game_results = []

    for game_num in range(args.games):
        print(f"\nPlaying  Game {game_num + 1}/{args.games}")

        try:
            rewards, game_info = run_single_game_offline(
                model_name=DEFAULT_MODEL_NAME,
                model_description=DEFAULT_MODEL_DESCRIPTION,
                team_hash=DEFAULT_TEAM_HASH,
                api_model_spec=DEFAULT_MODEL,
                small_category=small_category
            )

            game_results.append({
                "game_num": game_num + 1,
                "rewards": rewards,
                "game_info": game_info
            })
            reward_0 = rewards[0]
            lm.log_trivial({
                'game_num': game_num + 1,
                'reward_0': reward_0,
                'win_rate': game_info['stats']['win_rate'],
            })

        except Exception as e:
            print(f"\nError in game {game_num + 1}: {e}")
            import traceback
            traceback.print_exc()
            raise e
        if game_num % 10 == 0:
            lm.log_trivial_finalize()

    all_results[str(game_num)] = game_results
    print(all_results)

    # Print summary
    print("\n" + "="*60)
    print("Competition Summary")
    print("="*60)

if __name__ == "__main__":
    main()
