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

SMALL_CATEGORY = True

# # Game environments for Track 2
# TRACK2_GAMES = [
#     "Codenames-v0",
#     "ColonelBlotto-v0",
#     "ThreePlayerIPD-v0"
# ]


def run_single_game(
    model_name: str,
    model_description: str,
    team_hash: str,
    api_model_spec: str,
    small_category: bool = False
):
    print(f"\n{'='*60}")
    print(f"Model Name: {model_name}")
    print(f"Model: {api_model_spec}")
    print(f"Division: {'Efficient' if small_category else 'Open'}")
    print(f"{'='*60}\n")

    # Create specialized agent for this game
    agent = create_track2_agent(
        model_name=model_name,
        api_model_spec=api_model_spec,
        env_name="auto-detect",
        enable_logging=True,
        memory=True,
    )

    # Create online environment
    env = ta.make_mgc_online(
        track="Generalization",
        model_name=model_name,
        model_description=model_description,
        team_hash=team_hash,
        agent=agent,
        small_category=small_category
    )

    # Reset environment
    env.reset(num_players=1)

    # Game loop
    done = False
    turn_count = 0
    while not done:
        player_id, observation = env.get_observation()
        if hasattr(agent, '_IS_ROUTER'):
            REAL = agent.get(observation)
            if REAL is not None:
                agent = REAL
        action = agent(observation)
        done, step_info = env.step(action=action)
        turn_count += 1

    # Close environment and get results
    rewards, game_info = env.close()

    # Update agent memory with rewards and game_info
    if hasattr(agent, 'finalize_game'):
        agent.finalize_game(rewards=rewards, game_info=game_info)

    print(f"\n{'='*60}")
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

    # Run games
    game_results = []

    for game_num in range(args.games):
        print(f"\nPlaying  Game {game_num + 1}/{args.games}")

        try:
            rewards, game_info = run_single_game(
                model_name=DEFAULT_MODEL_NAME,
                model_description=DEFAULT_MODEL_DESCRIPTION,
                team_hash=DEFAULT_TEAM_HASH,
                api_model_spec=DEFAULT_MODEL,
                small_category=SMALL_CATEGORY
            )

            game_results.append({
                "game_num": game_num + 1,
                "rewards": rewards,
                "game_info": game_info
            })

        except Exception as e:
            print(f"\nError in game {game_num + 1}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "="*60)
    print("Competition Summary")
    print("="*60)

    total_games = len(game_results)
    successful_games = sum(1 for result in game_results if result.get("rewards") is not None)

    print(f"Total games played: {total_games}")
    print(f"Successful games: {successful_games}")
    print("="*60)

if __name__ == "__main__":
    main()
