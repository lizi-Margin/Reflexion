"""
Example: Single IPD Game with Memory-Enhanced Agent

This script shows how to use the IPD agent with inter-trial memory
for a single game.
"""

import textarena as ta
from reflexion.ipd_runs.ipd_agent import IPDAgent
from reflexion.ipd_runs.ipd_agent_baseline import IPDAgent as BaselineAgent
from reflexion.ipd_runs.ipd_memory import IPDMemory


def main():
    # Create shared memory (will load existing if available)
    memory = IPDMemory(api_model_spec='qwen3-8b')

    # Show memory statistics
    stats = memory.get_statistics()
    print(f"\n{'='*60}")
    print(f"Memory Statistics:")
    print(f"  Total Trials: {stats['total_trials']}")
    print(f"  Win Rate: {stats['win_rate']:.2%}")
    print(f"  Avg Rank: {stats['avg_rank']:.2f}")
    print(f"{'='*60}\n")

    # Create agents
    agents = {
        0: IPDAgent(
            model_name='memory_agent',
            api_model_spec='qwen3-8b',
            enable_logging=True,
            memory=memory  # Use shared memory
        ),
        1: BaselineAgent(
            model_name='baseline_1',
            api_model_spec='qwen3-8b',
            enable_logging=False
        ),
        2: BaselineAgent(
            model_name='baseline_2',
            api_model_spec='qwen3-8b',
            enable_logging=False
        )
    }

    # Create environment
    env_id = "ThreePlayerIPD-v0-train"
    env = ta.make(env_id=env_id)
    env.reset(num_players=len(agents))

    # Game loop
    done = False
    step = 0
    while not done:
        step += 1
        print(f"\nStep {step} {'>'*50}")

        player_id, observation = env.get_observation()

        if player_id == 0:  # Only show our agent's observations
            print(f"Player {player_id} observation:\n{observation[:200]}...")

        action = agents[player_id](observation)

        if player_id == 0:
            print(f"Player {player_id} action: {action}")

        done, step_info = env.step(action=action)

    # Get results
    rewards, game_info = env.close()

    # Finalize and update memory
    final_obs = game_info.get('final_observation', '')
    agents[0].finalize_game(final_observation=final_obs)

    print(f"\n{'='*60}")
    print(f"GAME RESULTS")
    print(f"Rewards: {rewards}")
    print(f"{'='*60}")

    # Show updated statistics
    stats = memory.get_statistics()
    print(f"\nUpdated Memory Statistics:")
    print(f"  Total Trials: {stats['total_trials']}")
    print(f"  Win Rate: {stats['win_rate']:.2%}")
    print(f"  Avg Rank: {stats['avg_rank']:.2f}")


if __name__ == "__main__":
    main()
