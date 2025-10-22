"""
Multi-Trial Training Script for IPD Agent with Reflexion Memory

This script runs multiple IPD game trials and uses reflexion to improve
performance across trials through inter-trial memory.

Usage:
    python reflexion/ipd_runs/train_with_memory.py
"""

import textarena as ta
from reflexion.ipd_runs.ipd_agent import IPDAgent
from reflexion.ipd_runs.ipd_agent import IPDAgent as BaselineAgent
from reflexion.ipd_runs.ipd_memory import IPDMemory
import sys


def create_baseline_agent(agent_id: int, api_model_spec: str = 'qwen3-8b'):
    """Create a baseline opponent agent"""
    return BaselineAgent(
        model_name=f'bsl_{agent_id}',
        api_model_spec=api_model_spec,
        enable_logging=False
    )

def create_selfplay_agent(agent_id: int, memory: IPDMemory, api_model_spec: str = 'qwen3-8b'):
    """Create a self-playing agent with memory"""
    return IPDAgent(
        model_name=f'selfplay_{agent_id}',
        api_model_spec=api_model_spec,
        memory=memory,
        enable_logging=False
    )


def run_single_trial(trial_num: int, agent: IPDAgent, memory: IPDMemory, env_id: str = "ThreePlayerIPD-v0-train",
                     api_model_spec: str = 'qwen3-8b', verbose: bool = True, selfplay: bool = True):
    """
    Run a single trial of IPD

    Args:
        trial_num: Trial number
        agent: The main IPD agent with memory
        env_id: Environment ID
        api_model_spec: API model specification for opponents
        verbose: Whether to print verbose output

    Returns:
        Dictionary with trial results
    """
    print(f"\n{'='*80}")
    print(f"TRIAL {trial_num}")
    print(f"{'='*80}")

    # Create baseline opponents
    if selfplay:
        print("Self-play mode enabled")
        agents = {
            0: agent,
            1: create_selfplay_agent(0, memory, api_model_spec),  # Our learning agent
            2: create_selfplay_agent(0, memory, api_model_spec),  # Our learning agent
        }
    else:
        agents = {
            0: agent,  # Our learning agent
            1: create_baseline_agent(1, api_model_spec),
            2: create_baseline_agent(2, api_model_spec)
        }

    # Create environment
    env = ta.make(env_id=env_id)
    env.reset(num_players=len(agents))

    # Game loop
    done = False
    step = 0
    while not done:
        step += 1
        if verbose:
            print(f"\nStep {step} {'>'*60}")

        player_id, observation = env.get_observation()

        if verbose and player_id == 0:  # Only print for our agent
            # print(f"Player {player_id} observation:\n{observation[:300]}...")
            print(f"Player {player_id} observation:\n{observation}...")

        action = agents[player_id](observation)

        if verbose and player_id == 0:
            print(f"Player {player_id} action: {action}")

        done, step_info = env.step(action=action)

    # Get final results
    rewards, game_info = env.close()

    # Update agent memory
    final_obs = game_info.get('final_observation', '')
    # agent.finalize_game(final_observation=final_obs)
    for agent_id in agents:
        agents[agent_id].finalize_game(final_observation=final_obs)

    if verbose:
        print(f"\n{'='*80}")
        print(f"TRIAL {trial_num} RESULTS")
        print(f"Rewards: {rewards}")
        print(f"{'='*80}\n")

    return {
        "trial_num": trial_num,
        "rewards": rewards,
        "game_info": game_info
    }


def main():
    """Main training loop"""
    # Configuration
    NUM_TRIALS = 5  # Number of trials to run
    ENV_ID = "ThreePlayerIPD-v0-train"
    # API_MODEL_SPEC = "qwen3-8b"  # Change to your preferred model
    API_MODEL_SPEC = "gpt-5-chat-latest"  # Change to your preferred model
    # API_MODEL_SPEC = "kimi-k2-250905"  # Change to your preferred model
    VERBOSE = True

    print(f"""
{'='*80}
IPD AGENT TRAINING WITH REFLEXION MEMORY
{'='*80}

Configuration:
- Number of trials: {NUM_TRIALS}
- Environment: {ENV_ID}
- API Model: {API_MODEL_SPEC}
- Memory will be saved to: reflexion/ipd_runs/memory/ipd_memory.json

The agent will learn from each trial through reflexion-based memory.
Watch how performance improves over trials!

{'='*80}
""")

    # Create shared memory for the agent
    memory = IPDMemory(api_model_spec=API_MODEL_SPEC)

    # Print initial memory statistics
    stats = memory.get_statistics()
    print(f"Initial Memory Statistics:")
    print(f"  Total Trials: {stats['total_trials']}")
    print(f"  Win Rate: {stats['win_rate']:.2%}")
    print(f"  Avg Rank: {stats['avg_rank']:.2f}")

    # Run multiple trials
    trial_results = []

    for trial_num in range(1, NUM_TRIALS + 1):
        # Create agent with shared memory
        # Note: We create a new agent instance each trial to simulate real competition
        # but share the memory across trials
        agent = IPDAgent(
            model_name=f'learning_agent_trial_{trial_num}',
            api_model_spec=API_MODEL_SPEC,
            enable_logging=True,
            memory=memory  # Shared memory!
        )

        # Run trial
        result = run_single_trial(
            trial_num=trial_num,
            agent=agent,
            memory=memory,  # Shared memory!
            env_id=ENV_ID,
            api_model_spec=API_MODEL_SPEC,
            verbose=VERBOSE
        )

        trial_results.append(result)

        # Print cumulative statistics after each trial
        stats = memory.get_statistics()
        print(f"\nCumulative Statistics (after {trial_num} trials):")
        print(f"  Total Trials: {stats['total_trials']}")
        print(f"  Win Rate: {stats['win_rate']:.2%}")
        print(f"  Avg Rank: {stats['avg_rank']:.2f}")
        # Show recent reflections
        print(f"\nRecent Lessons:")
        print(memory.get_guidance(max_reflections=3))

    # Final summary
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE - FINAL SUMMARY")
    print(f"{'='*80}")

    final_stats = memory.get_statistics()
    print(f"\nFinal Memory Statistics:")
    print(f"  Total Trials: {final_stats['total_trials']}")
    print(f"  Win Rate: {final_stats['win_rate']:.2%}")
    print(f"  Avg Rank: {final_stats['avg_rank']:.2f}")

    # Show recent reflections
    print(f"\nRecent Lessons:")
    print(memory.get_guidance(max_reflections=3))


    print(f"\n{'='*80}")
    print(f"Memory saved to: {memory.memory_file}")
    print(f"Game logs saved to: runs/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
