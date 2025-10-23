"""
Multi-Trial Training Script for Blotto Agent with Reflexion Memory

This script runs multiple Colonel Blotto game trials and uses reflexion to improve
performance across trials through inter-trial memory.

Multi-threading support: Runs multiple trials concurrently while safely
sharing memory across all threads.

Usage:
    python reflexion/blotto_runs/train_with_memory.py
"""

import textarena as ta
from reflexion.blotto_runs.blotto_agent import BlottoAgent
from reflexion.blotto_runs.blotto_memory import BlottoMemory
import sys, random
from uhtk.print_pack import *
from uhtk.mcv_log_manager import LogManager
from uhtk.VISUALIZE.mcom import mcom
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, traceback


def create_baseline_agent(agent_id: int, api_model_spec: str = 'qwen3-8b'):
    """Create a baseline opponent agent without memory"""
    # from reflexion.api_agent import ApiAgent as BaselineAgent
    # return BaselineAgent(
    #     model_name=f'bsl_{agent_id}',
    #     api_model_spec=api_model_spec,
    #     enable_logging=False
    from reflexion.blotto_runs.blotto_agent import BlottoAgent as BaselineAgent
    return BaselineAgent(
        model_name=f'bsl_{agent_id}',
        api_model_spec=api_model_spec,
        enable_logging=False,
        memory=None
    )


def create_selfplay_agent(agent_id: int, memory: BlottoMemory, api_model_spec: str = 'qwen3-8b'):
    """Create a self-playing agent with shared memory"""
    return BlottoAgent(
        model_name=f'selfplay_{agent_id}',
        api_model_spec=api_model_spec,
        memory=memory,
        enable_logging=False
    )


def run_single_trial(trial_num: int, agent: BlottoAgent, memory: BlottoMemory, env_id: str = "ColonelBlotto-v0-train",
                     api_model_spec: str = None, verbose: bool = True, selfplay: bool = False, print_lock: threading.Lock = None):
    """
    Run a single trial of Colonel Blotto (thread-safe)

    Args:
        trial_num: Trial number
        agent: The main Blotto agent with memory
        memory: Shared memory object (thread-safe)
        env_id: Environment ID
        api_model_spec: API model specification for opponents
        verbose: Whether to print verbose output
        selfplay: Whether to use self-play mode (both agents learn)
        print_lock: Lock for synchronized printing

    Returns:
        Dictionary with trial results
    """
    def thread_print(*args, **kwargs):
        """Thread-safe print function"""
        if print_lock:
            with print_lock:
                print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    thread_print(f"\n{'='*80}")
    thread_print(f"TRIAL {trial_num} [Thread: {threading.current_thread().name}]")
    thread_print(f"{'='*80}")

    if api_model_spec is None:
        all_available_models = [
            'qwen3-8b',
            'gemini-2.5-flash-lite-nothinking',
            'gemini-1.5-flash-8b',
        ]
        api_model_spec = random.choice(all_available_models)

    # Create agents (Colonel Blotto is 2-player)
    if selfplay:
        thread_print("Self-play mode enabled")
        agents = {
            0: agent,  # Our learning agent
            1: create_selfplay_agent(1, memory, api_model_spec),  # Another learning agent
        }
    else:
        agents = {
            0: agent,  # Our learning agent
            1: create_baseline_agent(1, api_model_spec),  # Baseline opponent
        }

    for agent_id, a in agents.items():
        thread_print(f"Agent {agent_id} ({a.model_name}) API Model Spec: {a.api_model_spec}")

    # Create environment
    env = ta.make(env_id=env_id)
    env.reset(num_players=len(agents))

    # Game loop
    done = False
    step = 0
    while not done:
        step += 1
        if verbose:
            thread_print(f"\nStep {step} {'>'*60}")

        player_id, observation = env.get_observation()

        if verbose and player_id == 0:  # Only print for our agent
            thread_print(f"Player {player_id} observation:\n{observation[-300:]}...")

        action = agents[player_id](observation)

        if verbose and player_id == 0:
            thread_print(f"Player {player_id} action: {action}")

        done, step_info = env.step(action=action)

    # Get final results
    rewards, game_info = env.close()
    # Rewards: {0: 0, 1: 0}

    # Update agent memory with rewards and game_info
    for agent_id, agent_instance in agents.items():
        if hasattr(agent_instance, 'finalize_game'):
            agent_instance.finalize_game(rewards=rewards, game_info=game_info)

    if verbose:
        thread_print(f"\n{'='*80}")
        thread_print(f"TRIAL {trial_num} RESULTS")
        thread_print(f"Rewards: {rewards}")
        thread_print(f"Game Info: {game_info}")
        thread_print(f"{'='*80}\n")

    return {
        "trial_num": trial_num,
        "rewards": rewards,
        "game_info": game_info
    }


def main():
    """Main training loop with multi-threading support"""
    # Configuration
    NUM_TRIALS = 10  # Number of trials to run
    MAX_WORKERS = 2  # Number of concurrent threads (adjust based on your needs)
    ENV_ID = "ColonelBlotto-v0-train"
    API_MODEL_SPEC = "qwen3-8b"  # Change to your preferred model
    VERBOSE = True
    OPP_API_MODEL_SPEC = "qwen3-8b"
    SELFPLAY = False  # Set True for self-play training

    # Create shared memory for the agent
    memory = BlottoMemory(api_model_spec=API_MODEL_SPEC)

    # Print initial memory statistics
    stats = memory.get_statistics()
    print(f"Initial Memory Statistics:")
    print(f"  Total Trials: {stats['total_trials']}")
    print(f"  Win Rate: {stats['win_rate']:.2%}")
    print(f"  Avg Rounds Won: {stats['avg_rounds_won']:.2f}")

    # Thread-safe print lock
    print_lock = threading.Lock()

    # Run multiple trials concurrently using ThreadPoolExecutor
    trial_results = []

    def run_trial_wrapper(trial_num):
        """Wrapper function to create agent and run trial"""
        # Create agent with shared memory for this trial
        agent = BlottoAgent(
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
            api_model_spec=OPP_API_MODEL_SPEC,
            verbose=VERBOSE,
            selfplay=SELFPLAY,
            print_lock=print_lock
        )

        return result

    # Initialize log manager
    lm = LogManager(mcv=mcom(path='./VISUALIZE_logdir/', logdir='./VISUALIZE_logdir/'), who='blotto_reflexion')

    # Submit trials to thread pool
    print(f"\nStarting {NUM_TRIALS} trials with {MAX_WORKERS} concurrent workers...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all trials
        future_to_trial = {
            executor.submit(run_trial_wrapper, trial_num): trial_num
            for trial_num in range(1, NUM_TRIALS + 1)
        }

        # Collect results as they complete
        for i, future in enumerate(as_completed(future_to_trial)):
            trial_num = future_to_trial[future]
            try:
                result = future.result()
                trial_results.append(result)

                # Print cumulative statistics after each trial completes
                stats = memory.get_statistics()
                with print_lock:
                    print(f"\n{'='*80}")
                    print(f"Cumulative Statistics (after {len(trial_results)} completed trials):")
                    print(f"  Total Trials: {stats['total_trials']}")
                    print(f"  Win Rate: {stats['win_rate']:.2%}")
                    print(f"  Avg Rounds Won: {stats['avg_rounds_won']:.2f}")
                    # Show recent reflections
                    print(f"\nRecent Lessons:")
                    print(memory.get_guidance(max_reflections=3))
                    print(f"{'='*80}\n")
                    # Rewards: {0: 0, 1: 0}
                    reward_0 = result['rewards'][0]
                    reward_1 = result['rewards'][1]
                    lm.log_trivial({
                        'trial_num': trial_num,
                        'iter': i,
                        'reward_0': reward_0,
                        'reward_1': reward_1,
                        'win_rate': stats['win_rate'],
                        'avg_rounds_won': stats['avg_rounds_won'],
                    })

            except Exception as exc:
                with print_lock:
                    print(f"Trial {trial_num} generated an exception: {exc}")
                    print(traceback.format_exc())

            # if (i+1) % MAX_WORKERS == 0:
            lm.log_trivial_finalize()

    # Sort results by trial number for final display
    trial_results.sort(key=lambda x: x['trial_num'])

    # Final summary
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE - FINAL SUMMARY")
    print(f"{'='*80}")

    final_stats = memory.get_statistics()
    print(f"\nFinal Memory Statistics:")
    print(f"  Total Trials: {final_stats['total_trials']}")
    print(f"  Win Rate: {final_stats['win_rate']:.2%}")
    print(f"  Avg Rounds Won: {final_stats['avg_rounds_won']:.2f}")

    # Show recent reflections
    print(f"\nRecent Lessons:")
    print(memory.get_guidance(max_reflections=5))

    print(f"\n{'='*80}")
    print(f"Memory saved to: {memory.memory_file}")
    print(f"Game logs saved to: runs/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
