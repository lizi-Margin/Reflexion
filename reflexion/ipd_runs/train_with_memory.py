"""
Multi-Trial Training Script for IPD Agent with Reflexion Memory

This script runs multiple IPD game trials and uses reflexion to improve
performance across trials through inter-trial memory.

Multi-threading support: Runs multiple trials concurrently while safely
sharing memory across all threads.

Usage:
    python reflexion/ipd_runs/train_with_memory.py
"""

import textarena as ta
from reflexion.ipd_runs.ipd_agent import IPDAgent
from reflexion.ipd_runs.ipd_memory import IPDMemory
import sys, random
from uhtk.print_pack import *
from uhtk.mcv_log_manager import LogManager
from uhtk.VISUALIZE.mcom import mcom
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, traceback


def create_baseline_agent(agent_id: int, api_model_spec: str = 'qwen3-8b'):
    """Create a baseline opponent agent"""
    # from reflexion.ipd_runs.ipd_agent import IPDAgent as BaselineAgent
    from reflexion.api_agent import ApiAgent as BaselineAgent
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
                     api_model_spec: str = None, verbose: bool = True, selfplay: bool = True, print_lock: threading.Lock = None):
    """
    Run a single trial of IPD (thread-safe)

    Args:
        trial_num: Trial number
        agent: The main IPD agent with memory
        memory: Shared memory object (thread-safe)
        env_id: Environment ID
        api_model_spec: API model specification for opponents
        verbose: Whether to print verbose output
        selfplay: Whether to use self-play mode
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
            # 'qwen3-8b',
            # 'qwen3-8b',
            # 'qwen3-8b',
            # 'qwen3-4b',
            # 'doubao-seed-1-6-lite-251015',
            # 'gpt-5-chat-latest',
            # 'gemini-2.5-flash-lite',
            # 'deepseek-v3.1-250821'
            'gemini-2.5-flash-lite-nothinking',
            'gemini-1.5-flash-8b',
        ]
        api_model_spec = random.choice(all_available_models)

    # Create baseline opponents
    if selfplay:
        thread_print("Self-play mode enabled")
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

    for a in agents.values():
        thread_print(f"Agent {a.model_name} API Model Spec: {a.api_model_spec}")

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
            # thread_print(f"Player {player_id} observation:\n{observation[:300]}...")
            thread_print(f"Player {player_id} observation:\n{observation}...")

        action = agents[player_id](observation)

        if verbose and player_id == 0:
            thread_print(f"Player {player_id} action: {action}")

        done, step_info = env.step(action=action)

    # Get final results
    rewards, game_info = env.close()

    # Update agent memory
    final_obs = game_info.get('final_observation', '')
    # agent.finalize_game(final_observation=final_obs)
    for agent_id in agents:
        agents[agent_id].finalize_game(final_observation=final_obs)

    if verbose:
        thread_print(f"\n{'='*80}")
        thread_print(f"TRIAL {trial_num} RESULTS")
        thread_print(f"Rewards: {rewards}")
        thread_print(f"{'='*80}\n")

    return {
        "trial_num": trial_num,
        "rewards": rewards,
        "game_info": game_info
    }


def main():
    """Main training loop with multi-threading support"""
    # Configuration
    NUM_TRIALS = 5  # Number of trials to run
    MAX_WORKERS = 5  # Number of concurrent threads (adjust based on your needs)
    ENV_ID = "ThreePlayerIPD-v0-train"
    API_MODEL_SPEC = "qwen3-8b"  # Change to your preferred model
    # API_MODEL_SPEC = 'doubao-seed-1-6-lite-251015'
    # API_MODEL_SPEC = "gpt-5-chat-latest"  # Change to your preferred model
    # API_MODEL_SPEC = "kimi-k2-250905"  # Change to your preferred model
    VERBOSE = True
    # OPP_API_MODEL_SPEC = "qwen3-8b"  # Change to your preferred model
    OPP_API_MODEL_SPEC = None
    # OPP_API_MODEL_SPEC = 'doubao-seed-1-6-lite-251015'

    # Create shared memory for the agent
    memory = IPDMemory(api_model_spec=API_MODEL_SPEC)

    # Print initial memory statistics
    stats = memory.get_statistics()
    print(f"Initial Memory Statistics:")
    print(f"  Total Trials: {stats['total_trials']}")
    print(f"  Win Rate: {stats['win_rate']:.2%}")
    print(f"  Avg Rank: {stats['avg_rank']:.2f}")

    # Thread-safe print lock
    print_lock = threading.Lock()

    # Run multiple trials concurrently using ThreadPoolExecutor
    trial_results = []

    def run_trial_wrapper(trial_num):
        """Wrapper function to create agent and run trial"""
        # Create agent with shared memory for this trial
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
            api_model_spec=OPP_API_MODEL_SPEC,
            verbose=VERBOSE,
            print_lock=print_lock
        )

        return result
    # run_trial_wrapper(1)
    # exit(0)
    lm = LogManager(mcv=mcom(path='./VISUALIZE_logdir/', logdir='./VISUALIZE_logdir/'), who='reflexion')

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
                    print(f"  Avg Rank: {stats['avg_rank']:.2f}")
                    # Show recent reflections
                    print(f"\nRecent Lessons:")
                    print(memory.get_guidance(max_reflections=3))
                    print(f"{'='*80}\n")
                    lm.log_trivial({
                        'win_rate': stats['win_rate'],
                        'avg_rank': stats['avg_rank'],
                    })

            except Exception as exc:
                with print_lock:
                    print(f"Trial {trial_num} generated an exception: {exc}")
                    print(traceback.format_exc())
            if (i+1) % MAX_WORKERS == 0:
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
