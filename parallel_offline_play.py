"""
Parallel offline play script for running multiple games simultaneously.
This script allows you to run Y games in batches of X games each using multiple threads.
All game sessions will be recorded for training and analysis purposes.
"""

import textarena as ta
from src.agent import LLMAgent, HumanAgent
from src.game_logger import GameLogger, LoggedAgent
from xushuhang.agents.family import Vito, Michael
import sys
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import argparse
from datetime import datetime

# Set UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Game configuration constants
MODEL_NAME = "Vito-1.3"
ENV_ID = "SecretMafia-v0"
ENABLE_LOGGING = True
LOG_DIR = "game_logs"

# Thread-safe counter for progress tracking
completed_games = threading.BoundedSemaphore(1)
game_counter = 0
results_lock = threading.Lock()
all_results = []

def create_agents() -> Dict[int, Any]:
    """Create a dictionary of agents for a single game."""
    raw_agents = {
        0: Michael(model_name="deepseek-r1"),
        1: Michael(model_name="deepseek-r1"),
        2: Michael(model_name="deepseek-r1"),
        3: Michael(model_name="deepseek-r1"),
        4: Michael(model_name="deepseek-r1"),
        5: Michael(model_name="deepseek-r1"),
    }
    return raw_agents

def run_single_game(game_id: int) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    """
    Run a single game and return the results.

    Args:
        game_id: Unique identifier for this game

    Returns:
        Tuple containing (game_id, rewards, game_info)
    """
    global game_counter

    # Create thread-specific log directory
    thread_id = threading.current_thread().ident
    thread_log_dir = os.path.join(LOG_DIR, f"thread_{thread_id}")

    # Initialize game logger for this thread
    logger = GameLogger(log_dir=thread_log_dir, enabled=ENABLE_LOGGING)

    # Create agents for this game
    raw_agents = create_agents()

    # Wrap agents with logging if enabled
    if ENABLE_LOGGING:
        agents = {}
        for player_id, agent in raw_agents.items():
            agents[player_id] = LoggedAgent(agent, logger, player_id)
    else:
        agents = raw_agents

    # Initialize the environment
    env = ta.make(env_id=ENV_ID)
    env.reset(num_players=len(agents))

    # Start logging session
    session_id = logger.start_session(
        env_id=ENV_ID,
        agents=raw_agents,
        num_players=len(agents),
        additional_info={
            "script": "parallel_offline_play.py",
            "model_name": MODEL_NAME,
            "game_id": game_id,
            "thread_id": thread_id,
            "description": f"Parallel game #{game_id} in thread {thread_id}"
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

    # End game and log results
    rewards, game_info = env.close()

    if ENABLE_LOGGING:
        logger.log_results(rewards, game_info)
        logger.end_session()

    # Update progress counter
    with completed_games:
        global game_counter
        game_counter += 1
        print(f"Completed game {game_counter}/{total_games} (Game ID: {game_id}, Thread: {thread_id})")

    return (game_id, rewards, game_info, session_id)

def run_parallel_games(batch_size: int, total_games: int, max_workers: int = None) -> List[Dict[str, Any]]:
    """
    Run multiple games in parallel using ThreadPoolExecutor.

    Args:
        batch_size: Number of games to run simultaneously
        total_games: Total number of games to run
        max_workers: Maximum number of worker threads (defaults to batch_size)

    Returns:
        List of game results
    """
    if max_workers is None:
        max_workers = batch_size

    print(f"Starting {total_games} games with batch size {batch_size} (max workers: {max_workers})")
    print(f"Log directory: {LOG_DIR}")
    print("-" * 60)

    results = []

    # Create games in batches
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all games to the executor
        future_to_game_id = {}

        for game_id in range(1, total_games + 1):
            future = executor.submit(run_single_game, game_id)
            future_to_game_id[future] = game_id

            # Control batch size by waiting for some futures to complete
            if len(future_to_game_id) >= batch_size:
                # Wait for at least one game to complete before submitting more
                completed_futures = []
                for future in as_completed(list(future_to_game_id.keys())):
                    completed_futures.append(future)
                    break

                # Remove completed futures from the dict and collect results
                for future in completed_futures:
                    game_result = future.result()
                    results.append({
                        'game_id': game_result[0],
                        'rewards': game_result[1],
                        'game_info': game_result[2],
                        'session_id': game_result[3]
                    })
                    del future_to_game_id[future]

        # Wait for remaining games to complete
        for future in as_completed(future_to_game_id.keys()):
            game_result = future.result()
            results.append({
                'game_id': game_result[0],
                'rewards': game_result[1],
                'game_info': game_result[2],
                'session_id': game_result[3]
            })

    return results

def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze and summarize game results.

    Args:
        results: List of game results

    Returns:
        Summary statistics
    """
    if not results:
        return {}

    # Analyze rewards
    all_rewards = []
    player_stats = {}

    for result in results:
        rewards = result['rewards']
        all_rewards.append(rewards)

        for player_id, reward in rewards.items():
            if player_id not in player_stats:
                player_stats[player_id] = []
            player_stats[player_id].append(reward)

    # Calculate statistics
    summary = {
        'total_games': len(results),
        'average_rewards_per_player': {},
        'win_rate_per_player': {},
        'total_rewards_per_player': {}
    }

    for player_id in player_stats:
        rewards = player_stats[player_id]
        summary['average_rewards_per_player'][player_id] = sum(rewards) / len(rewards)
        summary['total_rewards_per_player'][player_id] = sum(rewards)
        summary['win_rate_per_player'][player_id] = sum(1 for r in rewards if r > 0) / len(rewards)

    return summary

def main():
    parser = argparse.ArgumentParser(description='Run multiple games in parallel')
    parser.add_argument('--batch-size', '-b', type=int, default=4,
                       help='Number of games to run simultaneously (default: 4)')
    parser.add_argument('--total-games', '-t', type=int, default=12,
                       help='Total number of games to run (default: 10)')
    parser.add_argument('--max-workers', '-w', type=int, default=None,
                       help='Maximum number of worker threads (default: batch_size)')
    parser.add_argument('--no-logging', action='store_true',
                       help='Disable game logging')

    args = parser.parse_args()

    # Update global variables based on arguments
    global total_games, ENABLE_LOGGING
    total_games = args.total_games
    if args.no_logging:
        ENABLE_LOGGING = False

    # Validate arguments
    if args.batch_size <= 0 or args.total_games <= 0:
        print("Error: batch-size and total-games must be positive integers")
        return

    if args.batch_size > args.total_games:
        print("Warning: batch-size is larger than total-games, setting batch-size to total-games")
        args.batch_size = args.total_games

    # Create log directory if it doesn't exist
    if ENABLE_LOGGING and not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Record start time
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Parallel offline play started at: {start_datetime}")

    # Run games
    results = run_parallel_games(args.batch_size, args.total_games, args.max_workers)

    # Record end time
    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = end_time - start_time

    print("-" * 60)
    print(f"Parallel offline play completed at: {end_datetime}")
    print(f"Total duration: {duration:.2f} seconds")
    print(f"Average time per game: {duration/len(results):.2f} seconds")

    # Analyze and display results
    summary = analyze_results(results)

    print(f"\nResults Summary:")
    print(f"Total games played: {summary['total_games']}")

    print(f"\nPlayer Performance:")
    for player_id in sorted(summary['average_rewards_per_player'].keys()):
        avg_reward = summary['average_rewards_per_player'][player_id]
        win_rate = summary['win_rate_per_player'][player_id]
        total_reward = summary['total_rewards_per_player'][player_id]
        print(f"  Player {player_id}: Avg Reward: {avg_reward:.3f}, Win Rate: {win_rate:.2%}, Total Reward: {total_reward}")

    if ENABLE_LOGGING:
        print(f"\nGame logs saved to: {LOG_DIR}")
        print(f"Log directories organized by thread ID for parallel execution")

    # Save results summary to file
    summary_file = os.path.join(LOG_DIR if ENABLE_LOGGING else ".", f"summary_{int(start_time)}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Parallel Offline Play Summary\n")
        f.write(f"{'='*40}\n")
        f.write(f"Start Time: {start_datetime}\n")
        f.write(f"End Time: {end_datetime}\n")
        f.write(f"Duration: {duration:.2f} seconds\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Total Games: {args.total_games}\n")
        f.write(f"Max Workers: {args.max_workers}\n")
        f.write(f"\nResults:\n")
        f.write(f"Total games played: {summary['total_games']}\n")

        f.write(f"\nPlayer Performance:\n")
        for player_id in sorted(summary['average_rewards_per_player'].keys()):
            avg_reward = summary['average_rewards_per_player'][player_id]
            win_rate = summary['win_rate_per_player'][player_id]
            total_reward = summary['total_rewards_per_player'][player_id]
            f.write(f"  Player {player_id}: Avg Reward: {avg_reward:.3f}, Win Rate: {win_rate:.2%}, Total Reward: {total_reward}\n")

    print(f"\nResults summary saved to: {summary_file}")

if __name__ == "__main__":
    main()