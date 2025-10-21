#!/usr/bin/env python3
"""
Simple Self-Play Training for SecretMafia
6 Michael agents play against each other in SecretMafia environment
"""

import os
import sys
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import textarena as ta
from corleone.agents.track1.micheal import Michael


class SimpleSelfPlay:
    def __init__(self, model_name="qwen3-8b", num_games=10, use_strategy_pool=True, max_workers=2):
        """
        Initialize simple self-play training

        Args:
            model_name: LLM model to use for all Michael agents (qwen3-8b, qwen3-4b, deepseek-r1, etc.)
            num_games: Number of games to play
            use_strategy_pool: Whether to use strategy pool for Michael agents
            max_workers: Number of parallel games to run simultaneously
        """
        self.model_name = model_name
        self.num_games = num_games
        self.use_strategy_pool = use_strategy_pool
        self.max_workers = max_workers
        self.env = ta.make(env_id="SecretMafia-v0")

        # Statistics tracking (thread-safe)
        import threading
        self.stats_lock = threading.Lock()
        self.stats = {
            'total_games': 0,
            'mafia_wins': 0,
            'civilian_wins': 0
        }

        print(f"Initialized Simple Self-Play:")
        print(f"  Model: {model_name}")
        print(f"  Number of games: {num_games}")
        print(f"  Players per game: 6")
        print(f"  Strategy Pool: {'Enabled' if use_strategy_pool else 'Disabled'}")
        print(f"  Parallel games: {max_workers}")

    def create_agents(self):
        """Create 6 Michael agents with the same model"""
        agents = []
        for i in range(6):
            agent = Michael(self.model_name, use_strategy_pool=self.use_strategy_pool)
            agents.append(agent)
        return agents

    def play_single_game(self, game_id=None):
        """Play a single game of SecretMafia"""
        if game_id is not None:
            print(f"\n=== Game {game_id} Started ===")
        else:
            print(f"\n=== New Game Started ===")

        # Create new environment instance for thread safety
        env = ta.make(env_id="SecretMafia-v0")

        # Create agents as a dict
        agents = self.create_agents()
        agent_dict = {i: agents[i] for i in range(len(agents))}

        # Reset environment
        env.reset(num_players=len(agents))

        # Track game state
        game_log = []
        done = False
        step = 0

        while not done:
            step += 1

            # Get current player and observation using textarena API
            player_id, observation = env.get_observation()

            if player_id not in agent_dict:
                print(f"Unknown player: {player_id}, skipping...")
                break

            agent = agent_dict[player_id]

            try:
                # Handle observation format for agent
                if observation is None:
                    formatted_obs = "Game started. You are Player1."
                elif isinstance(observation, dict):
                    # Convert dict observation to string
                    if 'message' in observation:
                        formatted_obs = observation['message']
                    else:
                        formatted_obs = str(observation)
                else:
                    formatted_obs = str(observation)

                action = agent(formatted_obs)

                # Log the action
                game_log.append({
                    'step': step,
                    'player_id': player_id,
                    'observation': formatted_obs,
                    'action': action
                })

                if game_id is not None:
                    print(f"Game {game_id} - Step {step}: Player {player_id}")
                else:
                    print(f"Step {step}: Player {player_id}")
                print(f"  Action: {action[:100]}..." if len(action) > 100 else f"  Action: {action}")

                # Take action in environment using textarena API
                done, step_info = env.step(action=action)

            except Exception as e:
                print(f"Error from agent {player_id}: {e}")
                # Take a random action as fallback
                action = random.choice(["pass", "investigate random", "accuse random"])
                done, step_info = env.step(action=action)

        # Game finished - get results
        rewards, game_info = env.close()

        # Determine winner from game info
        winner = self._determine_winner(game_info, game_log)
        self._update_stats(winner)

        if game_id is not None:
            print(f"\nGame {game_id} Finished! Winner: {winner}")
        else:
            print(f"\nGame Finished! Winner: {winner}")

        return game_log, winner

    def play_single_game_parallel(self, game_id):
        """Play a single game in parallel (thread-safe version)"""
        try:
            return self.play_single_game(game_id)
        except Exception as e:
            print(f"Game {game_id} failed with error: {e}")
            return [], "Error"

    def _extract_current_player(self, observation):
        """Extract current player ID from observation"""
        try:
            # Handle different observation formats
            if isinstance(observation, dict):
                # If observation is a dict, try to get the current player
                return observation.get('current_player', 'Player1')
            elif isinstance(observation, str):
                # If observation is a string, use regex to extract player
                import re
                match = re.search(r"Current Player: (\w+)", observation)
                if match:
                    return match.group(1)
                # Try other patterns
                match = re.search(r"Player(\d+)'s turn", observation)
                if match:
                    return f"Player{match.group(1)}"
            return "Player1"  # Default fallback
        except Exception as e:
            print(f"Error extracting current player: {e}")
            return "Player1"

    def _determine_winner(self, info, game_log):
        """Determine the winner based on game info"""
        # Try to get winner from info
        if info and 'winner' in info:
            return info['winner']

        # Fallback: simple heuristic based on game length
        if len(game_log) > 20:
            return "Civilians"
        else:
            return "Mafia"

    def _update_stats(self, winner):
        """Update game statistics (thread-safe)"""
        with self.stats_lock:
            self.stats['total_games'] += 1

            if winner == "Mafia":
                self.stats['mafia_wins'] += 1
            else:
                self.stats['civilian_wins'] += 1

            # Simple player tracking (just by player position)
            for i in range(6):
                player_id = f"Player{i+1}"
                if player_id not in self.stats:
                    self.stats[player_id] = {'games': 0, 'wins': 0}

                self.stats[player_id]['games'] += 1

    def run_training(self):
        """Run the complete self-play training session with parallel processing"""
        print(f"\n{'='*60}")
        print(f"STARTING SIMPLE SELF-PLAY TRAINING (PARALLEL)")
        print(f"{'='*60}")
        print(f"Running {self.num_games} games with {self.max_workers} parallel workers")

        start_time = datetime.now()

        if self.max_workers <= 1:
            # Sequential execution (original behavior)
            for game_num in range(self.num_games):
                print(f"\n--- Game {game_num + 1}/{self.num_games} ---")

                try:
                    game_log, winner = self.play_single_game()

                    # Print intermediate stats every 5 games
                    if (game_num + 1) % 5 == 0:
                        self.print_intermediate_stats()

                except Exception as e:
                    print(f"Error in game {game_num + 1}: {e}")
                    continue
        else:
            # Parallel execution
            self.run_training_parallel()

        end_time = datetime.now()
        duration = end_time - start_time

        # Print final statistics
        self.print_final_stats(duration)

    def run_training_parallel(self):
        """Run training with parallel games"""
        print(f"\nStarting parallel training with {self.max_workers} workers...")

        completed_games = 0

        # Use ThreadPoolExecutor for parallel games
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all games
            future_to_game = {
                executor.submit(self.play_single_game_parallel, i+1): i+1
                for i in range(self.num_games)
            }

            # Collect results as they complete
            for future in as_completed(future_to_game):
                game_id = future_to_game[future]
                try:
                    game_log, winner = future.result()
                    completed_games += 1

                    # Progress update
                    print(f"Completed {completed_games}/{self.num_games} games (Game {game_id} finished)")

                    # Print intermediate stats every 5 games
                    if completed_games % 5 == 0:
                        self.print_intermediate_stats()

                except Exception as e:
                    print(f"Game {game_id} failed: {e}")
                    completed_games += 1

    def print_intermediate_stats(self):
        """Print statistics after every few games"""
        print(f"\n--- Intermediate Stats ---")
        total = self.stats['total_games']
        if total > 0:
            mafia_win_rate = self.stats['mafia_wins'] / total * 100
            civilian_win_rate = self.stats['civilian_wins'] / total * 100
            print(f"  Total Games: {total}")
            print(f"  Mafia Win Rate: {mafia_win_rate:.1f}%")
            print(f"  Civilian Win Rate: {civilian_win_rate:.1f}%")

    def print_final_stats(self, duration):
        """Print final training statistics"""
        print(f"\n{'='*60}")
        print(f"FINAL TRAINING RESULTS")
        print(f"{'='*60}")

        total = self.stats['total_games']
        if total > 0:
            mafia_win_rate = self.stats['mafia_wins'] / total * 100
            civilian_win_rate = self.stats['civilian_wins'] / total * 100

            print(f"Model: {self.model_name}")
            print(f"Total Games: {total}")
            print(f"Duration: {duration}")
            print(f"Mafia Wins: {self.stats['mafia_wins']} ({mafia_win_rate:.1f}%)")
            print(f"Civilian Wins: {self.stats['civilian_wins']} ({civilian_win_rate:.1f}%)")

            print(f"\nPlayer Performance:")
            for i in range(6):
                player_id = f"Player{i+1}"
                if player_id in self.stats:
                    games = self.stats[player_id]['games']
                    win_rate = 50.0  # Simplified - we don't track individual wins in this version
                    print(f"  {player_id}: {games} games played")
                else:
                    print(f"  {player_id}: 0 games played")


def main():
    """Main function with simple configuration"""

    # SIMPLE CONFIGURATION - JUST CHANGE THESE VALUES
    MODEL_NAME = "deepseek-v3.1"          # Options: qwen3-8b, qwen3-4b, deepseek-r1, etc.
    NUM_GAMES = 10                   # Number of games to play
    MAX_WORKERS = 2                  # Number of parallel games (1 = sequential)
    USE_STRATEGY_POOL = True         # Whether to use strategy pool (True/False)

    print("Simple Self-Play Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Games: {NUM_GAMES}")
    print(f"  Players: 6 Michael agents")
    print(f"  Parallel games: {MAX_WORKERS}")
    print(f"  Strategy Pool: {'Enabled' if USE_STRATEGY_POOL else 'Disabled'}")

    # Confirm and start
    confirm = input("\nStart training? (y/n): ").lower()
    if confirm != 'y':
        print("Training cancelled.")
        return

    # Run training
    trainer = SimpleSelfPlay(
        model_name=MODEL_NAME,
        num_games=NUM_GAMES,
        max_workers=MAX_WORKERS,
        use_strategy_pool=USE_STRATEGY_POOL
    )

    trainer.run_training()


if __name__ == "__main__":
    main()