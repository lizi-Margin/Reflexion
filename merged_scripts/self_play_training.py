"""
Self-Play Training System for SecretMafia with Dynamic Strategy Pool
This script orchestrates self-play games between Michael agents to optimize the strategy pool.
"""

import os
import sys
import time
import json
import random
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import textarena as ta
from .strategy_pool_manager import StrategyPoolManager, TrainingSession
from corleone.agents.track1.micheal import Michael
from .game_logger import GameLogger, LoggedAgent


class SelfPlayTrainer:
    """
    Self-play training system for optimizing strategy pools through
    competitive games between Michael agents.
    """

    def __init__(self,
                 strategy_pool_path: str = "strategy_pool.json",
                 log_dir: str = "self_play_logs",
                 max_workers: int = 4):
        self.strategy_pool_path = strategy_pool_path
        self.log_dir = log_dir
        self.max_workers = max_workers

        # Initialize components
        self.strategy_manager = StrategyPoolManager(strategy_pool_path)
        self.game_logger = GameLogger(log_dir=log_dir, enabled=True)

        # Training parameters
        self.games_per_session = 50
        self.optimization_interval = 10  # Optimize every N games
        self.min_players_per_game = 5
        self.max_players_per_game = 8

        # Training state
        self.current_session = None
        self.games_completed = 0
        self.total_games_played = 0
        self.training_results = []

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        print(f"Self-Play Trainer initialized with {len(self.strategy_manager.strategies)} strategies")

    def start_training_session(self, session_name: str = None) -> str:
        """Start a new training session"""
        if not session_name:
            session_name = f"training_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session_id = self.game_logger.start_session(
            env_id="SecretMafia-v0",
            agents={},  # Will be populated per game
            num_players=self.min_players_per_game,
            session_name=session_name
        )

        self.current_session = TrainingSession(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            games_played=0,
            strategies_used=[],
            results=[],
            performance_improvement=0.0
        )

        print(f"Started training session: {session_id}")
        return session_id

    def run_self_play_game(self, game_config: Dict = None) -> Dict:
        """
        Run a single self-play game between Michael agents.
        """
        try:
            # Configure game
            num_players = game_config.get('num_players', random.randint(self.min_players_per_game, self.max_players_per_game))

            # Select strategies for this game
            selected_strategy_ids = self.strategy_manager.select_strategies_for_training(num_players * 2)

            # Create agents with selected strategies
            agents = {}
            strategy_assignments = {}

            for player_id in range(num_players):
                # Randomly assign strategies to players
                strategy_id = random.choice(selected_strategy_ids)
                strategy_assignments[player_id] = strategy_id

                # Create Michael agent with strategy pool enabled
                agent = Michael(model_name="qwen3-8b")  # Use consistent model for training
                agents[player_id] = LoggedAgent(agent, self.game_logger, player_id)

            # Initialize environment
            env = ta.make(env_id="SecretMafia-v0")
            obs = env.reset(num_players=num_players)

            # Track game state
            game_start_time = time.time()
            game_data = {
                "session_id": self.current_session.session_id if self.current_session else None,
                "game_id": f"game_{self.total_games_played + 1}",
                "start_time": game_start_time,
                "num_players": num_players,
                "strategy_assignments": strategy_assignments,
                "strategies_used": list(set(selected_strategy_ids)),
                "game_events": [],
                "agent_states": []
            }

            # Run game loop
            done = False
            step_count = 0

            while not done and step_count < 100:  # Prevent infinite loops
                player_id, observation = env.get_observation()

                # Record observation
                game_data["game_events"].append({
                    "step": step_count,
                    "player_id": player_id,
                    "observation": observation,
                    "timestamp": time.time()
                })

                # Get agent action
                agent = agents[player_id]
                action = agent(observation)

                # Record action
                game_data["agent_states"].append({
                    "step": step_count,
                    "player_id": player_id,
                    "action": action,
                    "strategy_id": strategy_assignments.get(player_id),
                    "timestamp": time.time()
                })

                # Execute action
                done, step_info = env.step(action=action)
                step_count += 1

                if step_count % 10 == 0:  # Progress indicator
                    print(f"Game {self.total_games_played + 1}: Step {step_count}")

            # Get final results
            rewards, game_info = env.close()
            game_end_time = time.time()

            # Parse game results
            game_result = self._parse_game_results(game_info, rewards, strategy_assignments)
            game_data.update({
                "end_time": game_end_time,
                "duration": game_end_time - game_start_time,
                "total_steps": step_count,
                "result": game_result,
                "rewards": rewards
            })

            # Update strategy performance
            self._update_strategy_performance(game_data)

            print(f"Game {self.total_games_played + 1} completed in {game_data['duration']:.1f}s. "
                  f"Winner: {game_result.get('winner_team', 'Unknown')}")

            return game_data

        except Exception as e:
            print(f"Error in self-play game: {e}")
            return {
                "error": str(e),
                "game_id": f"game_{self.total_games_played + 1}",
                "session_id": self.current_session.session_id if self.current_session else None
            }

    def _parse_game_results(self, game_info: Dict, rewards: Dict, strategy_assignments: Dict) -> Dict:
        """Parse game results from environment output"""
        result = {
            "winner_team": "Unknown",
            "winning_players": [],
            "losing_players": [],
            "strategy_performance": {},
            "victory": False
        }

        # Extract winner information
        if "winner" in game_info:
            result["winner_team"] = game_info["winner"]
        elif "winners" in game_info:
            result["winning_players"] = game_info["winners"]

        # Calculate strategy performance
        for player_id, strategy_id in strategy_assignments.items():
            player_reward = rewards.get(player_id, 0)
            player_won = (player_id in result["winning_players"]) or \
                        (result["winner_team"] and self._player_in_winning_team(player_id, result["winner_team"], game_info))

            result["strategy_performance"][strategy_id] = {
                "player_id": player_id,
                "reward": player_reward,
                "victory": player_won,
                "role": game_info.get("player_roles", {}).get(player_id, "Unknown")
            }

        return result

    def _player_in_winning_team(self, player_id: int, winner_team: str, game_info: Dict) -> bool:
        """Determine if a player belongs to the winning team"""
        player_roles = game_info.get("player_roles", {})
        player_role = player_roles.get(player_id, "Unknown")

        if winner_team == "Mafia":
            return player_role == "Mafia"
        elif winner_team == "Village":
            return player_role in ["A regular villager", "Detective", "Doctor"]

        return False

    def _update_strategy_performance(self, game_data: Dict):
        """Update strategy performance based on game results"""
        if "strategy_performance" not in game_data:
            return

        strategy_performance = game_data["strategy_performance"]

        for strategy_id, perf_data in strategy_performance.items():
            victory = perf_data.get("victory", False)
            role = perf_data.get("role", "Unknown")

            # Calculate performance score
            base_score = 1.0 if victory else -0.5

            # Role-specific bonuses
            if victory:
                if role == "Detective":
                    base_score += 0.2
                elif role == "Doctor":
                    base_score += 0.1
                elif role == "Mafia":
                    base_score += 0.15

            # Update strategy in pool
            if strategy_id in self.strategy_manager.strategies:
                strategy = self.strategy_manager.strategies[strategy_id]
                strategy["success_score"] += base_score
                strategy["usage_count"] += 1
                strategy["avg_performance"] = strategy["success_score"] / strategy["usage_count"]
                strategy["last_used"] = datetime.now().isoformat()

    def run_training_batch(self, num_games: int = None) -> List[Dict]:
        """Run a batch of self-play games"""
        if not num_games:
            num_games = self.games_per_session

        print(f"Starting training batch of {num_games} games...")

        batch_results = []

        # Use ThreadPoolExecutor for parallel games
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all games
            future_to_game = {
                executor.submit(self.run_self_play_game): i
                for i in range(num_games)
            }

            # Collect results as they complete
            for future in as_completed(future_to_game):
                game_index = future_to_game[future]
                try:
                    result = future.result()
                    batch_results.append(result)
                    self.games_completed += 1
                    self.total_games_played += 1

                    # Progress update
                    if self.games_completed % 5 == 0:
                        print(f"Completed {self.games_completed}/{num_games} games in current batch")

                except Exception as e:
                    print(f"Game {game_index} failed: {e}")
                    batch_results.append({
                        "error": str(e),
                        "game_index": game_index
                    })

        # Update training session
        if self.current_session:
            self.current_session.games_played += len(batch_results)
            self.current_session.results.extend(batch_results)
            self.current_session.strategies_used.extend([
                strategy for game in batch_results
                for strategy in game.get("strategies_used", [])
            ])

        print(f"Training batch completed. {len(batch_results)} games finished.")
        return batch_results

    def run_training_session(self, num_batches: int = 5, games_per_batch: int = 20):
        """Run a complete training session with multiple batches"""
        print("="*60)
        print("STARTING SELF-PLAY TRAINING SESSION")
        print("="*60)

        # Start training session
        session_id = self.start_training_session()

        all_results = []

        for batch_num in range(num_batches):
            print(f"\n--- Batch {batch_num + 1}/{num_batches} ---")

            # Run batch
            batch_results = self.run_training_batch(games_per_batch)
            all_results.extend(batch_results)

            # Optimize strategy pool
            if (batch_num + 1) % self.optimization_interval == 0:
                print(f"\nOptimizing strategy pool after {batch_num + 1} batches...")
                self.strategy_manager.optimize_strategy_pool(all_results[-self.optimization_interval * games_per_batch:])

                # Generate interim report
                self.generate_training_report()

        # Final optimization
        print("\nFinal strategy pool optimization...")
        self.strategy_manager.optimize_strategy_pool(all_results)

        # Complete training session
        if self.current_session:
            self.current_session.end_time = datetime.now()
            session_duration = self.current_session.end_time - self.current_session.start_time
            self.current_session.performance_improvement = self._calculate_session_improvement(all_results)

            self.strategy_manager.training_history.append(self.current_session)

        # Generate final report
        print("\n" + "="*60)
        print("TRAINING SESSION COMPLETED")
        print("="*60)

        final_report = self.generate_training_report(save_to_file=True)
        self.strategy_manager.visualize_strategy_performance()

        return final_report

    def _calculate_session_improvement(self, results: List[Dict]) -> float:
        """Calculate performance improvement during the session"""
        if len(results) < 10:
            return 0.0

        # Split results into first half and second half
        mid_point = len(results) // 2
        first_half = results[:mid_point]
        second_half = results[mid_point:]

        # Calculate average win rate for each half
        first_half_wins = sum(1 for r in first_half if r.get("result", {}).get("strategy_performance", {}))
        second_half_wins = sum(1 for r in second_half if r.get("result", {}).get("strategy_performance", {}))

        first_half_rate = first_half_wins / len(first_half) if first_half else 0
        second_half_rate = second_half_wins / len(second_half) if second_half else 0

        return second_half_rate - first_half_rate

    def generate_training_report(self, save_to_file: bool = False) -> Dict:
        """Generate comprehensive training report"""
        report = self.strategy_manager.get_training_report()

        # Add session-specific data
        if self.current_session:
            session_data = {
                "current_session_id": self.current_session.session_id,
                "session_games_played": self.current_session.games_played,
                "session_duration": str(datetime.now() - self.current_session.start_time) if not self.current_session.end_time
                                   else str(self.current_session.end_time - self.current_session.start_time),
                "session_strategies_used": len(set(self.current_session.strategies_used))
            }
            report.update(session_data)

        # Add training statistics
        report.update({
            "total_games_completed": self.total_games_played,
            "games_in_current_session": self.games_completed,
            "training_batches_completed": self.games_completed // self.games_per_session
        })

        print("\n" + "="*50)
        print("TRAINING REPORT")
        print("="*50)
        print(f"Total Strategies: {report['total_strategies']}")
        print(f"Total Games Played: {report.get('total_games_completed', 0)}")
        print(f"Average Strategy Performance: {report['average_performance']:.3f}")
        print(f"Best Strategy: {report['best_strategy']['id']} (perf: {report['best_strategy']['performance']:.3f})")
        print(f"Strategy Types: {report['type_distribution']}")
        print(f"Strategy Roles: {report['role_distribution']}")

        if save_to_file:
            report_path = os.path.join(self.log_dir, f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Training report saved to: {report_path}")

        return report

    def load_existing_session(self, session_id: str) -> bool:
        """Load an existing training session"""
        # This would load session data from logs
        # Implementation depends on how sessions are stored
        print(f"Loading existing session: {session_id}")
        return False


def main():
    """Main function to run self-play training"""
    import argparse

    parser = argparse.ArgumentParser(description="Self-Play Training for SecretMafia")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of training batches")
    parser.add_argument("--games-per-batch", type=int, default=20, help="Games per batch")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--strategy-pool", type=str, default="strategy_pool.json", help="Strategy pool file")
    parser.add_argument("--log-dir", type=str, default="self_play_logs", help="Log directory")

    args = parser.parse_args()

    # Initialize trainer
    trainer = SelfPlayTrainer(
        strategy_pool_path=args.strategy_pool,
        log_dir=args.log_dir,
        max_workers=args.workers
    )

    # Run training
    try:
        final_report = trainer.run_training_session(
            num_batches=args.num_batches,
            games_per_batch=args.games_per_batch
        )

        print("\nTraining completed successfully!")
        print(f"Final average strategy performance: {final_report['average_performance']:.3f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.generate_training_report(save_to_file=True)
    except Exception as e:
        print(f"\nTraining failed: {e}")
        trainer.generate_training_report(save_to_file=True)


if __name__ == "__main__":
    main()