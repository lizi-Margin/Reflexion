"""
Enhanced Self-Play Training System for SecretMafia with Dynamic Strategy Pool
This script orchestrates self-play games between configurable agents to optimize the strategy pool.
"""

import os
import sys
import time
import json
import random
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import argparse
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import textarena as ta
from corleone.training.strategy_pool_manager import StrategyPoolManager, TrainingSession
from corleone.agents.track1.micheal import Michael
from corleone.agents.track1.vito import Vito
from corleone.agents.track2.ipd_agent import IPDAgent
from corleone.game_logger_merged import GameLogger


@dataclass
class AgentConfig:
    """Configuration for agents in training"""
    agent_type: str  # "Michael", "Vito", etc.
    model_name: str
    temperature: float = 0.4
    max_tokens: int = 1024
    strategy_pool_enabled: bool = True
    custom_params: Dict = None

    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}


@dataclass
class TrainingConfig:
    """Configuration for training sessions"""
    # Agent configurations
    agent_configs: List[AgentConfig]
    agent_distribution: List[float]  # Probability distribution for agent types

    # Game configuration
    min_players_per_game: int = 5
    max_players_per_game: int = 8
    env_id: str = "SecretMafia-v0"

    # Training parameters
    games_per_session: int = 50
    optimization_interval: int = 10
    max_workers: int = 4

    # Strategy pool configuration
    strategy_pool_path: str = "strategy_pool.json"
    enable_strategy_learning: bool = True
    strategy_mutation_rate: float = 0.1

    # Logging configuration
    log_dir: str = "self_play_logs"
    enable_logging: bool = True

    def __post_init__(self):
        if sum(self.agent_distribution) != 1.0:
            # Normalize distribution
            total = sum(self.agent_distribution)
            self.agent_distribution = [p/total for p in self.agent_distribution]


class AgentFactory:
    """Factory for creating agents with different configurations"""

    @staticmethod
    def create_agent(agent_config: AgentConfig, player_id: int, logger=None, strategy_pool_manager=None):
        """Create an agent based on configuration

        Args:
            agent_config: Agent configuration
            player_id: Player ID for this agent
            logger: Optional logger
            strategy_pool_manager: Optional strategy pool manager (required for IPDAgent)
        """

        if agent_config.agent_type == "Michael":
            agent = Michael(model_name=agent_config.model_name)
        elif agent_config.agent_type == "Vito":
            agent = Vito(model_name=agent_config.model_name)
        elif agent_config.agent_type == "IPDAgent":
            agent = IPDAgent(
                model_name=agent_config.model_name,
                api_model_spec=agent_config.model_name,
                enable_logging=True,
                strategy_pool_manager=strategy_pool_manager if agent_config.strategy_pool_enabled else None
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_config.agent_type}")

        # Configure agent parameters (skip for IPDAgent which handles this internally)
        if agent_config.agent_type != "IPDAgent":
            if hasattr(agent, 'temperature'):
                agent.temperature = agent_config.temperature

            if hasattr(agent, 'max_tokens'):
                agent.max_tokens = agent_config.max_tokens

            # Apply custom parameters
            for param, value in agent_config.custom_params.items():
                if hasattr(agent, param):
                    setattr(agent, param, value)

        # Wrap with logger if enabled (skip for IPDAgent which has its own logging)
        if logger and agent_config.strategy_pool_enabled and agent_config.agent_type != "IPDAgent":
            from corleone.game_logger_merged import LoggedAgent
            agent = LoggedAgent(agent, logger, player_id)

        return agent


class EnhancedSelfPlayTrainer:
    """
    Enhanced self-play training system with configurable agents.
    Supports mixed agent populations and comprehensive training strategies.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Initialize components
        # Strategy pool manager - used by all agents (SecretMafia and IPD)
        # SecretMafia agents (Michael/Vito) use trainer-managed strategies
        # IPD agents create and manage their own strategies but share the same pool
        self.strategy_manager = StrategyPoolManager(config.strategy_pool_path)
        self.game_logger = GameLogger(log_dir=config.log_dir, enabled=config.enable_logging)
        self.agent_factory = AgentFactory()

        # Training state
        self.current_session = None
        self.games_completed = 0
        self.total_games_played = 0
        self.training_results = []
        self.agent_performance_history = defaultdict(list)

        # Ensure log directory exists
        os.makedirs(config.log_dir, exist_ok=True)

        print(f"Enhanced Self-Play Trainer initialized")
        print(f"Agent types: {[ac.agent_type for ac in config.agent_configs]}")
        print(f"Strategy pool: {len(self.strategy_manager.strategies)} strategies")

    def select_agent_config_for_game(self) -> AgentConfig:
        """Select agent configuration based on distribution"""
        return np.random.choice(
            self.config.agent_configs,
            p=self.config.agent_distribution
        )

    def run_self_play_game(self, game_config: Dict = None) -> Dict:
        """
        Run a single self-play game with configurable agents.
        """
        try:
            # Configure game - adapt to environment type
            if "IPD" in self.config.env_id:
                num_players = 3  # IPD is always 3 players
            elif "Blotto" in self.config.env_id:
                num_players = 2  # Blotto is 1v1
            elif "Codenames" in self.config.env_id:
                num_players = 4  # Codenames is 2v2
            else:
                # SecretMafia or other games with variable players
                num_players = game_config.get('num_players',
                    random.randint(self.config.min_players_per_game, self.config.max_players_per_game))

            # Create agents with mixed configurations
            agents = {}
            strategy_assignments = {}
            agent_configs_used = []

            for player_id in range(num_players):
                # Select agent configuration for this player
                agent_config = self.select_agent_config_for_game()
                agent_configs_used.append(agent_config)

                # Create agent
                agent = self.agent_factory.create_agent(
                    agent_config, player_id,
                    self.game_logger if self.config.enable_logging else None,
                    strategy_pool_manager=self.strategy_manager
                )
                agents[player_id] = agent

                # Track strategy assignment for strategy pool agents
                if agent_config.strategy_pool_enabled and hasattr(agent, 'agent'):
                    if hasattr(agent.agent, 'current_behavior_strategy_id'):
                        strategy_assignments[player_id] = agent.agent.current_behavior_strategy_id
                    else:
                        # Create a default strategy assignment
                        strategy_assignments[player_id] = f"player_{player_id}_default"

            # Initialize environment
            env = ta.make(env_id=self.config.env_id)
            obs = env.reset(num_players=num_players)

            # Track game state
            game_start_time = time.time()
            game_data = {
                "session_id": self.current_session.session_id if self.current_session else None,
                "game_id": f"game_{self.total_games_played + 1}",
                "start_time": game_start_time,
                "num_players": num_players,
                "agent_configs": [
                    {
                        "agent_type": ac.agent_type,
                        "model_name": ac.model_name,
                        "temperature": ac.temperature
                    } for ac in agent_configs_used
                ],
                "strategy_assignments": strategy_assignments,
                "strategies_used": list(set(strategy_assignments.values())),
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
                    "observation": str(observation)[:500],  # Truncate for storage
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
                    "agent_type": agent_configs_used[player_id].agent_type,
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
            game_result = self._parse_game_results(game_info, rewards, strategy_assignments, agent_configs_used)
            game_data.update({
                "end_time": game_end_time,
                "duration": game_end_time - game_start_time,
                "total_steps": step_count,
                "result": game_result,
                "rewards": rewards
            })

            # Update strategy performance
            if self.config.enable_strategy_learning:
                self._update_strategy_performance(game_data)

            # Update agent performance history
            self._update_agent_performance_history(game_data)

            # Print game completion message with scores
            winner_msg = game_result.get('winner_team', 'Unknown')
            if rewards:
                scores_str = ", ".join([f"P{pid}:{r}" for pid, r in sorted(rewards.items())])
                print(f"Game {self.total_games_played + 1} completed in {game_data['duration']:.1f}s. "
                      f"Winner: {winner_msg}. Scores: [{scores_str}]")
            else:
                print(f"Game {self.total_games_played + 1} completed in {game_data['duration']:.1f}s. "
                      f"Winner: {winner_msg}")

            return game_data

        except Exception as e:
            print(f"Error in self-play game: {e}")
            return {
                "error": str(e),
                "game_id": f"game_{self.total_games_played + 1}",
                "session_id": self.current_session.session_id if self.current_session else None
            }

    def _parse_game_results(self, game_info: Dict, rewards: Dict,
                          strategy_assignments: Dict, agent_configs_used: List[AgentConfig]) -> Dict:
        """Parse game results from environment output"""
        result = {
            "winner_team": "Unknown",
            "winning_players": [],
            "losing_players": [],
            "strategy_performance": {},
            "agent_performance": {},
            "victory": False
        }

        # Extract winner information
        if "winner" in game_info:
            result["winner_team"] = game_info["winner"]
        elif "winners" in game_info:
            result["winning_players"] = game_info["winners"]
        else:
            # For games without explicit winner field (like IPD), determine from rewards
            if rewards:
                # Find player(s) with highest reward
                max_reward = max(rewards.values())
                winners = [pid for pid, r in rewards.items() if r == max_reward]
                result["winning_players"] = winners

                # Set winner_team for display (use player IDs)
                if len(winners) == 1:
                    result["winner_team"] = f"Player {winners[0]}"
                else:
                    result["winner_team"] = f"Players {', '.join(map(str, winners))} (tie)"

        # Calculate performance for strategies and agents
        for player_id, strategy_id in strategy_assignments.items():
            player_reward = rewards.get(player_id, 0)
            player_won = (player_id in result["winning_players"]) or \
                        (result["winner_team"] and self._player_in_winning_team(player_id, result["winner_team"], game_info))

            agent_config = agent_configs_used[player_id]

            result["strategy_performance"][strategy_id] = {
                "player_id": player_id,
                "reward": player_reward,
                "victory": player_won,
                "role": game_info.get("player_roles", {}).get(player_id, "Unknown"),
                "agent_type": agent_config.agent_type,
                "model_name": agent_config.model_name
            }

            result["agent_performance"][player_id] = {
                "strategy_id": strategy_id,
                "reward": player_reward,
                "victory": player_won,
                "role": game_info.get("player_roles", {}).get(player_id, "Unknown"),
                "agent_type": agent_config.agent_type,
                "model_name": agent_config.model_name,
                "temperature": agent_config.temperature
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
            reward = perf_data.get("reward", 0)

            # Calculate performance score
            base_score = 1.0 if victory else -0.5

            # Role-specific bonuses (SecretMafia only)
            if victory and role != "Unknown":
                if role == "Detective":
                    base_score += 0.2
                elif role == "Doctor":
                    base_score += 0.1
                elif role == "Mafia":
                    base_score += 0.15

            # For games without role system (IPD, Blotto, Codenames),
            # use reward-based adjustment if available
            if role == "Unknown" and reward != 0:
                # Normalize reward to [-0.5, 0.5] range as adjustment
                reward_adjustment = max(-0.5, min(0.5, reward / 10.0))
                base_score += reward_adjustment

            # Update strategy in pool
            if strategy_id in self.strategy_manager.strategies:
                strategy = self.strategy_manager.strategies[strategy_id]
                strategy["success_score"] += base_score
                strategy["usage_count"] += 1
                strategy["avg_performance"] = strategy["success_score"] / strategy["usage_count"]
                strategy["last_used"] = datetime.now().isoformat()

    def _update_agent_performance_history(self, game_data: Dict):
        """Update agent performance history for analysis"""
        if "agent_performance" not in game_data:
            return

        agent_performance = game_data["agent_performance"]

        for player_id, perf_data in agent_performance.items():
            agent_type = perf_data.get("agent_type", "Unknown")
            model_name = perf_data.get("model_name", "Unknown")
            victory = perf_data.get("victory", False)

            key = f"{agent_type}_{model_name}"
            self.agent_performance_history[key].append({
                "game_id": game_data["game_id"],
                "timestamp": game_data["end_time"],
                "victory": victory,
                "role": perf_data.get("role", "Unknown"),
                "reward": perf_data.get("reward", 0)
            })

    def start_training_session(self, session_name: str = None) -> str:
        """Start a new training session"""
        if not session_name:
            session_name = f"enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session_id = self.game_logger.start_session(
            env_id=self.config.env_id,
            agents={},  # Will be populated per game
            num_players=self.config.min_players_per_game,
            # session_name=session_name
            additional_info={
                "session_name": session_name
            }
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

        print(f"Started enhanced training session: {session_id}")
        return session_id

    def run_training_batch(self, num_games: int = None) -> List[Dict]:
        """Run a batch of self-play games"""
        if not num_games:
            num_games = self.config.games_per_session

        print(f"Starting enhanced training batch of {num_games} games...")

        batch_results = []

        # Use ThreadPoolExecutor for parallel games
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
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

        print(f"Enhanced training batch completed. {len(batch_results)} games finished.")
        return batch_results

    def run_training_session(self, num_batches: int = 5, games_per_batch: int = 20):
        """Run a complete enhanced training session"""
        print("="*60)
        print("STARTING ENHANCED SELF-PLAY TRAINING SESSION")
        print("="*60)
        print(f"Agent configurations: {len(self.config.agent_configs)}")
        print(f"Agent types: {[ac.agent_type for ac in self.config.agent_configs]}")
        print(f"Strategy learning: {self.config.enable_strategy_learning}")
        print("="*60)

        # Start training session
        session_id = self.start_training_session()

        all_results = []

        for batch_num in range(num_batches):
            print(f"\n--- Enhanced Batch {batch_num + 1}/{num_batches} ---")

            # Run batch
            batch_results = self.run_training_batch(games_per_batch)
            all_results.extend(batch_results)

            # Optimize strategy pool
            if self.config.enable_strategy_learning and (batch_num + 1) % self.optimization_interval == 0:
                print(f"\nOptimizing strategy pool after {batch_num + 1} batches...")
                self.strategy_manager.optimize_strategy_pool(all_results[-self.optimization_interval * games_per_batch:])

                # Generate interim report
                self.generate_training_report()

        # Final optimization
        if self.config.enable_strategy_learning:
            print("\nFinal strategy pool optimization...")
            self.strategy_manager.optimize_strategy_pool(all_results)

        # Complete training session
        if self.current_session:
            self.current_session.end_time = datetime.now()
            session_duration = self.current_session.end_time - self.current_session.start_time

        # Generate final report
        print("\n" + "="*60)
        print("ENHANCED TRAINING SESSION COMPLETED")
        print("="*60)

        final_report = self.generate_enhanced_training_report()

        if self.config.enable_strategy_learning:
            self.strategy_manager.visualize_strategy_performance()

        return final_report

    def generate_enhanced_training_report(self) -> Dict:
        """Generate comprehensive enhanced training report"""
        base_report = self.strategy_manager.get_training_report()

        # Add enhanced metrics
        agent_type_performance = {}
        for agent_key, history in self.agent_performance_history.items():
            if history:
                victories = sum(1 for h in history if h["victory"])
                total_games = len(history)
                win_rate = victories / total_games

                agent_type, model_name = agent_key.split("_", 1)
                if agent_type not in agent_type_performance:
                    agent_type_performance[agent_type] = []
                agent_type_performance[agent_type].append({
                    "model": model_name,
                    "win_rate": win_rate,
                    "games": total_games
                })

        # Calculate agent type averages
        agent_type_averages = {}
        for agent_type, performances in agent_type_performance.items():
            avg_win_rate = np.mean([p["win_rate"] for p in performances])
            total_games = sum(p["games"] for p in performances)
            agent_type_averages[agent_type] = {
                "average_win_rate": avg_win_rate,
                "total_games": total_games,
                "model_count": len(performances)
            }

        # Add enhanced data to report
        enhanced_report = {
            **base_report,
            "agent_type_performance": agent_type_performance,
            "agent_type_averages": agent_type_averages,
            "training_config": asdict(self.config),
            "total_agent_types": len(self.config.agent_configs),
            "strategy_learning_enabled": self.config.enable_strategy_learning
        }

        print("\n" + "="*50)
        print("ENHANCED TRAINING REPORT")
        print("="*50)
        print(f"Total Strategies: {enhanced_report['total_strategies']}")
        print(f"Total Games Played: {enhanced_report.get('total_games_completed', 0)}")
        print(f"Agent Types Used: {enhanced_report['total_agent_types']}")

        for agent_type, stats in agent_type_averages.items():
            print(f"{agent_type} Performance: {stats['average_win_rate']:.3f} ({stats['total_games']} games)")

        return enhanced_report

    def generate_training_report(self, save_to_file: bool = False) -> Dict:
        """Generate training report (backward compatibility)"""
        return self.generate_enhanced_training_report()

