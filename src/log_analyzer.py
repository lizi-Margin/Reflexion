"""
Log Analyzer for Mind Games Challenge
Tools for analyzing logged game data for training insights.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import pandas as pd


class LogAnalyzer:
    """
    Analyze logged game sessions to extract insights for training.
    """

    def __init__(self, log_dir: str = "game_logs"):
        """
        Initialize the log analyzer.

        Args:
            log_dir: Directory containing game logs
        """
        self.log_dir = Path(log_dir)
        self.sessions = []

    def load_sessions(self) -> None:
        """Load all game sessions from log directory."""
        self.sessions = []

        for file_path in self.log_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    session = json.load(f)
                    self.sessions.append(session)
            except Exception as e:
                print(f"Error loading session file {file_path}: {e}")

        print(f"Loaded {len(self.sessions)} sessions from {self.log_dir}")

    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the logged sessions."""
        if not self.sessions:
            return {"error": "No sessions loaded"}

        stats = {
            "total_sessions": len(self.sessions),
            "environments": Counter(),
            "agent_models": Counter(),
            "role_distribution": Counter(),
            "game_outcomes": Counter(),
            "avg_turns_per_game": 0,
            "total_turns": 0
        }

        for session in self.sessions:
            # Environment stats
            env = session.get("environment", "unknown")
            stats["environments"][env] += 1

            # Agent model stats
            for agent_info in session.get("agents", {}).values():
                model = agent_info.get("model_name", "unknown")
                stats["agent_models"][model] += 1

            # Role distribution
            for agent_info in session.get("agents", {}).values():
                init_info = agent_info.get("init_info", {})
                if init_info and "role" in init_info:
                    stats["role_distribution"][init_info["role"]] += 1

            # Game outcomes
            summary = session.get("summary", {})
            if "winners" in summary:
                winners = summary["winners"]
                if isinstance(winners, list) and winners:
                    winners_str = "_".join(map(str, winners))
                    stats["game_outcomes"][f"winners_{winners_str}"] += 1

            # Turn statistics
            total_turns = len(session.get("turns", []))
            stats["total_turns"] += total_turns

        if stats["total_sessions"] > 0:
            stats["avg_turns_per_game"] = stats["total_turns"] / stats["total_sessions"]

        return stats

    def extract_training_data(self) -> List[Dict[str, Any]]:
        """
        Extract training examples from logged sessions with enhanced context.

        Returns:
            List of training examples with observation-action pairs and rich context
        """
        training_examples = []

        for session in self.sessions:
            env = session.get("environment", "unknown")
            agents_info = session.get("agents", {})

            for turn in session.get("turns", []):
                player_id = turn.get("player_id")
                observation = turn.get("observation")
                agent_state = turn.get("agent_state", {})
                action = turn.get("action")
                game_context = turn.get("game_context", {})

                # Get agent information
                agent_info = agents_info.get(str(player_id), {})
                model_name = agent_info.get("model_name", "unknown")
                agent_class = agent_info.get("agent_class", "unknown")

                # Create enhanced training example
                example = {
                    "session_id": session.get("session_id"),
                    "environment": env,
                    "player_id": player_id,
                    "model_name": model_name,
                    "agent_class": agent_class,
                    "turn_number": turn.get("turn_number"),
                    "timestamp": turn.get("timestamp"),

                    # Core training data
                    "observation": observation,
                    "action": action,

                    # Agent internal state
                    "agent_belief": agent_state.get("belief", ""),
                    "agent_strategy": agent_state.get("strategy", ""),
                    "agent_round": agent_state.get("round", 0),
                    "agent_is_initialized": agent_state.get("is_initialized", False),

                    # Game context for better training
                    "game_phase": game_context.get("phase"),
                    "game_round": game_context.get("round"),
                    "agent_role": game_context.get("agent_role"),
                    "agent_team": game_context.get("agent_team"),
                    "action_type": game_context.get("action_type"),
                    "is_voting_action": game_context.get("is_voting_action", False),
                    "is_speaking_action": game_context.get("is_speaking_action", False),

                    # Enhanced observation analysis
                    "observation_analysis": game_context.get("formatted_observation", {}),

                    # Session-level context
                    "total_players": session.get("num_players"),
                    "session_info": session.get("game_info", {})
                }

                training_examples.append(example)

        return training_examples

    def export_training_data(self, output_path: str, format: str = "json") -> None:
        """
        Export training data in specified format.

        Args:
            output_path: Path to save the training data
            format: Export format ("json", "csv", "parquet")
        """
        training_data = self.extract_training_data()

        if format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)

        elif format.lower() == "csv":
            # Flatten nested data for CSV
            flattened_data = []
            for example in training_data:
                flat_example = example.copy()
                flat_example["observation"] = str(flat_example["observation"])
                flat_example["agent_belief"] = str(flat_example["agent_belief"])
                flat_example["agent_strategy"] = str(flat_example["agent_strategy"])
                flattened_data.append(flat_example)

            df = pd.DataFrame(flattened_data)
            df.to_csv(output_path, index=False)

        elif format.lower() == "parquet":
            # Flatten nested data for Parquet
            flattened_data = []
            for example in training_data:
                flat_example = example.copy()
                flat_example["observation"] = str(flat_example["observation"])
                flat_example["agent_belief"] = str(flat_example["agent_belief"])
                flat_example["agent_strategy"] = str(flat_example["agent_strategy"])
                flattened_data.append(flat_example)

            df = pd.DataFrame(flattened_data)
            df.to_parquet(output_path, index=False)

        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Training data exported to {output_path} in {format} format")

    def analyze_agent_performance(self) -> Dict[str, Any]:
        """
        Analyze agent performance metrics.

        Returns:
            Dictionary with performance analysis
        """
        performance = defaultdict(lambda: {
            "games_played": 0,
            "wins": 0,
            "win_rate": 0.0,
            "roles_played": Counter(),
            "avg_turns_to_win": 0.0,
            "total_turns": 0
        })

        for session in self.sessions:
            agents_info = session.get("agents", {})
            results = session.get("results")
            if not results:
                continue
            rewards = results.get("rewards", {})
            if not rewards:
                continue
            total_turns = len(session.get("turns", []))

            for player_id, reward in rewards.items():
                agent_info = agents_info.get(str(player_id), {})
                model_name = agent_info.get("model_name", f"agent_{player_id}")

                # Update games played
                performance[model_name]["games_played"] += 1
                performance[model_name]["total_turns"] += total_turns

                # Track roles
                init_info = agent_info.get("init_info", {})
                if init_info and "role" in init_info:
                    performance[model_name]["roles_played"][init_info["role"]] += 1

                # Update wins (reward = 1 indicates win)
                if reward == 1:
                    performance[model_name]["wins"] += 1
                    performance[model_name]["avg_turns_to_win"] += total_turns

        # Calculate win rates and averages
        for model_name, stats in performance.items():
            if stats["games_played"] > 0:
                stats["win_rate"] = stats["wins"] / stats["games_played"]
                stats["avg_turns_per_game"] = stats["total_turns"] / stats["games_played"]

            if stats["wins"] > 0:
                stats["avg_turns_to_win"] = stats["avg_turns_to_win"] / stats["wins"]

        return dict(performance)

    def generate_training_report(self, output_path: str) -> None:
        """
        Generate a comprehensive training analysis report.

        Args:
            output_path: Path to save the report
        """
        if not self.sessions:
            print("No sessions to analyze")
            return

        report = {
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "total_sessions": len(self.sessions),
            "basic_stats": self.get_basic_stats(),
            "agent_performance": self.analyze_agent_performance(),
            "training_examples_count": len(self.extract_training_data()),
            "data_quality": self._assess_data_quality()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Training analysis report saved to {output_path}")

    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess the quality of logged data."""
        quality_metrics = {
            "sessions_with_complete_data": 0,
            "sessions_with_missing_agent_state": 0,
            "sessions_with_missing_observations": 0,
            "avg_agent_state_completeness": 0.0
        }

        total_turns = 0
        turns_with_agent_state = 0
        turns_with_observation = 0

        for session in self.sessions:
            has_complete_data = True

            # Check session completeness
            if not session.get("session_id") or not session.get("environment"):
                has_complete_data = False

            # Check turn completeness
            for turn in session.get("turns", []):
                total_turns += 1

                if turn.get("agent_state"):
                    turns_with_agent_state += 1
                else:
                    has_complete_data = False

                if turn.get("observation"):
                    turns_with_observation += 1
                else:
                    has_complete_data = False

            if has_complete_data:
                quality_metrics["sessions_with_complete_data"] += 1

        quality_metrics["sessions_with_missing_agent_state"] = len(self.sessions) - turns_with_agent_state
        quality_metrics["sessions_with_missing_observations"] = len(self.sessions) - turns_with_observation

        if total_turns > 0:
            quality_metrics["avg_agent_state_completeness"] = turns_with_agent_state / total_turns

        return quality_metrics

    def print_summary(self) -> None:
        """Print a summary of the analysis."""
        if not self.sessions:
            print("No sessions loaded")
            return

        stats = self.get_basic_stats()
        performance = self.analyze_agent_performance()

        print("\n" + "="*60)
        print("GAME LOG ANALYSIS SUMMARY")
        print("="*60)

        print(f"\nBasic Statistics:")
        print(f"  Total Sessions: {stats['total_sessions']}")
        print(f"  Environments: {dict(stats['environments'])}")
        print(f"  Average Turns per Game: {stats['avg_turns_per_game']:.1f}")

        print(f"\nAgent Performance:")
        for model_name, perf in performance.items():
            print(f"  {model_name}:")
            print(f"    Games: {perf['games_played']}, Wins: {perf['wins']}")
            print(f"    Win Rate: {perf['win_rate']*100:.1f}%")
            print(f"    Roles: {dict(perf['roles_played'])}")

        quality = self._assess_data_quality()
        print(f"\nData Quality:")
        print(f"  Complete Sessions: {quality['sessions_with_complete_data']}/{stats['total_sessions']}")
        print(f"  Agent State Completeness: {quality['avg_agent_state_completeness']*100:.1f}%")