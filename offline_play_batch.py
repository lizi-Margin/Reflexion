"""
Batch testing tool for offline games.
Runs multiple games and collects statistics.
"""

import sys
import io
from collections import defaultdict, Counter

from offline_play import create_agents, play_game


class GameStatistics:
    """Tracks and reports statistics across multiple games"""

    def __init__(self):
        self.total_games = 0
        self.player_stats = defaultdict(lambda: {
            "wins": 0,
            "games": 0,
            "total_reward": 0.0
        })
        self.game_results = []

    def update(self, rewards, game_info):
        """Update statistics with results from a single game"""
        self.total_games += 1
        self.game_results.append((rewards, game_info))

        for player_id, reward in rewards.items():
            self.player_stats[player_id]["games"] += 1
            self.player_stats[player_id]["total_reward"] += reward
            if reward > 0:
                self.player_stats[player_id]["wins"] += 1

    def print_summary(self):
        """Print comprehensive statistics summary"""
        print(f"\n{'='*60}")
        print(f"BATCH STATISTICS - {self.total_games} GAMES COMPLETED")
        print(f"{'='*60}")

        print(f"\nPer-Player Statistics:")
        for player_id in sorted(self.player_stats.keys()):
            stats = self.player_stats[player_id]
            win_rate = stats["wins"] / stats["games"] * 100 if stats["games"] > 0 else 0
            avg_reward = stats["total_reward"] / stats["games"] if stats["games"] > 0 else 0

            print(f"  Player {player_id}:")
            print(f"    Games: {stats['games']}")
            print(f"    Wins: {stats['wins']} ({win_rate:.1f}%)")
            print(f"    Avg Reward: {avg_reward:.3f}")

        print(f"\n{'='*60}")


def run_batch(env_id, npc_num, num_games, model_name='test', api_model_spec='qwen3-8b'):
    """
    Run multiple games and collect statistics.

    Args:
        env_id: Environment ID (e.g., "ThreePlayerIPD-v0-train")
        npc_num: Number of NPC agents
        num_games: Number of games to run
        model_name: Name for the main agent
        api_model_spec: API model specification

    Returns:
        GameStatistics object with all results
    """
    print(f"Starting batch test: {num_games} games of {env_id}")
    print(f"Model: {model_name} | API: {api_model_spec}")
    print(f"{'='*60}\n")

    stats = GameStatistics()

    for game_num in range(1, num_games + 1):
        print(f"\n[Game {game_num}/{num_games}]")

        try:
            # Create fresh agents for each game
            agents = create_agents(
                env_id=env_id,
                npc_num=npc_num,
                model_name=model_name,
                api_model_spec=api_model_spec,
                enable_logging=False
            )

            # Play game with minimal verbosity
            rewards, game_info = play_game(env_id=env_id, agents=agents, verbose=True)

            # Update statistics
            stats.update(rewards, game_info)

            # Print brief result
            print(f"  Rewards: {rewards}")

        except Exception as e:
            print(f"  ERROR in game {game_num}: {e}")
            continue

    # Print final summary
    stats.print_summary()

    return stats


def main():
    """Main entry point for batch testing"""

    # Configuration
    # env_id = "Codenames-v0-train"; npc_num = 3  # 2v2
    # env_id = "ColonelBlotto-v0-train"; npc_num = 1  # 1v1
    env_id = "ThreePlayerIPD-v0-train"; npc_num = 2  # 3 players

    NUM_GAMES = 5  # Number of games to run
    MODEL_NAME = 'test'
    API_MODEL_SPEC = 'qwen3-8b'

    # Run batch
    stats = run_batch(
        env_id=env_id,
        npc_num=npc_num,
        num_games=NUM_GAMES,
        model_name=MODEL_NAME,
        api_model_spec=API_MODEL_SPEC
    )


if __name__ == "__main__":
    main()
