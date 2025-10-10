"""
This script allows you to play against a fixed model.
You (human player) will be player 0, and the AI model will be player 1.
"""

import textarena as ta
from src.agent import LLMAgent
from xushuhang.agents.family import Vito, Michael
import sys
import io
import os
import json
from collections import defaultdict, Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

MODEL_NAME = "Vito-1.3" # Replace with your model name

# Game statistics tracking
game_stats = {
    "total_games": 0,
    "mafia_wins": 0,
    "villager_wins": 0,
    "role_stats": defaultdict(lambda: {"wins": 0, "games": 0, "win_rate": 0.0}),
    "reasons": Counter(),
    "invalid_moves": 0,
    "agent_stats": defaultdict(lambda: {"wins": 0, "games": 0, "win_rate": 0.0, "roles": Counter()})  # 添加agent统计
}

def initialize_agents():
    """Initialize agents for a new game"""
    return {
        0: Vito(model_name=MODEL_NAME),
        1: Vito(model_name="bsl1"),
        2: Michael(model_name="bsl2"),
        3: Vito(model_name="bsl3"),
        4: Vito(model_name="bsl4"),
        5: Vito(model_name="bsl5"),
    }

def get_agent_name(player_id):
    """Get agent name based on player ID"""
    agent_names = {
        0: MODEL_NAME,
        1: "bsl1",
        2: "bsl2",
        3: "bsl3",
        4: "bsl4",
        5: "bsl5"
    }
    return agent_names.get(player_id, f"agent_{player_id}")

def play_single_game(game_id):
    """Play a single game and return results"""
    print(f"\n{'='*50}")
    print(f"Starting Game {game_id}")
    print(f"{'='*50}")
    
    # Initialize agents for this game
    agents = initialize_agents()
    
    # Initialize the environment
    env = ta.make(env_id="SecretMafia-v0")
    env.reset(num_players=len(agents))
    
    # Main game loop
    done = False 
    while not done:
        player_id, observation = env.get_observation()
        action = agents[player_id](observation)
        done, step_info = env.step(action=action)
    
    rewards, game_info = env.close()
    
    print(f"Game {game_id} completed.")
    print(f"Rewards: {rewards}")
    print(f"Game Info: {game_info}")
    
    return rewards, game_info, agents

def update_statistics(rewards, game_info, agents):
    """Update game statistics based on game results"""
    game_stats["total_games"] += 1
    
    # Count wins by team
    mafia_win = any(info["role"] in ["Mafia"] and reward == 1 
                   for info, reward in zip(game_info.values(), rewards.values()))
    
    if mafia_win:
        game_stats["mafia_wins"] += 1
    else:
        game_stats["villager_wins"] += 1
    
    # Track reasons for game end
    for player_info in game_info.values():
        game_stats["reasons"][player_info["reason"]] += 1
        if player_info["invalid_move"]:
            game_stats["invalid_moves"] += 1
    
    # Track role-based statistics
    for player_id, info in game_info.items():
        role = info["role"]
        reward = rewards[player_id]
        
        game_stats["role_stats"][role]["games"] += 1
        if reward == 1:  # Win
            game_stats["role_stats"][role]["wins"] += 1
        
        # Update win rate
        games = game_stats["role_stats"][role]["games"]
        wins = game_stats["role_stats"][role]["wins"]
        game_stats["role_stats"][role]["win_rate"] = wins / games if games > 0 else 0.0
        
        # Track agent-based statistics
        agent_name = get_agent_name(player_id)
        game_stats["agent_stats"][agent_name]["games"] += 1
        game_stats["agent_stats"][agent_name]["roles"][role] += 1
        if reward == 1:  # Win
            game_stats["agent_stats"][agent_name]["wins"] += 1
        
        # Update agent win rate
        agent_games = game_stats["agent_stats"][agent_name]["games"]
        agent_wins = game_stats["agent_stats"][agent_name]["wins"]
        game_stats["agent_stats"][agent_name]["win_rate"] = agent_wins / agent_games if agent_games > 0 else 0.0

def print_statistics():
    """Print comprehensive game statistics"""
    print(f"\n{'='*60}")
    print(f"FINAL STATISTICS AFTER {game_stats['total_games']} GAMES")
    print(f"{'='*60}")
    
    # Overall win rates
    print(f"\nOverall Results:")
    print(f"  Total Games: {game_stats['total_games']}")
    print(f"  Mafia Wins: {game_stats['mafia_wins']} "
          f"({game_stats['mafia_wins']/game_stats['total_games']*100:.1f}%)")
    print(f"  Villager Wins: {game_stats['villager_wins']} "
          f"({game_stats['villager_wins']/game_stats['total_games']*100:.1f}%)")
    
    # Agent-based statistics
    print(f"\nAgent-based Statistics:")
    for agent_name, stats in game_stats["agent_stats"].items():
        print(f"  {agent_name}: {stats['wins']}/{stats['games']} "
              f"({stats['win_rate']*100:.1f}%)")
        print(f"    Roles played: {dict(stats['roles'])}")
    
    # Role-based statistics
    print(f"\nRole-based Statistics:")
    for role, stats in game_stats["role_stats"].items():
        print(f"  {role}: {stats['wins']}/{stats['games']} "
              f"({stats['win_rate']*100:.1f}%)")
    
    # Game end reasons
    print(f"\nGame End Reasons:")
    for reason, count in game_stats["reasons"].most_common():
        print(f"  {reason}: {count}")
    
    # Invalid moves
    print(f"\nInvalid Moves: {game_stats['invalid_moves']}")

def main():
    """Main function to run multiple games"""
    NUM_GAMES = 5  # Change this to the number of games you want to run
    
    print(f"Starting {NUM_GAMES} games of Secret Mafia...")
    
    # Play multiple games
    for game_id in range(1, NUM_GAMES + 1):
        try:
            rewards, game_info, agents = play_single_game(game_id)
            update_statistics(rewards, game_info, agents)
        except Exception as e:
            print(f"Error in game {game_id}: {e}")
            continue
    
    # Print final statistics
    print_statistics()

if __name__ == "__main__":
    main()