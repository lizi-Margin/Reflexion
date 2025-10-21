"""
Test script for enhanced game logging functionality.
"""

import sys
import io
import os
import json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from src.game_logger import GameLogger
from src.log_analyzer import LogAnalyzer

def test_enhanced_logging():
    """Test enhanced logging with game context."""
    print("="*60)
    print("ENHANCED GAME LOGGING TEST")
    print("="*60)

    # Initialize logger
    logger = GameLogger(log_dir="enhanced_test_logs", enabled=True)

    # Start a session
    session_id = logger.start_session(
        env_id="SecretMafia-v0",
        agents={
            "0": {
                "player_id": 0,
                "agent_class": "Michael",
                "model_name": "qwen3-8b",
                "init_info": {
                    "player_id": 0,
                    "role": "Mafia",
                    "team": "Mafia"
                }
            },
            "1": {
                "player_id": 1,
                "agent_class": "Vito",
                "model_name": "deepseek-r1",
                "init_info": {
                    "player_id": 1,
                    "role": "Villager",
                    "team": "Village"
                }
            }
        },
        num_players=2,
        additional_info={
            "test": "enhanced_logging",
            "description": "Testing enhanced logging with game context"
        }
    )

    print(f"Started enhanced session: {session_id}")

    # Simulate game turns with different phases
    turns = [
        {
            "player_id": 0,
            "observation": "SYSTEM: Game started. You are Player 0. Your role: Mafia. Team: Mafia. The game will start at night.",
            "agent_state": {
                "belief": "=== PLAYER BELIEFS ===\nPlayer 0: Role: CONFIRMED as Mafia | Status: ALIVE\nPlayer 1: Role: UNKNOWN | Status: ALIVE",
                "strategy": "Pretend to be a villager and blend in during the day.",
                "round": 0,
                "is_initialized": True,
                "init_identity": "=== YOUR IDENTITY ===\nPlayer ID: Player 0\nRole: Mafia\nTeam: Mafia"
            },
            "action": "[1]",
            "game_phase": "night"
        },
        {
            "player_id": 1,
            "observation": "SYSTEM: Night phase begins. Player 0 was eliminated during the night.",
            "agent_state": {
                "belief": "=== PLAYER BELIEFS ===\nPlayer 0: Role: SUSPECTED as Mafia | Status: DEAD\nPlayer 1: Role: CONFIRMED as Villager | Status: ALIVE",
                "strategy": "Investigate remaining players to find mafia members.",
                "round": 1,
                "is_initialized": True
            },
            "action": "I think we need to be more careful. Someone was eliminated last night.",
            "game_phase": "day_speak"
        },
        {
            "player_id": 1,
            "observation": "SYSTEM: Voting phase begins. Please vote for who you think is the mafia.",
            "agent_state": {
                "belief": "=== PLAYER BELIEFS ===\nPlayer 0: Role: SUSPECTED as Mafia | Status: DEAD\nPlayer 1: Role: CONFIRMED as Villager | Status: ALIVE",
                "strategy": "Vote based on available evidence.",
                "round": 4,
                "is_initialized": True
            },
            "action": "Based on the available information, I vote for [0].",
            "game_phase": "day_vote"
        }
    ]

    # Log each turn with enhanced context
    for i, turn in enumerate(turns):
        logger.log_turn(
            player_id=turn["player_id"],
            observation=turn["observation"],
            agent_state=turn["agent_state"],
            action=turn["action"],
            game_phase=turn.get("game_phase"),
            round_number=turn["agent_state"].get("round")
        )
        print(f"Logged enhanced turn {i+1}: Player {turn['player_id']} in {turn.get('game_phase', 'unknown')} phase")

    # Log game results
    rewards = {"0": 0.0, "1": 1.0}  # Villagers win
    game_info = {
        "0": {"role": "Mafia", "reason": "Eliminated by vote", "invalid_move": False},
        "1": {"role": "Villager", "reason": "All mafia eliminated", "invalid_move": False}
    }

    logger.log_results(rewards, game_info)
    logger.end_session()

    print(f"✓ Enhanced session completed: {session_id}")
    return session_id

def test_enhanced_analysis():
    """Test analysis of enhanced logging data."""
    print("\n" + "="*60)
    print("ENHANCED DATA ANALYSIS TEST")
    print("="*60)

    # Initialize analyzer
    analyzer = LogAnalyzer(log_dir="enhanced_test_logs")

    # Load sessions
    analyzer.load_sessions()

    if not analyzer.sessions:
        print("No sessions found")
        return

    print(f"Loaded {len(analyzer.sessions)} enhanced sessions")

    # Extract training data
    training_data = analyzer.extract_training_data()
    print(f"Extracted {len(training_data)} enhanced training examples")

    if training_data:
        # Show first example with enhanced context
        example = training_data[0]
        print(f"\nFirst Enhanced Training Example:")
        print(f"  Environment: {example['environment']}")
        print(f"  Player ID: {example['player_id']}")
        print(f"  Agent Role: {example['agent_role']}")
        print(f"  Game Phase: {example['game_phase']}")
        print(f"  Action Type: {example['action_type']}")
        print(f"  Is Voting Action: {example['is_voting_action']}")
        print(f"  Is Speaking Action: {example['is_speaking_action']}")
        print(f"  Agent Round: {example['agent_round']}")
        print(f"  Observation Length: {example['observation_analysis'].get('length', 0)}")
        print(f"  Contains System Message: {example['observation_analysis'].get('contains_system_message', False)}")

    # Export enhanced training data
    analyzer.export_training_data("enhanced_test_logs/enhanced_training_data.json", "json")
    print(f"\n✓ Enhanced training data exported")

    # Generate analysis report
    analyzer.generate_training_report("enhanced_test_logs/enhanced_analysis_report.json")
    print(f"✓ Enhanced analysis report generated")

def inspect_enhanced_log():
    """Inspect the enhanced log file structure."""
    print("\n" + "="*60)
    print("ENHANCED LOG FILE INSPECTION")
    print("="*60)

    log_files = list(os.path.join("enhanced_test_logs", f) for f in os.listdir("enhanced_test_logs") if f.endswith('.json'))

    if not log_files:
        print("No enhanced log files found")
        return

    # Read the first enhanced log file
    log_file = log_files[0]
    print(f"Inspecting enhanced log: {log_file}")

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            session = json.load(f)

        # Show enhanced structure
        print(f"\nEnhanced Session Structure:")
        print(f"  Session ID: {session.get('session_id')}")
        print(f"  Environment: {session.get('environment')}")
        print(f"  Number of turns: {len(session.get('turns', []))}")

        turns = session.get('turns', [])
        if turns:
            print(f"\nEnhanced Turn Structure (first turn):")
            turn = turns[0]
            print(f"  Turn number: {turn.get('turn_number')}")
            print(f"  Player ID: {turn.get('player_id')}")
            print(f"  Has game_context: {'game_context' in turn}")

            if 'game_context' in turn:
                ctx = turn['game_context']
                print(f"  Game Phase: {ctx.get('phase')}")
                print(f"  Agent Role: {ctx.get('agent_role')}")
                print(f"  Action Type: {ctx.get('action_type')}")
                print(f"  Is Voting: {ctx.get('is_voting_action')}")
                print(f"  Is Speaking: {ctx.get('is_speaking_action')}")

            print(f"  Agent State Keys: {list(turn.get('agent_state', {}).keys())}")

    except Exception as e:
        print(f"Error reading enhanced log file: {e}")

def cleanup():
    """Clean up test files."""
    import shutil
    if os.path.exists("enhanced_test_logs"):
        shutil.rmtree("enhanced_test_logs")
        print("\n✓ Enhanced test files cleaned up")

def main():
    """Run enhanced logging tests."""
    try:
        # Test enhanced logging
        session_id = test_enhanced_logging()

        # Test enhanced analysis
        test_enhanced_analysis()

        # Inspect enhanced log structure
        inspect_enhanced_log()

        print("\n" + "="*60)
        print("✓ ALL ENHANCED LOGGING TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)

        print("\nEnhanced features now available:")
        print("  ✓ Game phase detection (night, day_speak, day_vote)")
        print("  ✓ Agent role and team tracking")
        print("  ✓ Action type classification (vote, speech, etc.)")
        print("  ✓ Enhanced observation analysis")
        print("  ✓ Rich training data context")
        print("  ✓ Better data for role-specific training")

        # Keep files for inspection
        print(f"\nEnhanced log files saved in: enhanced_test_logs/")
        print("You can inspect the enhanced JSON files to see the new structure.")
        print("Run cleanup() manually when you're done inspecting.")

    except Exception as e:
        print(f"\n❌ Enhanced test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()