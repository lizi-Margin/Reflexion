"""
Simple test script for game logging functionality without API calls.
"""

import sys
import io
import os
import json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from src.game_logger import GameLogger
from src.log_analyzer import LogAnalyzer

def test_basic_logging():
    """Test basic logging functionality."""
    print("="*50)
    print("BASIC LOGGING FUNCTIONALITY TEST")
    print("="*50)

    # Initialize logger
    logger = GameLogger(log_dir="simple_test_logs", enabled=True)

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
                    "role": "Villager",
                    "team": "Village"
                }
            },
            "1": {
                "player_id": 1,
                "agent_class": "Vito",
                "model_name": "deepseek-r1",
                "init_info": {
                    "player_id": 1,
                    "role": "Mafia",
                    "team": "Mafia"
                }
            }
        },
        num_players=2,
        additional_info={
            "script": "simple_logging_test.py",
            "description": "Basic logging test"
        }
    )

    print(f"Started session: {session_id}")

    # Simulate a few game turns
    turns = [
        {
            "player_id": 0,
            "observation": "SYSTEM: Game started. You are Player 0. Your role: Villager. Team: Village.",
            "agent_state": {
                "belief": "=== PLAYER BELIEFS ===\nPlayer 0: Role: CONFIRMED as Villager | Status: ALIVE\nPlayer 1: Role: UNKNOWN | Status: ALIVE",
                "strategy": "Gather information about other players.",
                "round": 1
            },
            "action": "Hello everyone, I'm a regular villager. Let's work together to find the mafia."
        },
        {
            "player_id": 1,
            "observation": "SYSTEM: Player 0 said: Hello everyone, I'm a regular villager. Let's work together to find the mafia.",
            "agent_state": {
                "belief": "=== PLAYER BELIEFS ===\nPlayer 0: Role: SUSPECTED as Villager | Status: ALIVE\nPlayer 1: Role: CONFIRMED as Mafia | Status: ALIVE",
                "strategy": "Pretend to be a villager and blend in.",
                "round": 1
            },
            "action": "I agree! I'm also a villager. We need to be careful about who we trust."
        },
        {
            "player_id": 0,
            "observation": "SYSTEM: Player 1 said: I agree! I'm also a villager. We need to be careful about who we trust.",
            "agent_state": {
                "belief": "=== PLAYER BELIEFS ===\nPlayer 0: Role: CONFIRMED as Villager | Status: ALIVE\nPlayer 1: Role: SUSPECTED as Villager | Status: ALIVE",
                "strategy": "Continue gathering information before making accusations.",
                "round": 2
            },
            "action": "Good point. Let's hear from everyone else before we make any decisions."
        }
    ]

    # Log each turn
    for i, turn in enumerate(turns):
        logger.log_turn(
            player_id=turn["player_id"],
            observation=turn["observation"],
            agent_state=turn["agent_state"],
            action=turn["action"],
            step_info={"turn_number": i+1}
        )
        print(f"Logged turn {i+1}: Player {turn['player_id']} acted")

    # Log game results
    rewards = {"0": 1.0, "1": 0.0}  # Villagers win
    game_info = {
        "0": {"role": "Villager", "reason": "All mafia eliminated", "invalid_move": False},
        "1": {"role": "Mafia", "reason": "Eliminated by vote", "invalid_move": False}
    }

    logger.log_results(rewards, game_info)
    logger.end_session()

    print(f"✓ Session completed and logged: {session_id}")
    return session_id

def test_log_analysis():
    """Test log analysis functionality."""
    print("\n" + "="*50)
    print("LOG ANALYSIS TEST")
    print("="*50)

    # Initialize analyzer
    analyzer = LogAnalyzer(log_dir="simple_test_logs")

    # Load sessions
    analyzer.load_sessions()

    if not analyzer.sessions:
        print("No sessions found")
        return

    print(f"Loaded {len(analyzer.sessions)} sessions")

    # Get basic statistics
    stats = analyzer.get_basic_stats()
    print(f"\nBasic Statistics:")
    print(f"  Total sessions: {stats['total_sessions']}")
    print(f"  Environments: {dict(stats['environments'])}")
    print(f"  Agent models: {dict(stats['agent_models'])}")
    print(f"  Average turns per game: {stats['avg_turns_per_game']:.1f}")

    # Extract training data
    training_data = analyzer.extract_training_data()
    print(f"\nTraining Data:")
    print(f"  Total examples: {len(training_data)}")

    if training_data:
        # Show first example
        example = training_data[0]
        print(f"\nFirst training example:")
        print(f"  Environment: {example['environment']}")
        print(f"  Player ID: {example['player_id']}")
        print(f"  Model: {example['model_name']}")
        print(f"  Observation: {example['observation'][:100]}...")
        print(f"  Action: {example['action']}")

    # Generate and save analysis report
    analyzer.generate_training_report("simple_test_logs/analysis_report.json")
    print(f"\n✓ Analysis report saved")

    # Export training data
    analyzer.export_training_data("simple_test_logs/training_data.json", "json")
    print(f"✓ Training data exported")

    # Print summary
    analyzer.print_summary()

def inspect_log_file():
    """Inspect the generated log file."""
    print("\n" + "="*50)
    print("LOG FILE INSPECTION")
    print("="*50)

    log_files = list(os.path.join("simple_test_logs", f) for f in os.listdir("simple_test_logs") if f.endswith('.json'))

    if not log_files:
        print("No log files found")
        return

    # Read the first log file
    log_file = log_files[0]
    print(f"Inspecting: {log_file}")

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            session = json.load(f)

        print(f"\nSession Structure:")
        print(f"  Session ID: {session.get('session_id')}")
        print(f"  Environment: {session.get('environment')}")
        print(f"  Timestamp: {session.get('timestamp')}")
        print(f"  Number of agents: {len(session.get('agents', {}))}")
        print(f"  Number of turns: {len(session.get('turns', []))}")
        print(f"  Has results: {'results' in session}")

        # Show first turn structure
        turns = session.get('turns', [])
        if turns:
            print(f"\nFirst Turn Structure:")
            turn = turns[0]
            print(f"  Turn number: {turn.get('turn_number')}")
            print(f"  Player ID: {turn.get('player_id')}")
            print(f"  Has observation: {'observation' in turn}")
            print(f"  Has agent_state: {'agent_state' in turn}")
            print(f"  Has action: {'action' in turn}")

        # Show agent structure
        agents = session.get('agents', {})
        if agents:
            print(f"\nAgent Structure:")
            for agent_id, agent_info in agents.items():
                print(f"  Agent {agent_id}:")
                print(f"    Class: {agent_info.get('agent_class')}")
                print(f"    Model: {agent_info.get('model_name')}")
                print(f"    Has init_info: {'init_info' in agent_info}")

    except Exception as e:
        print(f"Error reading log file: {e}")

def cleanup():
    """Clean up test files."""
    import shutil
    if os.path.exists("simple_test_logs"):
        shutil.rmtree("simple_test_logs")
        print("\n✓ Test files cleaned up")

def main():
    """Run all tests."""
    try:
        # Test basic logging
        session_id = test_basic_logging()

        # Test log analysis
        test_log_analysis()

        # Inspect log file
        inspect_log_file()

        print("\n" + "="*50)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*50)

        # Keep files for inspection
        print(f"\nLog files saved in: simple_test_logs/")
        print("You can inspect the generated JSON files to see the data structure.")
        print("Run cleanup() manually when you're done inspecting.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()