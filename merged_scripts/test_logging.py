"""
Test script for game logging functionality.
"""

import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from .game_logger import GameLogger, LoggedAgent
from .log_analyzer import LogAnalyzer
from src.agent import LLMAgent
from corleone.agents.track1.micheal import Vito, Michael

def test_game_logger():
    """Test basic game logger functionality."""
    print("Testing Game Logger...")

    # Initialize logger
    logger = GameLogger(log_dir="test_logs", enabled=True)

    # Start a session
    session_id = logger.start_session(
        env_id="TestEnv-v0",
        agents={0: Michael(model_name="test-model")},
        num_players=1,
        additional_info={"test": True}
    )

    print(f"Started session: {session_id}")

    # Log some test turns
    test_observation = "System: You are Player 0. Your role: Villager. Team: Village."
    test_agent_state = {
        "belief": "All players are unknown.",
        "strategy": "Gather information first.",
        "round": 1
    }
    test_action = "I think we should discuss our strategies."

    logger.log_turn(
        player_id=0,
        observation=test_observation,
        agent_state=test_agent_state,
        action=test_action
    )

    # Log results
    test_rewards = {0: 1.0}
    test_game_info = {0: {"role": "Villager", "reason": "Game completed", "invalid_move": False}}

    logger.log_results(test_rewards, test_game_info)
    logger.end_session()

    print("✓ Game logger test completed")

def test_logged_agent():
    """Test logged agent wrapper."""
    print("\nTesting Logged Agent Wrapper...")

    # Initialize logger
    logger = GameLogger(log_dir="test_logs", enabled=True)

    # Create base agent
    base_agent = Michael(model_name="test-model")

    # Wrap with logging
    logged_agent = LoggedAgent(base_agent, logger, 0)

    # Test session
    session_id = logger.start_session(
        env_id="TestEnv-v0",
        agents={0: base_agent},
        num_players=1,
        additional_info={"test": True}
    )

    # Simulate an observation and action
    test_observation = "System: You are Player 0. Your role: Villager. Team: Village."

    # This should automatically log the turn
    try:
        action = logged_agent(test_observation)
        print(f"Agent action: {action}")
        print("✓ Logged agent test completed")
    except Exception as e:
        print(f"Note: Agent execution failed (expected if no API keys): {e}")
        print("✓ Logged agent wrapper structure test completed")

    logger.end_session()

def test_log_analyzer():
    """Test log analyzer functionality."""
    print("\nTesting Log Analyzer...")

    # Initialize analyzer
    analyzer = LogAnalyzer(log_dir="test_logs")

    # Load sessions
    analyzer.load_sessions()

    if analyzer.sessions:
        # Get basic stats
        stats = analyzer.get_basic_stats()
        print(f"Loaded {stats['total_sessions']} sessions")

        # Extract training data
        training_data = analyzer.extract_training_data()
        print(f"Extracted {len(training_data)} training examples")

        # Print summary
        analyzer.print_summary()

        print("✓ Log analyzer test completed")
    else:
        print("No sessions found to analyze")

def run_simple_game_test():
    """Run a simple game test with logging."""
    print("\nRunning Simple Game Test...")

    try:
        import textarena as ta

        # Initialize logger
        logger = GameLogger(log_dir="test_logs", enabled=True)

        # Create simple agents
        raw_agents = {
            0: LLMAgent(model_name="microsoft/DialoGPT-medium"),
            1: LLMAgent(model_name="microsoft/DialoGPT-medium")
        }

        # Wrap with logging
        agents = {}
        for player_id, agent in raw_agents.items():
            agents[player_id] = LoggedAgent(agent, logger, player_id)

        # Create environment (if available)
        try:
            env = ta.make(env_id="SecretMafia-v0")
        except:
            print("TextArena environment not available, skipping full game test")
            return

        # Start session
        session_id = logger.start_session(
            env_id="SecretMafia-v0",
            agents=raw_agents,
            num_players=2,
            additional_info={"test": True}
        )

        # Reset environment
        env.reset(num_players=2)

        # Run a few turns (or until error)
        max_turns = 5
        for turn in range(max_turns):
            try:
                player_id, observation = env.get_observation()
                action = agents[player_id](observation)
                done, step_info = env.step(action=action)

                print(f"Turn {turn + 1}: Player {player_id} acted")

                if done:
                    break
            except Exception as e:
                print(f"Game stopped at turn {turn + 1} due to: {e}")
                break

        # Get results and end session
        rewards, game_info = env.close()
        logger.log_results(rewards, game_info)
        logger.end_session()

        print("✓ Simple game test completed")

    except ImportError:
        print("TextArena not available, skipping game test")
    except Exception as e:
        print(f"Game test failed: {e}")

def cleanup_test_logs():
    """Clean up test logs."""
    import shutil
    if os.path.exists("test_logs"):
        shutil.rmtree("test_logs")
        print("\n✓ Test logs cleaned up")

def main():
    """Run all tests."""
    print("="*60)
    print("GAME LOGGING FUNCTIONALITY TESTS")
    print("="*60)

    # Run tests
    test_game_logger()
    test_logged_agent()
    test_log_analyzer()
    run_simple_game_test()

    # Final analysis
    print("\n" + "="*60)
    print("FINAL ANALYSIS")
    print("="*60)

    analyzer = LogAnalyzer(log_dir="test_logs")
    analyzer.load_sessions()

    if analyzer.sessions:
        # Generate training report
        analyzer.generate_training_report("test_logs/analysis_report.json")

        # Export sample training data
        training_data = analyzer.extract_training_data()
        if training_data:
            # Export first 10 examples as sample
            sample_data = training_data[:10]
            with open("test_logs/sample_training_data.json", 'w', encoding='utf-8') as f:
                import json
                json.dump(sample_data, f, indent=2, ensure_ascii=False)
            print("✓ Sample training data exported")

    # Cleanup
    cleanup_test_logs()

    print("\n✓ All tests completed!")

if __name__ == "__main__":
    main()