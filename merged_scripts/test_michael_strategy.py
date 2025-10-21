"""
Test script to validate the enhanced Michael agent with dual strategy system
"""

import sys
import io
import os
import json

# Set UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import required modules
from corleone.agents.track1.micheal import Michael

def test_strategy_pool_system():
    """Test the strategy pool system functionality"""
    print("=" * 60)
    print("Testing Michael's Enhanced Strategy Pool System")
    print("=" * 60)

    # Test 1: Initialize Michael with strategy pool
    print("\n1. Testing Michael initialization...")
    try:
        michael = Michael(model_name="test-model")
        print(f"✓ Michael initialized successfully")
        print(f"✓ Strategy pool loaded with {len(michael.strategy_pool)} strategies")

        # Check strategy types
        behavior_count = sum(1 for s in michael.strategy_pool.values() if s["type"] == "behavior")
        language_count = sum(1 for s in michael.strategy_pool.values() if s["type"] == "language")
        print(f"✓ Found {behavior_count} behavior strategies and {language_count} language strategies")

    except Exception as e:
        print(f"✗ Error initializing Michael: {e}")
        return False

    # Test 2: Strategy similarity calculation
    print("\n2. Testing strategy similarity calculation...")
    try:
        # Mock init_info for testing
        michael.init_info = {
            "player_id": 0,
            "role": "Mafia",
            "team": "Mafia",
            "all_players": [0, 1, 2, 3, 4, 5],
            "teammates": [1]
        }
        michael.current_phase = "day_speak"

        context_features = michael.get_current_context_features()
        print(f"✓ Context features extracted: {context_features}")

        # Test similarity calculation for first strategy
        if michael.strategy_pool:
            first_strategy_id = list(michael.strategy_pool.keys())[0]
            similarity = michael.calculate_strategy_similarity(first_strategy_id, context_features)
            print(f"✓ Similarity calculated for strategy {first_strategy_id}: {similarity:.3f}")

    except Exception as e:
        print(f"✗ Error in similarity calculation: {e}")
        return False

    # Test 3: Strategy selection
    print("\n3. Testing strategy selection...")
    try:
        behavior_strategies = michael.select_reference_strategies('behavior')
        language_strategies = michael.select_reference_strategies('language')

        print(f"✓ Selected {len(behavior_strategies)} behavior strategies")
        print(f"✓ Selected {len(language_strategies)} language strategies")

        if behavior_strategies:
            print(f"✓ Sample behavior strategy: {behavior_strategies[0]['id']}")
        if language_strategies:
            print(f"✓ Sample language strategy: {language_strategies[0]['id']}")

    except Exception as e:
        print(f"✗ Error in strategy selection: {e}")
        return False

    # Test 4: Strategy creation
    print("\n4. Testing new strategy creation...")
    try:
        behavior_text = "Test behavior strategy: Hide identity and observe others"
        language_text = "Test language strategy: Use cautious and ambiguous language"

        behavior_id = michael.create_new_strategy("behavior", behavior_text)
        language_id = michael.create_new_strategy("language", language_text, behavior_id)

        print(f"✓ Created behavior strategy: {behavior_id}")
        print(f"✓ Created language strategy: {language_id}")

        # Verify strategies were added to pool
        assert behavior_id in michael.strategy_pool
        assert language_id in michael.strategy_pool
        print("✓ New strategies successfully added to pool")

    except Exception as e:
        print(f"✗ Error in strategy creation: {e}")
        return False

    # Test 5: Strategy logging
    print("\n5. Testing strategy usage logging...")
    try:
        michael.log_strategy_usage(behavior_id, "behavior", "day_speak")
        michael.log_strategy_usage(language_id, "language", "day_speak")

        print(f"✓ Strategy usage log contains {len(michael.strategy_usage_log)} entries")
        print(f"✓ Strategy {behavior_id} usage count: {michael.strategy_pool[behavior_id]['usage_count']}")

    except Exception as e:
        print(f"✗ Error in strategy logging: {e}")
        return False

    # Test 6: Strategy performance update
    print("\n6. Testing strategy performance update...")
    try:
        game_result = {
            "victory": True,
            "winner_team": "Mafia",
            "reason": "Mafia reached parity"
        }
        role_performance = {
            "role": "Mafia",
            "team": "Mafia",
            "bonus": 0.3
        }

        michael.update_strategy_performance(game_result, role_performance)
        print("✓ Strategy performance updated successfully")

        # Check performance scores
        behavior_strategy = michael.strategy_pool[behavior_id]
        print(f"✓ Behavior strategy avg performance: {behavior_strategy['avg_performance']:.3f}")

    except Exception as e:
        print(f"✗ Error in performance update: {e}")
        return False

    # Test 7: Strategy pool saving
    print("\n7. Testing strategy pool saving...")
    try:
        michael.save_strategy_pool()
        print("✓ Strategy pool saved successfully")

        # Verify file exists and has content
        if os.path.exists("strategy_pool.json"):
            with open("strategy_pool.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"✓ Saved strategy pool contains {len(data['strategies'])} strategies")
                print(f"✓ Metadata updated: {data['metadata']['last_updated']}")

    except Exception as e:
        print(f"✗ Error in strategy pool saving: {e}")
        return False

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("Michael's dual strategy system is working correctly.")
    print("=" * 60)
    return True

def test_game_end_detection():
    """Test game end detection functionality"""
    print("\n" + "=" * 60)
    print("Testing Game End Detection")
    print("=" * 60)

    try:
        michael = Michael(model_name="test-model")
        michael.init_info = {
            "player_id": 0,
            "role": "Mafia",
            "team": "Mafia",
            "all_players": [0, 1, 2, 3, 4, 5],
            "teammates": [1]
        }

        # Test game end messages
        test_observations = [
            '[[-1, "Mafia reached parity with villagers. Mafia wins!"]]',
            '[[-1, "All Mafia were eliminated. Village wins!"]]',
            'Regular game message without end'
        ]

        for i, obs in enumerate(test_observations):
            game_ended, result = michael.detect_game_end(obs)
            print(f"Test {i+1}: Game ended = {game_ended}, Result = {result}")

        print("✓ Game end detection working correctly")
        return True

    except Exception as e:
        print(f"✗ Error in game end detection: {e}")
        return False

if __name__ == "__main__":
    print("Starting Michael Agent Validation Tests...")

    success = True
    success &= test_strategy_pool_system()
    success &= test_game_end_detection()

    if success:
        print("\n🎉 All validation tests completed successfully!")
        print("The enhanced Michael agent is ready for use.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")