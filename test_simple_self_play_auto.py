#!/usr/bin/env python3
"""
Test script for Simple Self-Play Training (Auto-run version)
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_self_play import SimpleSelfPlay

def test_simple_self_play():
    """Test the simple self-play with minimal configuration"""
    print("Testing Simple Self-Play with parallel processing...")

    # Create trainer with minimal games for testing
    trainer = SimpleSelfPlay(
        model_name="deepseek-v3.1",
        num_games=1,  # Just 1 game for testing
        max_workers=1,  # Start with sequential mode
        use_strategy_pool=True
    )

    # Run a quick test
    print("Running 1 test game in sequential mode...")
    try:
        trainer.run_training()
        print("Test completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_self_play()