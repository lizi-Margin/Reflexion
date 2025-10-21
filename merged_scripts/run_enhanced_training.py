#!/usr/bin/env python3
"""
Enhanced Self-Play Training Launcher
Easy launcher with preset configurations and command-line interface.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from self_play_training_enhanced import EnhancedSelfPlayTrainer, TrainingConfig, AgentConfig


def load_config_from_file(config_path: str) -> TrainingConfig:
    """Load training configuration from JSON file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    # Convert agent configs
    agent_configs = []
    for ac_data in config_data['agent_configs']:
        agent_configs.append(AgentConfig(**ac_data))

    config_data['agent_configs'] = agent_configs
    return TrainingConfig(**config_data)


def load_preset_config(preset_name: str) -> TrainingConfig:
    """Load preset configuration from training_configs.json"""
    configs_path = "training_configs.json"

    if not os.path.exists(configs_path):
        print(f"Configuration file {configs_path} not found!")
        print("Using default configuration.")
        return create_default_config()

    with open(configs_path, 'r', encoding='utf-8') as f:
        all_configs = json.load(f)

    if preset_name not in all_configs['training_presets']:
        print(f"Preset '{preset_name}' not found!")
        print(f"Available presets: {list(all_configs['training_presets'].keys())}")
        return create_default_config()

    config_data = all_configs['training_presets'][preset_name]
    print(f"Using preset: {config_data['description']}")

    # Convert agent configs
    agent_configs = []
    for ac_data in config_data['agent_configs']:
        agent_configs.append(AgentConfig(**ac_data))

    config_data['agent_configs'] = agent_configs
    return TrainingConfig(**config_data)


def create_default_config() -> TrainingConfig:
    """Create a basic default configuration"""
    return TrainingConfig(
        agent_configs=[
            AgentConfig("Michael", "qwen3-8b", temperature=0.4)
        ],
        agent_distribution=[1.0],
        games_per_session=20,
        max_workers=4
    )


def list_available_presets():
    """List all available training presets"""
    configs_path = "training_configs.json"

    if not os.path.exists(configs_path):
        print("No configuration file found. Available presets: None")
        return

    with open(configs_path, 'r', encoding='utf-8') as f:
        all_configs = json.load(f)

    print("\nAvailable Training Presets:")
    print("=" * 50)

    for name, config in all_configs['training_presets'].items():
        print(f"\n{name}:")
        print(f"  Description: {config['description']}")
        print(f"  Agent Types: {[ac['agent_type'] for ac in config['agent_configs']]}")
        print(f"  Models: {[ac['model_name'] for ac in config['agent_configs']]}")
        print(f"  Games per Session: {config['games_per_session']}")
        print(f"  Max Workers: {config['max_workers']}")


def show_agent_info():
    """Show information about available agent types"""
    configs_path = "training_configs.json"

    if not os.path.exists(configs_path):
        print("No configuration file found.")
        return

    with open(configs_path, 'r', encoding='utf-8') as f:
        all_configs = json.load(f)

    print("\nAgent Type Information:")
    print("=" * 50)

    for agent_type, info in all_configs['agent_type_info'].items():
        print(f"\n{agent_type}:")
        print(f"  Description: {info['description']}")
        print(f"  Features: {', '.join(info['features'])}")
        print(f"  Recommended Temperature: {info['recommended_temperature']}")
        print(f"  Strategy Pool Compatible: {info['strategy_pool_compatible']}")


def show_model_info():
    """Show information about available models"""
    configs_path = "training_configs.json"

    if not os.path.exists(configs_path):
        print("No configuration file found.")
        return

    with open(configs_path, 'r', encoding='utf-8') as f:
        all_configs = json.load(f)

    print("\nModel Information:")
    print("=" * 50)

    for model, info in all_configs['model_info'].items():
        print(f"\n{model}:")
        print(f"  Description: {info['description']}")
        print(f"  Recommended Temperature: {info['recommended_temperature']}")
        print(f"  Context Length: {info['context_length']}")
        print(f"  Cost Efficiency: {info['cost_efficiency']}")


def show_training_tips():
    """Show training recommendations"""
    configs_path = "training_configs.json"

    if not os.path.exists(configs_path):
        print("No configuration file found.")
        return

    with open(configs_path, 'r', encoding='utf-8') as f:
        all_configs = json.load(f)

    print("\nTraining Recommendations:")
    print("=" * 50)

    for level, info in all_configs['training_tips'].items():
        print(f"\n{level.upper()}:")
        print(f"  Recommended Preset: {info['recommended_preset']}")
        print(f"  Description: {info['description']}")
        print(f"  Suggested Parameters: {info['parameters']}")


def create_custom_config_interactive() -> TrainingConfig:
    """Interactive configuration creator"""
    print("\nInteractive Configuration Creator")
    print("=" * 40)

    # Get basic settings
    print("\n1. Basic Settings:")
    games_per_session = int(input("Games per session (default 20): ") or "20")
    max_workers = int(input("Maximum workers (default 4): ") or "4")
    min_players = int(input("Minimum players per game (default 5): ") or "5")
    max_players = int(input("Maximum players per game (default 8): ") or "8")

    # Get agent configurations
    print("\n2. Agent Configuration:")
    agent_configs = []
    agent_distribution = []

    while True:
        print(f"\nAgent {len(agent_configs) + 1}:")
        agent_type = input("Agent type (Michael/Vito, default Michael): ") or "Michael"
        model_name = input("Model name (qwen3-8b/deepseek-r1/qwen3-4b, default qwen3-8b): ") or "qwen3-8b"
        temperature = float(input("Temperature (0.1-1.0, default 0.4): ") or "0.4")

        strategy_pool_enabled = input("Enable strategy pool (y/n, default y): ").lower() != 'n'

        agent_config = AgentConfig(
            agent_type=agent_type,
            model_name=model_name,
            temperature=temperature,
            strategy_pool_enabled=strategy_pool_enabled
        )
        agent_configs.append(agent_config)
        agent_distribution.append(1.0)  # Equal distribution initially

        another = input("Add another agent? (y/n): ").lower()
        if another != 'y':
            break

    # Normalize distribution
    total = sum(agent_distribution)
    agent_distribution = [p/total for p in agent_distribution]

    # Create configuration
    config = TrainingConfig(
        agent_configs=agent_configs,
        agent_distribution=agent_distribution,
        min_players_per_game=min_players,
        max_players_per_game=max_players,
        games_per_session=games_per_session,
        max_workers=max_workers,
        log_dir=f"custom_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Save configuration
    save_path = f"custom_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(save_path, 'w') as f:
        config_dict = {
            'agent_configs': [asdict(ac) for ac in config.agent_configs],
            'agent_distribution': config.agent_distribution,
            'min_players_per_game': config.min_players_per_game,
            'max_players_per_game': config.max_players_per_game,
            'games_per_session': config.games_per_session,
            'max_workers': config.max_workers,
            'log_dir': config.log_dir
        }
        json.dump(config_dict, f, indent=2)

    print(f"\nConfiguration saved to: {save_path}")
    return config


def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Enhanced Self-Play Training Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --preset michael_models            # Use Michael with different models
  %(prog)s --preset mixed_agents               # Use Michael and Vito agents
  %(prog)s --config my_config.json             # Use custom configuration file
  %(prog)s --interactive                      # Create configuration interactively
  %(prog)s --list-presets                     # Show available presets
        """
    )

    # Configuration options
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--preset", help="Use preset configuration")
    config_group.add_argument("--config", help="Path to custom configuration file")
    config_group.add_argument("--interactive", action="store_true", help="Create configuration interactively")

    # Training parameters
    parser.add_argument("--num-batches", type=int, default=5, help="Number of training batches")
    parser.add_argument("--games-per-batch", type=int, help="Override games per batch")
    parser.add_argument("--workers", type=int, help="Override number of workers")

    # Paths and options
    parser.add_argument("--strategy-pool", default="strategy_pool.json", help="Strategy pool file")
    parser.add_argument("--log-dir", help="Override log directory")
    parser.add_argument("--no-strategy-learning", action="store_true", help="Disable strategy learning")

    # Information options
    parser.add_argument("--list-presets", action="store_true", help="List available presets")
    parser.add_argument("--agent-info", action="store_true", help="Show agent information")
    parser.add_argument("--model-info", action="store_true", help="Show model information")
    parser.add_argument("--training-tips", action="store_true", help="Show training recommendations")

    args = parser.parse_args()

    # Handle information requests
    if args.list_presets:
        list_available_presets()
        return
    elif args.agent_info:
        show_agent_info()
        return
    elif args.model_info:
        show_model_info()
        return
    elif args.training_tips:
        show_training_tips()
        return

    # Get configuration
    if args.interactive:
        config = create_custom_config_interactive()
    elif args.config:
        config = load_config_from_file(args.config)
    elif args.preset:
        config = load_preset_config(args.preset)
    else:
        print("No configuration specified. Using default.")
        config = create_default_config()

    # Override with command line arguments
    if args.games_per_batch:
        config.games_per_session = args.games_per_batch
    if args.workers:
        config.max_workers = args.workers
    if args.log_dir:
        config.log_dir = args.log_dir
    if args.strategy_pool:
        config.strategy_pool_path = args.strategy_pool
    if args.no_strategy_learning:
        config.enable_strategy_learning = False

    # Show configuration summary
    print("\nConfiguration Summary:")
    print("=" * 30)
    print(f"Agent Types: {[ac.agent_type for ac in config.agent_configs]}")
    print(f"Models: {[ac.model_name for ac in config.agent_configs]}")
    print(f"Games per Session: {config.games_per_session}")
    print(f"Max Workers: {config.max_workers}")
    print(f"Strategy Learning: {config.enable_strategy_learning}")
    print(f"Log Directory: {config.log_dir}")

    # Confirm before starting
    confirm = input("\nStart training with this configuration? (y/n): ").lower()
    if confirm != 'y':
        print("Training cancelled.")
        return

    # Initialize and run trainer
    print("\n" + "=" * 60)
    print("STARTING ENHANCED SELF-PLAY TRAINING")
    print("=" * 60)

    try:
        trainer = EnhancedSelfPlayTrainer(config)

        final_report = trainer.run_training_session(
            num_batches=args.num_batches
        )

        print("\nTraining completed successfully!")

        # Show results
        if 'agent_type_averages' in final_report:
            print("\nFinal Results:")
            for agent_type, stats in final_report['agent_type_averages'].items():
                print(f"  {agent_type}: {stats['average_win_rate']:.3f} win rate "
                      f"({stats['total_games']} games)")

        # Save final report
        report_path = os.path.join(config.log_dir, "final_training_report.json")
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        print(f"Final report saved to: {report_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()