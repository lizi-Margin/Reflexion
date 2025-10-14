# Game Logging Guide

This guide explains how to use the game logging functionality to record game sessions for training and analysis purposes.

## Overview

The game logging system captures:
- Game metadata (environment, players, timestamps)
- Agent states (beliefs, strategies, rounds)
- Turn-by-turn observations and actions
- Game results and outcomes

## Components

### 1. GameLogger (`src/game_logger.py`)

Main logging class that:
- Manages game sessions
- Records turns and results
- Saves data in JSON format
- Provides session statistics

### 2. LoggedAgent (`src/game_logger.py`)

Wrapper class that:
- Automatically logs agent actions
- Extracts agent internal state
- Preserves original agent behavior

### 3. LogAnalyzer (`src/log_analyzer.py`)

Analysis tool that:
- Processes logged sessions
- Extracts training data
- Generates performance reports
- Exports data in multiple formats

## Quick Start

### 1. Enable Logging in Offline Testing

Edit your offline testing script and set:
```python
ENABLE_LOGGING = True
LOG_DIR = "game_logs"
```

### 2. Run Games with Logging

```python
# Simple example
from src.game_logger import GameLogger, LoggedAgent
from xushuhang.agents.family import Michael

# Initialize logger
logger = GameLogger(log_dir="game_logs", enabled=True)

# Create agents
raw_agents = {0: Michael(model_name="qwen3-8b"), 1: Michael(model_name="deepseek-r1")}

# Wrap with logging
agents = {}
for player_id, agent in raw_agents.items():
    agents[player_id] = LoggedAgent(agent, logger, player_id)

# Start session
session_id = logger.start_session(
    env_id="SecretMafia-v0",
    agents=raw_agents,
    num_players=len(agents)
)

# Play game (your normal game loop)
# ...

# Log results
logger.log_results(rewards, game_info)
logger.end_session()
```

### 3. Analyze Logged Data

```python
from src.log_analyzer import LogAnalyzer

# Initialize analyzer
analyzer = LogAnalyzer(log_dir="game_logs")

# Load all sessions
analyzer.load_sessions()

# Print summary statistics
analyzer.print_summary()

# Extract training data
training_data = analyzer.extract_training_data()

# Export for training
analyzer.export_training_data("training_data.json", "json")

# Generate comprehensive report
analyzer.generate_training_report("analysis_report.json")
```

## Data Structure

### Session Format
```json
{
  "session_id": "SecretMafia-v0_20251014_131543",
  "timestamp": "2025-10-14T13:15:43.695531",
  "environment": "SecretMafia-v0",
  "num_players": 2,
  "agents": {
    "0": {
      "player_id": "0",
      "agent_class": "Michael",
      "model_name": "qwen3-8b",
      "init_info": {...}
    }
  },
  "game_info": {...},
  "turns": [
    {
      "turn_number": 1,
      "timestamp": "2025-10-14T13:15:43.695531",
      "player_id": 0,
      "observation": "...",
      "agent_state": {
        "belief": "...",
        "strategy": "...",
        "round": 1
      },
      "action": "...",
      "step_info": {...}
    }
  ],
  "results": {
    "rewards": {"0": 1.0, "1": 0.0},
    "game_info": {...},
    "timestamp": "2025-10-14T13:15:43.695531"
  },
  "summary": {...}
}
```

### Training Example Format
```json
{
  "session_id": "SecretMafia-v0_20251014_131543",
  "environment": "SecretMafia-v0",
  "player_id": 0,
  "model_name": "qwen3-8b",
  "agent_class": "Michael",
  "turn_number": 1,
  "observation": "SYSTEM: You are Player 0...",
  "agent_belief": "=== PLAYER BELIEFS ===...",
  "agent_strategy": "Gather information...",
  "agent_round": 1,
  "action": "Hello everyone...",
  "timestamp": "2025-10-14T13:15:43.695531"
}
```

## Usage Examples

### 1. Batch Testing with Logging

Run multiple games and automatically log all sessions:

```bash
python offline_play_batch.py
```

This will:
- Run 5 games by default (modify NUM_GAMES)
- Log each session to `game_logs/`
- Print statistics and logging summary

### 2. Single Game Testing

Run one game with detailed logging:

```bash
python offline_play.py
```

### 3. Custom Logging Analysis

```python
# Load specific sessions
analyzer = LogAnalyzer(log_dir="game_logs")
analyzer.load_sessions()

# Filter by environment
mafia_sessions = [s for s in analyzer.sessions
                  if s.get("environment") == "SecretMafia-v0"]

# Extract specific patterns
training_examples = []
for session in mafia_sessions:
    for turn in session.get("turns", []):
        if turn.get("agent_state", {}).get("strategy"):
            training_examples.append({
                "observation": turn["observation"],
                "strategy": turn["agent_state"]["strategy"],
                "action": turn["action"]
            })
```

## Training Applications

### 1. Supervised Fine-Tuning

Extract observation-action pairs for supervised learning:
```python
analyzer = LogAnalyzer(log_dir="game_logs")
analyzer.load_sessions()

# Extract clean training pairs
training_pairs = []
for session in analyzer.sessions:
    for turn in session.get("turns", []):
        training_pairs.append({
            "input": turn["observation"],
            "output": turn["action"]
        })

# Save for fine-tuning
analyzer.export_training_data("fine_tuning_data.json", "json")
```

### 2. Behavior Cloning

Train smaller models to mimic expert behavior:
```python
# Filter for successful games
successful_sessions = [s for s in analyzer.sessions
                      if s.get("summary", {}).get("winners")]

# Extract expert demonstrations
expert_demonstrations = []
for session in successful_sessions:
    for turn in session.get("turns", []):
        if turn["player_id"] in session.get("summary", {}).get("winners", []):
            expert_demonstrations.append({
                "state": turn["observation"],
                "expert_action": turn["action"],
                "reasoning": turn.get("agent_state", {}).get("strategy", "")
            })
```

### 3. Reinforcement Learning

Use logged data for reward modeling or offline RL:
```python
# Extract reward information
rl_data = []
for session in analyzer.sessions:
    rewards = session.get("results", {}).get("rewards", {})
    for turn in session.get("turns", []):
        player_id = str(turn["player_id"])
        rl_data.append({
            "state": turn["observation"],
            "action": turn["action"],
            "reward": rewards.get(player_id, 0.0),
            "done": turn == session.get("turns", [])[-1]
        })
```

## Configuration Options

### GameLogger Options
```python
logger = GameLogger(
    log_dir="custom_logs",    # Custom log directory
    enabled=True              # Enable/disable logging
)
```

### Export Formats
- **JSON**: Human-readable, preserves all data
- **CSV**: Tabular format, easier for some ML frameworks
- **Parquet**: Efficient storage, good for large datasets

```python
analyzer.export_training_data("data.json", "json")
analyzer.export_training_data("data.csv", "csv")
analyzer.export_training_data("data.parquet", "parquet")
```

## Best Practices

1. **Organize by Environment**: Use separate log directories for different games
2. **Regular Cleanup**: Archive old logs to manage disk space
3. **Label Sessions**: Include descriptive information in `additional_info`
4. **Validate Data**: Use `_assess_data_quality()` to check log completeness
5. **Version Control**: Track model versions and log structure changes

## Troubleshooting

### Common Issues

1. **Missing Agent State**: Ensure agents expose internal state attributes
2. **Incomplete Sessions**: Check for errors during game execution
3. **Large Log Files**: Consider compression or archival for long-term storage
4. **API Failures**: Network issues don't affect logging, but may produce error actions

### Debug Mode

Enable verbose logging for debugging:
```python
# In your game loop
if ENABLE_LOGGING:
    print(f"Turn {turn_count}: Logged action for Player {player_id}")
```

## File Locations

- **Logger**: `src/game_logger.py`
- **Analyzer**: `src/log_analyzer.py`
- **Updated Scripts**: `offline_play.py`, `offline_play_batch.py`
- **Test Scripts**: `test_logging.py`, `simple_logging_test.py`
- **Example Logs**: `game_logs/` (after running games)

## Next Steps

1. Collect sufficient game data (100+ games recommended)
2. Analyze performance patterns
3. Prepare training datasets
4. Experiment with fine-tuning or distillation
5. Evaluate improvements on held-out test games

For questions or issues, refer to the code comments or run the test scripts to see examples in action.