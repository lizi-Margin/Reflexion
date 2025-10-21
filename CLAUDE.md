# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Mind Games Challenge starter kit for AI agents competing in text-based games. The project contains two competition tracks:

1. **Social Detection Track** - Uses SecretMafia-v0 environment for deception detection
2. **Generalization Track** - Uses Codenames-v0, ColonelBlotto-v0, and ThreePlayerIPD-v0 environments for cross-game performance

### Competition Divisions
- **Open Division**: No restrictions on model size or computational resources (default)
- **Efficient Division**: Resource-efficient agents with smaller models (`small_category=True`)

### Team Registration
- Each team receives a unique `team_hash` (format: MG25-XXXXXXXXXX) via registration email
- Model name and description must remain consistent across submissions for the same agent version

## Common Development Commands

### Installation and Setup
```bash
pip install textarena>=0.7.2
```

### Running the Agent

**Online Competition - Track 1 (Social Detection):**
```bash
python run.py                    # Connects to SecretMafia-v0 online arena
```

**Online Competition - Track 2 (Generalization):**
```bash
python src/online_play_track2.py # Connects to multi-game online arena
```

**Local Testing and Development:**
```bash
python offline_play.py           # Single game testing with logging enabled
python offline_play_batch.py     # Multiple game sessions for statistics
python run_batch.py              # Batch testing with different configurations
python parallel_offline_play.py  # Parallel testing for performance evaluation
```

**Logging System Validation:**
```bash
python test_logging.py           # Comprehensive logging system test
python test_enhanced_logging.py  # Enhanced logging features test
python simple_logging_test.py    # Quick logging validation
```

**Agent Strategy Testing:**
```bash
python test_michael_strategy.py  # Test specific agent strategies and reasoning
```

## Architecture Overview

### Core Components

1. **Agent Base Class** (`src/agent.py`):
   - `Agent`: Abstract base class for all agents
   - `LLMAgent`: Hugging Face model-based agent implementation
   - `HumanAgent`: Manual player input for testing

2. **Custom Agents** (`xushuhang/agents/family.py`):
   - `Vito`: Advanced reasoning agent with 4-step process (analyze → belief → strategy → action)
   - `Michael`: Enhanced version with improved prompting and strategy guidance

3. **Environment Files** (`envs/`):
   - Game-specific implementations for SecretMafia, Codenames, ColonelBlotto, ThreePlayerIPD
   - Each contains environment logic and renderers

4. **Game Logging System** (`src/game_logger.py`, `src/log_analyzer.py`):
   - `GameLogger`: Session management and comprehensive data recording
   - `LoggedAgent`: Wrapper for automatic agent state logging
   - `LogAnalyzer`: Data analysis and training data extraction tools

### Agent Architecture Pattern

The custom agents (Vito, Michael) implement a sophisticated 4-step reasoning process:
1. **Analysis**: Parse and understand new observations from the environment
2. **Belief Update**: Maintain and update beliefs about other players' roles/intentions
3. **Strategy**: Develop game strategies based on analysis and current beliefs
4. **Action**: Generate final speech content or voting action

This architecture enables agents to maintain internal state, track game history, and make strategic decisions based on accumulated knowledge.

### Game Data Pipeline Architecture

The comprehensive logging system captures complete game sessions for training and analysis:
- **Session Management**: Automatic session creation with environment metadata, player info, timestamps
- **Agent State Capture**: Internal beliefs, strategies, reasoning processes at each turn
- **Action Logging**: Turn-by-turn observations, agent responses, and game outcomes
- **Probability Data**: LLM token probabilities for knowledge distillation and model training
- **Multi-format Export**: JSON (complete data), CSV (tabular), Parquet (efficient storage)

### Configuration Management

- **Model Selection**: Support for multiple models (qwen3-8b, qwen3-4b, deepseek-v3.1, deepseek-r1)
- **API Integration**: External API access via environment variables (OPENAI_API_KEY, WWXQ_API_KEY)
- **Competition Compliance**: Model name/description consistency enforced across submissions
- **Debugging Control**: Logging enabled/disabled via `ENABLE_LOGGING` flag in test scripts

## Development Workflow

### Agent Development Pattern
To create a new agent:
1. Inherit from `Agent` base class in `src/agent.py`
2. Implement `__call__(self, observation: str) -> str` method
3. Handle game-specific logic (SecretMafia has night/day phases, voting, etc.)
4. Test both online and offline before competition submission

### Environment-Specific Implementation Details
- **SecretMafia-v0**: Night/day cycles, role detection, accusation/voting mechanics
- **Codenames-v0**: Word association, team coordination, clue giving mechanics
- **ColonelBlotto-v0**: Resource allocation, battlefield strategy, simultaneous moves
- **ThreePlayerIPD-v0**: Iterated prisoner's dilemma, cooperation/defection dynamics

### Testing and Validation Strategy
- Use `offline_play.py` for local testing and debugging with logging enabled
- Test with multiple agents simultaneously to evaluate strategy performance
- Validate action formats (e.g., voting must be `[player_id]` format)
- Monitor observation parsing and belief update logic
- **Logging Integration**: All test scripts support comprehensive game logging by default
- **Statistical Analysis**: Batch testing provides win rates, role performance, invalid move tracking

### Training Data Pipeline
- **Game Logs**: All sessions automatically saved to `game_logs/` directory when `ENABLE_LOGGING=True`
- **Training Data Extraction**: Use `src/log_analyzer.py` to convert logs to ML-ready datasets
- **Performance Analytics**: Win rates by agent, role, game outcome patterns, strategy effectiveness
- **Knowledge Distillation**: LLM token probabilities captured for training student models
- **Multi-format Export**: JSON (complete data), CSV (structured data), Parquet (efficient storage)

## Project Structure

### Core Files and Directories
- `run.py` - Track 1 (Social Detection) online competition entry point
- `src/` - Core agent implementations, online play scripts, comprehensive logging system
- `envs/` - Game environment implementations with rendering support
- `modal_lab/` - Modal Labs cloud deployment setup and configuration
- `xushuhang/agents/` - Custom advanced agent implementations (Vito, Michael)
- `game_logs/` - Auto-created directory for storing all logged game sessions

### Development Patterns and Best Practices

#### Game Logging Integration
```python
from src.game_logger import GameLogger, LoggedAgent

# Initialize logger with session management
logger = GameLogger(log_dir="game_logs", enabled=True)

# Wrap agents for automatic state logging
agents = {}
for player_id, agent in raw_agents.items():
    agents[player_id] = LoggedAgent(agent, logger, player_id)

# Start logging session with metadata
session_id = logger.start_session(
    env_id=ENV_ID,
    agents=raw_agents,
    num_players=len(agents)
)
```

#### Log Data Analysis and Training
```python
from src.log_analyzer import LogAnalyzer

# Load and analyze logged game sessions
analyzer = LogAnalyzer(log_dir="game_logs")
analyzer.load_sessions()
analyzer.print_summary()

# Extract training data for ML pipelines
training_data = analyzer.extract_training_data()
```

#### Model Configuration and API Management
```python
# Configure models in agent constructors
agent = Michael(model_name="qwen3-8b")  # Available: qwen3-8b, qwen3-4b, deepseek-v3.1, deepseek-r1

# API endpoints and keys are managed in xushuhang/agents/family.py api() method
# Environment variables: OPENAI_API_KEY, WWXQ_API_KEY
```

#### Competition Deployment Configuration
```python
# Update these values in run.py for online competition
MODEL_NAME = "YourTeam_Agent_v1"  # Must remain consistent across submissions
MODEL_DESCRIPTION = "Agent description for competition"
team_hash = "MG25-XXXXXXXXXX"     # Your unique team hash from registration
```