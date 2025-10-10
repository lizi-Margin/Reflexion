# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Mind Games Challenge** competition repository for developing AI agents that compete in strategic multi-agent games. The project focuses on two competition tracks:
- **Track 1 (Social Detection)**: SecretMafia-v0 - deception detection and social manipulation
- **Track 2 (Generalization)**: Multiple games (Codenames-v0, ColonelBlotto-v0, ThreePlayerIPD-v0)

The repository uses the `textarena` library for game environments and online competition infrastructure.

## Core Architecture

### Agent System

**Base Agent Interface** (`src/agent.py`):
- Abstract `Agent` class with `__call__(observation: str) -> str` method
- `LLMAgent`: Uses HuggingFace transformers for local model inference
- `HumanAgent`: Interactive command-line player

**Custom Agent Implementation** (`xushuhang/agents/family.py`):
- `Vito`: Advanced multi-phase reasoning agent using external LLM APIs
- **Four-phase decision process**:
  1. **Analysis**: Parse observations and extract key information
  2. **Belief Update**: Maintain and update beliefs about other players' roles
  3. **Strategy Update**: Determine high-level strategy based on beliefs
  4. **Final Action**: Generate game action or speech based on strategy
- Tracks game state: `belief`, `strategy`, `observation_history`, `turn_counter`
- Logging system integrated via `GameLogger`

### API Router System (`xushuhang/agents/`)

**Multi-Provider API Support** (`api.py` + `api_router.py`):
- Abstract `API` base class with standardized interface
- Provider implementations: `WWXQ_API`, `OpenAI_API`, `Volcano_API`
- Automatic provider selection via `get_api_class()` based on model name
- Retry logic with exponential backoff (4 attempts, 1-8 seconds)
- Supported models:
  - Qwen (via WWXQ): `qwen3-8b`
  - Doubao (via Volcano): `doubao-seed-1-6-250615`, `doubao-seed-1-6-flash-250828`
  - Kimi (via Volcano): `kimi-k2-250905`
  - DeepSeek (via Volcano): `deepseek-v3-1-250821`
  - OpenAI: `gpt-4o-mini`, etc.

**API Keys**: Store in `.env` file:
```
WWXQ_API_KEY=your_key_here
ARK_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### Logging System (`xushuhang/agents/game_logger.py`)

- `GameLogger`: Structured JSON logging per game run
- Creates timestamped directories: `runs/YYMMDD-H_MMAM/`
- Logs all four phases per turn: prompts, raw responses, parsed results
- Automatically saves after each turn completion

## Running Games

### Offline Testing (Development)
```bash
# Test locally without online submission
python src/offline_play.py
```
Modify the script to:
- Choose environment: `SecretMafia-v0`, `Codenames-v0`, `ColonelBlotto-v0`, `ThreePlayerIPD-v0`
- Select agents (HumanAgent vs AI agents)
- Test game mechanics and agent behavior

### Online Competition

**Track 1 (Social Detection)**:
```bash
python run.py
```
Current configuration:
- Model: Configurable via `MODEL_NAME` and `API_MODEL_SPEC` constants
- Environment: `SecretMafia-v0`
- Agent: `Vito` class
- Team hash: `MG25-F5C82328D3`

**Track 2 (Generalization)**:
```bash
python src/online_play_track2.py
```
(Template - needs agent configuration)

**Batch Testing**:
```bash
# Run 13 consecutive games with 5-second delays
python run_batch.py

# Custom number of runs
python run_batch.py 20
```

### Modal Labs Cloud Deployment

Deploy agents on cloud GPUs with $500 free credits:
```bash
# Setup
pip install modal
modal setup

# Deploy
modal run src/online_play_track1_with_modal_lab.py
```
See `modal_lab/MODAL_SETUP.md` for detailed instructions.

## Key Implementation Notes

### Competition Requirements
- `model_name` and `model_description` must **exactly match** across submissions of the same model
- Team hash format: `MG25-XXXXXXXXXX` (from registration)
- Online games: Always use `env.reset(num_players=1)` even for multiplayer games
- Two divisions: Open (default) and Efficient (`small_category=True`)

### Agent Development Guidelines

**For Vito Agent** (or similar multi-phase agents):
- System prompt in `prompt_system()` includes identity and game rules
- Each phase has dedicated prompt methods: `prompt_analyze()`, `prompt_belief()`, `prompt_strategy()`, `prompt_talk()`
- Response parsing uses tag markers: `#SUMMARY:`, `#BELIEF:`, `#STRATEGY:`, `#FINAL:`
- Action format: `[X]` where X is player ID (e.g., `[3]` to vote for player 3)
- Regex patterns extract player selections from LLM responses

**Game State Parsing**:
- Observations are JSON lists: `[[speaker_id, content, message_type], ...]`
- `speaker_id == -1`: System messages
- `content == "[X]"`: Vote actions
- Initialize with `parse_initialization_info()` to extract role, team, teammates

### Utility Libraries

**uhtk Package**: Internal utility toolkit (visualization, encoding, utilities)
- Not core to game logic, used for debugging/visualization
- Located in `uhtk/` subdirectory

## Testing

```bash
# Test API connections
python xushuhang/agents/api_router.py

# Run single game test
python test.py  # (content varies based on current test)
```

## Common Development Patterns

### Adding a New LLM Provider
1. Implement new class in `xushuhang/agents/api.py` inheriting from `API`
2. Add routing logic in `get_api_class()` in `api_router.py`
3. Add API key to `.env` file
4. Update retry logic if provider has special requirements

### Modifying Agent Strategy
- Edit prompt methods in `xushuhang/agents/family.py`
- Tune response parsing in `parse_llm_response()`
- Adjust phase logic in `__call__()` method
- Test with offline play before online submission

### Debugging Game Runs
- Check `runs/` directory for timestamped JSON logs
- Each log contains: metadata, all turns, phases with prompts/responses
- Use `game_log.json` for detailed turn-by-turn analysis
