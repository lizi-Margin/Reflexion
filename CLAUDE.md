# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a competition codebase for the **Mind Games Challenge**, an AI agent competition focused on strategic game-playing. The project implements specialized agents for two competition tracks:

- **Track 1: Social Detection** - SecretMafia game (deception detection)
- **Track 2: Generalization** - Codenames, ColonelBlotto, and ThreePlayerIPD games

The codebase uses a **reflexion-based architecture** where agents employ multi-phase reasoning (observation → analysis → strategy → action) to play games competitively.

## Architecture

### Core Components

1. **API Layer** (`api/`)
   - `api.py`: Abstract base class and implementations for multiple LLM providers (OpenAI, WWXQ, Volcano/Doubao)
   - `api_router.py`: Router that selects the correct API class based on model name patterns
   - Supports models: qwen, doubao, kimi, deepseek, gpt

2. **Agent Layer**
   - `envs/agent.py`: Base `Agent` abstract class and reference implementations (`LLMAgent`, `HumanAgent`)
   - `reflexion/api_agent.py`: Generic API-based agent for simple play
   - All agents implement `__call__(observation: str) -> str`

3. **Game-Specific Agents** (`reflexion/`)
   - `codename_runs/codenames_agent.py`: Codenames agent (Spymaster/Operative roles)
   - `blotto_runs/blotto_agent.py`: Colonel Blotto resource allocation agent
   - `ipd_runs/ipd_agent.py`: Three Player IPD main agent with reflexion
   - `ipd_runs/ipd_agent_baseline.py`: Baseline IPD agent (used for NPC opponents)
   - `track2_router.py`: Factory function that creates appropriate agent based on environment

4. **Game Logger** (`reflexion/game_logger.py`)
   - Structured logging system that saves game turns to timestamped JSON files
   - Creates logs in `runs/YYMMDD-H_MMAM/game_log.json` format
   - Tracks phases: analysis, belief_update, strategy_update, final_action

5. **Environment Configurations** (`envs/`)
   - Game environment definitions deployed via textarena package
   - Reference implementations for local testing

### Multi-Phase Reasoning Pattern

Most specialized agents follow a reflexion pattern with these phases:
1. **Analysis**: Parse observation, extract game state
2. **Belief Update**: Update beliefs about opponents/game dynamics
3. **Strategy Update**: Adjust strategy based on beliefs
4. **Final Action**: Generate action in required format

Each phase uses separate LLM calls with specialized prompts, logged via `GameLogger`.

## Common Development Tasks

### Running Offline Games

Use `offline_play.py` to test agents locally without connecting to the arena:

```bash
# Edit offline_play.py to select environment:
# env_id = "Codenames-v0-train"; npc_num = 3  # 2v2
# env_id = "ColonelBlotto-v0-train"; npc_num = 1  # 1v1
# env_id = "ThreePlayerIPD-v0-train"; npc_num = 2  # 3 players

python offline_play.py
```

The script creates multiple agents via `create_track2_agent()` and plays a single game with verbose output.

### Running Online Competition

For Track 2, use `run_track2.py`:

```bash
python run_track2.py
```

Key parameters in the script:
- `MODEL_NAME`: Unique identifier for leaderboard (must match exactly for resubmissions)
- `MODEL_DESCRIPTION`: Description of your agent
- `team_hash`: Your team verification code (format: `MG25-XXXXXXXXXX`)
- `api_model_spec`: Which LLM to use (e.g., `"qwen3-8b"`, `"kimi-k2-250905"`)

### Environment Setup

1. Install dependencies:
```bash
pip install textarena>=0.7.2
```

2. Configure API keys in `.env`:
```bash
OPENAI_API_KEY=sk-...
WWXQ_API_KEY=sk-...
ARK_API_KEY=...
```

The `api_router.py` automatically selects the correct API based on model name.

### Testing API Connections

```bash
python api/api_router.py
```

This runs unit tests for all configured API providers.

## Code Organization Patterns

### Creating a Track 2 Agent

The `track2_router.py` factory handles agent creation:

```python
from reflexion.track2_router import create_track2_agent

agent = create_track2_agent(
    model_name='my_agent',
    api_model_spec='qwen3-8b',
    env_name='ThreePlayerIPD-v0-train',  # or Codenames, ColonelBlotto
    enable_logging=True
)
```

For baseline opponents in IPD, use model names starting with `'bsl'` to get the baseline agent.

### Agent Lifecycle

1. **Initialization**: Agent receives first observation, parses player_id, role, team
2. **Turn Processing**:
   - `start_turn()` in logger
   - Multi-phase reasoning (each phase logged)
   - `end_turn()` with final output
3. **Finalization**: `finalize()` saves complete game log

### Game State Management

Each specialized agent maintains:
- `is_initialized`: Whether first observation has been processed
- `player_id`: Agent's player ID
- Role-specific state (e.g., `player_role`, `team` for Codenames)
- History tracking (observation history, opponent patterns)

## Important Patterns

### API Model Routing

Model names determine which API provider is used:
- `qwen*` → WWXQ_API
- `doubao*`, `kimi*`, `deepseek*` → Volcano_API
- `gpt*` → OpenAI_API
- Default → WWXQ_API

### Game Detection

`track2_router.py` can auto-detect games from observations:
- "codenames" or "spymaster" → Codenames
- "colonel blotto" or "commander" → ColonelBlotto
- "prisoner's dilemma" or "cooperate" → ThreePlayerIPD

### Error Handling

All API classes implement retry logic with exponential backoff (max 4 attempts). Agents should catch exceptions and log errors via the logger.

## Competition-Specific Notes

### Model Name Consistency

**Critical**: `model_name` and `model_description` must match EXACTLY for resubmissions to the same model. Case-sensitive, no spacing changes allowed.

### Division Selection

- `small_category=False`: Open Division (no restrictions)
- `small_category=True`: Efficient Division (resource-constrained)

### Online vs Offline Environments

- Online: Use `ta.make_mgc_online()`, always set `num_players=1`
- Offline: Use `ta.make(env_id=...)`, set actual number of players
- Offline env_ids end with `-train` (e.g., `"Codenames-v0-train"`)

## File References

When working on specific games, refer to these key files:

- Codenames logic: `reflexion/codename_runs/codenames_agent.py`
- Colonel Blotto logic: `reflexion/blotto_runs/blotto_agent.py`
- IPD main logic: `reflexion/ipd_runs/ipd_agent.py`
- IPD baseline: `reflexion/ipd_runs/ipd_agent_baseline.py`
- Agent routing: `reflexion/track2_router.py`
- API implementation: `api/api.py`
- API routing: `api/api_router.py`

The `reflexion/alfworld_runs/` directory contains reference implementation from the original Reflexion paper but is not actively used in the competition.
