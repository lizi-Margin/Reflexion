# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this 
epository.

## Overview

This is a Mind Games Challenge competition repository containing AI agents for two tracks:
- **Track 1 (Social Detection)**: SecretMafia-v0 game
- **Track 2 (Generalization)**: Codenames-v0, ColonelBlotto-v0, ThreePlayerIPD-v0

The codebase is built on the TextArena framework and implements sophisticated multi-phase 
easoning agents that compete online against other teams.

## Common Commands

### Running Games

**Offline testing (local play):**
```bash
python offline_play.py
```
Edit `offline_play.py` to configure which game to test (`env_id`, `npc_num`).

**Online competition - Track 1:**
```bash
python run.py
```
Uses the `Vito` agent for SecretMafia. Configure `MODEL_NAME`, `API_MODEL_SPEC`, and 
team_hash` in the script.

**Online competition - Track 2:**
```bash
python run_track2_adv.py
```
Uses specialized agents (Codenames/Blotto/IPD) with auto-detection. Configure `MODEL_NAME`, 
API_MODEL_SPEC`, and `TEAM_HASH`.

**Batch testing:**
```bash
python run_batch.py <script_name> [num_runs]
# Example: python run_batch.py run.py 15
```

### API Testing

```bash
python corleone/api/api_router.py
```
Tests all configured API models (Qwen, Doubao, Kimi, DeepSeek, GPT).

## Architecture

### Core Agent Structure

The repository uses a hierarchical agent architecture:

```
src/agent.py (base Agent ABC)
    └─ corleone/agents/api_agent.py (generic API-based agent)
        ├─ corleone/agents/track1/vito.py (SecretMafia specialist)
        └─ corleone/agents/track2/ (Track 2 specialists)
            ├─ track2_router.py (auto-detects game type)
            ├─ codenames_agent.py (4-phase reasoning)
            ├─ blotto_agent.py (resource allocation)
            └─ ipd_agent.py (prisoner's dilemma)
```

### Multi-Phase Reasoning Pattern (Track 1 - Vito)

The `Vito` agent implements a 4-step reasoning cycle:

1. **Analysis** (`prompt_analyze`): Parse new observations, identify key information
2. **Belief Update** (`prompt_belief`): Update beliefs about other players' identities
3. **Strategy Update** (`prompt_strategy`): Determine next actions based on goals/beliefs
4. **Action Generation** (`prompt_talk`/`prompt_vote`): Generate final speech or vote

This pattern is implemented via sequential API calls with structured prompts that extract 
ections using tags like `#SUMMARY:`, `#BELIEF:`, `#STRATEGY:`, `#FINAL:`.

### API Router System

`corleone/api/api_router.py` provides a unified interface to multiple LLM providers:

- **Qwen models** → `WWXQ_API`
- **Doubao/Kimi/DeepSeek** → `Volcano_API`
- **GPT models** → `OpenAI_API`

Usage: `get_api_class(model_spec)` returns the appropriate API client class.

API credentials are stored in `.env`:
- `OPENAI_API_KEY`
- `WWXQ_API_KEY`
- `ARK_API_KEY`

### Track 2 Agent Router

`track2_router.py` implements runtime game detection:

1. `create_track2_agent()` with `env_name="auto-detect"` returns `TrackTwoAutoAgent`
2. On first observation, `detect_game_from_observation()` identifies the game
3. Delegates to specialized agent (CodenamesAgent, BlottoAgent, or IPDAgent)

Each specialized agent implements game-specific multi-phase reasoning optimized for that game's
mechanics.

### Game Logging

`corleone/game_logger.py` provides structured logging:

- Creates timestamped directories: `runs/YYMMDD-H_MMAM/`
- Logs metadata (player_id, role, team) and turn-by-turn decisions
- Captures each phase (analysis, belief, strategy, action) with prompts and responses
- Saves to `game_log.json`

Enable with `enable_logging=True` when initializing agents.

## Key Implementation Details

### Action Format Detection

Agents must output actions in specific formats:
- **Voting/targeting**: `[X]` where X is player ID (e.g., `[2]`)
- **Codenames clue**: `[word number]` (e.g., `[wind 2]`)
- **Codenames guess**: `[word]` or `[pass]`

Agents use regex patterns to extract or validate these formats:
```python
bracket_match = re.search(r'\[(\d+)\]', response)
```

### Phase Detection (SecretMafia)

`Vito` tracks game phase using round counter modulo 5:
- Round % 5 == 0: "night" or "day_speak" (depends on role)
- Round % 5 in [1,2,3]: "day_speak"
- Round % 5 == 4: "day_vote"

Different prompts are used for speaking vs voting phases.

### Observation Parsing

Observations come as JSON lists: `[[speaker_id, content, message_type], ...]`

- `speaker_id == -1`: System message
- `speaker_id >= 0`: Player action/speech
- Content matching `^\[(\d+)\]$`: Vote action

Agents parse these into human-readable event logs before processing.

### Initialization Detection

Agents detect first observation by checking `self.is_initialized` flag. Initial observations 
ontain player role, team, and game rules. Agents extract:
- Player ID: `You are Player (\d+)`
- Role: `Your role: (.+)`
- Team: `Team: (.+)`
- Teammates: `Your teammates are: (.+)`

## Competition Configuration

**Team Hash**: Required for online play, format `MG25-XXXXXXXXXX`

**Model Naming**: Must be exact match across submissions for the same model version.

**Divisions**:
- Open Division: `small_category=False` (default)
- Efficient Division: `small_category=True`

## Important Files

- `corleone/family.py`: Exports `Vito` and `Michael` agents for Track 1
- `src/agent.py`: Base `Agent` ABC, `LLMAgent` (HuggingFace), `HumanAgent`
- `envs/`: Contains game environment implementations (for reference)
- `.env`: API credentials (not committed to git)

## Known Issues

- `run_track2_adv.action_format_invalid.issue.txt`: Documents action format validation issues
- `corleone/game_logger.concurrency_bug.issue.txt`: Notes about logging in concurrent scenarios
- `corleone/agents/track2/*.TODO.txt`: Development notes for Track 2 agents