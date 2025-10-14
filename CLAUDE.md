# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Mind Games Challenge starter kit for AI agents competing in text-based games. The project contains two competition tracks:

1. **Social Detection Track** - Uses SecretMafia-v0 environment
2. **Generalization Track** - Uses Codenames-v0, ColonelBlotto-v0, and ThreePlayerIPD-v0 environments

## Common Development Commands

### Installation and Setup
```bash
pip install textarena>=0.7.2
```

### Running the Agent

**Online Competition - Track 1 (Social Detection):**
```bash
python run.py
```

**Online Competition - Track 2 (Generalization):**
```bash
python src/online_play_track2.py
```

**Offline Testing:**
```bash
python offline_play.py
```

**Batch Testing:**
```bash
python offline_play_batch.py
python run_batch.py
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

### Agent Architecture

The custom agents (Vito, Michael) use a sophisticated 4-step reasoning process:
1. **Analysis**: Parse and understand new observations
2. **Belief Update**: Maintain and update beliefs about other players' roles
3. **Strategy**: Develop game strategies based on analysis and beliefs
4. **Action**: Generate final speech or voting action

### Key Configuration Points

- **Model Selection**: Agents support multiple models (qwen3-8b, qwen3-4b, deepseek-v3.1, deepseek-r1)
- **API Integration**: Uses external APIs through environment variables (OPENAI_API_KEY, WWXQ_API_KEY)
- **Competition Settings**: Model name and description must match exactly across submissions

## Important Development Notes

### Competition Requirements
- Each team receives a unique `team_hash` (format: MG25-XXXXXXXXXX)
- `model_name` and `model_description` must remain consistent across submissions
- Two divisions: Open Division (default) and Efficient Division (`small_category=True`)

### Agent Development Pattern
To create a new agent:
1. Inherit from `Agent` base class
2. Implement `__call__(self, observation: str) -> str` method
3. Handle game-specific logic (SecretMafia has night/day phases, voting, etc.)
4. Test both online and offline before competition submission

### Environment-Specific Logic
- **SecretMafia**: Night/day cycles, role detection, voting mechanics
- **Codenames**: Word association and team guessing
- **ColonelBlotto**: Resource allocation strategy
- **ThreePlayerIPD**: Iterated prisoner's dilemma

### Testing Strategy
- Use `offline_play.py` for local testing and debugging
- Test with multiple agents simultaneously
- Validate action formats (e.g., voting must be `[player_id]`)
- Monitor observation parsing and belief updates

## File Structure Notes

- `run.py` - Main entry point for Track 1 online competition
- `src/` - Core agent implementations and online play scripts
- `envs/` - Game environment implementations
- `modal_lab/` - Modal Labs cloud deployment setup
- `xushuhang/agents/` - Custom advanced agent implementations