# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation

1. Create Environment
```bash
# Using pip
pip install -r requirements.txt

# Using mamba
mamba env create -f ./mamba_env.yml -n MindGameCorleone
```

2. API Key Configuration
- Configure API keys in `.env`
```bash
OPENAI_API_KEY=sk-...
WWXQ_API_KEY=sk-...
ARK_API_KEY=...
```

### Running and Testing

1. Offline Game Testing
```bash
# Run multiple local games
python run_competition_test.py --games 100

# Single game testing (manual environment selection)
# Edit offline_play.py to select environment
python offline_play.py
```

2. Game Environment Selection
- Codenames: `env_id = "Codenames-v0-train"`
- Colonel Blotto: `env_id = "ColonelBlotto-v0-train"`
- Three Player IPD: `env_id = "ThreePlayerIPD-v0-train"`

3. API Connection Testing
```bash
python api/api_router.py
```

### Deployment

1. Online Competition Submission
```bash
python run_track2.py
```

## Key Development Considerations

### Agent Development

- Implement multi-phase reasoning:
  1. Analysis
  2. Belief Update
  3. Strategy Update
  4. Final Action

- Use `track2_router.py` for agent creation
- Implement `__call__(observation: str) -> str` method

### API Model Routing

Automatic API selection based on model name:
- `qwen*` → WWXQ_API
- `doubao*`, `kimi*`, `deepseek*` → Volcano_API
- `gpt*` → OpenAI_API
- Default → WWXQ_API

## Performance Optimization

1. Model Selection
- Open Division: `small_category=False`
- Efficient Division: `small_category=True`

2. Logging
- Game logs saved in `runs/YYMMDD-H_MMAM/game_log.json`
- Track game phases: analysis, belief_update, strategy_update, final_action

## Troubleshooting

1. Verify API keys and network connectivity
2. Check model name consistency for competition submissions
3. Ensure environment configurations match game requirements