# Blotto Training Script - `train_with_memory.py`

## Overview
Multi-trial training script for Colonel Blotto agent with reflexion memory, adapted from the IPD training script.

## Key Features

### 1. **Shared Memory Architecture**
- One `BlottoMemory` instance shared across all trials
- Thread-safe operations for concurrent training
- Memory persists and improves across trials

### 2. **Two Training Modes**

#### Mode 1: Against Baseline (Default)
```python
SELFPLAY = False
```
- Learning agent (with memory) vs baseline opponent (no memory)
- Baseline uses simple ApiAgent without learning
- Good for learning against diverse opponents

#### Mode 2: Self-Play
```python
SELFPLAY = True
```
- Both agents share the same memory
- Both agents learn and improve together
- More challenging, faster convergence

### 3. **Multi-Threading Support**
- Runs multiple trials concurrently
- Configurable with `MAX_WORKERS` parameter
- Thread-safe printing and memory updates

### 4. **Real-time Statistics**
After each trial completes:
- Win rate
- Average rounds won
- Recent lessons learned (top 3 reflections)

## Configuration

```python
NUM_TRIALS = 10       # Number of games to play
MAX_WORKERS = 3       # Concurrent threads
API_MODEL_SPEC = "qwen3-8b"  # Model for learning agent
SELFPLAY = False      # Self-play mode
```

## Usage

```bash
cd /home/hulc/Desktop/Corleone
python reflexion/blotto_runs/train_with_memory.py
```

## What It Does

### For Each Trial:
1. Creates a learning agent with shared memory
2. Creates opponent (baseline or self-play)
3. Runs complete game (9 rounds by default)
4. Agent calls `finalize_game()` with rewards/game_info
5. Generates reflections on strategy
6. Updates shared memory
7. Prints cumulative statistics

### Example Output:
```
================================================================================
Cumulative Statistics (after 3 completed trials):
  Total Trials: 3
  Win Rate: 66.67%
  Avg Rounds Won: 5.33

Recent Lessons:
### Lessons from Past Games:
##########################################
1. [WON]
Our Strategy: Focus on fields A and B while abandoning C
Strategy Reflection: Concentrated strategy worked well when opponent played uniform...
Opponent Analysis: Opponent used uniform distribution across all rounds...
##########################################
...
================================================================================
```

## Key Differences from IPD Training

| Feature | IPD | Blotto |
|---------|-----|--------|
| Players | 3 players | 2 players (Commander Alpha vs Beta) |
| Environment | ThreePlayerIPD-v0-train | ColonelBlotto-v0-train |
| Finalization | `finalize_game(final_observation)` | `finalize_game(rewards, game_info)` |
| Memory Stat | Avg Rank | Avg Rounds Won |
| Self-play default | True (common) | False (1v1 baseline) |

## Memory Storage

Memory saved to: `reflexion/blotto_runs/memory/blotto_memory.json`

Format:
```json
{
  "memory": [
    {
      "strategy": "Agent's strategic approach",
      "reflection": "What worked/failed",
      "opponent_analysis": "Opponent patterns",
      "won": true,
      "timestamp": "2025-10-23T..."
    }
  ],
  "trial_history": [...],
  "metadata": {...}
}
```

## Training Strategy

1. **Start with baseline opponents**: Learn diverse strategies
2. **Monitor win rate**: Should improve over trials
3. **Review reflections**: Check if agent is learning patterns
4. **Switch to self-play**: For advanced training once baseline is mastered
5. **Adjust models**: Try different opponent models for variety

## Tips

- Use `MAX_WORKERS=1` for debugging (sequential execution)
- Set `VERBOSE=False` to reduce output
- Monitor memory file growth
- Check `runs/` directory for detailed game logs
- Use self-play once win rate > 60% against baseline

## Integration with Competition

After training, use the learned memory in competition:
```python
# Load trained memory
memory = BlottoMemory()  # Auto-loads from saved file

# Create agent with trained memory
agent = BlottoAgent(
    model_name='competition_agent',
    api_model_spec='qwen3-8b',
    memory=memory,
    enable_logging=True
)
```

The agent will use learned strategies from training!
