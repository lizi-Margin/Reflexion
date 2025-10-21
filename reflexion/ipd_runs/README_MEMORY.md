# IPD Agent with Reflexion-Based Inter-Trial Memory

This directory implements an enhanced IPD (Iterated Prisoner's Dilemma) agent that uses the **Reflexion framework** to learn across multiple trials through inter-trial memory.

## Overview

The memory-enhanced IPD agent addresses a key limitation of single-trial agents: **inability to learn from past games**. Using the reflexion framework (https://arxiv.org/abs/2303.11366), the agent:

1. **Plays games** using sophisticated multi-phase reasoning (conversation + decision policies)
2. **Reflects on outcomes** after each trial, generating actionable lessons
3. **Integrates learnings** into future games through memory-guided prompts
4. **Improves over time** by accumulating strategic insights

## Architecture

### Core Components

#### 1. `ipd_memory.py` - Inter-Trial Memory Manager

Manages persistent memory across trials:

```python
from reflexion.ipd_runs.ipd_memory import IPDMemory

# Create/load memory
memory = IPDMemory(api_model_spec='qwen3-8b')

# After each game, update memory
memory.update_memory_from_trial(
    trial_log_path='path/to/game_log.json',
    trial_rank=2,  # 1=won, 2=2nd place, 3=last
    won=False,
    should_reflect=True  # Generate reflections for losses
)

# Retrieve guidance for next game
conv_guidance = memory.get_conversation_guidance(max_reflections=3)
dec_guidance = memory.get_decision_guidance(max_reflections=3)
```

**Memory Structure:**
- `conversation_reflections`: Lessons about communication strategies
- `decision_reflections`: Lessons about cooperation/defection decisions
- `trial_history`: Record of all past trials
- `metadata`: Statistics and timestamps

#### 2. `ipd_agent.py` - Memory-Enhanced Agent

Extended IPD agent with memory integration:

```python
from reflexion.ipd_runs.ipd_agent import IPDAgent
from reflexion.ipd_runs.ipd_memory import IPDMemory

# Create shared memory
memory = IPDMemory()

# Create agent with memory
agent = IPDAgent(
    model_name='learning_agent',
    api_model_spec='qwen3-8b',
    enable_logging=True,
    memory=memory  # Share memory across trials
)

# After game ends
agent.finalize_game(final_observation=final_obs)
```

**Key Enhancements:**
- `memory` parameter in `__init__` for shared memory
- `finalize_game()` method to update memory after each trial
- Memory guidance injected into conversation and decision prompts

#### 3. `train_with_memory.py` - Multi-Trial Training Script

Runs multiple trials with memory accumulation:

```bash
python reflexion/ipd_runs/train_with_memory.py
```

**Features:**
- Runs N trials with the same memory instance
- Tracks performance improvement over time
- Prints statistics and reflections after each trial
- Saves memory persistently to disk

#### 4. `example_memory_agent.py` - Single Game Example

Simple example of using memory-enhanced agent:

```bash
python reflexion/ipd_runs/example_memory_agent.py
```

## How Reflexion Works in IPD

### Phase 1: Game Execution

Agent plays IPD using multi-phase reasoning:
- **Conversation phase**: Build trust, deceive, coordinate
- **Decision phase**: Choose cooperate/defect based on analysis

All turns are logged to `runs/YYMMDD-H_MMAM/game_log.json`.

### Phase 2: Reflection Generation

After game ends (if lost or didn't win):

1. **Load game log** with all conversation and decision turns
2. **Generate conversation reflection**:
   ```
   "In past games, being overly cooperative in early rounds led to exploitation.
   Focus on conditional cooperation: cooperate only with proven cooperators."
   ```

3. **Generate decision reflection**:
   ```
   "When trailing in final rounds, aggressive defection against the leader
   is necessary to close the score gap and win."
   ```

4. **Store reflections** in memory with metadata (rank, won, timestamp)

### Phase 3: Memory Integration

In next game, reflections are injected into prompts:

**Conversation Strategy Prompt:**
```
Based on your analysis:
[current game analysis]

### Lessons from Past Games (Conversation Strategy):
1. [WON] Build trust early but verify through actions
2. [Rank 2] Being overly cooperative led to exploitation
3. [Rank 3] Deceptive promises backfired when exposed

Apply these lessons to improve your conversation strategy.

Now, determine your conversation strategy...
```

**Decision Strategy Prompt:**
```
Based on your analysis and payoff evaluation:
[current analysis and payoffs]

### Lessons from Past Games (Decision Strategy):
1. [Rank 2] In final rounds, trailing players must defect aggressively
2. [WON] Punishing early defectors builds credible deterrence
3. [Rank 3] Cooperating with everyone results in being exploited

Apply these lessons to improve your decision-making.

Now, make your final decisions...
```

## Usage Examples

### Example 1: Train Over Multiple Trials

```python
from reflexion.ipd_runs.ipd_memory import IPDMemory
from reflexion.ipd_runs.ipd_agent import IPDAgent
import textarena as ta

# Create shared memory
memory = IPDMemory(api_model_spec='qwen3-8b')

# Run 10 trials
for trial in range(10):
    # Create agent with shared memory
    agent = IPDAgent(
        model_name=f'agent_trial_{trial}',
        api_model_spec='qwen3-8b',
        memory=memory  # Same memory!
    )

    # Play game (setup agents, env, etc.)
    # ... game loop ...

    # After game ends
    agent.finalize_game(final_observation=final_obs)

    # Check improvement
    stats = memory.get_statistics()
    print(f"Win rate after {trial+1} trials: {stats['win_rate']:.2%}")
```

### Example 2: Load Existing Memory

```python
# Memory is automatically saved to reflexion/ipd_runs/memory/ipd_memory.json
# Loading existing memory:

memory = IPDMemory()  # Automatically loads from default path

stats = memory.get_statistics()
print(f"Loaded memory with {stats['total_trials']} past trials")
print(f"Current win rate: {stats['win_rate']:.2%}")

# Use this memory for new games
agent = IPDAgent(memory=memory, ...)
```

### Example 3: Analyze Memory Contents

```python
from reflexion.ipd_runs.ipd_memory import IPDMemory

memory = IPDMemory()

# Get conversation insights
print(memory.get_conversation_guidance(max_reflections=5))

# Get decision insights
print(memory.get_decision_guidance(max_reflections=5))

# View trial history
for trial in memory.memory['trial_history']:
    print(f"Trial {trial['trial_id']}: Rank {trial['rank']}, Won: {trial['won']}")
```

## Configuration

### Memory Parameters

**File location:**
```python
memory = IPDMemory(
    memory_file='path/to/custom_memory.json',  # Default: reflexion/ipd_runs/memory/ipd_memory.json
    api_model_spec='qwen3-8b'  # LLM for generating reflections
)
```

**Reflection limits:**
- Maximum reflections kept: 10 (prevents prompt bloat)
- Reflections shown in prompts: 3 (most recent)

### When Reflections Are Generated

```python
# Only generate reflections for:
# 1. Losses (won=False)
# 2. Non-first-place finishes (rank > 1)

agent.finalize_game()  # Internally calls:
memory.update_memory_from_trial(
    ...,
    should_reflect=(not won or rank > 1)
)
```

To force reflection even on wins:
```python
memory.update_memory_from_trial(..., should_reflect=True)
```

## File Structure

```
reflexion/ipd_runs/
├── ipd_agent.py              # Original agent (now with memory support)
├── ipd_agent_baseline.py     # Baseline agent for opponents
├── ipd_memory.py             # NEW: Inter-trial memory manager
├── train_with_memory.py      # NEW: Multi-trial training script
├── example_memory_agent.py   # NEW: Single game example
└── memory/                   # NEW: Memory storage directory
    └── ipd_memory.json       # Persistent memory file
```

## Key Design Decisions

### 1. Separated Conversation and Decision Reflections

IPD has distinct phases requiring different strategies:
- **Conversation**: Trust-building, deception, coordination
- **Decision**: Actual cooperate/defect choices

Reflections are generated separately for each, allowing targeted learning.

### 2. Context-Aware Reflection Prompts

Reflection prompts include:
- Final rank and win status
- Conversation/decision summaries from the trial
- Previous reflections for context

This ensures high-quality, actionable reflections.

### 3. Automatic Memory Management

- Memory auto-saves after each trial
- Memory auto-loads on initialization
- Reflection limits prevent prompt bloat
- Old reflections are pruned but trial history is kept

### 4. LLM-Generated Reflections

Unlike hand-coded rules, reflections are generated by LLM analyzing game logs:
- Adapts to any game dynamics
- Captures nuanced strategic insights
- Natural language format integrates seamlessly into prompts

## Performance Tips

### 1. Start with Baseline Memory

Run initial trials to build foundational memory:
```bash
python reflexion/ipd_runs/train_with_memory.py
```

### 2. Monitor Win Rate Trends

```python
stats = memory.get_statistics()
print(f"Trials: {stats['total_trials']}, Win Rate: {stats['win_rate']:.2%}")
```

Expected pattern: Win rate should increase over trials.

### 3. Review Reflections Periodically

```python
# Check if reflections make sense
print(memory.get_conversation_guidance(max_reflections=10))
print(memory.get_decision_guidance(max_reflections=10))
```

If reflections seem off, you can manually edit `memory/ipd_memory.json`.

### 4. Use Stronger Models for Reflection

```python
# Use a stronger model for generating reflections
memory = IPDMemory(api_model_spec='gpt-4o-mini')

# But use faster model for actual gameplay
agent = IPDAgent(api_model_spec='qwen3-8b', memory=memory)
```

## Comparison with Original Agent

| Feature | Original `ipd_agent.py` | Memory-Enhanced Agent |
|---------|------------------------|----------------------|
| Multi-phase reasoning | ✅ | ✅ |
| Conversation policy | ✅ | ✅ |
| Decision policy | ✅ | ✅ |
| Game logging | ✅ | ✅ |
| **Inter-trial learning** | ❌ | ✅ **NEW** |
| **Reflection generation** | ❌ | ✅ **NEW** |
| **Memory-guided prompts** | ❌ | ✅ **NEW** |
| **Performance improvement** | Static | **Improves over time** |

## Troubleshooting

### Memory not loading

```python
# Check if file exists
import os
print(os.path.exists('reflexion/ipd_runs/memory/ipd_memory.json'))

# Force create new memory
memory = IPDMemory(memory_file='new_memory.json')
```

### Reflections seem poor quality

Try:
1. Use a stronger model: `IPDMemory(api_model_spec='gpt-4o-mini')`
2. Check game logs are detailed enough
3. Ensure games finish properly (not truncated)

### Agent not improving over trials

Check:
1. Memory is being shared across trials (same instance)
2. `finalize_game()` is called after each trial
3. Reflections are being generated (check memory file)
4. Guidance is non-empty in prompts

## References

- **Reflexion Paper**: https://arxiv.org/abs/2303.11366
- **AlfWorld Implementation**: `reflexion/alfworld_runs/` (reference)
- **Original IPD Agent**: `ipd_agent.py`
- **Mind Games Challenge**: https://www.mindgamesarena.com/

## Future Enhancements

Potential improvements:
- [ ] Opponent-specific memory (learn patterns per opponent)
- [ ] Strategy pool integration (combine with existing strategy manager)
- [ ] Multi-agent shared memory (agents learn from each other)
- [ ] Memory pruning based on performance (keep only good reflections)
- [ ] Hierarchical memory (general lessons + specific tactics)
