"""
Inter-trial Memory System for Blotto Agent using Reflexion Framework

This module implements:
1. Memory storage for cross-trial learning
2. Reflection generation after each trial
3. Memory retrieval for strategy improvement
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from api.api_router import get_api_class
from typing import TYPE_CHECKING
import threading
if TYPE_CHECKING:
    from .blotto_agent import BlottoAgent


class BlottoMemory:
    """
    Manages inter-trial memory for Blotto agents using reflexion framework

    Memory is organized by:
    - Allocation strategies and outcomes
    - Opponent behavior patterns
    - Reflections on failed strategies
    """

    def __init__(self, memory_file: str = None, api_model_spec: str = 'qwen3-8b'):
        """
        Initialize Blotto Memory

        Args:
            memory_file: Path to save/load memory (default: reflexion/blotto_runs/memory/blotto_memory.json)
            api_model_spec: API model for generating reflections
        """
        if memory_file is None:
            memory_dir = Path(__file__).parent / "memory"
            memory_dir.mkdir(exist_ok=True)
            memory_file = memory_dir / "blotto_memory.json"

        self.memory_file = Path(memory_file)
        self.api = get_api_class(api_model_spec)(model=api_model_spec)

        # Thread safety lock for concurrent access
        self.lock = threading.Lock()

        # Memory structure
        self.memory = {
            "memory": [],  # List of strategy reflections
            "trial_history": [],  # History of trial outcomes
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_trials": 0
            }
        }

        # Load existing memory if available
        self.load()

    def load(self):
        """Load memory from file (thread-safe)"""
        with self.lock:
            if self.memory_file.exists():
                try:
                    with open(self.memory_file, 'r', encoding='utf-8') as f:
                        loaded = json.load(f)
                        self.memory.update(loaded)
                    print(f"[BlottoMemory] Loaded memory from {self.memory_file}")
                    print(f"[BlottoMemory] Total trials: {self.memory['metadata']['total_trials']}")
                    print(f"[BlottoMemory] Reflections: {len(self.memory['memory'])}")
                except Exception as e:
                    print(f"[BlottoMemory] Error loading memory: {e}")

    def save(self):
        """Save memory to file (thread-safe)"""
        with self.lock:
            self.memory["metadata"]["last_updated"] = datetime.now().isoformat()

            try:
                with open(self.memory_file, 'w', encoding='utf-8') as f:
                    json.dump(self.memory, f, indent=2, ensure_ascii=False)
                print(f"[BlottoMemory] Saved memory to {self.memory_file}")
            except Exception as e:
                print(f"[BlottoMemory] Error saving memory: {e}")

    def add_trial_result(self, trial_data: Dict[str, Any]):
        """
        Add trial result to memory (thread-safe)

        Args:
            trial_data: Dictionary containing:
                - won: Boolean indicating if won
                - rounds_won: Number of rounds won
                - total_rounds: Total number of rounds
                - final_score: Final score
                - opponent_score: Opponent's final score
                - game_log: Path to detailed game log
        """
        trial_record = {
            "trial_id": len(self.memory["trial_history"]),
            "timestamp": datetime.now().isoformat(),
            **trial_data
        }

        self.memory["trial_history"].append(trial_record)
        self.memory["metadata"]["total_trials"] += 1
        self.save()

    def generate_reflection(self, agent: 'BlottoAgent', won: bool) -> Dict[str, str]:
        """
        Generate reflection from a trial using LLM

        Args:
            agent: BlottoAgent instance with game history
            won: Whether the agent won the game

        Returns:
            Dictionary with strategy, reflection, and opponent_analysis
        """
        strategy = agent.strategy
        assert strategy is not None, "Agent strategy must be set before generating reflection"

        # Generate reflection on allocation strategies
        reflection = self._generate_strategy_reflection(agent, won)

        # Generate opponent behavior analysis
        opponent_analysis = self._generate_opponent_analysis(agent, won)

        return {
            "strategy": strategy,
            "reflection": reflection,
            "opponent_analysis": opponent_analysis,
        }

    def _generate_opponent_analysis(self, agent: 'BlottoAgent', won: bool) -> str:
        """Generate analysis of opponent behavior"""
        # Build round history from round_results (format: [alpha_alloc_str, beta_alloc_str, winner_str])
        rounds_summary = ""
        if len(agent.round_results) > 0:
            for i, round_result in enumerate(agent.round_results):
                rounds_summary += f"Round {i+1}/{agent.num_rounds}:\n"
                rounds_summary += f"{round_result[0]}\n"  # Alpha allocation
                rounds_summary += f"{round_result[1]}\n"  # Beta allocation
                rounds_summary += f"{round_result[2]}\n\n"  # Winner

        prompt = f"""You are analyzing a Colonel Blotto game.
Your Commander: {agent.commander_role}
Total Rounds: {agent.num_rounds}
Fields: {', '.join(agent.fields)}
Total Units: {agent.total_units}

Final Score: You {agent.scores[agent.player_id]} - Opponent {agent.scores[1 - agent.player_id]}
OUTCOME: {"You WON the game!" if won else "You LOST the game."}

Game History:
{rounds_summary}

Based on the opponent's allocations across all rounds, analyze:
1. Did they use a consistent strategy (uniform, concentrated, adaptive)?
2. Did they favor specific fields?
3. Did they respond to your moves or play independently?
4. What pattern could you exploit in future games?

Provide brief analysis (2-3 sentences, key points only).

Analysis:"""

        try:
            analysis = self.api(input_messages=[
                {"role": "system", "content": "You are an expert at analyzing Colonel Blotto game strategies."},
                {"role": "user", "content": prompt}
            ])
            return analysis.strip()
        except Exception as e:
            print(f"[BlottoMemory] Error generating opponent analysis: {e}")
            return ""

    def _generate_strategy_reflection(self, agent: 'BlottoAgent', won: bool) -> str:
        """Generate reflection on allocation strategies"""
        assert agent.strategy is not None, "Agent strategy must be set before generating reflection"

        # Build round history from round_results
        rounds_summary = ""
        if len(agent.round_results) > 0:
            for i, round_result in enumerate(agent.round_results):
                rounds_summary += f"Round {i+1}/{agent.num_rounds}:\n"
                rounds_summary += f"{round_result[0]}\n"  # Alpha allocation
                rounds_summary += f"{round_result[1]}\n"  # Beta allocation
                rounds_summary += f"{round_result[2]}\n\n"  # Winner

        # Get previous reflections for context
        previous_reflections = self.memory["memory"][-3:]
        previous_text = ""
        if previous_reflections:
            previous_text = "\n\nPrevious strategy reflections:\n"
            for i, ref in enumerate(previous_reflections):
                previous_text += f"{i+1}. {ref['reflection']}\n"

        prompt = f"""You are analyzing a Colonel Blotto game.
Your Commander: {agent.commander_role}
Your strategy: {agent.strategy}

Final Score: You {agent.scores[agent.player_id]} - Opponent {agent.scores[1 - agent.player_id]}
OUTCOME: {"You WON the game!" if won else "You LOST the game."}

Full Game History:
{rounds_summary}

{previous_text}

Based on this outcome, reflect on your allocation strategy:
1. What allocation patterns worked well or poorly?
2. Were you too uniform, too concentrated, or too adaptive?
3. Did you predict opponent moves effectively?
4. What strategy adjustments would improve your win rate?

Provide a concise, actionable reflection (2-3 sentences) that identifies the KEY lesson learned.

Reflection:"""

        try:
            reflection = self.api(input_messages=[
                {"role": "system", "content": "You are an expert at analyzing game strategies and extracting actionable lessons."},
                {"role": "user", "content": prompt}
            ])
            return reflection.strip()
        except Exception as e:
            print(f"[BlottoMemory] Error generating strategy reflection: {e}")
            return ""

    def update_memory_from_trial(self, agent: 'BlottoAgent', won: bool, should_reflect: bool = True):
        """
        Update memory with reflections from completed trial

        Args:
            agent: BlottoAgent with completed game history
            won: Whether the agent won
            should_reflect: Whether to generate reflections
        """
        # Generate reflections if requested
        if should_reflect:
            print(f"\n[BlottoMemory] Generating reflections for {'winning' if won else 'losing'} trial...")
            reflections = self.generate_reflection(agent, won)

            self.memory["memory"].append({
                "strategy": reflections["strategy"],
                "reflection": reflections["reflection"],
                "opponent_analysis": reflections["opponent_analysis"],
                "won": won,
                "timestamp": datetime.now().isoformat()
            })
            print(f"[BlottoMemory] Reflection: {self.memory['memory'][-1]['reflection']}")
            self.save()

    def get_guidance(self, max_reflections: int = 3) -> str:
        """
        Get strategic guidance from past trials

        Args:
            max_reflections: Maximum number of recent reflections to include

        Returns:
            Formatted guidance string
        """
        reflections = self.memory["memory"][-max_reflections:]

        if not reflections:
            return ""

        guidance = "\n### Lessons from Past Games:\n"
        for i, ref in enumerate(reflections, 1):
            outcome = "WON" if ref["won"] else "LOST"
            guidance += f"##########################################\n"
            guidance += f"{i}. [{outcome}]\n"
            guidance += f"Our Strategy: {ref['strategy']}\n"
            guidance += f"Strategy Reflection: {ref['reflection']}\n"
            guidance += f"Opponent Analysis: {ref['opponent_analysis']}\n"
            guidance += f"##########################################\n"

        guidance += "\nApply these lessons to improve your allocation strategy.\n"
        return guidance

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        trials = self.memory["trial_history"].copy()

        if not trials:
            return {
                "total_trials": 0,
                "win_rate": 0.0,
                "avg_rounds_won": 0.0
            }

        wins = sum(1 for t in trials if t.get("won", False))
        avg_rounds = sum(t.get("rounds_won", 0) for t in trials) / len(trials) if trials else 0.0

        return {
            "total_trials": len(trials),
            "win_rate": wins / len(trials) if trials else 0.0,
            "avg_rounds_won": avg_rounds,
            "total_reflections": len(self.memory["memory"])
        }
