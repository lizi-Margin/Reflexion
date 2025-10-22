"""
Inter-trial Memory System for IPD Agent using Reflexion Framework

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
    from .ipd_agent import IPDAgent


class IPDMemory:
    """
    Manages inter-trial memory for IPD agents using reflexion framework

    Memory is organized by:
    - Conversation strategies and outcomes
    - Decision strategies and outcomes
    - Opponent behavior patterns
    - Reflections on failed strategies
    """

    def __init__(self, memory_file: str = None, api_model_spec: str = 'qwen3-8b'):
        """
        Initialize IPD Memory

        Args:
            memory_file: Path to save/load memory (default: reflexion/ipd_runs/memory/ipd_memory.json)
            api_model_spec: API model for generating reflections
        """
        if memory_file is None:
            memory_dir = Path(__file__).parent / "memory"
            memory_dir.mkdir(exist_ok=True)
            memory_file = memory_dir / "ipd_memory.json"

        self.memory_file = Path(memory_file)
        self.api = get_api_class(api_model_spec)(model=api_model_spec)

        # Thread safety lock for concurrent access
        self.lock = threading.Lock()

        # Memory structure
        self.memory = {
            "memory": [],  # List of conversation strategy reflections
            "trial_history": [],             # History of trial outcomes
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
                    print(f"[IPDMemory] Loaded memory from {self.memory_file}")
                    print(f"[IPDMemory] Total trials: {self.memory['metadata']['total_trials']}")
                    print(f"[IPDMemory] memory: {len(self.memory['memory'])}")
                except Exception as e:
                    print(f"[IPDMemory] Error loading memory: {e}")

    def save(self):
        """Save memory to file (thread-safe)"""
        with self.lock:
            self.memory["metadata"]["last_updated"] = datetime.now().isoformat()

            try:
                with open(self.memory_file, 'w', encoding='utf-8') as f:
                    json.dump(self.memory, f, indent=2, ensure_ascii=False)
                print(f"[IPDMemory] Saved memory to {self.memory_file}")
            except Exception as e:
                print(f"[IPDMemory] Error saving memory: {e}")

    def add_trial_result(self, trial_data: Dict[str, Any]):
        """
        Add trial result to memory (thread-safe)

        Args:
            trial_data: Dictionary containing:
                - rank: Final rank (1, 2, or 3)
                - score: Final score
                - opponent_scores: {player_id: score}
                - won: Boolean indicating if won
                - game_log: Path to detailed game log
                - strategy_summary: Summary of strategies used
        """
        trial_record = {
            "trial_id": len(self.memory["trial_history"]),
            "timestamp": datetime.now().isoformat(),
            **trial_data
        }

        self.memory["trial_history"].append(trial_record)
        self.memory["metadata"]["total_trials"] += 1
        self.save()

    def generate_reflection(self, agent: 'IPDAgent', trial_rank: int, won: bool) -> Dict[str, str]:
        """
        Generate reflection from a trial using LLM

        Args:
            trial_log_path: Path to the game log JSON file
            trial_rank: Final rank achieved (1=best, 3=worst)
            won: Whether the agent won the game

        Returns:
            Dictionary with conversation_reflection and decision_reflection
        """
        strategy = agent.strategy
        assert strategy is not None, "Agent strategy must be set before generating reflection"

        # Generate opponent behavior analysis
        opponent_analysis = self._generate_opponent_behavior_analysis(
            agent, trial_rank, won
        )

        # Generate conversation reflection
        reflection = self._generate_reflection(
            agent, trial_rank, won
        )

        return {
            "strategy": strategy,
            "reflection": reflection,
            "opponent_analysis": opponent_analysis,
        }

    def _generate_strategy_conclusion(self, agent: 'IPDAgent', trial_rank: int, won: bool) -> str:
        """Generate conclusion on strategy performance"""
    
    def _generate_opponent_behavior_analysis(self, agent: 'IPDAgent', trial_rank: int, won: bool) -> str:
        """Generate analysis of opponent behavior"""
        full_obs = agent.observation_history[-1]
        prompt = f"""You are analyzing a Three Player Iterated Prisoner's Dilemma game where conversation happened before decisions.
Player ID: You are the player {agent.player_id}.
Opponent Player IDs: {", ".join([str(p) for p in agent.opponent_ids])}
Full observation (You are the player {agent.player_id}):
{full_obs}

Based on this outcome, analyze the behavior of your 2 opponents:
1. How they behaved in the whole game (mostly cooperate or defect)?
2. Did they truthful or deceptive? Are they saying cooperate in conversation but defect in decision?
3. Did they defect in the last round (optimum strategy)?
Please provide your brief analysis in a clear format, every opponent in just 1 sentences (only key points).

Analysis:"""
        try:
            opponent_analysis = self.api(input_messages=[
                {"role": "system", "content": "You are an expert at analyzing game strategies."},
                {"role": "user", "content": prompt}
            ])
            return opponent_analysis.strip()
        except Exception as e:
            print(f"[IPDMemory] Error generating opponent behavior analysis: {e}")
            return ""

    def _generate_reflection(self, agent: 'IPDAgent', trial_rank: int, won: bool) -> str:
        """Generate reflection on conversation strategies"""
        assert agent.strategy is not None, "Agent strategy must be set before generating reflection"

        # Build context from conversation turns
        full_obs = agent.observation_history[-1]
        # Get previous conversation reflections for context
        previous_reflections = self.memory["memory"][-3:]
        previous_text = ""
        if previous_reflections:
            previous_text = "\n\nPrevious conversation reflections:\n"
            for i, ref in enumerate(previous_reflections):
                previous_text += f"{i+1}. {ref['reflection']}\n"

        # Generate reflection prompt
        prompt = f"""You are analyzing a Three Player Iterated Prisoner's Dilemma game where conversation happened before decisions.
Player ID: You are the player {agent.player_id}.
Your strategy: {agent.strategy}

OUTCOME: You finished rank {trial_rank} out of 3 players. {"You WON the game!" if won else "You did NOT win."}

Full observation (You are the player {agent.player_id}):
{full_obs}

{previous_text}

Based on this outcome, reflect on your strategy:
About conversation:
1. What conversation approaches worked well or poorly?
2. Were you too truthful or too deceptive in your communication?

About decision:
1. Were you too cooperative or too aggressive?
2. Did you punish defectors effectively?
3. Did you choose defect in the last round (optimum strategy)?
4. What decision patterns would improve your win rate?


Provide a concise, actionable reflection (2-3 sentences) that identifies the KEY lesson learned about strategy in IPD, key point only.

Reflection:"""

        try:
            reflection = self.api(input_messages=[
                {"role": "system", "content": "You are an expert at analyzing game strategies and extracting actionable lessons."},
                {"role": "user", "content": prompt}
            ])
            return reflection.strip()
        except Exception as e:
            print(f"[IPDMemory] Error generating conversation reflection: {e}")
            return ""

    def update_memory_from_trial(self, agent: 'IPDAgent', trial_rank: int,
                                 won: bool, should_reflect: bool = True):
        # Only generate reflections if we didn't win or performed poorly
        if should_reflect and (not won or trial_rank > 1):
            print(f"\n[IPDMemory] Generating reflections for rank {trial_rank} trial...")
            reflections = self.generate_reflection(agent, trial_rank, won)

            self.memory["memory"].append({
                "strategy": reflections["strategy"],
                "reflection": reflections["reflection"],
                "opponent_analysis": reflections["opponent_analysis"],
                "trial_rank": trial_rank,
                "won": won,
                "timestamp": datetime.now().isoformat()
            })
            print(f"[IPDMemory] reflection: {self.memory['memory'][-1]['reflection']}")
            self.save()

    def get_guidance(self, max_reflections: int = 3) -> str:
        reflections = self.memory["memory"][-max_reflections:]

        if not reflections:
            return ""

        guidance = "\n### Lessons from Past Games (Conversation Strategy):\n"
        for i, ref in enumerate(reflections, 1):
            outcome = "WON" if ref["won"] else f"Rank {ref['trial_rank']}"
            guidance += f"##########################################\n"
            guidance += f"{i}. [{outcome}]\n"
            # guidance += f"Our Strategy: {ref['strategy']}\n"
            guidance += f"Reflection to our strategy: {ref['reflection']}\n"
            guidance += f"Opponent behavior: {ref['opponent_analysis']}\n"
            guidance += f"##########################################\n"

        guidance += "\nApply these lessons to improve your strategy.\n"
        return guidance

    def get_statistics(self) -> Dict[str, Any]:
        trials = self.memory["trial_history"].copy()

        if not trials:
            return {
                "total_trials": 0,
                "win_rate": 0.0,
                "avg_rank": 0.0
            }

        wins = sum(1 for t in trials if t.get("won", False))
        ranks = [t.get("rank", 3) for t in trials]

        return {
            "total_trials": len(trials),
            "win_rate": wins / len(trials) if trials else 0.0,
            "avg_rank": sum(ranks) / len(ranks) if ranks else 0.0,
            "memory": len(self.memory["memory"])
        }
