"""
Inter-trial Memory System for Codenames Agent using Reflexion Framework

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
    from .codenames_agent import CodenamesAgent


class CodenamesMemory:
    """
    Manages inter-trial memory for Codenames agents using reflexion framework

    Memory is organized by:
    - Clue giving strategies and outcomes
    - Guessing approaches
    - Opponent team's behavior patterns
    - Reflections on failed strategies
    """

    def __init__(self, memory_file: str = None, api_model_spec: str = 'qwen3-8b'):
        """
        Initialize Codenames Memory

        Args:
            memory_file: Path to save/load memory
            api_model_spec: API model for generating reflections
        """
        if memory_file is None:
            memory_dir = Path(__file__).parent / "memory"
            memory_dir.mkdir(exist_ok=True)
            memory_file = memory_dir / "codenames_memory.json"
            print(memory_file)

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
                    print(f"[CodenamesMemory] Loaded memory from {self.memory_file}")
                    print(f"[CodenamesMemory] Total trials: {self.memory['metadata']['total_trials']}")
                    print(f"[CodenamesMemory] memory: {len(self.memory['memory'])}")
                except Exception as e:
                    print(f"[CodenamesMemory] Error loading memory: {e}")

    def save(self):
        """Save memory to file (thread-safe)"""
        with self.lock:
            self.memory["metadata"]["last_updated"] = datetime.now().isoformat()

            try:
                with open(self.memory_file, 'w', encoding='utf-8') as f:
                    json.dump(self.memory, f, indent=2, ensure_ascii=False)
                print(f"[CodenamesMemory] Saved memory to {self.memory_file}")
            except Exception as e:
                print(f"[CodenamesMemory] Error saving memory: {e}")

    def add_trial_result(self, trial_data: Dict[str, Any]):
        """
        Add trial result to memory (thread-safe)

        Args:
            trial_data: Dictionary containing:
                - won: Boolean indicating if won
                - team: 'Red' or 'Blue'
                - role: 'Spymaster' or 'Operative'
                - final_score: Final game score
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

    def generate_reflection(self, agent: 'CodenamesAgent', won: bool) -> Dict[str, str]:
        """
        Generate reflection from a trial using LLM

        Args:
            agent: CodenamesAgent instance with game history
            won: Whether the agent won the game

        Returns:
            Dictionary with strategy, reflection, and opponent_analysis
        """
        # Use last clue or role as strategy proxy
        strategy = (
            getattr(agent, 'last_clue', None) or
            agent.player_role or
            "Default Strategy"
        )

        # Generate reflection on game strategies
        reflection = self._generate_strategy_reflection(agent, won)

        # Generate opponent behavior analysis
        opponent_analysis = self._generate_opponent_analysis(agent, won)

        return {
            "strategy": strategy,
            "reflection": reflection,
            "opponent_analysis": opponent_analysis,
        }

    def _generate_opponent_analysis(self, agent: 'CodenamesAgent', won: bool) -> str:
        """Generate analysis of opponent team's behavior"""
        # Build context from game history
        prompt = f"""You are analyzing a Codenames game.

Team Role: {agent.team} team, {agent.player_role}
Final Outcome: {"Won" if won else "Lost"}

Game Context:
Teammates: Spymaster and Operative from the {agent.team} team
Opponent Team: {"Blue" if agent.team == "Red" else "Red"}

Key Observations:
1. How effectively did the opposing team's Spymaster give clues?
2. Did the opposing team's Operative successfully guess words?
3. Were there any notable strategic decisions by the opposing team?
4. What communication or guessing patterns did you notice from the opposing team?

Provide a brief analysis focusing on the opponent's strategic approach, 1-2 sentences.

Opponent Analysis:"""

        try:
            analysis = self.api(input_messages=[
                {"role": "system", "content": "You are an expert at analyzing Codenames team strategies."},
                {"role": "user", "content": prompt}
            ])
            return analysis.strip()
        except Exception as e:
            print(f"[CodenamesMemory] Error generating opponent analysis: {e}")
            return ""

    def _generate_strategy_reflection(self, agent: 'CodenamesAgent', won: bool) -> str:
        """Generate reflection on Codenames strategies"""
        # Get previous reflections for context
        previous_reflections = self.memory["memory"][-3:]
        previous_text = ""
        if previous_reflections:
            previous_text = "\n\nPrevious strategy reflections:\n"
            for i, ref in enumerate(previous_reflections):
                previous_text += f"{i+1}. {ref['reflection']}\n"

        prompt = f"""You are analyzing a Codenames game.

Team Role: {agent.team} team, {agent.player_role}
Final Outcome: {"Won" if won else "Lost"}

Game Context:
Total Words: {len(agent.board)} (Red: {sum(1 for w, l in agent.board.items() if l == 'R')},
                Blue: {sum(1 for w, l in agent.board.items() if l == 'B')},
                Neutral: {sum(1 for w, l in agent.board.items() if l == 'N')},
                Assassin: 1)

{previous_text}

Reflect on your performance, focusing on:
Spymaster Perspective:
1. How effective were your clues in connecting multiple words?
2. Did you successfully avoid guiding the team towards neutral or opponent words?
3. How well did you manage the risk of the assassin word?

Operative Perspective:
1. How accurately did you interpret the Spymaster's clues?
2. Did you make any incorrect or risky guesses?
3. How did your guessing strategy evolve during the game?

Provide a concise, actionable reflection (2-3 sentences) that identifies the KEY lesson learned about your strategy.

Strategy Reflection:"""

        try:
            reflection = self.api(input_messages=[
                {"role": "system", "content": "You are an expert at analyzing game strategies and extracting actionable lessons."},
                {"role": "user", "content": prompt}
            ])
            return reflection.strip()
        except Exception as e:
            print(f"[CodenamesMemory] Error generating strategy reflection: {e}")
            return ""

    def update_memory_from_trial(self, agent: 'CodenamesAgent', won: bool, should_reflect: bool = True):
        """
        Update memory with reflections from completed trial

        Args:
            agent: CodenamesAgent with completed game history
            won: Whether the agent won
            should_reflect: Whether to generate reflections
        """
        # Generate reflections if requested
        if should_reflect:
            print(f"\n[CodenamesMemory] Generating reflections for {'winning' if won else 'losing'} trial...")
            reflections = self.generate_reflection(agent, won)

            self.memory["memory"].append({
                "strategy": reflections["strategy"],
                "reflection": reflections["reflection"],
                "opponent_analysis": reflections["opponent_analysis"],
                "won": won,
                "role": agent.player_role,
                "team": agent.team,
                "timestamp": datetime.now().isoformat()
            })
            print(f"[CodenamesMemory] Reflection: {self.memory['memory'][-1]['reflection']}")
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

        guidance = "\n### Lessons from Past Codenames Games:\n"
        for i, ref in enumerate(reflections, 1):
            outcome = "WON" if ref["won"] else "LOST"
            guidance += f"##########################################\n"
            guidance += f"{i}. [{outcome}] {ref['role']} for {ref['team']} Team\n"
            guidance += f"Strategy Reflection: {ref['reflection']}\n"
            guidance += f"Opponent Analysis: {ref['opponent_analysis']}\n"
            guidance += f"##########################################\n"

        guidance += "\nApply these lessons to improve your Codenames strategy.\n"
        return guidance

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        trials = self.memory["trial_history"].copy()

        if not trials:
            return {
                "total_trials": 0,
                "win_rate": 0.0,
                "total_reflections": 0
            }

        wins = sum(1 for t in trials if t.get("won", False))

        return {
            "total_trials": len(trials),
            "win_rate": wins / len(trials) if trials else 0.0,
            "total_reflections": len(self.memory["memory"])
        }