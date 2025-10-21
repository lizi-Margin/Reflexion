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

        # Memory structure
        self.memory = {
            "conversation_reflections": [],  # List of conversation strategy reflections
            "decision_reflections": [],      # List of decision strategy reflections
            "opponent_patterns": {},         # Patterns observed across trials
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
        """Load memory from file"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self.memory.update(loaded)
                print(f"[IPDMemory] Loaded memory from {self.memory_file}")
                print(f"[IPDMemory] Total trials: {self.memory['metadata']['total_trials']}")
                print(f"[IPDMemory] Conversation reflections: {len(self.memory['conversation_reflections'])}")
                print(f"[IPDMemory] Decision reflections: {len(self.memory['decision_reflections'])}")
            except Exception as e:
                print(f"[IPDMemory] Error loading memory: {e}")

    def save(self):
        """Save memory to file"""
        self.memory["metadata"]["last_updated"] = datetime.now().isoformat()

        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
            print(f"[IPDMemory] Saved memory to {self.memory_file}")
        except Exception as e:
            print(f"[IPDMemory] Error saving memory: {e}")

    def add_trial_result(self, trial_data: Dict[str, Any]):
        """
        Add trial result to memory

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

    def generate_reflection(self, trial_log_path: str, trial_rank: int, won: bool) -> Dict[str, str]:
        """
        Generate reflection from a trial using LLM

        Args:
            trial_log_path: Path to the game log JSON file
            trial_rank: Final rank achieved (1=best, 3=worst)
            won: Whether the agent won the game

        Returns:
            Dictionary with conversation_reflection and decision_reflection
        """
        # Load trial log
        try:
            with open(trial_log_path, 'r', encoding='utf-8') as f:
                game_log = json.load(f)
        except Exception as e:
            print(f"[IPDMemory] Error loading trial log: {e}")
            return {"conversation_reflection": "", "decision_reflection": ""}

        # Extract key information
        turns = game_log.get("turns", [])
        metadata = game_log.get("metadata", {})

        # Separate conversation and decision turns
        conversation_turns = []
        decision_turns = []

        for turn in turns:
            observation = turn.get("observation", "")
            if "Submit your decisions" in observation or "Chat finished" in observation:
                decision_turns.append(turn)
            else:
                conversation_turns.append(turn)

        # Generate conversation reflection
        conversation_reflection = self._generate_conversation_reflection(
            conversation_turns, trial_rank, won
        )

        # Generate decision reflection
        decision_reflection = self._generate_decision_reflection(
            decision_turns, trial_rank, won
        )

        return {
            "conversation_reflection": conversation_reflection,
            "decision_reflection": decision_reflection
        }

    def _generate_conversation_reflection(self, conversation_turns: List[Dict],
                                         trial_rank: int, won: bool) -> str:
        """Generate reflection on conversation strategies"""
        if not conversation_turns:
            return ""

        # Build context from conversation turns
        conversation_summary = self._summarize_conversations(conversation_turns)

        # Get previous conversation reflections for context
        previous_reflections = self.memory["conversation_reflections"][-3:]
        previous_text = ""
        if previous_reflections:
            previous_text = "\n\nPrevious conversation reflections:\n"
            for i, ref in enumerate(previous_reflections):
                previous_text += f"{i+1}. {ref['reflection']}\n"

        # Generate reflection prompt
        prompt = f"""You are analyzing a Three Player Iterated Prisoner's Dilemma game where conversation happened before decisions.

OUTCOME: You finished rank {trial_rank} out of 3 players. {"You WON the game!" if won else "You did NOT win."}

CONVERSATION SUMMARY:
{conversation_summary}

{previous_text}

Based on this outcome, reflect on your conversation strategy:
1. What conversation approaches worked well or poorly?
2. Did your messages build trust effectively or create suspicion?
3. Were you too truthful or too deceptive in your communication?
4. How did opponents respond to your messages?
5. What should you do differently in future conversations?

Provide a concise, actionable reflection (2-3 sentences) that identifies the KEY lesson learned about conversation strategy in IPD.

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

    def _generate_decision_reflection(self, decision_turns: List[Dict],
                                      trial_rank: int, won: bool) -> str:
        """Generate reflection on decision strategies"""
        if not decision_turns:
            return ""

        # Build context from decision turns
        decision_summary = self._summarize_decisions(decision_turns)

        # Get previous decision reflections for context
        previous_reflections = self.memory["decision_reflections"][-3:]
        previous_text = ""
        if previous_reflections:
            previous_text = "\n\nPrevious decision reflections:\n"
            for i, ref in enumerate(previous_reflections):
                previous_text += f"{i+1}. {ref['reflection']}\n"

        # Generate reflection prompt
        prompt = f"""You are analyzing a Three Player Iterated Prisoner's Dilemma game's decision-making.

OUTCOME: You finished rank {trial_rank} out of 3 players. {"You WON the game!" if won else "You did NOT win."}

DECISION SUMMARY:
{decision_summary}

{previous_text}

Based on this outcome, reflect on your decision strategy:
1. Were you too cooperative or too aggressive?
2. Did you correctly predict opponents' actions?
3. Did you punish defectors effectively?
4. Did you adapt your strategy based on game state (rank, remaining rounds)?
5. What decision patterns would improve your win rate?

Provide a concise, actionable reflection (2-3 sentences) that identifies the KEY lesson learned about decision-making in IPD.

Reflection:"""

        try:
            reflection = self.api(input_messages=[
                {"role": "system", "content": "You are an expert at analyzing game strategies and extracting actionable lessons."},
                {"role": "user", "content": prompt}
            ])
            return reflection.strip()
        except Exception as e:
            print(f"[IPDMemory] Error generating decision reflection: {e}")
            return ""

    def _summarize_conversations(self, conversation_turns: List[Dict]) -> str:
        """Summarize conversation turns"""
        summary_parts = []

        for i, turn in enumerate(conversation_turns[:10]):  # Limit to first 10 turns
            obs = turn.get("observation", "")
            final = turn.get("final_output", "")

            # Extract round info
            round_match = None
            if "Round" in obs:
                import re
                round_match = re.search(r"Round (\d+)", obs)

            round_info = f"Round {round_match.group(1)}" if round_match else f"Turn {i+1}"

            if final:
                summary_parts.append(f"{round_info}: You said: {final}")

        return "\n".join(summary_parts) if summary_parts else "No conversation data available."

    def _summarize_decisions(self, decision_turns: List[Dict]) -> str:
        """Summarize decision turns"""
        summary_parts = []

        for i, turn in enumerate(decision_turns):
            obs = turn.get("observation", "")
            final = turn.get("final_output", "")

            # Extract round results from observation
            import re
            round_match = re.search(r"Round (\d+)", obs)
            round_info = f"Round {round_match.group(1)}" if round_match else f"Decision {i+1}"

            if final:
                summary_parts.append(f"{round_info}: Your decisions: {final}")

            # Extract results if available
            if "Results:" in obs:
                results_section = obs.split("Results:")[1].split("Current scores:")[0]
                summary_parts.append(f"  Results: {results_section.strip()[:200]}")

        return "\n".join(summary_parts) if summary_parts else "No decision data available."

    def update_memory_from_trial(self, trial_log_path: str, trial_rank: int,
                                 won: bool, should_reflect: bool = True):
        """
        Update memory after a trial

        Args:
            trial_log_path: Path to game log
            trial_rank: Final rank (1-3)
            won: Whether won the game
            should_reflect: Whether to generate reflections (only for losses or poor performance)
        """
        # Only generate reflections if we didn't win or performed poorly
        if should_reflect and (not won or trial_rank > 1):
            print(f"\n[IPDMemory] Generating reflections for rank {trial_rank} trial...")
            reflections = self.generate_reflection(trial_log_path, trial_rank, won)

            # Add conversation reflection
            if reflections["conversation_reflection"]:
                self.memory["conversation_reflections"].append({
                    "reflection": reflections["conversation_reflection"],
                    "trial_rank": trial_rank,
                    "won": won,
                    "timestamp": datetime.now().isoformat()
                })
                print(f"[IPDMemory] Conversation reflection: {reflections['conversation_reflection']}")

            # Add decision reflection
            if reflections["decision_reflection"]:
                self.memory["decision_reflections"].append({
                    "reflection": reflections["decision_reflection"],
                    "trial_rank": trial_rank,
                    "won": won,
                    "timestamp": datetime.now().isoformat()
                })
                print(f"[IPDMemory] Decision reflection: {reflections['decision_reflection']}")

            # Keep only last N reflections to avoid prompt bloat
            max_reflections = 10
            self.memory["conversation_reflections"] = self.memory["conversation_reflections"][-max_reflections:]
            self.memory["decision_reflections"] = self.memory["decision_reflections"][-max_reflections:]

            self.save()

    def get_conversation_guidance(self, max_reflections: int = 3) -> str:
        """
        Get conversation guidance from past reflections

        Args:
            max_reflections: Maximum number of recent reflections to include

        Returns:
            Formatted string with reflection guidance
        """
        reflections = self.memory["conversation_reflections"][-max_reflections:]

        if not reflections:
            return ""

        guidance = "\n### Lessons from Past Games (Conversation Strategy):\n"
        for i, ref in enumerate(reflections, 1):
            outcome = "WON" if ref["won"] else f"Rank {ref['trial_rank']}"
            guidance += f"{i}. [{outcome}] {ref['reflection']}\n"

        guidance += "\nApply these lessons to improve your conversation strategy.\n"
        return guidance

    def get_decision_guidance(self, max_reflections: int = 3) -> str:
        """
        Get decision guidance from past reflections

        Args:
            max_reflections: Maximum number of recent reflections to include

        Returns:
            Formatted string with reflection guidance
        """
        reflections = self.memory["decision_reflections"][-max_reflections:]

        if not reflections:
            return ""

        guidance = "\n### Lessons from Past Games (Decision Strategy):\n"
        for i, ref in enumerate(reflections, 1):
            outcome = "WON" if ref["won"] else f"Rank {ref['trial_rank']}"
            guidance += f"{i}. [{outcome}] {ref['reflection']}\n"

        guidance += "\nApply these lessons to improve your decision-making.\n"
        return guidance

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        trials = self.memory["trial_history"]

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
            "conversation_reflections": len(self.memory["conversation_reflections"]),
            "decision_reflections": len(self.memory["decision_reflections"])
        }
