"""
Colonel Blotto Agent - Mind Games Challenge Track 2

This module implements a specialized agent for the Colonel Blotto game.

Game Rules:
- 2-player resource allocation game
- Players simultaneously allocate resources across multiple battlefields
- Player with more resources on a battlefield wins that battlefield
- Win a round by winning majority of battlefields
- Win the game by winning majority of rounds
"""

import re
import random
import copy
from typing import Dict, List, Tuple, Optional
from envs.agent import Agent
from api.api_router import get_api_class
from reflexion.game_logger import GameLogger


class BlottoAgent(Agent):
    """
    Specialized agent for playing Colonel Blotto in Mind Games Challenge Track 2

    The agent tracks opponent's allocation patterns and adapts its strategy
    across multiple rounds.
    """

    def __init__(self, model_name: str, api_model_spec='qwen3-8b', enable_logging: bool = True):
        """
        Initialize the BlottoAgent

        Args:
            model_name: Name of the model for identification
            api_model_spec: Which API/model to use for generation
            enable_logging: Whether to enable logging
        """
        self.model_name = model_name
        self.api = get_api_class(api_model_spec)(model=api_model_spec)

        # Game state tracking
        self.is_initialized = False
        self.commander_role = None  # "Alpha" or "Beta"
        self.player_id = None  # 0 or 1
        self.fields = []  # List of field names (e.g., ["A", "B", "C"])
        self.total_units = 0  # Total units to allocate
        self.num_rounds = 0  # Total number of rounds
        self.current_round = 0  # Current round
        self.scores = {0: 0, 1: 0}  # Scores per player

        # History tracking
        self.allocation_history = []  # List of past allocations
        self.opponent_history = []  # List of opponent's past allocations
        self.round_results = []  # Results of each round
        self.observation_history = []
        self.turn_counter = 0

        # Initialize logger
        self.logger = GameLogger() if enable_logging else None

    def __call__(self, observation: str) -> str:
        """
        Process observation and generate appropriate allocation

        Args:
            observation: The current game observation string

        Returns:
            Allocation string in format "[A4 B2 C14]"
        """
        try:
            # First turn initialization
            if not self.is_initialized:
                self._initialize_from_observation(observation)

            # Regular turn processing
            self.observation_history.append(observation)
            self.turn_counter += 1

            # Start logging this turn
            if self.logger:
                self.logger.start_turn(self.turn_counter, observation)

            # Update game state from observation
            self._update_game_state_from_observation(observation)

            # Generate allocation
            result = self._generate_allocation()

            # End logging for this turn
            if self.logger:
                self.logger.end_turn(result)

            return result

        except Exception as e:
            error_msg = f"Error in BlottoAgent: {e}"
            print(error_msg)

            if self.logger:
                self.logger.end_turn(f"ERROR: {str(e)}")

            # In case of error, return a uniform allocation as fallback
            return self._generate_uniform_allocation()

    def _initialize_from_observation(self, observation: str):
        """
        Parse initial observation to determine role, fields, and units

        Args:
            observation: Initial game observation string
        """
        # Extract commander role (Alpha or Beta)
        commander_match = re.search(r"You are Commander (Alpha|Beta)", observation)
        if commander_match:
            self.commander_role = commander_match.group(1)
            self.player_id = 0 if self.commander_role == "Alpha" else 1

        # Extract available fields
        fields_match = re.search(r"fields:\s+([A-Z ,]+)", observation)
        if fields_match:
            self.fields = [f.strip() for f in fields_match.group(1).split(",")]

        # Extract total units
        units_match = re.search(r"Units to allocate: (\d+)", observation)
        if units_match:
            self.total_units = int(units_match.group(1))

        # Extract number of rounds (if available)
        rounds_match = re.search(r"Round (\d+)/(\d+)", observation)
        if rounds_match:
            self.current_round = int(rounds_match.group(1))
            self.num_rounds = int(rounds_match.group(2))

        self.is_initialized = True

        # Set player info in logger
        if self.logger:
            self.logger.set_player_info(
                self.player_id,
                f"Commander {self.commander_role}",
                "N/A"  # No teams in Colonel Blotto
            )

    def _update_game_state_from_observation(self, observation: str):
        """
        Update game state based on new observation

        Args:
            observation: Current game observation
        """
        # Update current round if present
        rounds_match = re.search(r"Round (\d+)/(\d+)", observation)
        if rounds_match:
            self.current_round = int(rounds_match.group(1))
            self.num_rounds = int(rounds_match.group(2))

        # Check for previous round results
        result_match = re.search(r"Round (\d+)\nCommander Alpha allocated: (.*?)\nCommander Beta allocated:\s+(.*?)\nWinner: (.+)", observation, re.DOTALL)
        if result_match:
            round_num = int(result_match.group(1))
            alpha_allocation_str = result_match.group(2)
            beta_allocation_str = result_match.group(3)
            winner = result_match.group(4).strip()

            # Parse allocations
            alpha_allocation = self._parse_allocation(alpha_allocation_str)
            beta_allocation = self._parse_allocation(beta_allocation_str)

            # Store in history
            if self.player_id == 0:  # Alpha
                self.allocation_history.append(alpha_allocation)
                self.opponent_history.append(beta_allocation)
            else:  # Beta
                self.allocation_history.append(beta_allocation)
                self.opponent_history.append(alpha_allocation)

            # Update scores
            if "Alpha" in winner:
                self.scores[0] += 1
            elif "Beta" in winner:
                self.scores[1] += 1

            # Store round result
            self.round_results.append({
                "round": round_num,
                "alpha_allocation": alpha_allocation,
                "beta_allocation": beta_allocation,
                "winner": winner
            })

    def _parse_allocation(self, allocation_str: str) -> Dict[str, int]:
        """
        Parse allocation string into field->units dictionary

        Args:
            allocation_str: Allocation string (e.g., "A: 4, B: 2, C: 14")

        Returns:
            Dictionary mapping field names to unit counts
        """
        allocation = {}
        # Handle different possible formats
        pattern = r"([A-Z]):\s*(\d+)"
        matches = re.findall(pattern, allocation_str)

        for field, units in matches:
            allocation[field] = int(units)

        return allocation

    def _generate_allocation(self) -> str:
        """
        Generate allocation using multi-phase reasoning

        Returns:
            Allocation string in format "[A4 B2 C14]"
        """
        # Phase 1: Analysis of opponent's history
        analysis_prompt = self._prompt_analysis()

        analysis_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_system()},
            {"role": "user", "content": analysis_prompt}
        ])

        # Extract analysis results
        analysis = self._parse_tag_section(analysis_response, "#ANALYSIS:")

        if self.logger:
            self.logger.log_phase("analysis", analysis_prompt, analysis_response, analysis)

        # Phase 2: Strategy selection
        strategy_prompt = self._prompt_strategy(analysis)

        strategy_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_system()},
            {"role": "user", "content": strategy_prompt}
        ])

        # Extract strategy
        strategy = self._parse_tag_section(strategy_response, "#STRATEGY:")

        if self.logger:
            self.logger.log_phase("strategy", strategy_prompt, strategy_response, strategy)

        # Phase 3: Final allocation
        allocation_prompt = self._prompt_allocation(strategy)

        allocation_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_system()},
            {"role": "user", "content": allocation_prompt}
        ])

        # Extract final allocation
        allocation = self._parse_tag_section(allocation_response, "#ALLOCATION:")

        if self.logger:
            self.logger.log_phase("allocation", allocation_prompt, allocation_response, allocation)

        # Parse and validate the allocation
        return self._parse_and_validate_allocation(allocation)
        # return allocation

    def _parse_and_validate_allocation(self, allocation_text: str) -> str:
        """
        Parse and validate the allocation text, ensuring it follows game rules

        Args:
            allocation_text: The raw allocation text from the LLM

        Returns:
            Validated allocation string in format "[A4 B2 C14]"
        """
        # Try to extract using bracket format [A4 B6 C10]
        bracket_match = re.search(r"\[(.*?)\]", allocation_text)
        if bracket_match:
            content = bracket_match.group(1).strip()
        else:
            # Try to extract field-unit pairs without brackets
            # Look for patterns like "A: 5, B: 10, C: 5" or "A4 B6 C10"
            content = allocation_text.strip()

        # Parse the field allocations
        allocation = {}

        # Try format "A: 5, B: 10, C: 5"
        colon_pattern = r"([A-Z])\s*:\s*(\d+)"
        colon_matches = re.findall(colon_pattern, content)

        if colon_matches:
            for field, units in colon_matches:
                if field in self.fields:
                    allocation[field] = int(units)
        else:
            # Try format "A4 B6 C10"
            simple_pattern = r"([A-Z])(\d+)"
            simple_matches = re.findall(simple_pattern, content)

            for field, units in simple_matches:
                if field in self.fields:
                    allocation[field] = int(units)

        # Check if we parsed any valid allocations
        if not allocation:
            return self._generate_uniform_allocation()

        # Validate total units
        total = sum(allocation.values())
        if total > self.total_units:
            # Scale down proportionally
            scale_factor = self.total_units / total
            for field in allocation:
                allocation[field] = int(allocation[field] * scale_factor)

        # Ensure all fields are included
        for field in self.fields:
            if field not in allocation:
                allocation[field] = 0

        # Convert to required format
        result = "[" + ", ".join(f"{field}:{allocation[field]}" for field in self.fields) + "]"
        return result

    def _generate_uniform_allocation(self) -> str:
        """
        Generate a uniform allocation as fallback strategy

        Returns:
            Uniform allocation string
        """
        base_units = self.total_units // len(self.fields)
        extra_units = self.total_units % len(self.fields)

        allocation = {}
        for i, field in enumerate(self.fields):
            allocation[field] = base_units + (1 if i < extra_units else 0)

        return "[" + " ".join(f"{field}{allocation[field]}" for field in self.fields) + "]"

    # Prompt methods

    def _prompt_system(self) -> str:
        """System prompt for Blotto agent"""
        return (
            f"You are Commander {self.commander_role} in the Colonel Blotto game. "
            f"Your goal is to allocate your {self.total_units} units across {len(self.fields)} fields "
            f"({', '.join(self.fields)}) to win more battlefields than your opponent. "
            f"The player who allocates more units to a battlefield wins that battlefield. "
            f"The player who wins the majority of battlefields wins the round. "
            f"You need to strategically anticipate your opponent's moves and counter them."
        )

    def _prompt_analysis(self) -> str:
        """Prompt for analysis phase"""
        prompt = (
            f"As Commander {self.commander_role}, you need to analyze the current game state "
            f"and your opponent's past behavior.\n\n"

            f"Current round: {self.current_round}/{self.num_rounds}\n"
            f"Current score: Commander Alpha {self.scores[0]} - Commander Beta {self.scores[1]}\n"
            f"Fields: {', '.join(self.fields)}\n"
            f"Total units: {self.total_units}\n\n"
        )

        if not self.opponent_history:
            prompt += (
                f"This is the first round, so you have no information about your opponent yet.\n"
                f"Please analyze the game structure:\n"
                f"1. What allocation strategies are possible with {self.total_units} units across {len(self.fields)} fields?\n"
                f"2. What are the advantages of different approaches (uniform, concentrated, random)?\n"
                f"3. What would be a good probing strategy for the first round?\n\n"
            )
        else:
            prompt += "Previous rounds:\n"

            for i, (my_alloc, opp_alloc) in enumerate(zip(self.allocation_history, self.opponent_history)):
                prompt += f"Round {i+1}:\n"
                prompt += f"- Your allocation: {', '.join([f'{f}: {my_alloc.get(f, 0)}' for f in self.fields])}\n"
                prompt += f"- Opponent's allocation: {', '.join([f'{f}: {opp_alloc.get(f, 0)}' for f in self.fields])}\n"
                prompt += f"- Result: {self.round_results[i]['winner']}\n\n"

            prompt += (
                f"Please analyze the opponent's behavior:\n"
                f"1. Is there a pattern in how they allocate units?\n"
                f"2. Do they favor particular fields?\n"
                f"3. How do they respond to your allocations?\n"
                f"4. What might they do next based on the current score and round?\n\n"
            )

        prompt += "Begin your analysis and start with a symbol: '#ANALYSIS:'"
        return prompt

    def _prompt_strategy(self, analysis) -> str:
        """Prompt for strategy selection phase"""
        prompt = (
            f"Based on your analysis:\n\n{analysis}\n\n"

            f"Now, determine your overall strategy for this round. Consider:\n"
            f"1. Current game state (round {self.current_round}/{self.num_rounds}, score {self.scores[0]}-{self.scores[1]})\n"
            f"2. Opponent's patterns and tendencies\n"
            f"3. Available strategic approaches:\n"
            f"   - Uniform: Spread units evenly across fields\n"
            f"   - Concentrated: Focus units on select fields\n"
            f"   - Adaptive: Counter opponent's expected moves\n"
            f"   - Random: Use unpredictability to your advantage\n\n"

            f"Determine your overall approach and reasoning. What fields should you prioritize and why?\n\n"

            f"Begin your strategy and start with a symbol: '#STRATEGY:'"
        )

        return prompt

    def _prompt_allocation(self, strategy) -> str:
        """Prompt for final allocation phase"""
        prompt = (
            f"Based on your strategy:\n\n{strategy}\n\n"

            f"Now, make your final allocation decision for round {self.current_round}. You have {self.total_units} units "
            f"to allocate across fields {', '.join(self.fields)}.\n\n"

            f"Calculate precise unit counts for each field. Remember:\n"
            f"1. The total must equal exactly {self.total_units} units\n"
            f"2. Each field must have a non-negative integer number of units\n"
            f"3. Your allocation must implement your chosen strategy effectively\n\n"

            f"Provide your allocation in EXACTLY and STRICTLY this format: [A:4, B:7, C:9]\n"
            f"Where each letter is a field name followed immediately by the number of units (no spaces between).\n\n"

            f"Begin your allocation and start with a symbol: '#ALLOCATION:'"
        )

        return prompt

    # Utility methods

    def _parse_tag_section(self, text: str, tag: str) -> str:
        """Extract content after a specific tag"""
        index = text.find(tag)
        if index != -1:
            return text[index + len(tag):].strip()
        return text  # Return full text if tag not found