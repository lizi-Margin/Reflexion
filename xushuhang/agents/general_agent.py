"""
API Agent - Generic agent for Track 2 games

This module provides a simple API-based agent for all Track 2 games.
Unlike the specialized agents, this agent does not implement game-specific
reasoning but uses a generic prompt approach for all games.
"""

import re
from typing import Dict, List, Optional
from src.agent import Agent
from xushuhang.agents.api_router import get_api_class
from xushuhang.agents.game_logger import GameLogger


class GeneralAgent(Agent):
    """
    A generic API-based agent for Track 2 games.

    This agent uses a simpler approach than the specialized agents,
    but can handle all Track 2 games (Codenames, Colonel Blotto, ThreePlayerIPD).
    """

    def __init__(self, model_name: str, api_model_spec='qwen3-8b', enable_logging: bool = True):
        """
        Initialize the ApiAgent

        Args:
            model_name: Name of the model for identification
            api_model_spec: Which API/model to use for generation
            enable_logging: Whether to enable logging
        """
        self.model_name = model_name
        self.api_model_spec = api_model_spec
        self.api = get_api_class(api_model_spec)(model=api_model_spec)

        # Game state tracking
        self.is_initialized = False
        self.game_type = None  # "Codenames", "ColonelBlotto", or "ThreePlayerIPD"
        self.player_id = None
        self.player_role = None
        self.observation_history = []
        self.turn_counter = 0

        # Initialize logger
        self.logger = GameLogger() if enable_logging else None

    def __call__(self, observation: str) -> str:
        """
        Process observation and generate appropriate response

        Args:
            observation: The current game observation string

        Returns:
            Action string in the format required by the game
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

            # Generate action using the API
            result = self._generate_action(observation)

            # End logging for this turn
            if self.logger:
                self.logger.end_turn(result)

            return result

        except Exception as e:
            error_msg = f"Error in ApiAgent: {e}"
            print(error_msg)

            if self.logger:
                self.logger.end_turn(f"ERROR: {str(e)}")

            # Provide a safe default response based on game type
            return self._get_safe_default()

    def _initialize_from_observation(self, observation: str):
        """
        Parse initial observation to determine game type and role

        Args:
            observation: Initial game observation string
        """
        # Detect game type
        observation_lower = observation.lower()

        if "codenames" in observation_lower or "spymaster" in observation_lower:
            self.game_type = "Codenames"
        elif "colonel blotto" in observation_lower or "commander" in observation_lower:
            self.game_type = "ColonelBlotto"
        elif "prisoner's dilemma" in observation_lower or "cooperate" in observation_lower:
            self.game_type = "ThreePlayerIPD"
        else:
            # Default fallback
            self.game_type = "Unknown"

        # Extract player info
        player_id_match = re.search(r"Player (\d+)", observation)
        if player_id_match:
            self.player_id = int(player_id_match.group(1))

        # Extract role (game-specific)
        if self.game_type == "Codenames":
            self.player_role = "Spymaster" if self.player_id in [0, 2] else "Operative"
        elif self.game_type == "ColonelBlotto":
            self.player_role = "Commander Alpha" if "Alpha" in observation else "Commander Beta"
        elif self.game_type == "ThreePlayerIPD":
            self.player_role = f"Player {self.player_id}"

        self.is_initialized = True

        # Set player info in logger
        if self.logger:
            self.logger.set_player_info(
                self.player_id if self.player_id is not None else -1,
                self.player_role if self.player_role is not None else "Unknown",
                "N/A"  # No teams in Track 2 games
            )

    def _generate_action(self, observation: str) -> str:
        """
        Generate action using a single prompt approach

        Args:
            observation: Current observation

        Returns:
            Action string
        """
        # Create game-specific prompt
        prompt = self._create_prompt(observation)

        # Call API
        response = self.api(input_messages=[
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": prompt}
        ])

        # Log the raw response
        if self.logger:
            self.logger.log_phase("api_response", prompt, response, None)

        # Parse and validate the response
        parsed_response = self._parse_response(response, observation)

        return parsed_response

    def _system_prompt(self) -> str:
        """Generate system prompt based on game type"""
        common_prompt = (
            "You are an expert game player participating in the Mind Games Challenge. "
            "Follow all game rules precisely and provide responses in the exact format required. "
            "Focus solely on making the optimal move based on the current game state."
        )

        if self.game_type == "Codenames":
            return common_prompt + (
                " In Codenames, you must either give clues as Spymaster ([word number]) "
                "or guess words as Operative ([word]). Be precise and strategic."
            )
        elif self.game_type == "ColonelBlotto":
            return common_prompt + (
                " In Colonel Blotto, you must allocate your units across battlefields "
                "in the format [A5 B10 C5]. Allocate exactly the available units."
            )
        elif self.game_type == "ThreePlayerIPD":
            return common_prompt + (
                " In the Three-Player Iterated Prisoner's Dilemma, either communicate "
                "strategically during discussion or make decisions in the format "
                "[player_id cooperate/defect]. Choose your actions carefully."
            )
        else:
            return common_prompt

    def _create_prompt(self, observation: str) -> str:
        """Create a game-specific prompt"""
        if self.game_type == "Codenames":
            return self._create_codenames_prompt(observation)
        elif self.game_type == "ColonelBlotto":
            return self._create_blotto_prompt(observation)
        elif self.game_type == "ThreePlayerIPD":
            return self._create_ipd_prompt(observation)
        else:
            return self._create_generic_prompt(observation)

    def _create_codenames_prompt(self, observation: str) -> str:
        """Create Codenames-specific prompt"""
        is_spymaster = self.player_role == "Spymaster"

        prompt = (
            f"You are playing Codenames as the {self.player_role}. "
            f"Carefully analyze the current game state and make a strategic move.\n\n"

            f"Current observation:\n{observation}\n\n"
        )

        if is_spymaster:
            prompt += (
                "As the Spymaster, you must provide a one-word clue and a number to help your "
                "Operative guess words belonging to your team. Format: [word number]\n\n"

                "Do not use any form or part of the words on the board. Your clue should be a single "
                "word that connects multiple words belonging to your team.\n\n"

                "Your response should be in the EXACT format: [word number]\n"
            )
        else:  # Operative
            prompt += (
                "As the Operative, you must guess a word based on your Spymaster's clue. "
                "Format: [word] or [pass]\n\n"

                "Choose a word that you believe belongs to your team based on the clue "
                "provided by your Spymaster. If you're uncertain, you can pass.\n\n"

                "Your response should be in the EXACT format: [word] or [pass]\n"
            )

        return prompt

    def _create_blotto_prompt(self, observation: str) -> str:
        """Create Colonel Blotto-specific prompt"""
        prompt = (
            f"You are playing Colonel Blotto as {self.player_role}. "
            f"Carefully analyze the current game state and make a strategic allocation.\n\n"

            f"Current observation:\n{observation}\n\n"

            "Allocate your units across the available fields. Your total units must equal exactly "
            "the number of units mentioned in the observation. Each field must have a non-negative "
            "integer number of units.\n\n"

            "Your response should be in the EXACT format: [A5 B10 C5]\n"
            "Where A, B, C are field names and 5, 10, 5 are the number of units allocated to each field.\n"
        )

        return prompt

    def _create_ipd_prompt(self, observation: str) -> str:
        """Create IPD-specific prompt"""
        is_conversation = "You can converse freely" in observation or not "Submit your decisions" in observation

        prompt = (
            f"You are playing a 3-player Iterated Prisoner's Dilemma as Player {self.player_id}. "
            f"Carefully analyze the current game state.\n\n"

            f"Current observation:\n{observation}\n\n"
        )

        if is_conversation:
            prompt += (
                "This is a conversation phase. You can freely communicate with other players "
                "to build trust, coordinate strategies, or negotiate agreements.\n\n"

                "Craft a strategic message that advances your interests while encouraging cooperation. "
                "Be persuasive but genuine.\n\n"

                "Your response should be natural language without any special formatting.\n"
            )
        else:  # decision phase
            opponent_ids = [i for i in range(3) if i != self.player_id]
            opponent_str = " and ".join([f"Player {i}" for i in opponent_ids])

            prompt += (
                "This is a decision phase. You must decide whether to cooperate or defect with "
                f"each opponent ({opponent_str}).\n\n"

                "Format your decision for each opponent as: [opponent_id cooperate] or [opponent_id defect]\n"
                "Example: [0 cooperate] [2 defect]\n\n"

                "Remember the payoff matrix:\n"
                "- Both cooperate: Mutual reward\n"
                "- Both defect: Mutual punishment\n"
                "- You defect, they cooperate: You get temptation payoff, they get sucker payoff\n"
                "- You cooperate, they defect: You get sucker payoff, they get temptation payoff\n\n"

                "Your response should be in the EXACT format: [opponent_id cooperate/defect] [opponent_id cooperate/defect]\n"
            )

        return prompt

    def _create_generic_prompt(self, observation: str) -> str:
        """Create a generic prompt when game type is unknown"""
        prompt = (
            "You are playing a strategic game in the Mind Games Challenge. "
            "Carefully analyze the current observation and make the best move possible.\n\n"

            f"Current observation:\n{observation}\n\n"

            "Provide your response in the exact format required by the game. "
            "Look for hints in the observation about the expected format.\n"
        )

        return prompt

    def _parse_response(self, response: str, observation: str) -> str:
        """
        Parse and validate the response based on game type

        Args:
            response: Raw response from API
            observation: Current observation

        Returns:
            Validated response
        """
        if self.game_type == "Codenames":
            return self._parse_codenames_response(response, observation)
        elif self.game_type == "ColonelBlotto":
            return self._parse_blotto_response(response, observation)
        elif self.game_type == "ThreePlayerIPD":
            return self._parse_ipd_response(response, observation)
        else:
            return response.strip()

    def _parse_codenames_response(self, response: str, observation: str) -> str:
        """Parse and validate Codenames response"""
        is_spymaster = self.player_role == "Spymaster"

        if is_spymaster:
            # Look for [word number] format
            clue_match = re.search(r'\[(\w+)\s+(\d+)\]', response)
            if clue_match:
                word, number = clue_match.group(1), clue_match.group(2)
                return f"[{word} {number}]"
            else:
                # Fallback: try to find any word and number
                word_match = re.search(r'[^\[\]]*(\w+)[^\[\]]*(\d+)', response)
                if word_match:
                    word, number = word_match.group(1), word_match.group(2)
                    return f"[{word} {number}]"
                return "[clue 1]"  # Safe default
        else:
            # Look for [word] format or [pass]
            word_match = re.search(r'\[(\w+)\]', response)
            if word_match:
                word = word_match.group(1).lower()
                if word == "pass":
                    return "[pass]"
                else:
                    return f"[{word}]"
            else:
                return "[pass]"  # Safe default

    def _parse_blotto_response(self, response: str, observation: str) -> str:
        """Parse and validate Colonel Blotto response"""
        # Extract total units and fields from observation
        units_match = re.search(r"allocate:?\s*(\d+)", observation)
        total_units = int(units_match.group(1)) if units_match else 20  # Default

        fields_match = re.search(r"fields:?\s+([A-Z,\s]+)", observation)
        fields = [f.strip() for f in fields_match.group(1).split(",")] if fields_match else ["A", "B", "C"]

        # Try to extract using bracket format [A4 B6 C10]
        bracket_match = re.search(r"\[(.*?)\]", response)
        if bracket_match:
            content = bracket_match.group(1).strip()
        else:
            # Try to extract field-unit pairs without brackets
            content = response.strip()

        # Parse the field allocations
        allocation = {}

        # Try format "A4 B6 C10"
        simple_pattern = r"([A-Z])(\d+)"
        simple_matches = re.findall(simple_pattern, content)

        for field, units in simple_matches:
            if field in fields:
                allocation[field] = int(units)

        # Check if we parsed any valid allocations
        if not allocation:
            # Generate uniform allocation
            base_units = total_units // len(fields)
            extra_units = total_units % len(fields)

            for i, field in enumerate(fields):
                allocation[field] = base_units + (1 if i < extra_units else 0)

        # Validate total units
        total = sum(allocation.values())
        if total > total_units:
            # Scale down proportionally
            scale_factor = total_units / total
            for field in allocation:
                allocation[field] = max(1, int(allocation[field] * scale_factor))

            # Adjust to exactly match total_units
            while sum(allocation.values()) > total_units:
                # Find max value and decrement
                max_field = max(allocation, key=allocation.get)
                allocation[max_field] -= 1

            while sum(allocation.values()) < total_units:
                # Find min value and increment
                min_field = min(allocation, key=allocation.get)
                allocation[min_field] += 1

        # Ensure all fields are included
        for field in fields:
            if field not in allocation:
                allocation[field] = 0

        # Convert to required format
        result = "[" + " ".join(f"{field}{allocation[field]}" for field in fields) + "]"
        return result

    def _parse_ipd_response(self, response: str, observation: str) -> str:
        """Parse and validate IPD response"""
        is_conversation = "You can converse freely" in observation or not "Submit your decisions" in observation

        if is_conversation:
            # For conversation, just clean up the response
            # Remove any decision format if present
            clean_response = re.sub(r'\[\d+\s+(cooperate|defect)\]', '', response, flags=re.IGNORECASE)
            return clean_response.strip()
        else:
            # For decision phase, extract player decisions
            decision_pattern = r'\[(\d+)\s+(cooperate|defect)\]'
            matches = re.findall(decision_pattern, response, re.IGNORECASE)

            # Get opponent IDs
            opponent_ids = [i for i in range(3) if i != self.player_id]

            # Create a dict to store decisions for each opponent
            decisions = {}
            for opponent_id, action in matches:
                opponent_id = int(opponent_id)
                if opponent_id != self.player_id and opponent_id in range(3):
                    decisions[opponent_id] = "cooperate" if "cooperate".startswith(action.lower()) else "defect"

            # Ensure all opponents have a decision (default to cooperate if missing)
            for opponent_id in opponent_ids:
                if opponent_id not in decisions:
                    decisions[opponent_id] = "cooperate"

            # Format the final decision string
            result = " ".join([f"[{opponent_id} {decisions[opponent_id]}]" for opponent_id in sorted(decisions.keys())])
            return result

    def _get_safe_default(self) -> str:
        """Provide a safe default response based on game type"""
        if self.game_type == "Codenames":
            if self.player_role == "Spymaster":
                return "[clue 1]"
            else:
                return "[pass]"
        elif self.game_type == "ColonelBlotto":
            return "[A7 B7 C6]"  # Assuming 20 total units
        elif self.game_type == "ThreePlayerIPD":
            opponent_ids = [i for i in range(3) if i != self.player_id]
            return " ".join([f"[{opponent_id} cooperate]" for opponent_id in opponent_ids])
        else:
            return "I need more information about the game."