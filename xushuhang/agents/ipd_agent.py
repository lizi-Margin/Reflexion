"""
Three Player IPD Agent - Mind Games Challenge Track 2

This module implements a specialized agent for the Three Player Iterated Prisoner's Dilemma game.

Game Rules:
- 3-player game with multiple rounds
- Each round has conversation phase followed by decision phase
- During conversation, players can freely communicate
- During decision, each player must decide to cooperate or defect with each other player
- Payoff matrix for each pair:
  - Both cooperate: R points each
  - Both defect: P points each
  - One defects, other cooperates: T points for defector, S points for cooperator
- Player(s) with highest score at the end win
"""

import re
import json
from typing import Dict, List, Set, Tuple, Optional
from src.agent import Agent
from xushuhang.agents.api_router import get_api_class
from xushuhang.agents.game_logger import GameLogger


class IPDAgent(Agent):
    """
    Specialized agent for playing Three Player IPD in Mind Games Challenge Track 2

    The agent uses multi-phase reasoning to:
    1. During conversation: Build trust and coordinate strategies
    2. During decision: Make strategic cooperate/defect choices
    """

    def __init__(self, model_name: str, api_model_spec='qwen3-8b', enable_logging: bool = True):
        """
        Initialize the IPDAgent

        Args:
            model_name: Name of the model for identification
            api_model_spec: Which API/model to use for generation
            enable_logging: Whether to enable logging
        """
        self.model_name = model_name
        self.api = get_api_class(api_model_spec)(model=api_model_spec)

        # Game state tracking
        self.is_initialized = False
        self.player_id = None  # 0, 1, or 2
        self.num_rounds = 0  # Total number of rounds
        self.current_round = 1  # Current round
        self.phase = None  # "conversation" or "decision"
        self.conversation_round = 0  # Current conversation round within the round
        self.total_conversation_rounds = 0  # Total conversation rounds per round

        # Game parameters
        self.R = 0  # Reward for mutual cooperation
        self.T = 0  # Temptation to defect
        self.S = 0  # Sucker's payoff
        self.P = 0  # Punishment for mutual defection

        # Player tracking
        self.scores = {0: 0, 1: 0, 2: 0}  # Current scores
        self.opponent_ids = []  # IDs of the opponents
        self.conversation_history = []  # List of conversation messages
        self.decision_history = {}  # {round -> {player -> {opponent -> action}}}
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
            Message for conversation or decisions for decision phase
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

            # Generate response based on phase
            if self.phase == "conversation":
                result = self._generate_conversation()
            else:  # decision phase
                result = self._generate_decisions()

            # End logging for this turn
            if self.logger:
                self.logger.end_turn(result)

            return result

        except Exception as e:
            error_msg = f"Error in IPDAgent: {e}"
            print(error_msg)

            if self.logger:
                self.logger.end_turn(f"ERROR: {str(e)}")

            # In case of error, return a safe default
            if self.phase == "conversation":
                return "I propose we all cooperate to maximize our collective score."
            else:
                # Default to cooperation for decisions
                return " ".join([f"[{opponent_id} cooperate]" for opponent_id in self.opponent_ids])

    def _initialize_from_observation(self, observation: str):
        """
        Parse initial observation to determine game parameters

        Args:
            observation: Initial game observation string
        """
        # Extract player ID
        player_id_match = re.search(r"You are Player (\d+)", observation)
        if player_id_match:
            self.player_id = int(player_id_match.group(1))
            self.opponent_ids = [i for i in range(3) if i != self.player_id]

        # Extract game parameters
        rounds_match = re.search(r"match lasts (\d+) rounds", observation)
        if rounds_match:
            self.num_rounds = int(rounds_match.group(1))

        conversation_match = re.search(r"(\d+) free-chat turns", observation)
        if conversation_match:
            self.total_conversation_rounds = int(conversation_match.group(1))

        # Extract payoff parameters
        self.R = self._extract_payoff(observation, r"Both cooperate\s*->\s*(\d+)")
        self.T = self._extract_payoff(observation, r"You defect, they cooperate -> (\d+)")
        self.S = self._extract_payoff(observation, r"You cooperate, they defect -> (\d+)")
        self.P = self._extract_payoff(observation, r"Both defect\s*->\s*(\d+)")

        # Set initial phase
        self.phase = "conversation"
        self.conversation_round = 0

        self.is_initialized = True

        # Set player info in logger
        if self.logger:
            self.logger.set_player_info(
                self.player_id,
                f"Player {self.player_id}",
                "N/A"  # No teams in IPD
            )

    def _extract_payoff(self, text: str, pattern: str) -> int:
        """Extract payoff value using regex pattern"""
        match = re.search(pattern, text)
        return int(match.group(1)) if match else 0

    def _update_game_state_from_observation(self, observation: str):
        """
        Update game state based on new observation

        Args:
            observation: Current game observation
        """
        # Check for round indicator
        round_start_match = re.search(r"Starting Round (\d+)", observation)
        if round_start_match:
            self.current_round = int(round_start_match.group(1))
            self.phase = "conversation"
            self.conversation_round = 0
            self.conversation_history = []

        # Check for conversation vs decision phase
        if "Chat finished for round" in observation or "Submit your decisions" in observation:
            self.phase = "decision"

        # Parse player messages (for conversation phase)
        player_msg_pattern = r"Player (\d+): (.*?)(?:\n|$)"
        player_messages = re.findall(player_msg_pattern, observation)
        for pid, msg in player_messages:
            speaker_id = int(pid)
            if speaker_id != self.player_id:
                self.conversation_history.append((speaker_id, msg.strip()))

        # Extract round results
        results_match = re.search(r"### Round (\d+) - Results:(.*?)Current scores:", observation, re.DOTALL)
        if results_match:
            round_num = int(results_match.group(1))
            results_text = results_match.group(2)

            # Parse pair-wise decisions
            decision_pattern = r"Player (\d+) vs Player (\d+) chose to (\w+) and (\w+) respectively"
            decisions = re.findall(decision_pattern, results_text)

            # Record decisions
            if round_num not in self.decision_history:
                self.decision_history[round_num] = {p: {} for p in range(3)}

            for p1, p2, p1_action, p2_action in decisions:
                p1, p2 = int(p1), int(p2)
                self.decision_history[round_num][p1][p2] = p1_action
                self.decision_history[round_num][p2][p1] = p2_action

            # Update scores
            scores_pattern = r"Player (\d+) \((\d+)\)"
            scores = re.findall(scores_pattern, observation)
            for player, score in scores:
                self.scores[int(player)] = int(score)

    def _generate_conversation(self) -> str:
        """
        Generate conversation message using multi-phase reasoning

        Returns:
            Conversation message
        """
        # Phase 1: Analysis of game state
        analysis_prompt = self._prompt_conversation_analysis()

        analysis_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_system()},
            {"role": "user", "content": analysis_prompt}
        ])

        # Extract analysis results
        analysis = self._parse_tag_section(analysis_response, "#ANALYSIS:")

        if self.logger:
            self.logger.log_phase("analysis", analysis_prompt, analysis_response, analysis)

        # Phase 2: Determine conversation strategy
        strategy_prompt = self._prompt_conversation_strategy(analysis)

        strategy_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_system()},
            {"role": "user", "content": strategy_prompt}
        ])

        # Extract strategy
        strategy = self._parse_tag_section(strategy_response, "#STRATEGY:")

        if self.logger:
            self.logger.log_phase("strategy", strategy_prompt, strategy_response, strategy)

        # Phase 3: Generate final message
        message_prompt = self._prompt_conversation_message(strategy)

        message_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_system()},
            {"role": "user", "content": message_prompt}
        ])

        # Extract message
        message = self._parse_tag_section(message_response, "#MESSAGE:")

        if self.logger:
            self.logger.log_phase("message", message_prompt, message_response, message)

        return message

    def _generate_decisions(self) -> str:
        """
        Generate decisions for both opponents using multi-phase reasoning

        Returns:
            Decision string in format "[0 cooperate] [2 defect]"
        """
        # Phase 1: Analysis of conversation and history
        analysis_prompt = self._prompt_decision_analysis()

        analysis_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_system()},
            {"role": "user", "content": analysis_prompt}
        ])

        # Extract analysis results
        analysis = self._parse_tag_section(analysis_response, "#ANALYSIS:")

        if self.logger:
            self.logger.log_phase("analysis", analysis_prompt, analysis_response, analysis)

        # Phase 2: Evaluate expected payoffs
        payoff_prompt = self._prompt_decision_payoff(analysis)

        payoff_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_system()},
            {"role": "user", "content": payoff_prompt}
        ])

        # Extract payoff evaluation
        payoff = self._parse_tag_section(payoff_response, "#PAYOFF:")

        if self.logger:
            self.logger.log_phase("payoff", payoff_prompt, payoff_response, payoff)

        # Phase 3: Make final decisions
        decision_prompt = self._prompt_decision_final(analysis, payoff)

        decision_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_system()},
            {"role": "user", "content": decision_prompt}
        ])

        # Extract decisions
        decisions = self._parse_tag_section(decision_response, "#DECISIONS:")

        if self.logger:
            self.logger.log_phase("decisions", decision_prompt, decision_response, decisions)

        # Parse and validate the decisions
        return self._parse_and_validate_decisions(decisions)

    def _parse_and_validate_decisions(self, decisions_text: str) -> str:
        """
        Parse and validate decision text, ensuring it follows game rules

        Args:
            decisions_text: Raw decisions text from the LLM

        Returns:
            Validated decision string like "[0 cooperate] [2 defect]"
        """
        # Define the decision pattern
        decision_pattern = r'\[(\d+)\s+(cooperate|defect)\]'
        matches = re.findall(decision_pattern, decisions_text, re.IGNORECASE)

        # Create a dict to store decisions for each opponent
        decisions = {}
        for opponent_id, action in matches:
            opponent_id = int(opponent_id)
            if opponent_id != self.player_id and opponent_id in range(3):
                decisions[opponent_id] = "cooperate" if "cooperate".startswith(action.lower()) else "defect"

        # Ensure all opponents have a decision (default to cooperate if missing)
        for opponent_id in self.opponent_ids:
            if opponent_id not in decisions:
                decisions[opponent_id] = "cooperate"

        # Format the final decision string
        result = " ".join([f"[{opponent_id} {decisions[opponent_id]}]" for opponent_id in sorted(decisions.keys())])
        return result

    # Prompt methods

    def _prompt_system(self) -> str:
        """System prompt for IPD agent"""
        return (
            f"You are Player {self.player_id} in a 3-player Iterated Prisoner's Dilemma game. "
            f"The game consists of {self.num_rounds} rounds, each with {self.total_conversation_rounds} conversation turns "
            f"followed by decisions. For each pair of players, the payoff matrix is:\n"
            f"- Both cooperate: {self.R} points each\n"
            f"- Both defect: {self.P} points each\n"
            f"- You defect, they cooperate: {self.T} points for you\n"
            f"- You cooperate, they defect: {self.S} points for you\n"
            f"Your goal is to maximize your score through strategic cooperation and defection."
        )

    def _prompt_conversation_analysis(self) -> str:
        """Prompt for conversation analysis phase"""
        prompt = (
            f"As Player {self.player_id} in Round {self.current_round}/{self.num_rounds}, "
            f"Conversation turn {self.conversation_round+1}/{self.total_conversation_rounds}, "
            f"you need to analyze the current game state and conversation history.\n\n"

            f"Current scores: {', '.join([f'Player {p}: {s}' for p, s in self.scores.items()])}\n\n"
        )

        # Add decision history if available
        if self.decision_history:
            prompt += "Previous rounds:\n"
            for round_num in sorted(self.decision_history.keys()):
                prompt += f"Round {round_num} decisions:\n"
                for p1 in range(3):
                    for p2 in range(p1 + 1, 3):
                        if p2 in self.decision_history[round_num][p1]:
                            prompt += f"- Player {p1} -> Player {p2}: {self.decision_history[round_num][p1][p2]}\n"
                            prompt += f"- Player {p2} -> Player {p1}: {self.decision_history[round_num][p2][p1]}\n"
            prompt += "\n"

        # Add conversation history for current round
        if self.conversation_history:
            prompt += "Conversation in current round:\n"
            for speaker_id, message in self.conversation_history:
                prompt += f"Player {speaker_id}: {message}\n"
            prompt += "\n"

        prompt += (
            f"Please analyze the current state:\n"
            f"1. What patterns of cooperation/defection have emerged?\n"
            f"2. Has any player been consistently cooperative or defective?\n"
            f"3. What promises or commitments have been made in conversation?\n"
            f"4. What is each player's likely strategy based on their behavior?\n"
            f"5. Who appears most trustworthy and who seems deceptive?\n\n"

            f"Begin your analysis and start with a symbol: '#ANALYSIS:'"
        )

        return prompt

    def _prompt_conversation_strategy(self, analysis) -> str:
        """Prompt for conversation strategy phase"""
        prompt = (
            f"Based on your analysis:\n\n{analysis}\n\n"

            f"Now, determine your conversation strategy for this turn. Consider:\n"
            f"1. Your goals in this conversation (build trust, coordinate, negotiate, signal intentions)\n"
            f"2. How to position yourself with each opponent\n"
            f"3. What information to reveal or conceal about your intentions\n"
            f"4. Whether to propose explicit agreements or strategies\n\n"

            f"Remember the game parameters:\n"
            f"- Both cooperate: {self.R} points each\n"
            f"- Both defect: {self.P} points each\n"
            f"- You defect, they cooperate: {self.T} points for you\n"
            f"- You cooperate, they defect: {self.S} points for you\n\n"

            f"What overall approach will you take in this conversation?\n\n"

            f"Begin your strategy and start with a symbol: '#STRATEGY:'"
        )

        return prompt

    def _prompt_conversation_message(self, strategy) -> str:
        """Prompt for generating the final conversation message"""
        prompt = (
            f"Based on your strategy:\n\n{strategy}\n\n"

            f"Now, craft your message to the other players. This message will be seen by all players.\n"
            f"Create a natural, conversational message that implements your strategy effectively.\n\n"

            f"Begin your message and start with a symbol: '#MESSAGE:'"
        )

        return prompt

    def _prompt_decision_analysis(self) -> str:
        """Prompt for decision analysis phase"""
        prompt = (
            f"As Player {self.player_id} in Round {self.current_round}/{self.num_rounds}, "
            f"you now need to decide whether to cooperate or defect with each opponent (Players {self.opponent_ids[0]} and {self.opponent_ids[1]}).\n\n"

            f"Current scores: {', '.join([f'Player {p}: {s}' for p, s in self.scores.items()])}\n\n"
        )

        # Add decision history if available
        if self.decision_history:
            prompt += "Previous rounds:\n"
            for round_num in sorted(self.decision_history.keys()):
                prompt += f"Round {round_num} decisions:\n"
                for p1 in range(3):
                    for p2 in range(p1 + 1, 3):
                        if p2 in self.decision_history[round_num][p1]:
                            prompt += f"- Player {p1} -> Player {p2}: {self.decision_history[round_num][p1][p2]}\n"
                            prompt += f"- Player {p2} -> Player {p1}: {self.decision_history[round_num][p2][p1]}\n"
            prompt += "\n"

        # Add conversation history for current round
        if self.conversation_history:
            prompt += "Conversation in current round:\n"
            for speaker_id, message in self.conversation_history:
                prompt += f"Player {speaker_id}: {message}\n"
            prompt += "\n"

        prompt += (
            f"Please analyze each opponent separately:\n"
            f"1. What is Player {self.opponent_ids[0]}'s likely strategy based on past behavior?\n"
            f"2. What is Player {self.opponent_ids[1]}'s likely strategy based on past behavior?\n"
            f"3. What promises or commitments have been made by each player?\n"
            f"4. How trustworthy does each player appear to be?\n"
            f"5. What signals have you given about your own intentions?\n\n"

            f"Begin your analysis and start with a symbol: '#ANALYSIS:'"
        )

        return prompt

    def _prompt_decision_payoff(self, analysis) -> str:
        """Prompt for decision payoff evaluation phase"""
        prompt = (
            f"Based on your analysis:\n\n{analysis}\n\n"

            f"Now, evaluate the expected payoffs for different decision combinations. "
            f"You need to decide whether to cooperate or defect with Players {self.opponent_ids[0]} and {self.opponent_ids[1]}.\n\n"

            f"Game parameters:\n"
            f"- Both cooperate: {self.R} points each\n"
            f"- Both defect: {self.P} points each\n"
            f"- You defect, they cooperate: {self.T} points for you\n"
            f"- You cooperate, they defect: {self.S} points for you\n\n"

            f"Consider these four possible strategies and their expected outcomes:\n"
            f"1. Cooperate with both opponents\n"
            f"2. Cooperate with Player {self.opponent_ids[0]}, defect against Player {self.opponent_ids[1]}\n"
            f"3. Defect against Player {self.opponent_ids[0]}, cooperate with Player {self.opponent_ids[1]}\n"
            f"4. Defect against both opponents\n\n"

            f"For each strategy, estimate:\n"
            f"- The likelihood each opponent will cooperate or defect\n"
            f"- Your expected payoff based on these probabilities\n"
            f"- The impact on future rounds and your reputation\n\n"

            f"Begin your payoff evaluation and start with a symbol: '#PAYOFF:'"
        )

        return prompt

    def _prompt_decision_final(self, analysis, payoff) -> str:
        """Prompt for final decision phase"""
        prompt = (
            f"Based on your analysis and payoff evaluation:\n\n"
            f"Analysis: {analysis}\n\n"
            f"Payoff evaluation: {payoff}\n\n"

            f"Now, make your final decisions for Player {self.opponent_ids[0]} and Player {self.opponent_ids[1]}.\n\n"

            f"Consider:\n"
            f"1. Your short-term payoff for this round\n"
            f"2. The long-term impact on your reputation and future cooperation\n"
            f"3. Your overall strategy for the game\n"
            f"4. The current scores and round number ({self.current_round}/{self.num_rounds})\n\n"

            f"Provide your decisions in EXACTLY this format:\n"
            f"[{self.opponent_ids[0]} cooperate/defect] [{self.opponent_ids[1]} cooperate/defect]\n\n"

            f"Where you replace 'cooperate/defect' with your actual decision for each opponent.\n\n"

            f"Begin your decisions and start with a symbol: '#DECISIONS:'"
        )

        return prompt

    # Utility methods

    def _parse_tag_section(self, text: str, tag: str) -> str:
        """Extract content after a specific tag"""
        index = text.find(tag)
        if index != -1:
            return text[index + len(tag):].strip()
        return text  # Return full text if tag not found