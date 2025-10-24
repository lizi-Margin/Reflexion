import re
import random
import copy, traceback
from typing import Dict, List, Tuple, Optional
from envs.agent import Agent
from api.api_router import get_api_class
from reflexion.game_logger import GameLogger
from uhtk.print_pack import *
from reflexion.blotto_runs.blotto_memory import BlottoMemory


class BlottoAgent(Agent):
    def __init__(self, model_name: str, api_model_spec='qwen3-8b', enable_logging: bool = True, memory: BlottoMemory = None):
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
        self.round_results = []  # Results of each round
        self.observation_history = []
        self.turn_counter = 0

        self.strategy = None
        self.max_reflections = 6
        # Inter-trial memory (reflexion) - optional, can be None
        self.memory = memory

        self.api_model_spec = api_model_spec

        # Initialize logger
        self.logger = GameLogger() if enable_logging else None

    def __call__(self, observation: str) -> str:
        try:
            # First turn initialization
            if not self.is_initialized:
                self._initialize_from_observation(observation)

            # Regular turn processing
            self.observation_history.append(observation)
            self.turn_counter += 1

            # Update game state from observation
            self._update_game_state_from_observation(observation)

            # Start logging this turn
            if self.logger:
                self.logger.start_turn(self.turn_counter, observation)
                print("=" * 60)
                print(f"My id: {self.player_id}")
                print(f"Turn Cnt: {self.turn_counter}:")
                print(f"Round Cnt: {self.current_round}:")
                print(f"Scores: {self.scores}")
                print(f"round_results: {self.round_results}")



            # Generate allocation
            result = self._generate_allocation()

            # End logging for this turn
            if self.logger:
                self.logger.end_turn(result)
                print_bold_blue("result:")
                print(result)
                print("=" * 60)

            return result

        except Exception as e:
            error_msg = f"Error in BlottoAgent: {e}"
            print(error_msg)
            print(traceback.format_exc())

            if self.logger:
                self.logger.end_turn(f"ERROR: {str(e)}")

            # In case of error, return a uniform allocation as fallback
            return self._generate_uniform_allocation()

    def _initialize_from_observation(self, observation: str):
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

        rounds_match = re.findall(r"Round (\d+)/(\d+)", observation)
        if rounds_match:
            self.current_round = int(rounds_match[-1][0])
            self.num_rounds = int(rounds_match[-1][1])
        
        # Round 3
        # Commander Alpha allocated: A: 7 , B: 7 , C: 6 
        # Commander Beta allocated:  A: 7 , B: 7 , C: 6 
        # Tie!
        result_match = re.findall(
            r"Round (\d+)\s*\n(Commander Alpha allocated:.*?)\n(Commander Beta allocated:.*?)\n(.*?)\n",
            observation, re.DOTALL)  
        if result_match:
            round_num = int(result_match[-1][0])
            # assert round_num == len(self.round_results) + 1, f"round_num {round_num} != len(self.round_results) + 1 {len(self.round_results) + 1}"
            if round_num >= len(self.round_results) + 1:
                self.round_results.append(result_match[-1][1:])
        else:
            if self.current_round > 1:
                assert False, f"current_round {self.current_round} > 1, but no round result found in observation"

        # Check for previous round results
        score_match = re.findall(r"Rounds Won - Commander Alpha: (\d+), Commander Beta: (\d+)", observation, re.DOTALL)
        if score_match:
            alpha_score = int(score_match[-1][0])
            beta_score = int(score_match[-1][1])
            self.scores[0] = alpha_score
            self.scores[1] = beta_score
   

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
        if self.logger:
            print_bold_blue("start _generate_allocation()...")
        # Phase 1: Analysis of opponent's history
        if self.current_round == 1:
            analysis = analysis_response = analysis_prompt = "None, now is first round, you need to decide your strategy."
        else:
            analysis_prompt = self._prompt_analysis()

            analysis_response = self.api(input_messages=[
                {"role": "system", "content": self._prompt_system()},
                {"role": "user", "content": analysis_prompt}
            ])

            # Extract analysis results
            analysis = self._parse_tag_section(analysis_response, "#ANALYSIS:")

        if self.logger:
            print_bold_blue("analysis_prompt:")
            print_indigo(analysis_prompt)
            print_bold_blue("analysis:")
            print(analysis)
            self.logger.log_phase("analysis", analysis_prompt, analysis_response, analysis)

        # Phase 2: Strategy selection
        strategy_prompt = self._prompt_strategy(analysis)

        strategy_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_system()},
            {"role": "user", "content": strategy_prompt}
        ])

        # Extract strategy
        strategy = self._parse_tag_section(strategy_response, "#STRATEGY:")
        self.strategy = strategy

        if self.logger:
            self.logger.log_phase("strategy", strategy_prompt, strategy_response, strategy)
            print_bold_blue("strategy_prompt:")
            print(strategy_prompt)
            print_bold_blue("strategy:")
            print(strategy)

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
            print_bold_blue("allocation_prompt:")
            print(allocation_prompt)
            print_bold_blue("allocation:")
            print(allocation)

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

    
        prompt += "Previous rounds:\n"
        for i, round_result in enumerate(self.round_results):
            prompt += f"Round {i+1}\{self.num_rounds}:\n"
            prompt += f"{round_result[0]}\n"
            prompt += f"{round_result[1]}\n"
            prompt += f"{round_result[2]}\n\n"

        prompt += (
            f"Please analyze the opponent's behavior:\n"
            f"1. Is there a pattern in how they allocate units?\n"
            f"2. Do they favor particular fields?\n"
            f"3. How do they respond to your allocations?\n"
            f"4. What might they do next based on the current score and round?\n\n"

            f"You can briefly anayze the opponent's behavior in Past rounds, Now round, Future predictions (it is okay to say unpredictable or random)\n"
            f"State the status briefly and clearly, DO NOT suggest any strategy since they will be considered later.\n\n"
        )

        prompt += "Begin your analysis and start with a symbol: '#ANALYSIS:'"
        return prompt

    def _prompt_strategy(self, analysis) -> str:
        """Prompt for strategy selection phase"""
        prompt = (
            f"Based on your analysis:\n\n{analysis}\n\n"

            f"Now, persist or refine your overall strategy (multi-round, early-stage, mid-stage, and final-stage) for this game. Consider:\n"
            f"1. Current game state (round {self.current_round}/{self.num_rounds}, score {self.scores[0]}-{self.scores[1]})\n"
            f"2. Opponent's patterns and tendencies\n"
            f"3. Posible strategic to defeat the opponent pattern:\n"
            f"   There are some simple strategy for each round:\n"
            f"   - Uniform: Spread units evenly across fields\n"
            f"   - Concentrated-2: Focus units on 2 fields (like [A:0 B:10 C:10])\n"
            f"   - Adaptive: Predict and counter opponent's expected moves\n"
            f"   And some more complexed examples:"
            f"   - Due to unknown opponent's strategy, some player might use Uniform strategy in the first round. If you use Concentrated-2 strategy, you might win. But if the opponent use a Concentrated-2 strategy as well, it may be ramdom (win or lose in this round).\n"
            f"   - This game is a multi-round ({self.num_rounds}) game, opponent will see your action after the first round, so you need to consider the long-term strategy.\n"
        )

        if not self.strategy:
            prompt += (
                f"This is the first round, so you have no information about your opponent yet.\n"
                f"Please construct your strategy based on past leasons:\n"
            )
        else:
            prompt += (
                f"Please consider your strategy based on past leasons:\n"
            )
        if self.memory:
            prompt += self.memory.get_guidance(max_reflections=self.max_reflections)
        else:
            prompt += "None, you have no past leasons.\n"


        prompt += (
            f"Determine your overall approach and reasoning. Make it brief but show the clear intention.\n\n"
            f"To state your strategy clearly, you can discribe it in Intention-Methodology structure.\n\n"
            f"Begin your strategy and start with a symbol: '#STRATEGY:'"
        )

        return prompt

    def _prompt_allocation(self, strategy) -> str:
        """Prompt for final allocation phase"""
        if  len(self.round_results) > 0:
            prompt = "Previous rounds:\n"
            for i, round_result in enumerate(self.round_results):
                prompt += f"Round {i+1}\{self.num_rounds}:\n"
                prompt += f"{round_result[0]}\n"
                prompt += f"{round_result[1]}\n"
                prompt += f"{round_result[2]}\n\n"
        else:
            prompt = "This is the first turn.\n"

        prompt += (
            f"Based on your strategy:\n\n{strategy}\n\n"
            f"Now, make your final allocation decision for round {self.current_round}. You have {self.total_units} units "
            f"to allocate across fields {', '.join(self.fields)}.\n\n"

            f"Calculate precise unit counts for each field. Remember:\n"
            f"1. The total must equal exactly {self.total_units} units\n"
            f"2. Each field must have a non-negative integer number of units\n"
            f"3. Your allocation must implement your chosen strategy effectively\n\n"

            f"Provide your allocation in EXACTLY and STRICTLY this format: [A:0, B:10, C:10]\n"
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

    def finalize_game(self, rewards: dict = None, game_info: dict = None):
        """
        Finalize the game and update memory with reflections (if memory is enabled)

        Args:
            rewards: Rewards dict from env.close()
            game_info: Game info dict from env.close()
        """
        # If no memory, skip finalization
        if self.memory is None:
            print(f"\n[BlottoAgent] Game finished! No memory enabled, skipping reflection.")
            return

        # Determine winner from scores (already updated from last observation)
        won = self.scores[self.player_id] > self.scores[1 - self.player_id]
        rounds_won = self.scores[self.player_id]
        opponent_score = self.scores[1 - self.player_id]

        print(f"\n[BlottoAgent] Game finished! Score: {rounds_won}-{opponent_score}, Won: {won}")

        # Generate reflections and update memory
        self.memory.update_memory_from_trial(
            self,
            won=won,
            should_reflect=True  # Always reflect to learn
        )

        # Add trial result
        if self.logger:
            trial_log_path = self.logger.run_dir / "game_log.json"
        else:
            trial_log_path = 'None'

        self.memory.add_trial_result({
            "won": won,
            "rounds_won": rounds_won,
            "total_rounds": self.num_rounds,
            "final_score": rounds_won,
            "opponent_score": opponent_score,
            "game_log": str(trial_log_path)
        })

        # Print statistics
        stats = self.memory.get_statistics()
        print(f"[BlottoAgent] Memory Statistics:")
        print(f"  Total Trials: {stats['total_trials']}")
        print(f"  Win Rate: {stats['win_rate']:.2%}")
        print(f"  Avg Rounds Won: {stats['avg_rounds_won']:.2f}")

        # Finalize logger
        if self.logger and self.logger.run_dir:
            self.logger.finalize(outcome=f"Score: {rounds_won}-{opponent_score}, Won: {won}")