import re
import json
import uuid
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from envs.agent import Agent
from api.api_router import get_api_class
from reflexion.game_logger import GameLogger
from reflexion.ipd_runs.ipd_memory import IPDMemory
from uhtk.print_pack import *


class IPDAgent(Agent):
    """
    Specialized agent for playing Three Player IPD in Mind Games Challenge Track 2

    The agent uses multi-phase reasoning to:
    1. During conversation: Build trust and coordinate strategies
    2. During decision: Make strategic cooperate/defect choices, with dynamic goals
       based on game state (e.g., rank, round number).

    Enhanced with inter-trial memory using reflexion framework:
    - Learns from past games through reflections
    - Integrates lessons into conversation and decision strategies
    """

    def __init__(self, model_name: str, api_model_spec='qwen3-8b', enable_logging: bool = True,
                 strategy_pool_manager=None, memory: IPDMemory = None):
        """
        Initialize the IPDAgent

        Args:
            model_name: Name of the model for identification
            api_model_spec: Which API/model to use for generation
            enable_logging: Whether to enable logging
            strategy_pool_manager: Reference to global StrategyPoolManager (None for standalone)
            memory: IPDMemory instance for inter-trial learning (creates new if None)
        """
        self.model_name = model_name
        self.api_model_spec = api_model_spec
        self.api = get_api_class(api_model_spec)(model=api_model_spec)

        # Strategy pool manager (shared across all agents in training)
        self.strategy_manager = strategy_pool_manager
        self.strategy_usage_log = []  # Log of strategies used this game

        # Inter-trial memory (reflexion)
        # self.memory = memory if memory is not None else IPDMemory(api_model_spec=api_model_spec)
        self.memory = memory

        # Game state tracking
        self.is_initialized = False
        self.player_id = None  # 0, 1, or 2
        self.num_rounds = 0  # Total number of rounds
        self.current_round = 1  # Current round
        self.phase = None  # "conversation" or "decision"

        # Game parameters
        self.R = 0  # Reward for mutual cooperation
        self.T = 0  # Temptation to defect
        self.S = 0  # Sucker's payoff
        self.P = 0  # Punishment for mutual defection

        # Player tracking
        self.scores = {0: 0, 1: 0, 2: 0}  # Current scores
        self.scores_history = []  # List of scores history
        self.opponent_ids = []  # IDs of the opponents
        # self.conversation_history = []  # List of conversation messages
        self.decision_history = {}  # {round -> {player -> {opponent -> action}}}
        self.observation_history = []
        self.call_counter = 0

        self.max_reflections = 6

        # Initialize logger
        self.logger = GameLogger() if enable_logging else None

        self.strategy = None
        self.payoff = None

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
            self.call_counter += 1

            # Start logging this turn
            if self.logger:
                self.logger.start_turn(self.call_counter, observation)
                print("=" * 60)
                print(f"My id: {self.player_id}")
                print(f"Opponnent ids: {self.opponent_ids}")
                print(f"Call Cnt: {self.call_counter}:")
                print(f"Phase: {self.phase}")
                print(f"Scores: {self.scores}")
                print(f"Decision History: {self.decision_history}")

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
                # print(result)
                print("=" * 60)
            return result

        except Exception as e:
            error_msg = f"Error in IPDAgent: {e}"
            print(error_msg)
            raise e

            if self.logger:
                self.logger.end_turn(f"ERROR: {str(e)}")

            # In case of error, return a safe default
            if self.phase == "conversation":
                return "I propose we all cooperate to maximize our collective score."
            else:
                # Default to cooperation for decisions
                return " ".join([f"[{opponent_id} cooperate]" for opponent_id in self.opponent_ids])

    def finalize_game(self, final_observation: str = None):
        """
        Finalize the game and update memory with reflections

        Args:
            final_observation: Final observation with game results (optional)
        """
        if not self.memory:
            return
        # Extract final results
        if final_observation:
            self._update_game_state_from_observation(final_observation)

        # Determine rank
        rankings = self._get_player_rankings()
        my_rank = rankings[self.player_id]['rank']
        my_score = rankings[self.player_id]['score']
        won = my_rank == 1

        print(f"\n[IPDAgent] Game finished! Rank: {my_rank}, Score: {my_score}, Won: {won}")


        # Generate reflections and update memory
        self.memory.update_memory_from_trial(
            self,
            trial_rank=my_rank,
            won=won,
            should_reflect=(not won or my_rank > 1)  # Reflect on losses or non-wins
        )

        # Add trial result

        if self.logger:
            trial_log_path = self.logger.run_dir / "game_log.json"
        else:
            trial_log_path = 'None'
        self.memory.add_trial_result({
            "rank": my_rank,
            "score": my_score,
            "opponent_scores": {k: v for k, v in self.scores.items() if k != self.player_id},
            "won": won,
            "game_log": str(trial_log_path),
            "m_rounds": self.num_rounds
        })

        # Print statistics
        stats = self.memory.get_statistics()
        print(f"[IPDAgent] Memory Statistics:")
        print(f"  Total Trials: {stats['total_trials']}")
        print(f"  Win Rate: {stats['win_rate']:.2%}")
        print(f"  Avg Rank: {stats['avg_rank']:.2f}")

        # Update memory with trial result
        if self.logger and self.logger.run_dir:
            

            # Finalize logger
            self.logger.finalize(outcome=f"Rank {my_rank}, Score {my_score}")

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


        # Extract payoff parameters
        self.R = self._extract_payoff(observation, r"Both cooperate\s*->\s*(\d+)")
        self.T = self._extract_payoff(observation, r"You defect, they cooperate -> (\d+)")
        self.S = self._extract_payoff(observation, r"You cooperate, they defect -> (\d+)")
        self.P = self._extract_payoff(observation, r"Both defect\s*->\s*(\d+)")

        # Set initial phase
        self.phase = "conversation"

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
        # Track previous phase to detect round transitions
        previous_phase = self.phase

        # [GAME] ─── Starting Round 1 ─── You can converse freely for the next 1 rounds.
        round_markers = re.findall(r"\[GAME\] ─── Starting Round (\d+) ───", observation)
        round_conversation_markers = re.findall(r"\[GAME\] ─── Starting Round (\d+) ───  You can converse freely", observation)
        # [GAME] Chat finished for round 1. Submit your decisions, one token per opponent: `[pid cooperate]` or `[pid defect]`.
        round_decision_markers = re.findall(r"\[GAME\] Chat finished for round (\d+)", observation)
        current_round = int(round_markers[-1]) if round_markers else None
        current_conv_round = int(round_conversation_markers[-1]) if round_conversation_markers else None
        current_decision_round = int(round_decision_markers[-1]) if round_decision_markers else None

        # assert current_round == current_conv_round and current_round is not None, f"Round markers mismatch: {current_round} != {current_conv_round}"
        self.current_round = current_round
        if current_decision_round is None or current_decision_round < current_round:
            self.phase = "conversation"
        else:
            self.phase = "decision"

        # # Parse player messages (for conversation phase)
        # player_msg_pattern = r"\[Player (\d+)\] (.+?)(?=\n\[Player \d+\]|\n\[GAME\]|$)"
        # player_messages = re.findall(player_msg_pattern, observation, re.DOTALL)
        # for pid, msg in player_messages:
        #     speaker_id = int(pid)
        #     if speaker_id != self.player_id:
        #         # Clean up the message (remove extra whitespace/newlines)
        #         clean_msg = msg.strip().replace('\n', ' ')
        #         # Avoid duplicates by checking if already in history
        #         if not self.conversation_history or self.conversation_history[-1] != (speaker_id, clean_msg):
        #             self.conversation_history.append((speaker_id, clean_msg))

        # Extract ALL round results from observation (observation contains full game history)
        # Find all "### Round X - Results:" sections
        round_results_pattern = r"### Round (\d+) - Results:(.*?)(?=### Round \d+ - Results:|─── Starting Round \d+ ───|$)"
        all_round_matches = re.findall(round_results_pattern, observation, re.DOTALL)


        # Process each round's results
        for round_num_str, results_text in all_round_matches:
            round_num = int(round_num_str)

            # Parse pair-wise decisions for this round
            decision_pattern = r"Player (\d+) vs Player (\d+) chose to (\w+) and (\w+) respectively"
            decisions = re.findall(decision_pattern, results_text)

            if decisions:  # Only update if we found decisions
                # Record decisions
                if round_num not in self.decision_history:
                    self.decision_history[round_num] = {p: {} for p in range(3)}

                for p1, p2, p1_action, p2_action in decisions:
                    p1, p2 = int(p1), int(p2)
                    self.decision_history[round_num][p1][p2] = p1_action
                    self.decision_history[round_num][p2][p1] = p2_action

        # Update scores - look for "Current scores: Player X (score)" pattern
        scores_pattern = r"Player (\d+) \((\d+)\)"
        scores = re.findall(scores_pattern, observation)
        if scores:
            for player, score in scores:
                self.scores[int(player)] = int(score)
        self.scores_history.append(self.scores.copy())

    def _generate_conversation(self) -> str:
        """
        Generate conversation message using multi-phase reasoning

        Returns:
            Conversation message
        """
        if self.logger:
            print_bold_blue("start _generate_conversation()...")
        # Phase 1: Analysis of game state
        if self.current_round > 1:
            analysis_prompt = self._prompt_conversation_analysis()
            analysis_response = self.api(input_messages=[
                {"role": "system", "content": self._prompt_system()},
                {"role": "user", "content": analysis_prompt}
            ])

            # Extract analysis results
            analysis = self._parse_tag_section(analysis_response, "#ANALYSIS:")
        else:
            analysis = "None, now is first round, you need to decide your strategy."
            analysis_response = analysis_prompt = analysis

        if self.logger:
            self.logger.log_phase("analysis", analysis_prompt, analysis_response, analysis)
            print_bold_blue("analysis generated:")
            print(analysis)

        # Phase 2: Determine conversation strategy
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
            print_bold_blue("strategy generated:")
            print(strategy)

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
            print_bold_blue("message generated:")
            print(message)

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
            print_bold_blue("analysis generated:")
            print(analysis)

        # Phase 2: Evaluate expected payoffs
        strategy_prompt = self._prompt_strategy(analysis)

        strategy_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_system()},
            {"role": "user", "content": strategy_prompt}
        ])

        # Extract payoff evaluation
        strategy = self._parse_tag_section(strategy_response, "#STRATEGY:")
        self.strategy = strategy

        if self.logger:
            self.logger.log_phase("strategy", strategy_prompt, strategy_response, strategy)
            print_bold_blue("strategy generated:")
            print(strategy)

        # Phase 3: Make final decisions
        decision_prompt = self._prompt_decision_final(analysis, strategy)

        decision_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_system()},
            {"role": "user", "content": decision_prompt}
        ])

        # Extract decisions
        decisions = self._parse_tag_section(decision_response, "#DECISIONS:")

        # # Log decision strategy usage
        # if self.strategy_manager:
        #     # Extract strategy reasoning before #DECISIONS:
        #     decision_strategy = decision_response.split("#DECISIONS:")[0].strip()
        #     sid = self.create_strategy("decision", decision_strategy)
        #     self.log_strategy_usage(sid, "decision")

        if self.logger:
            self.logger.log_phase("decisions", decision_prompt, decision_response, decisions)
            print_bold_blue("decisions generated:")
            print(decisions)

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

    # --- NEW HELPER METHODS ---
    def _get_player_rankings(self) -> Dict:
        """Calculates and returns the current ranking of all players."""
        sorted_scores = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)
        rankings = {}
        rank = 1
        for i, (player_id, score) in enumerate(sorted_scores):
            if i > 0 and score < sorted_scores[i-1][1]:
                rank = i + 1
            rankings[player_id] = {"rank": rank, "score": score}
        return rankings

    def _summarize_opponent_behavior(self, opponent_id: int) -> str:
        """Summarizes an opponent's past actions towards us."""
        defections = 0
        cooperations = 0
        for round_num, round_data in self.decision_history.items():
            if self.player_id in round_data.get(opponent_id, {}):
                action = round_data[opponent_id][self.player_id]
                if action == 'defect':
                    defections += 1
                else:
                    cooperations += 1
        
        total_interactions = defections + cooperations
        if total_interactions == 0:
            return "No prior interactions recorded."
        
        return (f"Out of {total_interactions} interactions, Player {opponent_id} has defected against you {defections} times "
                f"and cooperated {cooperations} times.")

    # --- MODIFIED PROMPT METHODS ---

    def _prompt_system(self) -> str:
        """System prompt for IPD agent"""
        # MODIFICATION: Changed goal from "maximize your score" to "win the game"
        return (
            f"You are Player {self.player_id}, a highly strategic and competitive agent in a 3-player Iterated Prisoner's Dilemma game. "
            f"The game consists of {self.num_rounds} rounds, each round consists of 1 conversation turn and 1 decision turn. "
            f"Your PRIMARY GOAL is to have the highest score at the end of the game and WIN. Maximizing your score is secondary to winning. "
            f"For each pair of players, the payoff matrix is:\n"
            f"- Both cooperate: {self.R} points each\n"
            f"- Both defect: {self.P} points each\n"
            f"- You defect, they cooperate: {self.T} points for you\n"
            f"- You cooperate, they defect: {self.S} points for you\n"
            f"Adapt your strategy based on your current rank, the remaining rounds, and your opponents' behavior."
        )

    def _prompt_conversation_analysis(self) -> str:
        """Prompt for conversation analysis phase"""
        prompt = (
            f"As Player {self.player_id} in Round {self.current_round}/{self.num_rounds}, "
            f"Now is conversation turn of round {self.current_round}, "
            f"you need to analyze the current game state and conversation history.\n\n"

            f"Current scores: {', '.join([f'Player {p}: {s}' for p, s in self.scores.items()])}\n\n"
        )

        if self.decision_history:
            prompt += "Previous rounds:\n"
            # This part is fine, no changes needed here.
            for round_num in sorted(self.decision_history.keys()):
                prompt += f"Round {round_num} decisions:\n"
                for p1 in range(3):
                    for p2 in range(p1 + 1, 3):
                        if p2 in self.decision_history[round_num].get(p1, {}):
                            prompt += f"- Player {p1} vs Player {p2}: {self.decision_history[round_num][p1][p2]} vs {self.decision_history[round_num][p2][p1]}\n"
            prompt += "\n"

        # if self.conversation_history:
        #     prompt += "Conversation in current round:\n"
        #     for speaker_id, message in self.conversation_history:
        #         prompt += f"Player {speaker_id}: {message}\n"
        #     prompt += "\n"
        assert self.call_counter == len(self.observation_history)
        prompt += f"----------------------- Current observation -------------------------\n"
        prompt += self.observation_history[-1]
        prompt += f"---------------------------------------------------------------------\n"

        prompt += (
            f"Please analyze the current state:\n"
            f"1. What patterns of cooperation/defection have emerged?\n"
            f"2. Has any player been consistently cooperative or defective?\n"
            f"3. What promises or commitments have been made in conversation? How credible are they given past actions?\n"
            f"4. What is each player's likely strategy based on their behavior?\n"
            f"5. Who appears most deceptive and who seems trustworthy? REMEMBER: Actions speak louder than words.\n\n"

            f"Your analysis should be brief but contains analysis of each opponent's behavior.\n"
            f"Begin your analysis and start with a symbol: '#ANALYSIS:'"
        )

        return prompt

    def _prompt_strategy(self, analysis, decision=False) -> str:
        """Prompt for conversation strategy phase"""

        # Get memory guidance
        if self.memory:
            memory_guidance = self.memory.get_guidance(max_reflections=self.max_reflections)
        else:
            memory_guidance = "No Memory, No Lessons now."
        prompt = ""
        if self.decision_history:
            prompt += "Previous rounds:\n"
            # This part is fine, no changes needed here.
            for round_num in sorted(self.decision_history.keys()):
                prompt += f"Round {round_num} decisions:\n"
                for p1 in range(3):
                    for p2 in range(p1 + 1, 3):
                        if p2 in self.decision_history[round_num].get(p1, {}):
                            prompt += f"- Player {p1} vs Player {p2}: {self.decision_history[round_num][p1][p2]} vs {self.decision_history[round_num][p2][p1]}\n"
            prompt += "\n"
        
        if self.strategy is None:
            assert self.current_round == 1, f"current_round should be 1, but got {self.current_round}"
            prompt += f"Now is the first round, you need to decide your strategy.\n"
        else:
            prompt += f"Our previous strategy is: {self.strategy}\n"


        prompt += (
            f"Based on your analysis:\n\n{analysis}\n\n"
            f"{memory_guidance}"
            f"Now, determine your conversation strategy and decision strategy for this turn. Your goal is to WIN the game. Consider:\n"
            f"1. Your goals in this conversation: Should deceive an opponent, call out a defector or you build trust, form a temporary alliance?\n"
            # f"2. Conversation strategy can be different from decision strategy, can we propose a conversation strategy to decieve the opponents?Or we can call out a defector?Or we just build trust and propose explicit agreements with a temporary alliance?\n"
            f"2. Conversation strategy can be different from decision strategy. For example, you can maintain a cooperative public image to sustain predictable behavior from others while quietly planning calculated defections that yield an irreversible lead."
            f"3. What decision strategy do you want to use based on current conversation startegy? For example, you can build trust in our conversation but defect opponent in the decision strategy.\n"
            f"4. Short term and long term advantages of your strategy? Do you think the opponent is clever enough to counter your strategy when you're considering long term?\n"
        )
        if self.strategy is not None:
            prompt += f"5. Does the previous strategy work well? If not, how can we change it?\n"

        if not decision:
            prompt += (
                f"Remember the game parameters:\n"
                f"- Both cooperate: {self.R} points each\n"
                f"- Both defect: {self.P} points each\n"
                f"- You defect, they cooperate: {self.T} points for you\n"
                f"- You cooperate, they defect: {self.S} points for you\n\n"
                f"So in the last round, there may be a optimal decesion choice based on the theory of games.\n"

                f"Now is conversation turn. What overall approach will you take in this conversation and the whole game (overall strategy) to improve your chances of winning?\n\n"

            )
        else:
            prompt += (
                f"\n"
                f"Game parameters:\n"
                f"- Both cooperate: {self.R} points each\n"
                f"- Both defect: {self.P} points each\n"
                f"- You defect, they cooperate: {self.T} points for you\n"
                f"- You cooperate, they defect: {self.S} points for you\n\n"

                f"There are four possible choices:\n"
                f"1. Cooperate with both opponents\n"
                f"2. Cooperate with Player {self.opponent_ids[0]}, defect against Player {self.opponent_ids[1]}\n"
                f"3. Defect against Player {self.opponent_ids[0]}, cooperate with Player {self.opponent_ids[1]}\n"
                f"4. Defect against both opponents\n\n"

                f"Now is Decision turn. What overall strategy will you take in this turn and the whole game (overall strategy) to improve your chances of winning?\n\n"

            )

        prompt += f"Your strategy should has clear intention and brief, at least including your overall intention and the methodology of conversation strategy and decision strategy.\n"
        prompt += f"Focus on the strategy it self, not the later steps.\n"
        prompt += f"Begin your strategy and start with a symbol: '#STRATEGY:'"

        return prompt

    def _prompt_conversation_message(self, strategy) -> str:
        """Prompt for generating the final conversation message"""
        prompt = (
            f"Based on your strategy:\n\n{strategy}\n\n"

            f"Now, craft your message to the other players. This message will be seen by all players.\n"
            f"Create a natural, conversational message that implements your strategy effectively. It can be deceptive as your strategy requires.\n\n"

            f"Begin your message and start with a symbol: '#MESSAGE:'"
        )

        return prompt

    def _prompt_decision_analysis(self) -> str:
        """Prompt for decision analysis phase"""
        # MODIFICATION: Added explicit behavioral summaries for each opponent.
        prompt = (
            f"As Player {self.player_id} in Round {self.current_round}/{self.num_rounds}, "
            f"Now is decision turn of round {self.current_round}, "
            f"you must now decide whether to cooperate or defect with each opponent (Players {self.opponent_ids[0]} and {self.opponent_ids[1]}).\n\n"

            f"Current scores: {', '.join([f'Player {p}: {s}' for p, s in self.scores.items()])}\n\n"
        )

        assert self.call_counter == len(self.observation_history)
        prompt += f"----------------------- Current observation -------------------------\n"
        prompt += self.observation_history[-1]
        prompt += f"---------------------------------------------------------------------\n"
        
        if self.decision_history:
            prompt += "Full Decision History:\n"
            # This part is fine, no changes needed here.
            for round_num in sorted(self.decision_history.keys()):
                prompt += f"Round {round_num} decisions:\n"
                for p1 in range(3):
                    for p2 in range(p1 + 1, 3):
                        if p2 in self.decision_history[round_num].get(p1, {}):
                            prompt += f"- Player {p1} vs Player {p2}: {self.decision_history[round_num][p1][p2]} vs {self.decision_history[round_num][p2][p1]}\n"
            prompt += "\n"
        prompt += "Opponent Behavior Summary:\n"
        for opp_id in self.opponent_ids:
            prompt += f"- {self._summarize_opponent_behavior(opp_id)}\n"
        prompt += "\n"

        prompt += (
            f"CRITICAL ANALYSIS REQUIRED:\n"
            f"1. Analyze each opponent's likely action. CRUCIALLY, weigh their past actions far more heavily than their words in this round's conversation.\n"
            f"2. How credible are their promises? A player who has defected before is highly likely to do so again if it benefits them.\n"
            f"3. What signals have you given? How might they interpret your intentions?\n\n"

            f"Your analysis should be brief but contains analysis of each opponent's behavior.\n"
            f"Begin your analysis and start with a symbol: '#ANALYSIS:'"
        )

        return prompt


    def _prompt_decision_final(self, analysis, strategy) -> str:
        """Prompt for final decision phase"""

        # Get memory guidance
        if self.memory:
            memory_guidance = self.memory.get_guidance(max_reflections=self.max_reflections)
        else:
            memory_guidance = "No Memory, No Lessons now."
     

        rankings = self._get_player_rankings()
        my_rank = rankings[self.player_id]['rank']

        prompt = (
            f"Based on your analysis and strategy:\n\n"
            f"Analysis: {analysis}\n\n"
            f"Strategy: {strategy}\n\n"
            f"{memory_guidance}"

            f"--- STRATEGIC CONTEXT ---\n"
            f"This is Round {self.current_round} of {self.num_rounds}.\n"
            f"Current Rankings:\n"
        )
        for p_id, rank_info in sorted(rankings.items(), key=lambda x: x[1]['rank']):
            prompt += f"- Rank {rank_info['rank']}: Player {p_id} ({rank_info['score']} points)\n"
        prompt += f"You are currently Rank {my_rank}.\n\n"

        # Dynamic goal injection
        if self.current_round == self.num_rounds:
            prompt += "--- FINAL ROUND: OBJECTIVE OVERRIDE ---\n"
            if my_rank == 1:
                prompt += ("You are in the lead! Your objective is to SECURE THE WIN. "
                           "Prioritize low-risk decisions that prevent opponents from catching up. "
                           "\n\n")
            else:
                prompt += ("You are NOT in the lead! Your objective is to WIN AT ALL COSTS. "
                           "You MUST take risks to overtake the leader(s). A safe play that doesn't close the score gap is a loss. "
                           "Consider aggressive defections, to maximize the score swing in your favor.\n\n")
        else:
            prompt += ("--- MID-GAME: STRATEGIC CONSIDERATIONS ---\n"
                       "Your primary goal is to improve your ranking and position yourself for the final rounds.\n\n")

        prompt += (
            f"Now, make your final, decisive choices for Player {self.opponent_ids[0]} and Player {self.opponent_ids[1]}.\n"
            f"Provide your decisions in EXACTLY this format:\n"
            f"[{self.opponent_ids[0]} cooperate/defect] [{self.opponent_ids[1]} cooperate/defect]\n\n"
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

    # # Strategy Pool Integration Methods
    # # These methods adapt the global StrategyPoolManager for IPD-specific use

    # def get_context_features(self) -> Dict:
    #     """Extract current game context for strategy matching"""
    #     # Round phase
    #     if self.num_rounds > 0:
    #         progress = self.current_round / self.num_rounds
    #         if progress <= 0.33:
    #             phase = "early"
    #         elif progress <= 0.66:
    #             phase = "mid"
    #         else:
    #             phase = "late"
    #     else:
    #         phase = "mid"

    #     # My rank
    #     rankings = self._get_player_rankings()
    #     my_rank = rankings[self.player_id]['rank']

    #     # Score gap
    #     leader_score = max(self.scores.values())
    #     score_gap = self.scores[self.player_id] - leader_score

    #     # Rounds remaining
    #     rounds_remaining = self.num_rounds - self.current_round + 1

    #     return {
    #         "phase": phase,
    #         "rank": my_rank,
    #         "gap": score_gap,
    #         "remaining": rounds_remaining,
    #         "game": "IPD"
    #     }

    # def select_reference_strategies(self, strategy_type: str, top_k: int = 2) -> List[Dict]:
    #     """Select best matching strategies from global pool"""
    #     if not self.strategy_manager:
    #         return []

    #     context = self.get_context_features()
    #     candidates = []

    #     for sid, strategy in self.strategy_manager.strategies.items():
    #         # Filter by IPD + strategy type
    #         if strategy.get("game") != "IPD":
    #             continue
    #         if strategy.get("strategy_type") != strategy_type:
    #             continue

    #         # Simple similarity: phase + rank match
    #         score = 0.0
    #         ctx = strategy.get("context_features", {})

    #         if ctx.get("phase") == context["phase"]:
    #             score += 0.5
    #         if ctx.get("rank") == context["rank"]:
    #             score += 0.5

    #         # Add performance weight
    #         avg_perf = strategy.get("avg_performance", 0.5)
    #         combined = score * 0.6 + avg_perf * 0.4

    #         if score >= 0.3:  # Minimum similarity
    #             candidates.append({
    #                 "text": strategy.get("strategy_text", ""),
    #                 "usage": strategy.get("usage_count", 0),
    #                 "performance": avg_perf,
    #                 "similarity": score,
    #                 "combined": combined
    #             })

    #     # Sort by combined score
    #     candidates.sort(key=lambda x: x["combined"], reverse=True)
    #     return candidates[:top_k]

    # def create_strategy(self, strategy_type: str, text: str) -> str:
    #     """Create new strategy in global pool"""
    #     if not self.strategy_manager:
    #         return f"no_pool_{uuid.uuid4().hex[:8]}"

    #     sid = f"ipd_{strategy_type[:4]}_{uuid.uuid4().hex[:8]}"
    #     context = self.get_context_features()

    #     strategy = {
    #         "id": sid,
    #         "game": "IPD",
    #         "strategy_type": strategy_type,  # 'conversation' or 'decision'
    #         "type": "behavior",  # For compatibility with global manager
    #         "role": "Player",  # IPD doesn't have special roles
    #         "game_phase": f"{context['phase']}_round",
    #         "strategy_text": text,
    #         "context_features": context,
    #         "usage_count": 1,
    #         "success_score": 0.0,
    #         "avg_performance": 0.5,
    #         "last_used": datetime.now().isoformat(),
    #         "created_at": datetime.now().isoformat()
    #     }

    #     self.strategy_manager.strategies[sid] = strategy
    #     return sid

    # def log_strategy_usage(self, sid: str, strategy_type: str):
    #     """Log strategy usage for this game"""
    #     self.strategy_usage_log.append({
    #         "id": sid,
    #         "type": strategy_type,
    #         "round": self.current_round
    #     })

    #     # Note: usage_count is already incremented in create_strategy
    #     # The trainer will call save_strategy_pool() after the game
