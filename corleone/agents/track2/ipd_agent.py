import re
import json
import uuid
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from src.agent import Agent
from corleone.api.api_router import get_api_class
from corleone.game_logger import GameLogger


class IPDAgent(Agent):
    """
    Specialized agent for playing Three Player IPD in Mind Games Challenge Track 2

    The agent uses multi-phase reasoning to:
    1. During conversation: Build trust and coordinate strategies
    2. During decision: Make strategic cooperate/defect choices, with dynamic goals
       based on game state (e.g., rank, round number).

    Uses the global StrategyPoolManager for learning across games.
    """

    def __init__(self, model_name: str, api_model_spec='qwen3-8b', enable_logging: bool = True,
                 strategy_pool_manager=None):
        """
        Initialize the IPDAgent

        Args:
            model_name: Name of the model for identification
            api_model_spec: Which API/model to use for generation
            enable_logging: Whether to enable logging
            strategy_pool_manager: Reference to global StrategyPoolManager (None for standalone)
        """
        self.model_name = model_name
        self.api = get_api_class(api_model_spec)(model=api_model_spec)

        # Strategy pool manager (shared across all agents in training)
        self.strategy_manager = strategy_pool_manager
        self.strategy_usage_log = []  # Log of strategies used this game

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

            if self.enable_logging:
                print("=" * 60)
                print(f"Turn {self.turn_counter}:")
                print(f"Phase: {self.phase}")
                print(result)
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

        # Log strategy usage
        if self.strategy_manager:
            sid = self.create_strategy("conversation", strategy)
            self.log_strategy_usage(sid, "conversation")

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

        # Log decision strategy usage
        if self.strategy_manager:
            # Extract strategy reasoning before #DECISIONS:
            decision_strategy = decision_response.split("#DECISIONS:")[0].strip()
            sid = self.create_strategy("decision", decision_strategy)
            self.log_strategy_usage(sid, "decision")

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
            f"The game consists of {self.num_rounds} rounds. "
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
            f"Conversation turn {self.conversation_round+1}/{self.total_conversation_rounds}, "
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

        if self.conversation_history:
            prompt += "Conversation in current round:\n"
            for speaker_id, message in self.conversation_history:
                prompt += f"Player {speaker_id}: {message}\n"
            prompt += "\n"

        prompt += (
            f"Please analyze the current state:\n"
            f"1. What patterns of cooperation/defection have emerged?\n"
            f"2. Has any player been consistently cooperative or defective?\n"
            f"3. What promises or commitments have been made in conversation? How credible are they given past actions?\n"
            f"4. What is each player's likely strategy based on their behavior?\n"
            f"5. Who appears most trustworthy and who seems deceptive? REMEMBER: Actions speak louder than words.\n\n"

            f"Begin your analysis and start with a symbol: '#ANALYSIS:'"
        )

        return prompt

    def _prompt_conversation_strategy(self, analysis) -> str:
        """Prompt for conversation strategy phase"""

        # Get reference strategies
        reference_section = ""
        if self.strategy_manager:
            ref_strategies = self.select_reference_strategies('conversation')
            if ref_strategies:
                reference_section = "\n### Reference Conversation Strategies (proven in similar situations):\n"
                for i, ref in enumerate(ref_strategies, 1):
                    reference_section += f"{i}. {ref['text']}\n   (Used {ref['usage']} times, performance: {ref['performance']:.2f})\n\n"
                reference_section += "You can adopt, modify, or ignore these.\n\n"

        prompt = (
            f"Based on your analysis:\n\n{analysis}\n\n"
            f"{reference_section}"
            f"Now, determine your conversation strategy for this turn. Your goal is to WIN the game. Consider:\n"
            f"1. Your goals in this conversation: Should you build trust, form a temporary alliance, deceive an opponent, or call out a defector?\n"
            f"2. How to position yourself with each opponent based on your rank and their behavior.\n"
            f"3. What information to reveal or conceal. Is it beneficial to lie about your intentions?\n"
            f"4. Whether to propose explicit agreements. If so, who is the target of this agreement?\n\n"

            f"Remember the game parameters:\n"
            f"- Both cooperate: {self.R} points each\n"
            f"- Both defect: {self.P} points each\n"
            f"- You defect, they cooperate: {self.T} points for you\n"
            f"- You cooperate, they defect: {self.S} points for you\n\n"

            f"What overall approach will you take in this conversation to improve your chances of winning?\n\n"

            f"Begin your strategy and start with a symbol: '#STRATEGY:'"
        )

        return prompt

    def _prompt_conversation_message(self, strategy) -> str:
        """Prompt for generating the final conversation message"""
        prompt = (
            f"Based on your strategy:\n\n{strategy}\n\n"

            f"Now, craft your message to the other players. This message will be seen by all players.\n"
            f"Create a natural, conversational message that implements your strategy effectively. It can be truthful or deceptive as your strategy requires.\n\n"

            f"Begin your message and start with a symbol: '#MESSAGE:'"
        )

        return prompt

    def _prompt_decision_analysis(self) -> str:
        """Prompt for decision analysis phase"""
        # MODIFICATION: Added explicit behavioral summaries for each opponent.
        prompt = (
            f"As Player {self.player_id} in Round {self.current_round}/{self.num_rounds}, "
            f"you must now decide whether to cooperate or defect with each opponent (Players {self.opponent_ids[0]} and {self.opponent_ids[1]}).\n\n"

            f"Current scores: {', '.join([f'Player {p}: {s}' for p, s in self.scores.items()])}\n\n"
        )
        
        # Add detailed behavioral summaries
        prompt += "Opponent Behavior Summary:\n"
        for opp_id in self.opponent_ids:
            prompt += f"- {self._summarize_opponent_behavior(opp_id)}\n"
        prompt += "\n"

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

        if self.conversation_history:
            prompt += "Conversation in This Round:\n"
            for speaker_id, message in self.conversation_history:
                prompt += f"Player {speaker_id}: {message}\n"
            prompt += "\n"

        prompt += (
            f"CRITICAL ANALYSIS REQUIRED:\n"
            f"1. Analyze each opponent's likely action. CRUCIALLY, weigh their past actions far more heavily than their words in this round's conversation.\n"
            f"2. How credible are their promises? A player who has defected before is highly likely to do so again if it benefits them.\n"
            f"3. What signals have you given? How might they interpret your intentions?\n\n"

            f"Begin your analysis and start with a symbol: '#ANALYSIS:'"
        )

        return prompt

    def _prompt_decision_payoff(self, analysis) -> str:
        """Prompt for decision payoff evaluation phase"""
        prompt = (
            f"Based on your analysis:\n\n{analysis}\n\n"

            f"Now, evaluate the expected payoffs for different decision combinations with Players {self.opponent_ids[0]} and {self.opponent_ids[1]}.\n\n"

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
            f"- Your expected payoff based on your prediction of opponents' actions.\n"
            f"- The impact on your score relative to your opponents, especially the current leader.\n"
            f"- The impact on future rounds and your reputation (less important in later rounds).\n\n"

            f"Begin your payoff evaluation and start with a symbol: '#PAYOFF:'"
        )

        return prompt

    def _prompt_decision_final(self, analysis, payoff) -> str:
        """Prompt for final decision phase"""

        # Get reference decision strategies
        reference_section = ""
        if self.strategy_manager:
            ref_strategies = self.select_reference_strategies('decision')
            if ref_strategies:
                reference_section = "\n### Reference Decision Strategies (proven approaches):\n"
                for i, ref in enumerate(ref_strategies, 1):
                    reference_section += f"{i}. {ref['text']}\n   (Performance: {ref['performance']:.2f})\n\n"
                reference_section += "Consider these when deciding.\n\n"

        rankings = self._get_player_rankings()
        my_rank = rankings[self.player_id]['rank']

        prompt = (
            f"Based on your analysis and payoff evaluation:\n\n"
            f"Analysis: {analysis}\n\n"
            f"Payoff evaluation: {payoff}\n\n"
            f"{reference_section}"

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
                           "Cooperating with others who are not a threat to your lead is a safe way to maintain your advantage.\n\n")
            else:
                prompt += ("You are NOT in the lead! Your objective is to WIN AT ALL COSTS. "
                           "You MUST take risks to overtake the leader(s). A safe play that doesn't close the score gap is a loss. "
                           "Consider aggressive defections, especially against the player(s) ahead of you, to maximize the score swing in your favor.\n\n")
        else:
            prompt += ("--- MID-GAME: STRATEGIC CONSIDERATIONS ---\n"
                       "Balance short-term gains with your long-term reputation. However, do not be afraid to punish defectors to establish credibility. "
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

    # Strategy Pool Integration Methods
    # These methods adapt the global StrategyPoolManager for IPD-specific use

    def get_context_features(self) -> Dict:
        """Extract current game context for strategy matching"""
        # Round phase
        if self.num_rounds > 0:
            progress = self.current_round / self.num_rounds
            if progress <= 0.33:
                phase = "early"
            elif progress <= 0.66:
                phase = "mid"
            else:
                phase = "late"
        else:
            phase = "mid"

        # My rank
        rankings = self._get_player_rankings()
        my_rank = rankings[self.player_id]['rank']

        # Score gap
        leader_score = max(self.scores.values())
        score_gap = self.scores[self.player_id] - leader_score

        # Rounds remaining
        rounds_remaining = self.num_rounds - self.current_round + 1

        return {
            "phase": phase,
            "rank": my_rank,
            "gap": score_gap,
            "remaining": rounds_remaining,
            "game": "IPD"
        }

    def select_reference_strategies(self, strategy_type: str, top_k: int = 2) -> List[Dict]:
        """Select best matching strategies from global pool"""
        if not self.strategy_manager:
            return []

        context = self.get_context_features()
        candidates = []

        for sid, strategy in self.strategy_manager.strategies.items():
            # Filter by IPD + strategy type
            if strategy.get("game") != "IPD":
                continue
            if strategy.get("strategy_type") != strategy_type:
                continue

            # Simple similarity: phase + rank match
            score = 0.0
            ctx = strategy.get("context_features", {})

            if ctx.get("phase") == context["phase"]:
                score += 0.5
            if ctx.get("rank") == context["rank"]:
                score += 0.5

            # Add performance weight
            avg_perf = strategy.get("avg_performance", 0.5)
            combined = score * 0.6 + avg_perf * 0.4

            if score >= 0.3:  # Minimum similarity
                candidates.append({
                    "text": strategy.get("strategy_text", ""),
                    "usage": strategy.get("usage_count", 0),
                    "performance": avg_perf,
                    "similarity": score,
                    "combined": combined
                })

        # Sort by combined score
        candidates.sort(key=lambda x: x["combined"], reverse=True)
        return candidates[:top_k]

    def create_strategy(self, strategy_type: str, text: str) -> str:
        """Create new strategy in global pool"""
        if not self.strategy_manager:
            return f"no_pool_{uuid.uuid4().hex[:8]}"

        sid = f"ipd_{strategy_type[:4]}_{uuid.uuid4().hex[:8]}"
        context = self.get_context_features()

        strategy = {
            "id": sid,
            "game": "IPD",
            "strategy_type": strategy_type,  # 'conversation' or 'decision'
            "type": "behavior",  # For compatibility with global manager
            "role": "Player",  # IPD doesn't have special roles
            "game_phase": f"{context['phase']}_round",
            "strategy_text": text,
            "context_features": context,
            "usage_count": 1,
            "success_score": 0.0,
            "avg_performance": 0.5,
            "last_used": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        }

        self.strategy_manager.strategies[sid] = strategy
        return sid

    def log_strategy_usage(self, sid: str, strategy_type: str):
        """Log strategy usage for this game"""
        self.strategy_usage_log.append({
            "id": sid,
            "type": strategy_type,
            "round": self.current_round
        })

        # Note: usage_count is already incremented in create_strategy
        # The trainer will call save_strategy_pool() after the game
