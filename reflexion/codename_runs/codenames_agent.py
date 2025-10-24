"""
Codenames Agent - Mind Games Challenge Track 2

This module implements a specialized agent for the Codenames game.

Game Rules:
- 4-player team game (2v2)
- 25 words on board: 9 Red, 8 Blue, 7 Neutral, 1 Assassin
- Teams take turns, Red goes first
- Each team has a Spymaster (who sees word types) and an Operative (who doesn't)
- Spymaster gives a one-word clue + number (e.g., "[wind 2]")
- Operative guesses words based on clue (e.g., "[breeze]") or "[pass]"
- Teams win by uncovering all their words or if opponents hit assassin
"""

import re
import json
from typing import Dict, List, Set, Tuple, Optional
from envs.agent import Agent
from api.api_router import get_api_class
from reflexion.game_logger import GameLogger
from reflexion.codename_runs.codenames_memory import CodenamesMemory
from uhtk.print_pack import *


class CodenamesAgent(Agent):
    """
    Specialized agent for playing Codenames in Mind Games Challenge Track 2

    The agent can function as either:
    - Spymaster (players 0, 2): Gives clues linking team words
    - Operative (players 1, 3): Guesses words based on clues
    """

    def __init__(self, model_name: str, api_model_spec='qwen3-8b',
                 memory: Optional[CodenamesMemory] = None, enable_logging: bool = True, verbose: bool = True):
        """
        Initialize the CodenamesAgent

        Args:
            model_name: Name of the model for identification
            api_model_spec: Which API/model to use for generation
            memory: Optional memory system for reflexion
            enable_logging: Whether to enable logging
            verbose: Whether to print verbose output
        """
        self.model_name = model_name
        self.api = get_api_class(api_model_spec)(model=api_model_spec)

        # Game state tracking
        self.is_initialized = False
        self.player_id = None
        self.player_role = None  # "Spymaster" or "Operative"
        self.team = None  # "Red" or "Blue"
        self.board = {}  # word -> team label mapping (R, B, N, A)
        self.guessed_words = set()  # Set of already guessed words
        self.last_clue = None  # Last clue provided by a Spymaster
        self.last_clue_number = None  # Number associated with last clue
        self.observation_history = []
        self.turn_counter = 0

        # Reflexion memory system
        self.memory = memory
        self.max_reflections = 6

        # Initialize logger
        self.logger = GameLogger() if enable_logging else None
        self.verbose = verbose

    def finalize_game(self, rewards: dict = None, game_info: dict = None):
        """
        Finalize the game and update memory with reflections (if memory is enabled)

        Args:
            rewards: Rewards dict from env.close()
            game_info: Game info dict from env.close()
        """
        # If no memory, skip finalization
        if self.memory is None:
            print(f"\n[CodenamesAgent] Game finished! No memory enabled, skipping reflection.")
            return

        # Simple win determination based on game context
        won = False
        if game_info and 'rewards' in game_info:
            # Adjust win condition based on Codenames game specifics
            won = game_info['rewards'][self.player_id] > 0

        print(f"\n[CodenamesAgent] Game finished! Won: {won}")

        # Generate reflections and update memory
        self.memory.update_memory_from_trial(
            self,
            won=won,
            should_reflect=True  # Always reflect to learn
        )

        # Add trial result
        trial_log_path = (
            self.logger.run_dir / "game_log.json"
            if self.logger and self.logger.run_dir
            else 'None'
        )

        self.memory.add_trial_result({
            "won": won,
            "team": self.team,
            "role": self.player_role,
            "final_score": None,  # Add logic to capture final score
            "opponent_score": None,
            "game_log": str(trial_log_path)
        })

        # Print statistics
        stats = self.memory.get_statistics()
        print(f"[CodenamesAgent] Memory Statistics:")
        print(f"  Total Trials: {stats['total_trials']}")
        print(f"  Win Rate: {stats['win_rate']:.2%}")

        # Finalize logger
        if self.logger and self.logger.run_dir:
            self.logger.finalize(outcome=f"Won: {won}")

    def __call__(self, observation: str) -> str:
        """
        Process observation and generate appropriate response based on game state

        Args:
            observation: The current game observation string

        Returns:
            Action string in the format required by the game
        """
        try:
            if self.verbose:
                print(f"model_name={self.model_name} observation: {observation[:30]}...")
            # First turn initialization
            if not self.is_initialized:
                self._initialize_from_observation(observation)

            # Regular turn processing
            self.observation_history.append(observation)
            self.turn_counter += 1

            # Start logging this turn
            if self.logger:
                self.logger.start_turn(self.turn_counter, observation)

            # Process observation and update game state
            self._update_game_state_from_observation(observation)

            if self.logger:
                print("=" * 60)
                print(f"My id: {self.player_id}")
                print(f"Turn Cnt: {self.turn_counter}:")
                print(f"self.last_clue: {self.last_clue}")
                print(f"self.last_clue_number: {self.last_clue_number}")
                print(f"self.guessed_words: {self.guessed_words}")
                print(f"self.board: {self.board}")

            # Generate action based on role
            if self.player_role == "Spymaster":
                result = self._generate_spymaster_clue()
            else:  # Operative
                result = self._generate_operative_guess()

            # End logging for this turn
            if self.logger:
                self.logger.end_turn(result)
            if self.verbose:
                print(f"{self.model_name} action: {result}")
            return result

        except Exception as e:
            error_msg = f"Error in CodenamesAgent: {e}"
            print(error_msg)

            if self.logger:
                self.logger.end_turn(f"ERROR: {str(e)}")

            # In case of error, return a safe default based on role
            if self.player_role == "Spymaster":
                return "[safe 1]"  # Safe default for Spymaster
            else:
                return "[pass]"  # Safe default for Operative

    def _initialize_from_observation(self, observation: str):
        """
        Parse initial observation to determine role, team, and board state

        Args:
            observation: Initial game observation string
        """
        # Extract player ID and role
        player_id_match = re.search(r"Player (\d+)", observation)
        if player_id_match:
            self.player_id = int(player_id_match.group(1))
        #     print(f"DEBUG: Matched player ID: {self.player_id}")
        # else:
        #     print(f"DEBUG: NO PLAYER ID MATCH in observation: {observation}")

        # Determine role and team based on player ID
        if self.player_id in [0, 2]:
            self.player_role = "Spymaster"
            self.team = "Red" if self.player_id == 0 else "Blue"
            # print(f"DEBUG: Role set as Spymaster, Team: {self.team}")
        else:
            self.player_role = "Operative"
            self.team = "Red" if self.player_id == 1 else "Blue"
            # print(f"DEBUG: Role set as Operative, Team: {self.team}")

        # Parse initial board state
        # print("DEBUG: Parsing initial board state...")
        self._parse_board_state(observation)

        self.is_initialized = True

        # Set player info in logger
        if self.logger:
            self.logger.set_player_info(
                self.player_id,
                self.player_role,
                self.team
            )

        # # Debug print full initialization details
        # print(f"DEBUG: Initialization complete - ID: {self.player_id}, Role: {self.player_role}, Team: {self.team}")

    def _parse_board_state(self, observation: str):
        """
        Parse board state from observation

        Args:
            observation: Game observation containing board state
        """
        # Extract words and their teams (if visible)
        words_section = re.search(r"Codenames Words:(.*?)(?:\n\n|\Z)", observation, re.DOTALL)
        if not words_section:
            return

        word_lines = words_section.group(1).strip().split('\n')

        for line in word_lines:
            parts = line.strip().split()
            if not parts:
                continue

            word = parts[0].lower()
            team_label = None

            # For Spymasters, team labels are visible
            if len(parts) > 1 and parts[1] in ['R', 'B', 'N', 'A']:
                team_label = parts[1]

            # Add to board if not already present
            if word not in self.board and word:
                self.board[word] = team_label

            # Check if word is revealed
            if "revealed" in line:
                self.guessed_words.add(word)

    def _update_game_state_from_observation(self, observation: str):
        """
        Update game state based on new observation

        Args:
            observation: Current game observation
        """
        # Update board state
        self._parse_board_state(observation)

        # Extract last clue if present
        clue_match = re.search(r"Spymaster of .* submitted \[(\w+) (\d+)\]", observation)
        if clue_match:
            self.last_clue = clue_match.group(1)
            self.last_clue_number = int(clue_match.group(2))

        # Extract guessed words if present
        guess_match = re.search(r"Operator of .* guessed \[(\w+)\]", observation)
        if guess_match:
            guessed_word = guess_match.group(1).lower()
            self.guessed_words.add(guessed_word)

    def _generate_spymaster_clue(self) -> str:
        """
        Generate a clue as Spymaster using multi-phase reasoning

        Returns:
            Clue string in format "[word number]"
        """
        # Get team words
        team_words = self._get_team_words()
        opponent_words = self._get_opponent_words()
        neutral_words = self._get_neutral_words()
        assassin_word = self._get_assassin_word()

        # Skip already guessed words
        remaining_team_words = [w for w in team_words if w not in self.guessed_words]

        # If no words left, return dummy clue
        if not remaining_team_words:
            return "[dummy 0]"

        # Phase 1: Analysis prompt
        analysis_prompt = self._prompt_spymaster_analysis(
            remaining_team_words, opponent_words,
            neutral_words, assassin_word, self.guessed_words
        )

        analysis_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_spymaster_system()},
            {"role": "user", "content": analysis_prompt}
        ])

        # Extract analysis results
        analysis = self._parse_tag_section(analysis_response, "#ANALYSIS:")

        if self.logger:
            self.logger.log_phase("analysis", analysis_prompt, analysis_response, analysis)
            print_bold_blue(f"Analysis:")
            print(analysis)

        # Phase 2: Candidate generation prompt
        candidate_prompt = self._prompt_spymaster_candidates(
            analysis, remaining_team_words, opponent_words,
            neutral_words, assassin_word
        )

        candidate_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_spymaster_system()},
            {"role": "user", "content": candidate_prompt}
        ])

        # Extract candidates
        candidates = self._parse_tag_section(candidate_response, "#CANDIDATES:")

        if self.logger:
            self.logger.log_phase("candidates", candidate_prompt, candidate_response, candidates)
            print_bold_blue(f"Candidates:")
            print(candidates)

        # Phase 3: Final clue selection prompt
        final_prompt = self._prompt_spymaster_final(candidates)

        final_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_spymaster_system()},
            {"role": "user", "content": final_prompt}
        ])

        # Extract final clue
        final_clue = self._parse_tag_section(final_response, "#FINAL:")

        if self.logger:
            self.logger.log_phase("final_clue", final_prompt, final_response, final_clue)
            print_bold_blue(f"Final Clue:")
            print(final_clue)

        # Make sure output matches required format [word number]
        clue_match = re.search(r'\[(\w+)\s+(\d+)\]', final_clue)
        if clue_match:
            word = clue_match.group(1)
            number = clue_match.group(2)
            return f"[{word} {number}]"
        else:
            # Fallback: provide a simple clue for one word
            if remaining_team_words:
                return f"[{remaining_team_words[0]} 1]"
            return "[safe 1]"

    def _generate_operative_guess(self) -> str:
        """
        Generate a guess as Operative using multi-phase reasoning

        Returns:
            Guess string in format "[word]" or "[pass]"
        """
        # If no clue available, or clue number is 0, pass
        if not self.last_clue or self.last_clue_number == 0:
            return "[pass]"

        # Get available words (not yet guessed)
        available_words = [word for word in self.board.keys() if word not in self.guessed_words]

        # If no words left, pass
        if not available_words:
            return "[pass]"

        # Phase 1: Analysis prompt
        analysis_prompt = self._prompt_operative_analysis(
            self.last_clue, self.last_clue_number,
            available_words, self.guessed_words
        )

        analysis_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_operative_system()},
            {"role": "user", "content": analysis_prompt}
        ])

        # Extract analysis results
        analysis = self._parse_tag_section(analysis_response, "#ANALYSIS:")

        if self.logger:
            self.logger.log_phase("analysis", analysis_prompt, analysis_response, analysis)

        # Phase 2: Ranking prompt
        ranking_prompt = self._prompt_operative_ranking(
            self.last_clue, self.last_clue_number,
            analysis, available_words
        )

        ranking_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_operative_system()},
            {"role": "user", "content": ranking_prompt}
        ])

        # Extract ranked words
        ranked_words = self._parse_tag_section(ranking_response, "#RANKING:")

        if self.logger:
            self.logger.log_phase("ranking", ranking_prompt, ranking_response, ranked_words)

        # Phase 3: Final guess prompt
        final_prompt = self._prompt_operative_final(
            self.last_clue, self.last_clue_number,
            ranked_words, available_words
        )

        final_response = self.api(input_messages=[
            {"role": "system", "content": self._prompt_operative_system()},
            {"role": "user", "content": final_prompt}
        ])

        # Extract final guess
        final_guess = self._parse_tag_section(final_response, "#FINAL:")

        if self.logger:
            self.logger.log_phase("final_guess", final_prompt, final_response, final_guess)

        # Make sure output matches required format [word]
        guess_match = re.search(r'\[(\w+)\]', final_guess)
        if guess_match:
            word = guess_match.group(1)
            # Check if the guessed word is valid (exists and not guessed)
            if word.lower() == "pass":
                return "[pass]"
            elif word.lower() in available_words:
                return f"[{word}]"
            else:
                # Invalid word, pass
                return "[pass]"
        else:
            # Fallback: pass
            return "[pass]"

    # Helper methods for game state

    def _get_team_words(self) -> List[str]:
        """Get words belonging to agent's team"""
        team_label = "R" if self.team == "Red" else "B"
        return [word for word, label in self.board.items() if label == team_label]

    def _get_opponent_words(self) -> List[str]:
        """Get words belonging to opponent team"""
        opponent_label = "B" if self.team == "Red" else "R"
        return [word for word, label in self.board.items() if label == opponent_label]

    def _get_neutral_words(self) -> List[str]:
        """Get neutral words"""
        return [word for word, label in self.board.items() if label == "N"]

    def _get_assassin_word(self) -> Optional[str]:
        """Get the assassin word"""
        for word, label in self.board.items():
            if label == "A":
                return word
        return None

    # Prompt methods

    def _prompt_spymaster_system(self) -> str:
        """System prompt for Spymaster role"""
        return (
            "You are the Spymaster in the word-guessing game Codenames. "
            "Your goal is to help your team identify all their words while avoiding opponent words, neutral words, and especially the assassin word. "
            "You must provide a one-word clue followed by a number (e.g., '[wind 2]') that helps connect multiple words of your team. "
            "The clue cannot be any form or part of the words on the board."
        )

    def _prompt_operative_system(self) -> str:
        """System prompt for Operative role"""
        return (
            "You are the Operative in the word-guessing game Codenames. "
            "Your goal is to correctly identify words belonging to your team based on the Spymaster's clue. "
            "You receive a one-word clue and a number, indicating how many words this clue relates to. "
            "You must respond with a word guess in the format '[word]' or '[pass]'."
        )

    def _prompt_spymaster_analysis(self, team_words, opponent_words, neutral_words, assassin_word, guessed_words) -> str:
        """Prompt for Spymaster analysis phase"""
        prompt = (
            f"As the {self.team} team Spymaster, you need to analyze the current game state "
            f"and find connections between your team's words.\n\n"

            f"Your team ({self.team}) words:\n"
            f"{', '.join(team_words)}\n\n"

            f"Opponent words (avoid these):\n"
            f"{', '.join(opponent_words)}\n\n"

            f"Neutral words (avoid these):\n"
            f"{', '.join(neutral_words)}\n\n"
        )

        if assassin_word:
            prompt += f"Assassin word (absolutely avoid this):\n{assassin_word}\n\n"

        prompt += (
            f"Already guessed words:\n"
            f"{', '.join(guessed_words) if guessed_words else 'None'}\n\n"

            f"Please analyze these words carefully and identify:\n"
            f"1. Potential thematic connections between 2 or more of your team's words\n"
            f"2. Word clusters that could be connected with a single clue\n"
            f"3. Potential high-risk words that could be confused with the assassin or opponent words\n\n"

            f"Begin your analysis and start with a symbol: '#ANALYSIS:'"
        )

        return prompt

    def _prompt_spymaster_candidates(self, analysis, team_words, opponent_words, neutral_words, assassin_word) -> str:
        """Prompt for Spymaster candidate generation phase"""
        # Initialize memory guidance
        memory_guidance = ""
        if self.memory and self.memory.memory['memory']:
            memory_guidance = self.memory.get_guidance(max_reflections=self.max_reflections)
            memory_guidance = f"\n\nPrevious Strategic Lessons:\n{memory_guidance}"

        prompt = (
            f"Based on your analysis:\n\n{analysis}\n\n"

            f"Now, generate 3-5 candidate clue words that could connect multiple words from your team.\n"
            f"For each candidate:\n"
            f"1. Specify which of your team's words it connects\n"
            f"2. Evaluate the risk of the opponent guessing their words or the assassin\n"
            f"3. Assign a confidence score (1-10)\n\n"

            f"Format each candidate as:\n"
            f"- Clue: [word] Number: [n]\n"
            f"  Connected words: [list of team words]\n"
            f"  Risk assessment: [low/medium/high]\n"
            f"  Confidence: [1-10]\n"
            f"  Explanation: [brief rationale]\n\n"

            f"Remember:\n"
            f"- Clues must be a single word\n"
            f"- Clues cannot be any form of the words on the board\n"
            f"- Aim for clues that connect the most team words safely\n\n"

            f"{memory_guidance}\n\n"

            f"Begin your candidate list and start with a symbol: '#CANDIDATES:'"
        )

        return prompt

    def _prompt_spymaster_final(self, candidates) -> str:
        """Prompt for Spymaster final clue selection phase"""
        prompt = (
            f"Based on your candidates:\n\n{candidates}\n\n"

            f"Select the best clue based on:\n"
            f"1. Number of team words it connects effectively\n"
            f"2. Minimal risk of confusion with opponent or assassin words\n"
            f"3. Clarity and distinctiveness\n\n"

            f"Provide your final decision in EXACTLY this format: [word n]\n"
            f"Where:\n"
            f"- 'word' is your chosen clue word\n"
            f"- 'n' is the number of team words it connects\n\n"

            f"Begin your final selection and start with a symbol: '#FINAL:'"
        )

        return prompt

    def _prompt_operative_analysis(self, clue, clue_number, available_words, guessed_words) -> str:
        """Prompt for Operative analysis phase"""
        prompt = (
            f"As the {self.team} team Operative, you've received a clue from your Spymaster:\n"
            f"Clue: '{clue}' Number: {clue_number}\n\n"

            f"Available words on the board:\n"
            f"{', '.join(available_words)}\n\n"

            f"Already guessed words:\n"
            f"{', '.join(guessed_words) if guessed_words else 'None'}\n\n"

            f"Please analyze this clue carefully:\n"
            f"1. What does this clue likely refer to?\n"
            f"2. What semantic connections exist between the clue and available words?\n"
            f"3. Consider synonyms, categories, associations, and other relationships\n\n"

            f"Begin your analysis and start with a symbol: '#ANALYSIS:'"
        )

        return prompt

    def _prompt_operative_ranking(self, clue, clue_number, analysis, available_words) -> str:
        """Prompt for Operative word ranking phase"""
        # Initialize memory guidance
        memory_guidance = ""
        if self.memory and self.memory.memory['memory']:
            memory_guidance = self.memory.get_guidance(max_reflections=self.max_reflections)
            memory_guidance = f"\n\nPrevious Strategic Lessons:\n{memory_guidance}"

        prompt = (
            f"Based on your analysis of the clue '{clue}' with number {clue_number}:\n\n{analysis}\n\n"

            f"Now, rank the top {min(5, len(available_words))} words from the board that you believe best match this clue.\n"
            f"For each word, provide:\n"
            f"1. A confidence score (1-10)\n"
            f"2. Your reasoning for why this word connects to the clue\n\n"

            f"Format your ranking as:\n"
            f"1. [word] - Confidence: [score] - Reason: [brief explanation]\n"
            f"2. [word] - Confidence: [score] - Reason: [brief explanation]\n"
            f"...\n\n"

            f"{memory_guidance}\n\n"

            f"Begin your ranking and start with a symbol: '#RANKING:'"
        )

        return prompt

    def _prompt_operative_final(self, clue, clue_number, ranking, available_words) -> str:
        """Prompt for Operative final guess selection phase"""
        prompt = (
            f"Based on your ranking for the clue '{clue}' with number {clue_number}:\n\n{ranking}\n\n"

            f"Now make your final decision on which word to guess. Consider:\n"
            f"1. The Spymaster indicated {clue_number} words relate to this clue\n"
            f"2. The confidence level of your top candidates\n"
            f"3. The risk of guessing an opponent's word or the assassin\n\n"

            f"If you're confident in your guess, provide the word in EXACTLY this format: [word]\n"
            f"If you're not confident enough, you can pass by responding: [pass]\n\n"

            f"Begin your final selection and start with a symbol: '#FINAL:'"
        )

        return prompt

    # Utility methods

    def _parse_tag_section(self, text: str, tag: str) -> str:
        """Extract content after a specific tag"""
        index = text.find(tag)
        if index != -1:
            return text[index + len(tag):].strip()
        return text  # Return full text if tag not found