"""
Game Logger for Mind Games Challenge
Records game sessions for training and analysis purposes.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class GameLogger:
    """
    Logger for recording game sessions including observations, actions, and outcomes.
    Supports structured data storage for subsequent training and analysis.
    """

    def __init__(self, log_dir: str = "game_logs", enabled: bool = True):
        """
        Initialize the game logger.

        Args:
            log_dir: Directory to store log files
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.current_session = None
        self.session_file = None

    def start_session(self,
                     env_id: str,
                     agents: Dict[int, Any],
                     num_players: int,
                     additional_info: Optional[Dict] = None) -> str:
        """
        Start a new game session.

        Args:
            env_id: Environment identifier
            agents: Dictionary mapping player IDs to agent instances
            num_players: Number of players in the game
            additional_info: Additional game information

        Returns:
            Session ID
        """
        if not self.enabled:
            return None

        session_id = f"{env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize session data
        self.current_session = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "environment": env_id,
            "num_players": num_players,
            "agents": {},
            "game_info": additional_info or {},
            "turns": [],
            "results": None,
            "summary": {}
        }

        # Record agent information
        for player_id, agent in agents.items():
            agent_info = {
                "player_id": player_id,
                "agent_class": agent.__class__.__name__,
                "model_name": getattr(agent, 'model_name', 'unknown'),
                "init_info": getattr(agent, 'init_info', None)
            }
            self.current_session["agents"][player_id] = agent_info

        # Create session file
        self.session_file = self.log_dir / f"{session_id}.json"

        print(f"Started game logging session: {session_id}")
        return session_id

    def log_turn(self,
                 player_id: int,
                 observation: Any,
                 agent_state: Dict[str, Any],
                 action: str,
                 step_info: Optional[Dict] = None,
                 game_phase: Optional[str] = None,
                 round_number: Optional[int] = None) -> None:
        """
        Log a single turn in the game.

        Args:
            player_id: ID of the player taking the action
            observation: Raw observation from the environment
            agent_state: Agent's internal state (beliefs, strategy, etc.)
            action: Action taken by the agent
            step_info: Additional step information from environment
            game_phase: Current phase of the game (e.g., "night", "day_speak", "day_vote")
            round_number: Current round number in the game
        """
        if not self.enabled or self.current_session is None:
            return

        # Extract role information from agent state if available
        agent_role = None
        agent_team = None
        if agent_state.get("init_info"):
            agent_role = agent_state["init_info"].get("role")
            agent_team = agent_state["init_info"].get("team")

        # Parse game phase from observation if not provided
        if not game_phase and isinstance(observation, str):
            game_phase = self._extract_game_phase(observation)

        turn_data = {
            "turn_number": len(self.current_session["turns"]) + 1,
            "timestamp": datetime.now().isoformat(),
            "player_id": player_id,
            "observation": self._serialize_observation(observation),
            "agent_state": agent_state,
            "action": action,
            "step_info": step_info or {},
            # Enhanced context information
            "game_context": {
                "phase": game_phase,
                "round": round_number,
                "agent_role": agent_role,
                "agent_team": agent_team,
                "formatted_observation": self._format_observation_for_analysis(observation),
                "action_type": self._classify_action_type(action, game_phase),
                "is_voting_action": self._is_voting_action(action),
                "is_speaking_action": self._is_speaking_action(action)
            }
        }

        self.current_session["turns"].append(turn_data)

    def log_results(self, rewards: Dict[int, float], game_info: Dict) -> None:
        """
        Log the final game results.

        Args:
            rewards: Dictionary mapping player IDs to their rewards
            game_info: Final game information from environment
        """
        if not self.enabled or self.current_session is None:
            return

        self.current_session["results"] = {
            "rewards": rewards,
            "game_info": game_info,
            "timestamp": datetime.now().isoformat()
        }

        # Generate summary
        self._generate_summary()

        # Save the complete session
        self._save_session()

    def _serialize_observation(self, observation: Any) -> Any:
        """
        Serialize observation for JSON storage.

        Args:
            observation: Raw observation data

        Returns:
            Serializable observation data
        """
        if isinstance(observation, str):
            return observation
        elif isinstance(observation, list):
            # Handle list observations (common in this codebase)
            try:
                # Try to parse if it looks like a JSON string in a list
                if len(observation) > 0 and isinstance(observation[0], str):
                    if observation[0].startswith('['):
                        return json.loads(observation[0])
                return observation
            except:
                return str(observation)
        else:
            return str(observation)

    def _extract_game_phase(self, observation: str) -> Optional[str]:
        """
        Extract current game phase from observation text.

        Args:
            observation: Raw observation text

        Returns:
            Game phase string (e.g., "night", "day_speak", "day_vote") or None
        """
        if not isinstance(observation, str):
            return None

        # Look for phase indicators in Secret Mafia
        if "night" in observation.lower():
            return "night"
        elif "day" in observation.lower() and "vote" in observation.lower():
            return "day_vote"
        elif "day" in observation.lower() and ("speak" in observation.lower() or "discuss" in observation.lower()):
            return "day_speak"
        elif "day" in observation.lower():
            return "day"

        return None

    def _format_observation_for_analysis(self, observation: Any) -> Dict[str, Any]:
        """
        Format observation for better analysis during training.

        Args:
            observation: Raw observation

        Returns:
            Structured observation data
        """
        formatted = {
            "raw": str(observation),
            "length": len(str(observation)),
            "type": type(observation).__name__
        }

        if isinstance(observation, str):
            # Extract key information from Secret Mafia observations
            formatted.update({
                "contains_system_message": observation.startswith("SYSTEM:") or "SYSTEM:" in observation,
                "contains_player_speech": any(f"Player {i}:" in observation for i in range(10)),
                "contains_vote": any(f"[{i}]" in observation for i in range(10)),
                "is_initialization": "You are Player" in observation and "Your role:" in observation
            })

        return formatted

    def _classify_action_type(self, action: str, phase: Optional[str] = None) -> str:
        """
        Classify the type of action taken.

        Args:
            action: The action string
            phase: Current game phase

        Returns:
            Action type classification
        """
        if not action:
            return "empty"

        action_lower = action.lower()

        if self._is_voting_action(action):
            return "vote"
        elif self._is_speaking_action(action):
            return "speech"
        elif any(word in action_lower for word in ["investigate", "detect", "check"]):
            return "investigate"
        elif any(word in action_lower for word in ["protect", "save", "heal"]):
            return "protect"
        elif any(word in action_lower for word in ["kill", "eliminate", "attack"]):
            return "attack"
        elif phase == "night":
            return "night_action"
        else:
            return "other"

    def _is_voting_action(self, action: str) -> bool:
        """
        Check if action is a voting action.

        Args:
            action: The action string

        Returns:
            True if this is a voting action
        """
        if not action:
            return False

        # Check for bracket format [X] where X is a number
        import re
        bracket_match = re.search(r'^\[\d+\]$', action.strip())
        if bracket_match:
            return True

        # Check for vote-related keywords
        vote_keywords = ["vote", "eliminate", "expel", "remove", "lynch"]
        return any(keyword in action.lower() for keyword in vote_keywords)

    def _is_speaking_action(self, action: str) -> bool:
        """
        Check if action is a speaking action.

        Args:
            action: The action string

        Returns:
            True if this is a speaking action
        """
        if not action:
            return False

        # If it's not a voting action and has substantial content, assume it's speech
        if not self._is_voting_action(action) and len(action) > 10:
            return True

        return False

    def _generate_summary(self) -> None:
        """Generate session summary statistics."""
        if not self.current_session:
            return

        turns = self.current_session["turns"]
        agents = self.current_session["agents"]
        results = self.current_session["results"]

        summary = {
            "total_turns": len(turns),
            "turns_per_player": {},
            "winner": None,
            "total_rewards": results["rewards"]
        }

        # Count turns per player
        for turn in turns:
            player_id = turn["player_id"]
            summary["turns_per_player"][player_id] = summary["turns_per_player"].get(player_id, 0) + 1

        # Determine winner(s) - players with maximum reward
        if rewards := results.get("rewards"):
            max_reward = max(rewards.values())
            winners = [pid for pid, reward in rewards.items() if reward == max_reward]
            summary["winners"] = winners

        self.current_session["summary"] = summary

    def _save_session(self) -> None:
        """Save the current session to file."""
        if not self.enabled or self.current_session is None or not self.session_file:
            return

        try:
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_session, f, indent=2, ensure_ascii=False)
            print(f"Game session saved to: {self.session_file}")
        except Exception as e:
            print(f"Error saving game session: {e}")

    def end_session(self) -> None:
        """End the current session and save if not already saved."""
        if self.current_session and not self.current_session.get("results"):
            # Auto-save incomplete session
            self._save_session()

        self.current_session = None
        self.session_file = None

    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about logged sessions.

        Returns:
            Dictionary with session statistics
        """
        if not self.enabled:
            return {"enabled": False}

        stats = {
            "enabled": True,
            "log_dir": str(self.log_dir),
            "total_sessions": 0,
            "sessions_by_env": {},
            "latest_session": None
        }

        # Count sessions by environment
        for file_path in self.log_dir.glob("*.json"):
            stats["total_sessions"] += 1

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    session = json.load(f)
                    env = session.get("environment", "unknown")
                    stats["sessions_by_env"][env] = stats["sessions_by_env"].get(env, 0) + 1

                    if (stats["latest_session"] is None or
                        session["timestamp"] > stats["latest_session"]["timestamp"]):
                        stats["latest_session"] = {
                            "session_id": session["session_id"],
                            "timestamp": session["timestamp"],
                            "environment": env
                        }
            except Exception as e:
                print(f"Error reading session file {file_path}: {e}")

        return stats


class LoggedAgent:
    """
    Wrapper class that adds logging capabilities to any agent.
    """

    def __init__(self, base_agent: Any, logger: GameLogger, player_id: int):
        """
        Initialize logged agent wrapper.

        Args:
            base_agent: The original agent instance
            logger: GameLogger instance
            player_id: Player ID for this agent
        """
        self.base_agent = base_agent
        self.logger = logger
        self.player_id = player_id

    def __call__(self, observation: str) -> str:
        """
        Forward call to base agent with logging.

        Args:
            observation: Game observation

        Returns:
            Agent action
        """
        # Get agent state before action (without probability data)
        agent_state_before = self._extract_agent_state()

        # Extract additional context from agent
        game_phase = self._extract_game_phase_from_agent()
        round_number = self._extract_round_from_agent()

        # Check if agent supports probability capture
        if hasattr(self.base_agent, 'api') and hasattr(self.base_agent.api, '__code__'):
            import inspect
            api_sig = inspect.signature(self.base_agent.api)
            if 'return_full_response' in api_sig.parameters:
                # Enable probability capture by calling with return_full_response=True
                # This will be captured in the next call to the agent's think method
                if hasattr(self.base_agent, '_enable_probability_capture'):
                    self.base_agent._enable_probability_capture = True

        # Get action from base agent
        action = self.base_agent(observation)

        # Get agent state after action (including probability data if available)
        agent_state_after = self._extract_agent_state()

        # Merge states, prioritizing after state for probability information
        merged_agent_state = {**agent_state_before, **agent_state_after}

        # Log the turn with enhanced context including probability data
        self.logger.log_turn(
            player_id=self.player_id,
            observation=observation,
            agent_state=merged_agent_state,
            action=action,
            game_phase=game_phase,
            round_number=round_number
        )

        return action

    def _extract_agent_state(self) -> Dict[str, Any]:
        """
        Extract relevant state information from the agent.

        Returns:
            Dictionary with agent state information
        """
        state = {}

        # Extract common state variables
        if hasattr(self.base_agent, 'belief'):
            state['belief'] = getattr(self.base_agent, 'belief')

        if hasattr(self.base_agent, 'strategy'):
            state['strategy'] = getattr(self.base_agent, 'strategy')

        if hasattr(self.base_agent, 'round'):
            state['round'] = getattr(self.base_agent, 'round')

        if hasattr(self.base_agent, 'observation_history'):
            state['history_length'] = len(getattr(self.base_agent, 'observation_history', []))

        if hasattr(self.base_agent, 'init_info'):
            state['init_info'] = getattr(self.base_agent, 'init_info')

        # Enhanced state extraction for Michael/Vito agents
        if hasattr(self.base_agent, 'is_initialized'):
            state['is_initialized'] = getattr(self.base_agent, 'is_initialized')

        if hasattr(self.base_agent, 'init_identity'):
            state['init_identity'] = getattr(self.base_agent, 'init_identity')

        # Extract LLM probability information (soft labels) for distillation
        if hasattr(self.base_agent, 'current_step_responses'):
            current_step_responses = getattr(self.base_agent, 'current_step_responses')
            if current_step_responses:
                # Get the most recent API response with probabilities
                latest_response = current_step_responses.get('latest')
                if latest_response:
                    state['llm_response'] = self._extract_probability_data(latest_response)

        # Extract API response history if available
        if hasattr(self.base_agent, 'api_responses'):
            api_responses = getattr(self.base_agent, 'api_responses')
            if api_responses:
                # Store recent API responses with probability information
                state['recent_api_responses'] = [
                    self._extract_probability_data(resp) for resp in api_responses[-3:]
                ]

        # Store the full agent for deeper analysis if needed
        state['agent_class'] = self.base_agent.__class__.__name__

        return state

    def _extract_probability_data(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract probability data from API response for distillation.

        Args:
            api_response: Raw API response from LLM

        Returns:
            Structured probability data suitable for training
        """
        if not api_response:
            return {}

        probability_data = {
            'text': api_response.get('text', ''),
            'model': api_response.get('model', ''),
            'has_probabilities': False,
            'top_tokens': [],
            'full_response': api_response
        }

        # Extract logprobs information from different API formats
        if 'logprobs' in api_response:
            logprobs = api_response['logprobs']
            if logprobs and isinstance(logprobs, dict):
                probability_data['has_probabilities'] = True

                # Handle OpenAI API format
                if 'top_logprobs' in logprobs:
                    probability_data['top_tokens'] = []
                    for token_data in logprobs['top_logprobs'][:10]:  # Keep top 10 tokens
                        if isinstance(token_data, dict) and 'token' in token_data:
                            probability_data['top_tokens'].append({
                                'token': token_data['token'],
                                'logprob': token_data.get('logprob', 0.0),
                                'linear_prob': token_data.get('linear_prob', 0.0)
                            })
                        elif isinstance(token_data, str):
                            probability_data['top_tokens'].append({
                                'token': token_data,
                                'logprob': 0.0,
                                'linear_prob': 0.0
                            })

                # Handle other possible formats
                elif 'content' in logprobs:
                    content = logprobs['content']
                    if isinstance(content, list) and content:
                        first_token = content[0]
                        if 'top_logprobs' in first_token:
                            probability_data['top_tokens'] = []
                            for token_data in first_token['top_logprobs'][:10]:
                                probability_data['top_tokens'].append({
                                    'token': token_data.get('token', ''),
                                    'logprob': token_data.get('logprob', 0.0),
                                    'linear_prob': token_data.get('linear_prob', 0.0)
                                })

        # Calculate additional probability statistics
        if probability_data['top_tokens']:
            total_prob = sum(token.get('linear_prob', 0.0) for token in probability_data['top_tokens'])
            probability_data['probability_mass'] = total_prob

            # Calculate entropy using natural logarithm
            import math
            entropy = 0.0
            for token in probability_data['top_tokens']:
                prob = token.get('linear_prob', 0.0)
                if prob > 0:
                    entropy -= prob * math.log(prob)
            probability_data['entropy'] = entropy

        return probability_data

    def _extract_game_phase_from_agent(self) -> Optional[str]:
        """
        Extract current game phase from agent state.

        Returns:
            Current game phase or None
        """
        # Try to get phase from agent's internal logic (specifically for Michael/Vito)
        if hasattr(self.base_agent, 'round') and hasattr(self.base_agent, 'init_info'):
            round_num = getattr(self.base_agent, 'round', 0)
            init_info = getattr(self.base_agent, 'init_info', {})

            # Replicate the phase logic from Michael's code
            current_round = round_num % 5
            if current_round == 0:
                if init_info.get("role") == "Villager":
                    return "day_speak"
                else:
                    return "night"
            elif current_round in [1, 2, 3]:
                return "day_speak"
            else:
                return "day_vote"

        return None

    def _extract_round_from_agent(self) -> Optional[int]:
        """
        Extract current round number from agent.

        Returns:
            Current round number or None
        """
        if hasattr(self.base_agent, 'round'):
            return getattr(self.base_agent, 'round')

        return None

    def __getattr__(self, name):
        """Delegate attribute access to the base agent."""
        return getattr(self.base_agent, name)