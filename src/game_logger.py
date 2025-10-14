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
                 step_info: Optional[Dict] = None) -> None:
        """
        Log a single turn in the game.

        Args:
            player_id: ID of the player taking the action
            observation: Raw observation from the environment
            agent_state: Agent's internal state (beliefs, strategy, etc.)
            action: Action taken by the agent
            step_info: Additional step information from environment
        """
        if not self.enabled or self.current_session is None:
            return

        turn_data = {
            "turn_number": len(self.current_session["turns"]) + 1,
            "timestamp": datetime.now().isoformat(),
            "player_id": player_id,
            "observation": self._serialize_observation(observation),
            "agent_state": agent_state,
            "action": action,
            "step_info": step_info or {}
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
        # Get agent state before action
        agent_state = self._extract_agent_state()

        # Get action from base agent
        action = self.base_agent(observation)

        # Log the turn
        self.logger.log_turn(
            player_id=self.player_id,
            observation=observation,
            agent_state=agent_state,
            action=action
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

        return state

    def __getattr__(self, name):
        """Delegate attribute access to the base agent."""
        return getattr(self.base_agent, name)