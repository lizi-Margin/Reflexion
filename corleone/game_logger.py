import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class GameLogger:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.getcwd()

        # Create timestamp directory: runs/251003-3_08PM
        now = datetime.now()
        date_str = now.strftime("%y%m%d")
        time_str = now.strftime("%I_%M%p").lstrip('0')  # Remove leading zero from hour
        timestamp = f"{date_str}-{time_str}"

        self.run_dir = Path(base_dir) / "runs" / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize game log structure
        self.game_log = {
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "player_id": None,
                "role": None,
                "team": None
            },
            "turns": []
        }

        self.current_turn = None

    def set_player_info(self, player_id: int, role: str, team: str):
        """Set player metadata"""
        self.game_log["metadata"]["player_id"] = player_id
        self.game_log["metadata"]["role"] = role
        self.game_log["metadata"]["team"] = team

    def start_turn(self, turn_number: int, observation: str):
        """Start a new turn"""
        self.current_turn = {
            "turn": turn_number,
            "observation": observation,
            "phases": {
                "analysis": {},
                "belief_update": {},
                "strategy_update": {},
                "final_action": {}
            }
        }

    def log_phase(self, phase_name: str, prompt: str, response: str, parsed_result: Optional[str] = None):
        """Log a phase of the turn"""
        if self.current_turn is None:
            return

        phase_data = {
            "prompt": prompt,
            "raw_response": response
        }

        if parsed_result is not None:
            phase_data["parsed_result"] = parsed_result

        self.current_turn["phases"][phase_name] = phase_data

    def end_turn(self, final_output: str):
        """End the current turn and save it"""
        if self.current_turn is None:
            return

        self.current_turn["final_output"] = final_output
        self.game_log["turns"].append(self.current_turn)
        self.current_turn = None

        # Save after each turn
        self.save()

    def save(self):
        """Save the game log to JSON file"""
        log_file = self.run_dir / "game_log.json"

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.game_log, f, indent=2, ensure_ascii=False)

    def finalize(self, outcome: Optional[str] = None):
        """Finalize the game log"""
        self.game_log["metadata"]["end_time"] = datetime.now().isoformat()
        if outcome:
            self.game_log["metadata"]["outcome"] = outcome
        self.save()
