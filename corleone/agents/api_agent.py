import json, re
from src.agent import LLMAgent, Agent
from typing import List, Dict, Optional
from corleone.api.api_router import get_api_class
from corleone.game_logger import GameLogger


class ApiAgent(Agent):
    def __init__(self, model_name: str, api_model_spec='qwen3-8b', enable_logging: bool = True):
        #super().__init__(model_name)
        self.system_prompt = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format."

        self.model_name = model_name or "qwen3-8b"
        self.observation_history = []
        self.turn_counter = 0

        self.api = get_api_class(api_model_spec)(model=api_model_spec)

        # Initialize logger
        self.logger = GameLogger() if enable_logging else None

    def __call__(self, observation: str) -> str:
        try: # Generate a response
            # Parse observation into events
            self.observation_history.append(observation)

            response = self.api(input_messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": observation}
            ])
            # Extract player ID from response
            bracket_match = re.search(r'\[(\d+)\]', response)
            if bracket_match:
                final_output = f"[{bracket_match.group(1)}]"
            else:
                final_output = response

            return final_output

        except Exception as e:
            error_msg = f"An error occurred: {e}"
            print(f"\n!!! ERROR in Vito agent: {error_msg}")

            # Log error if logger exists
            if self.logger:
                self.logger.end_turn(f"ERROR: {str(e)}")

            # Re-raise to see full traceback during development
            # Comment this out in production if you want to continue playing
            raise e
        

