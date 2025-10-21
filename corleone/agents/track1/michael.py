
from corleone.agents.agent import LLMAgent
from openai import OpenAI
import random
import itertools
import time
import json
import http.client
from datetime import datetime
import csv
from typing import List, Dict, Optional
import os
import requests
import os
import re
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some advanced features may be limited.")
from dotenv import load_dotenv   




load_dotenv()                     

openai_key = os.getenv("OPENAI_API_KEY")
wwxq_key = os.getenv("WWXQ_API_KEY")




class Michael(LLMAgent):
    def __init__(self, model_name: str, use_strategy_pool: bool = True):
        #super().__init__(model_name)
        self.model_name = model_name
        self.is_initialized = False
        self.init_info = None
        self.belief = ""
        self.strategy = ""
        self.observation_history = []
        self.round = 0
        # Store API responses for logging
        self.api_responses = []  # Store full API responses with probabilities
        self.current_step_responses = {}  # Store responses for current step

        # Strategy pool system
        self.use_strategy_pool = use_strategy_pool
        if use_strategy_pool:
            self.strategy_pool = {}  # Load strategy pool from JSON
            self.current_behavior_strategy_id = None
            self.current_language_strategy_id = None
            self.strategy_usage_log = []  # Track strategy usage in current game
            self.similarity_threshold = 0.7  # Threshold for strategy matching

            # Load strategy pool
            self.load_strategy_pool()
            print("Initializing Michael with dynamic strategy pool...")
        else:
            print("Initializing Michael without strategy pool...")
        """
        night: 0, 5,10,...
        day speak:1,2,3,6,7,8,11,12,13...
        day vote:4,9,14...

        """



    def __call__(self, observation: str) -> str:
        try: # Generate a response
            # return self.api(input_messages=[
            #     {"role": "system", "content": "You are participating in the game Secret Mafia and playing one of the roles. Then you need to understand the game rules, understand your identity and game goals, recognize whether you are currently in a dialogue or voting phase, and then analyze and make decisions based on the current information. Note: During the conversation phase, you are free to speak up; During the voting phase, your voting target format must be the player ID in square brackets, such as: [2]"},
            #     {"role": "user", "content": observation},
            # ],
            # temperature=0.6,
            # model="qwen3-8b",
            # max_tokens=2048)

            # Parse observation into events
            print("\n=========================================   MICHAEL   =============================================")
            print("Observation:", observation)
            obs_list = json.loads(observation) if isinstance(observation, str) and observation.startswith('[') else observation


            
            # First observation - initialization
            if not self.is_initialized:
                self.init_info = self.parse_initialization_info(observation)
                self.init_identity = self.generate_identity_prompt(self.init_info)
                self.belief = self.generate_belief_prompt(self.init_info)
                self.strategy = "No strategy set yet. Will develop based on game progress."
                self.is_initialized = True
                print("Michael: Game information initialized.\n")

            #count round
            current_round = self.round % 5
            if current_round == 0:
                if self.init_info and self.init_info.get("role") == "Villager":
                    self.round += 1
                    phase = "day_speak"
                else:
                    phase = "night"
            elif current_round in [1,2,3]:
                phase = "day_speak"
            else:
                phase = "day_vote"
            self.round += 1

            # Set current phase for strategy matching
            self.current_phase = phase
            print(f"Current Round: {current_round}, Phase: {phase}")

            # Check for game end and update strategy performance
            game_ended, game_result = self.detect_game_end(observation)
            if game_ended:
                print(f"Game ended! Winner: {game_result.get('winner_team')}, Victory: {game_result.get('victory')}")
                if self.use_strategy_pool:
                    self.update_strategy_performance(game_result, game_result.get("role_performance", {}))
                    self.save_strategy_pool()
                    print("Strategy pool updated and saved at game end.")

            # Regular observation processing
            formatted_obs = self.parse_observation_events(obs_list) if isinstance(obs_list, list) else observation
            self.observation_history.append(formatted_obs)
            


            # Step 1: Analyze new information
            analysis = self.parse_llm_response(
            self.api(input_messages=[
                {"role": "system", "content": self.prompt_system()},
                {"role": "user", "content": self.prompt_analyze(formatted_obs)}
            ]),
            "#SUMMARY:")
            
            # Step 2: Update beliefs
            self.belief = self.parse_llm_response(
            self.api(input_messages=[
                {"role": "system", "content": self.prompt_system()},
                {"role": "user", "content": self.prompt_belief(analysis, self.belief)}
            ]),
            "#BELIEF:")
            
            # Step 3: Update strategy with dual strategy system
            strategy_response = self.api(input_messages=[
                {"role": "system", "content": self.prompt_system()},
                {"role": "user", "content": self.prompt_strategy(analysis, self.belief, self.strategy)}
            ])

            # Parse and log strategy usage
            self.strategy = self.parse_llm_response(strategy_response, "#STRATEGY:")

            # Extract and log behavior strategy
            behavior_strategy_match = re.search(r'#BEHAVIOR_STRATEGY:(.*?)(?=#LANGUAGE_STRATEGY:|#FINAL:|$)', self.strategy, re.DOTALL)
            if behavior_strategy_match:
                behavior_strategy_text = behavior_strategy_match.group(1).strip()
                # Check if it's a reference to existing strategy or new
                new_strategy_id = self.create_new_strategy("behavior", behavior_strategy_text)
                self.current_behavior_strategy_id = new_strategy_id
                self.log_strategy_usage(new_strategy_id, "behavior", phase)
            else:
                # Fallback: treat entire strategy as behavior strategy
                new_strategy_id = self.create_new_strategy("behavior", self.strategy)
                self.current_behavior_strategy_id = new_strategy_id
                self.log_strategy_usage(new_strategy_id, "behavior", phase)

            # Extract and log language strategy (only for day_speak phase)
            if phase == "day_speak":
                language_strategy_match = re.search(r'#LANGUAGE_STRATEGY:(.*?)(?=#FINAL:|$)', self.strategy, re.DOTALL)
                if language_strategy_match:
                    language_strategy_text = language_strategy_match.group(1).strip()
                    # Check if it's a reference to existing strategy or new
                    new_strategy_id = self.create_new_strategy("language", language_strategy_text, self.current_behavior_strategy_id)
                    self.current_language_strategy_id = new_strategy_id
                    self.log_strategy_usage(new_strategy_id, "language", phase)
            
            # Step 4: Generate final action/speech



            if phase == "day_speak":
                final_response = self.parse_llm_response(
                self.api(input_messages=[
                    {"role": "system", "content": self.prompt_system()},
                    {"role": "user", "content": self.prompt_talk(self.belief, self.strategy)}
                ]),
                "#FINAL:")
                final_output = final_response


            else:
                final_response = self.parse_llm_response(
                self.api(input_messages=[
                    {"role": "system", "content": self.prompt_system()},
                    {"role": "user", "content": self.prompt_vote(self.belief, self.strategy)}
                ]),
                "#FINAL:")

                bracket_match = re.search(r'\[(\d+)\]', final_response)
                if bracket_match:
                    final_output = f"[{bracket_match.group(1)}]"
                else:
                    patterns = [
                        r'vote[^\d]{0,10}(\d+)',
                        r'detect[^\d]{0,10}(\d+)', 
                        r'eliminate[^\d]{0,10}(\d+)',
                        r'kill[^\d]{0,10}(\d+)',
                        r'protect[^\d]{0,10}(\d+)',
                        r'rescue[^\d]{0,10}(\d+)',
                        r'investigate[^\d]{0,10}(\d+)'
                    ]
                    
                    reversed_text = final_output[::-1]
                    
                    found_number = None
                    for pattern in patterns:
                        reversed_pattern = pattern[::-1]
                        match = re.search(reversed_pattern, reversed_text)
                        if match:
                            found_number = match.group(1)[::-1]
                            break
                    
                    if found_number:
                        final_output = f"[{found_number}]"


            
            # print("Observation:\n")
            # print(observation)
            # print("=" * 20)
            # print("\n\n\n\n\nSYSTEM PROMPT:\n", self.prompt_system())
            # print("=" * 20)
            # print("\n\n\n\n\nANALYSIS PROMPT:\n", self.prompt_analyze(formatted_obs))
            # print("\nANALYSIS RESULT:\n", analysis)
            # print("=" * 20)
            # print("\n\n\n\n\nBELIEF PROMPT:\n", self.prompt_belief(analysis, self.belief))
            # print("\nBELIEF RESULT:\n", self.belief)
            # print("=" * 20)
            # print("\n\n\n\n\nSTRATEGY PROMPT:\n", self.prompt_strategy(analysis, self.belief, self.strategy))
            # print("\nSTRATEGY RESULT:\n", self.strategy)
            if phase == "day_speak":
                print("TALK PROMPT:\n", self.prompt_talk(self.belief, self.strategy))
            else:
                print("VOTE PROMPT:\n", self.prompt_vote(self.belief, self.strategy))
            print("response:\n", final_response)
            print("\nFINAL OUTPUT:\n", final_output)
            print("\n\n" + "=" * 50 + "\n\n\n\n\n\n")


            return final_output



        except Exception as e:
            return f"An error occurred: {e}"
        



    
    
    
    def parse_observation_events(self, observation: List[List]) -> str:
        """
        Parse observation list into formatted event statements
        
        Args:
            observation: List of events, each event is [speaker_id, content, message_type]
                        speaker_id: -1 for system, 0-N for players
                        content: message content
                        message_type: typically 2 for speech, 4 for system announcement
            
        Returns:
            Formatted event description text
        """
        prompt_parts = []
        
        for event in observation:
            if len(event) < 2:
                continue
                
            speaker_id = event[0]
            content = event[1]
            
            # Check if content contains a vote pattern like [0], [1], etc.
            vote_match = re.search(r'^\[(\d+)\]$', content.strip())
            
            if speaker_id == -1:
                # System message
                prompt_parts.append(f"SYSTEM: {content}")
            elif vote_match:
                # Vote action
                voted_player = vote_match.group(1)
                prompt_parts.append(f"Player {speaker_id} VOTED: [Player {voted_player}]")
            else:
                # Regular speech
                # Check if it's wrapped in brackets (internal thoughts)
                if content.strip().startswith('[') and content.strip().endswith(']'):
                    prompt_parts.append(f"Player {speaker_id} (internal): {content}")
                else:
                    prompt_parts.append(f"Player {speaker_id}: {content}")
        
        return "\n".join(prompt_parts)
        
    
    
    
    
    def parse_initialization_info(self, observation_text: str) -> Dict:

        init_info = {
            "player_id": None,
            "role": None,
            "team": None,
            "description": None,
            "all_players": [],
            "teammates": []
        }
        
        # 提取玩家ID
        player_match = re.search(r'You are Player (\d+)', observation_text)
        if player_match:
            init_info["player_id"] = int(player_match.group(1))
        
        # 提取角色信息
        role_match = re.search(r'Your role: (.+)', observation_text)
        if role_match:
            init_info["role"] = role_match.group(1).strip()
        
        # 提取队伍信息
        team_match = re.search(r'Team: (.+)', observation_text)
        if team_match:
            init_info["team"] = team_match.group(1).strip()
        
        # 提取描述信息
        desc_match = re.search(r'Description: (.+?)(?:\n\n|Players:)', observation_text, re.DOTALL)
        if desc_match:
            init_info["description"] = desc_match.group(1).strip()
        
        # 提取所有玩家列表
        players_match = re.search(r'Players: (.+)', observation_text)
        if players_match:
            players_str = players_match.group(1)
            # 提取所有 Player X 格式的玩家
            player_ids = re.findall(r'Player (\d+)', players_str)
            init_info["all_players"] = [int(pid) for pid in player_ids]
        
        # 提取队友信息
        teammates_match = re.search(r'Your teammates are: (.+?)\.', observation_text)
        if teammates_match:
            teammates_str = teammates_match.group(1)
            teammate_ids = re.findall(r'Player (\d+)', teammates_str)
            init_info["teammates"] = [int(tid) for tid in teammate_ids]
        
        return init_info


    def generate_identity_prompt(self, init_info: Dict) -> str:
        """
        Generate identity description prompt from initialization info
        
        Args:
            init_info: Dictionary returned by parse_initialization_info
            
        Returns:
            Formatted identity description text
        """
        prompt_parts = []
        
        prompt_parts.append("=== YOUR IDENTITY ===")
        prompt_parts.append(f"Player ID: Player {init_info['player_id']}")
        prompt_parts.append(f"Role: {init_info['role']}")
        prompt_parts.append(f"Team: {init_info['team']}")
        prompt_parts.append(f"Description: {init_info['description']}")

        if init_info['role'] == "Mafia":
            prompt_parts.append("Goals: Try to deceive other players to conceal yourself and your companions until enough villagers are eliminated.")
        elif init_info['role'] == "A regular villager":
            prompt_parts.append("Goals: Try to identify and eliminate all mafia members through discussion and voting.")
        elif init_info['role'] == "Detective":
            prompt_parts.append("Goals: Try to Protect yourself and identify mafia members through investigation and help villagers eliminate them.")
        elif init_info['role'] == "Doctor":
            prompt_parts.append("Goals: Try to protect key villagers, especially yourself and the detective, from being eliminated by the mafia during the night.")
        
        return "\n".join(prompt_parts)


    def generate_belief_prompt(self, init_info: Dict, custom_beliefs: Optional[Dict[int, Dict]] = None) -> str:
        """
        Generate belief information prompt for all players
        
        Args:
            init_info: Dictionary returned by parse_initialization_info
            custom_beliefs: Custom belief information, format:
                {
                    player_id: {
                        "role": "role name",
                        "status": "unknown" | "suspected" | "confirmed",
                        "notes": "additional notes",
                        "alive": True | False
                    }
                }
                If None, auto-initialize basic beliefs
            
        Returns:
            Formatted belief description text
        """
        prompt_parts = []
        prompt_parts.append("=== PLAYER BELIEFS ===")
        
        # If no custom_beliefs provided, initialize based on init_info
        if custom_beliefs is None:
            custom_beliefs = {}
            
            for player_id in init_info['all_players']:
                if player_id == init_info['player_id']:
                    # Self
                    custom_beliefs[player_id] = {
                        "role": init_info['role'],
                        "status": "confirmed",
                        "notes": "This is yourself.",
                        "alive": True
                    }
                elif player_id in init_info['teammates']:
                    # Teammates
                    custom_beliefs[player_id] = {
                        "role": init_info['role'],  # same role as teammates
                        "status": "confirmed",
                        "notes": "Confirmed by system.",
                        "alive": True
                    }
                else:
                    # Other players
                    custom_beliefs[player_id] = {
                        "role": "Unknown",
                        "status": "unknown",
                        "notes": "There is currently no further information available.",
                        "alive": True
                    }
        
        # Sort by player ID
        sorted_players = sorted(custom_beliefs.keys())
        
        for player_id in sorted_players:
            belief = custom_beliefs[player_id]
            status = belief["status"]
            
            # Format status and role
            if status == "confirmed":
                role_info = f"Role: CONFIRMED as {belief['role']}"
            elif status == "suspected":
                role_info = f"Role: SUSPECTED as {belief['role']}"
            else:
                role_info = "Role: UNKNOWN"
            
            # Add alive/dead status
            life_status = "ALIVE" if belief.get("alive", True) else "DEAD"
            
            # Build entry with consistent formatting
            entry = f"Player {player_id}: {role_info} | Status: {life_status}"
            
            if belief.get("notes"):
                entry += f" | Notes: {belief['notes']}"
            
            prompt_parts.append(entry)
        
        return "\n".join(prompt_parts)




    
    def prompt_system(self) -> str:
        ret = f"""
        You are participating in the game Secret Mafia and playing one of the roles.\n
        The game will start at night and alternate between night and day. At night, Mafia will secretly eliminate players, detectives can investigate a player's identity, and doctors can choose to protect a player from being eliminated by Mafia. During each daytime stage, players will have 3 rounds of discussion and then vote to eliminate the player with the most votes.
        \n\n
        Here are some information about this game:\n
        {self.init_identity}\n

    """
        return ret


    def prompt_analyze(self, observation) -> str:
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 1. Analyze newly acquired information.

    # You got these new information:
    {observation}

    # Please follow the steps:
    1. What information does the system declaration reflect? You may need to analyze this - for example, if a player is eliminated at night, it can reflect Mafia's strategy, and if no player is eliminated at night, it indicates that the doctor successfully protected a player.
    2. For other players' comments, try to empathize with their perspective one by one: why do they speak like this? What is the purpose? This may reflect their identity or strategy.
    3. Summarize your analysis results and start with a symbol: "#SUMMARY:"
    """
        return ret



    def prompt_belief(self, analysis, belief) -> str:
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 2. You should update your beliefs.

    Here is your analysis results:
    {analysis}


    # Previous belief:
    {belief}

    # Please follow the steps:
    1. Based on system message, which players' survival status needs to be modified?
    2. Based on your analysis just now, which players' identities can be guessed? Note that identity confirmation can only be set through system messages from Mafia and Detection, otherwise you can only suspect their roles.
    3. Modify your BELIEF and generate a new BELIEF, maintain the format: [player_id: player identity guess | survival status | explanation of identity guess and elimination reason.], starting with the symbol: "#BELIEF:" 

    """
        return ret



    def prompt_strategy(self, analysis, belief, strategy) -> str:
        """Generate strategy prompt with behavior and language strategy support"""
        # Get current phase for strategy selection
        current_phase = getattr(self, 'current_phase', 'day_speak')

        # Select reference strategies
        behavior_strategies = self.select_reference_strategies('behavior')

        # Prepare behavior strategy section
        behavior_section = ""
        if behavior_strategies:
            behavior_section = "\n# Reference Behavior Strategies:\n"
            for i, ref_strategy in enumerate(behavior_strategies[:2]):  # Show top 2
                behavior_section += f"{i+1}. {ref_strategy['strategy_text']}\n"
            behavior_section += "\nThese behavior strategies are provided as reference. You can adopt one, modify it, or create your own based on the current situation.\n"

        # Prepare language strategy section (only for day_speak phase)
        language_section = ""
        if current_phase == 'day_speak':
            language_strategies = self.select_reference_strategies('language')
            if language_strategies:
                language_section = "\n# Reference Language Strategies:\n"
                for i, ref_strategy in enumerate(language_strategies[:2]):  # Show top 2
                    language_section += f"{i+1}. {ref_strategy['strategy_text']}\n"
                language_section += "\nThese language strategies show how to express yourself effectively. You can use these techniques or develop your own approach.\n"

        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 3. Please refer to your goals, analysis, and beliefs, then decide your strategy.

    # Your analysis:
    {analysis}

    # Your belief:
    {belief}

    # Your current strategy:
    {strategy}

    # Strategy Selection Guidance:
    Based on the current game phase ({current_phase}), you need to consider both BEHAVIOR and LANGUAGE strategies:

    ## BEHAVIOR STRATEGY (All Phases)
    This defines your overall goal and approach for this round (e.g., "Hide identity and mislead", "Protect key players", "Investigate suspicious behavior").

    ## LANGUAGE STRATEGY (Day Discussion Only)
    This defines HOW you will communicate and express yourself when speaking (e.g., "Use logical reasoning to point out contradictions", "Create emotional appeal to gain trust", "Use misleading information to confuse others").

    {behavior_section}
    {language_section}
    # Please follow the steps:
    1. First, decide your BEHAVIOR STRATEGY - what is your primary goal for this round?
    2. If it's a discussion phase, decide your LANGUAGE STRATEGY - how will you communicate effectively?
    3. Generate your strategies using the format below:

    #BEHAVIOR_STRATEGY: [Your behavior strategy description]

    #LANGUAGE_STRATEGY: [Your language strategy description]

    # Final combined strategy:
    #STRATEGY: [Your final integrated strategy description]

    # Important: You can reference the provided strategies, modify them, or create completely new ones based on your analysis and beliefs.
    """
        return ret
    



    def prompt_talk(self, belief, strategy) -> str:
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 4. Decide on your own speech or action.

    You need to speak now, you need to generate the final speech content based on your beliefs and strategies, and the speech content should not contain your thinking process.

    Now, please refer to your beliefs and predetermined strategies to generate your final speech. Start with a symbol: "#FINAL:".

    # BELIEF:
    {belief}

    # STRATEGY:
    {strategy}

    """
        return ret


    def prompt_vote(self, belief, strategy) -> str:
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 4. Decide on your own action.
    
    Now you need to select a player for action (including voting, rescue, detect, etc.), you need to output the final goal in the format of "[X]", where X represents the player's ID and is a number. For example, "[1]" means you want to vote for player 1.

    Now, please refer to your beliefs and predetermined strategies to generate your final action goal. Start with a symbol: "#FINAL:".

    # BELIEF:
    {belief}

    # STRATEGY:
    {strategy}

    """
        return ret









    def api(
            self,
            input_messages: Optional[List[Dict]] = None,
            temperature: float = 0.4,
            model: str = 'qwen3-8b',
            max_tokens: int = 1024,
            return_full_response: bool = False,
            full_logprobs: bool = False,  # New parameter for complete probability distribution
    ):
        if input_messages is None:
            raise ValueError("messages should not be None!")

        if self.model_name in ['qwen3-8b', 'qwen3-4b','deepseek-v3.1','deepseek-r1']:
            print(self.model_name)
            url = "cloud.infini-ai.com"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {wwxq_key}"
            }

            payload = {
                "model": self.model_name,
                "messages": input_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                # Add logprobs to get token probabilities
                "logprobs": True,
                "top_logprobs": 5,  # Get top 5 tokens for each position
            }

            MAX_RETRIES = 4
            attempts = 0
            RETRY_INTERVAL = 1

            while attempts < MAX_RETRIES:
                try:
                    conn = http.client.HTTPSConnection(url)
                    conn.request("POST", f"/maas/{self.model_name}/nvidia/chat/completions",
                                json.dumps(payload), headers)
                    res = conn.getresponse()
                    data = res.read()
                    response_json = json.loads(data.decode("utf-8"))
                    print(f"{self.model_name} HTTP Status:", res.status)

                    # Store full response for logging
                    response_data = {
                        "timestamp": datetime.now().isoformat(),
                        "model": self.model_name,
                        "temperature": temperature,
                        "input_messages": input_messages,
                        "response": response_json
                    }
                    self.api_responses.append(response_data)

                    if return_full_response:
                        return response_data
                    else:
                        return response_json["choices"][0]["message"]["content"]

                except KeyError as e:
                    if attempts < MAX_RETRIES - 1:
                        print(f"KeyError: {e}. Retrying in {RETRY_INTERVAL} seconds...")
                        time.sleep(RETRY_INTERVAL)
                        attempts += 1
                        RETRY_INTERVAL = RETRY_INTERVAL * 2
                    else:
                        raise Exception(f"Failed to get 'choices' after {MAX_RETRIES} attempts.") from e
        else:
            api_key = openai_key

            MAX_RETRIES = 5
            attempts = 0
            RETRY_INTERVAL = 1

            while attempts < MAX_RETRIES:
                try:
                    client = OpenAI(
                        api_key=api_key,
                        base_url="https://xiaoai.plus/v1",
                    )
                    completion = client.chat.completions.create(
                        model=self.model_name,
                        messages=input_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        logprobs=True,
                        top_logprobs=5
                    )
                    print(f"GPT ({self.model_name}) working.")

                    # Store full response for logging
                    response_data = {
                        "timestamp": datetime.now().isoformat(),
                        "model": self.model_name,
                        "temperature": temperature,
                        "input_messages": input_messages,
                        "response": completion.model_dump()  # Convert to dict
                    }
                    self.api_responses.append(response_data)

                    if return_full_response:
                        return response_data
                    else:
                        return completion.choices[0].message.content
                except Exception as e:
                    if attempts < MAX_RETRIES - 1:
                        print(f"HTTP Exception or Timeout Error occurred: {e}. Retrying in {RETRY_INTERVAL} seconds...")
                        time.sleep(RETRY_INTERVAL)
                        attempts += 1
                        RETRY_INTERVAL = RETRY_INTERVAL * 2
                    else:
                        raise Exception(f"Failed to get response after {MAX_RETRIES} attempts.") from e
            


    def parse_llm_response(self, response_text, tag_name):

        index = response_text.find(tag_name)
        if index != -1:
            return response_text[index + len(tag_name):].strip()
        else:
            return response_text
        


  # Strategy Pool System Methods
    def load_strategy_pool(self):
        """Load strategy pool from JSON file"""
        try:
            import json
            strategy_pool_path = os.path.join(os.getcwd(), "strategy_pool.json")
            if os.path.exists(strategy_pool_path):
                with open(strategy_pool_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.strategy_pool = {s["id"]: s for s in data.get("strategies", [])}
                print(f"Loaded {len(self.strategy_pool)} strategies from pool")
            else:
                print(f"Strategy pool file not found at {strategy_pool_path}, using empty pool")
                self.strategy_pool = {}
        except Exception as e:
            print(f"Error loading strategy pool: {e}")
            self.strategy_pool = {}

    def save_strategy_pool(self):
        """Save strategy pool to JSON file"""
        if not self.use_strategy_pool:
            return

        try:
            import json
            strategy_pool_path = os.path.join(os.getcwd(), "strategy_pool.json")

            # Load existing data to preserve metadata
            existing_data = {}
            if os.path.exists(strategy_pool_path):
                with open(strategy_pool_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)

            # Update strategies
            existing_data["strategies"] = list(self.strategy_pool.values())
            existing_data["metadata"]["last_updated"] = datetime.now().isoformat()
            existing_data["metadata"]["total_strategies"] = len(self.strategy_pool)

            # Count strategy types
            behavior_count = sum(1 for s in self.strategy_pool.values() if s["type"] == "behavior")
            language_count = sum(1 for s in self.strategy_pool.values() if s["type"] == "language")
            existing_data["metadata"]["behavior_strategies"] = behavior_count
            existing_data["metadata"]["language_strategies"] = language_count

            with open(strategy_pool_path, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(self.strategy_pool)} strategies to pool")
        except Exception as e:
            print(f"Error saving strategy pool: {e}")

    def get_current_context_features(self):
        """Extract current game context features for strategy matching"""
        if not self.init_info:
            return {}

        alive_players = len([p for p in self.init_info.get("all_players", [])])

        # Extract known threats from beliefs
        known_threats = 0
        if self.belief:
            # Simple heuristic: count suspected/confirmed mafia in beliefs
            for line in self.belief.split("\n"):
                if "mafia" in line.lower() and ("suspected" in line.lower() or "confirmed" in line.lower()):
                    known_threats += 1

        # Determine urgency based on game state
        urgency = "medium"
        if alive_players <= 4:
            urgency = "high"
        elif alive_players >= 6:
            urgency = "low"

        return {
            "alive_players": alive_players,
            "known_threats": known_threats,
            "urgency_level": urgency
        }

    def calculate_strategy_similarity(self, strategy_id, current_context):
        """Calculate similarity score between a strategy and current context"""
        if strategy_id not in self.strategy_pool:
            return 0.0

        strategy = self.strategy_pool[strategy_id]
        total_score = 0.0

        # Role matching (40% weight)
        if self.init_info and strategy["role"] == self.init_info.get("role"):
            total_score += 0.4

        # Game phase matching (30% weight)
        current_phase = getattr(self, "current_phase", "day_speak")
        if strategy["game_phase"] == current_phase:
            total_score += 0.3

        # Context features matching (20% weight)
        strategy_context = strategy.get("context_features", {})

        # Check alive players range
        min_players = strategy_context.get("min_alive_players", 0)
        max_players = strategy_context.get("max_alive_players", 10)
        current_players = current_context.get("alive_players", 0)
        if min_players <= current_players <= max_players:
            total_score += 0.1

        # Check urgency level
        allowed_urgency = strategy_context.get("urgency_levels", [])
        current_urgency = current_context.get("urgency_level", "medium")
        if current_urgency in allowed_urgency:
            total_score += 0.1

        # Strategy type context matching (10% weight)
        strategy_type = strategy.get("type", "behavior")
        if strategy_type == "behavior":
            total_score += 0.1  # Behavior strategies always get full context score
        elif strategy_type == "language" and current_phase == "day_speak":
            total_score += 0.1  # Language strategies only in discussion phase

        return total_score

    def select_reference_strategies(self, strategy_type, behavior_context=None):
        """Select reference strategies based on current context"""
        current_context = self.get_current_context_features()

        # Filter strategies by type and relevance
        candidate_strategies = []
        for strategy_id, strategy in self.strategy_pool.items():
            if strategy["type"] != strategy_type:
                continue

            # Additional filtering for language strategies
            if strategy_type == "language" and behavior_context:
                # Check if language strategy relates to current behavior strategy
                related_behavior_id = strategy.get("related_behavior_id")
                if related_behavior_id and related_behavior_id != behavior_context:
                    continue

            # Calculate similarity
            similarity = self.calculate_strategy_similarity(strategy_id, current_context)
            if similarity > 0.3:  # Minimum threshold
                candidate_strategies.append((strategy_id, similarity))

        # Sort by similarity score
        candidate_strategies.sort(key=lambda x: x[1], reverse=True)

        # Select strategies based on threshold
        selected_strategies = []
        for strategy_id, similarity in candidate_strategies[:3]:  # Top 3
            if similarity >= self.similarity_threshold:
                selected_strategies.append(self.strategy_pool[strategy_id])

        return selected_strategies

    def log_strategy_usage(self, strategy_id, strategy_type, phase):
        """Log strategy usage for performance tracking"""
        log_entry = {
            "strategy_id": strategy_id,
            "strategy_type": strategy_type,
            "phase": phase,
            "round": self.round,
            "context": self.get_current_context_features(),
            "belief_summary": self.belief[:200] if self.belief else "",  # Truncate for storage
            "timestamp": datetime.now().isoformat()
        }
        self.strategy_usage_log.append(log_entry)

        # Update strategy usage count
        if strategy_id in self.strategy_pool:
            self.strategy_pool[strategy_id]["usage_count"] += 1
            self.strategy_pool[strategy_id]["last_used"] = datetime.now().isoformat()

    def update_strategy_performance(self, game_result, role_performance):
        """Update strategy performance based on game outcome"""
        try:
            # Calculate performance scores for each used strategy
            base_score = 1.0 if game_result.get("victory", False) else -1.0

            # Role-specific performance bonus
            role_bonus = role_performance.get("bonus", 0.0)
            total_score = base_score + role_bonus

            # Update each used strategy
            for log_entry in self.strategy_usage_log:
                strategy_id = log_entry["strategy_id"]
                if strategy_id in self.strategy_pool:
                    strategy = self.strategy_pool[strategy_id]

                    # Update success score
                    strategy["success_score"] += total_score

                    # Update average performance
                    usage_count = strategy["usage_count"]
                    if usage_count > 0:
                        strategy["avg_performance"] = strategy["success_score"] / usage_count

                    # Ensure performance stays within reasonable bounds
                    strategy["avg_performance"] = max(0.0, min(1.0, strategy["avg_performance"]))

            print(f"Updated strategy performance. Total strategies updated: {len(self.strategy_usage_log)}")

        except Exception as e:
            print(f"Error updating strategy performance: {e}")

    def create_new_strategy(self, strategy_type, strategy_text, behavior_context=None):
        """Create a new strategy and add it to the pool"""
        import uuid

        new_strategy = {
            "id": f"{strategy_type}_{uuid.uuid4().hex[:8]}",
            "type": strategy_type,
            "role": self.init_info.get("role", "Unknown") if self.init_info else "Unknown",
            "game_phase": getattr(self, "current_phase", "day_speak"),
            "strategy_text": strategy_text,
            "context_features": self.get_current_context_features(),
            "usage_count": 1,
            "success_score": 0.0,
            "avg_performance": 0.5,  # Start with neutral performance
            "related_behavior_id": behavior_context if strategy_type == "language" else None,
            "last_used": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        }

        self.strategy_pool[new_strategy["id"]] = new_strategy
        print(f"Created new {strategy_type} strategy: {new_strategy['id']}")

        return new_strategy["id"]

    def detect_game_end(self, observation: str) -> tuple[bool, Dict]:
        """Detect if the game has ended and extract game results"""
        try:
            # Parse observation
            obs_list = json.loads(observation) if isinstance(observation, str) and observation.startswith('[') else observation

            if isinstance(obs_list, list):
                # Look through all events for game end messages
                for event in obs_list:
                    if len(event) >= 2 and event[0] == -1:  # System message
                        message = event[1]

                        # Check for game end patterns
                        if "wins!" in message or "All Mafia were eliminated" in message or "Mafia reached parity" in message:
                            game_result = self._parse_game_result(message)
                            if game_result:
                                return True, game_result

            # Also check formatted observations
            if isinstance(observation, str):
                if "wins!" in observation or "All Mafia were eliminated" in observation or "Mafia reached parity" in observation:
                    game_result = self._parse_game_result(observation)
                    if game_result:
                        return True, game_result

            return False, {}

        except Exception as e:
            print(f"Error detecting game end: {e}")
            return False, {}

    def _parse_game_result(self, message: str) -> Dict:
        """Parse game result from system message"""
        try:
            game_result = {
                "victory": False,
                "winner_team": None,
                "reason": "",
                "role_performance": {}
            }

            # Determine winner and reason
            if "All Mafia were eliminated" in message or "Village wins!" in message:
                game_result["winner_team"] = "Village"
                game_result["reason"] = "All Mafia eliminated"
            elif "Mafia reached parity" in message or "Mafia wins!" in message:
                game_result["winner_team"] = "Mafia"
                game_result["reason"] = "Mafia reached parity"

            # Determine if current player won
            if self.init_info:
                player_role = self.init_info.get("role", "")
                player_team = "Mafia" if player_role == "Mafia" else "Village"
                game_result["victory"] = (player_team == game_result["winner_team"])

                # Calculate role-specific performance bonus
                if game_result["victory"]:
                    bonus = 0.2  # Base victory bonus
                    # Additional bonuses for special roles
                    if player_role == "Detective" and game_result["winner_team"] == "Village":
                        bonus += 0.1  # Detective bonus for village victory
                    elif player_role == "Doctor" and game_result["winner_team"] == "Village":
                        bonus += 0.1  # Doctor bonus for protecting village
                    elif player_role == "Mafia" and game_result["winner_team"] == "Mafia":
                        bonus += 0.1  # Mafia bonus for successful deception
                else:
                    bonus = -0.1  # Small penalty for losing

                game_result["role_performance"] = {
                    "role": player_role,
                    "team": player_team,
                    "bonus": bonus
                }

            return game_result

        except Exception as e:
            print(f"Error parsing game result: {e}")
            return {}