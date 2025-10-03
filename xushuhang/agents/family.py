import json, re
from src.agent import LLMAgent
from typing import List, Dict, Optional
from xushuhang.agents.api_router import get_api_class


class Vito(LLMAgent):
    def __init__(self, model_name: str, api_model_spec='qwen3-8b'):
        #super().__init__(model_name)
        self.model_name = model_name or "qwen3-8b"
        self.is_initialized = False
        self.init_info = None
        self.belief = ""
        self.strategy = ""
        self.observation_history = []

        self.api = get_api_class(api_model_spec)(model_name=model_name)


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
            obs_list = json.loads(observation) if isinstance(observation, str) and observation.startswith('[') else observation
            
            # First observation - initialization
            if not self.is_initialized:
                self.init_info = self.parse_initialization_info(observation)
                self.init_identity = self.generate_identity_prompt(self.init_info)
                self.belief = self.generate_belief_prompt(self.init_info)
                self.strategy = "No strategy set yet. Will develop based on game progress."
                self.is_initialized = True

            
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
            
            # Step 3: Update strategy
            self.strategy = self.parse_llm_response(
            self.api(input_messages=[
                {"role": "system", "content": self.prompt_system()},
                {"role": "user", "content": self.prompt_strategy(analysis, self.belief, self.strategy)}
            ]),
            "#STRATEGY:")
            
            # Step 4: Generate final action/speech
            #TODO: [x] format
            final_output = self.parse_llm_response(
            self.api(input_messages=[
                {"role": "system", "content": self.prompt_system()},
                {"role": "user", "content": self.prompt_talk(self.belief, self.strategy)}
            ]),
            "#FINAL:")

            bracket_match = re.search(r'\[(\d+)\]', final_output)
            if bracket_match:
                final_output = f"[{bracket_match.group(1)}]"
            else:
                patterns = [
                    r'vote[^\d]{0,10}(\d+)',
                    r'detect[^\d]{0,10}(\d+)', 
                    r'eliminate[^\d]{0,10}(\d+)'
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
            print("\n\n\n\n\n" + "=" * 20)
            print("TALK PROMPT:\n", self.prompt_talk(self.belief, self.strategy))
            print("\nFINAL OUTPUT:\n", final_output)
            print("\n\n\n\n\n" + "=" * 20)


            return final_output



        except Exception as e:
            return f"An error occurred: {e}"
        




    # def log_turn_info(self, observation, formatted_obs, analysis, analysis_response, 
    #                 belief_response, strategy_response, final_response, final_output):
    #     """
    #     将回合信息记录到日志文件中
    #     """
    #     # 创建logs目录（如果不存在）
    #     log_dir = "logs"
    #     if not os.path.exists(log_dir):
    #         os.makedirs(log_dir)
        
    #     # 生成日志文件名（包含时间戳）
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     player_id = self.init_info['player_id'] if self.init_info and self.init_info['player_id'] else "unknown"
    #     log_filename = f"{log_dir}/player_{player_id}_turn_{timestamp}.log"
        
    #     # 写入日志文件
    #     with open(log_filename, 'w', encoding='utf-8') as f:
    #         f.write("=" * 80 + "\n")
    #         f.write(f"TURN LOG - Player {player_id} - {timestamp}\n")
    #         f.write("=" * 80 + "\n\n")
            
    #         f.write("OBSERVATION:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(observation + "\n\n")
            
    #         f.write("FORMATTED OBSERVATION:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(formatted_obs + "\n\n")
            
    #         f.write("SYSTEM PROMPT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(self.prompt_system() + "\n\n")
            
    #         f.write("ANALYSIS PROMPT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(self.prompt_analyze(formatted_obs) + "\n\n")
            
    #         f.write("ANALYSIS RAW RESPONSE:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(analysis_response + "\n\n")
            
    #         f.write("ANALYSIS RESULT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(analysis + "\n\n")
            
    #         f.write("BELIEF PROMPT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(self.prompt_belief(analysis, self.belief) + "\n\n")
            
    #         f.write("BELIEF RAW RESPONSE:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(belief_response + "\n\n")
            
    #         f.write("BELIEF RESULT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(self.belief + "\n\n")
            
    #         f.write("STRATEGY PROMPT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(self.prompt_strategy(analysis, self.belief, self.strategy) + "\n\n")
            
    #         f.write("STRATEGY RAW RESPONSE:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(strategy_response + "\n\n")
            
    #         f.write("STRATEGY RESULT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(self.strategy + "\n\n")
            
    #         f.write("TALK PROMPT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(self.prompt_talk(self.belief, self.strategy) + "\n\n")
            
    #         f.write("TALK RAW RESPONSE:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(final_response + "\n\n")
            
    #         f.write("FINAL OUTPUT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(final_output + "\n\n")
            
    #         f.write("=" * 80 + "\n")
    #         f.write("END OF TURN LOG\n")
    #         f.write("=" * 80 + "\n")
        
    #     print(f"Turn log saved to: {log_filename}")



# PROMPTING===================================================================================================================
    
    
    
    
    
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

    You got these new information:
    {observation}

    Please follow the steps:
    1. What key information do these records reveal?
    2. For other players' comments, try to empathize with their perspective: why do they speak like this? What is the purpose? This may reflect their identity or strategy.
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

    Please follow the steps:
    1. Based on system message, which players' survival status needs to be modified?
    2. Based on your analysis just now, which players' identities can be guessed? Note that identity confirmation can only be set through system messages from Mafia and Detection, otherwise you can only suspect their roles.
    3. Modify your BELIEF and generate a new BELIEF, maintain the format: [player_id: player identity guess | survival status | explanation of identity guess and elimination reason.], starting with the symbol: "#BELIEF:" 

    """
        return ret



    def prompt_strategy(self, analysis, belief, strategy) -> str:
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 3. Please refer to your goals, analysis, and beliefs, then decide your strategy.

    # Your analysis:
    {analysis}

    # Your belief:
    {belief}

    # Your strategy:
    {strategy}

    Please follow the steps:
    1. What is your goal?
    2. Based on your analysis and belief, what is your strategy? For example, you can decide whether to claim which character you are, encourage everyone to expel which player, explain your words and actions to everyone, and so on.
    3. Generate a new STRATEGY, starting with the symbol: "#STRATEGY:"

    """
        return ret
    



    def prompt_talk(self, belief, strategy) -> str:
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 4. Decide on your own speech or action.

    If you need to speak now, you need to generate the final speech content based on your beliefs and strategies, and the speech content should not contain any thinking process.

    If you need to select a player for action (including voting, rescue, reconnaissance, etc.), you need to output the final goal in the format of "[X]", where X represents the player's ID and is a number. For example, "[1]" means you want to vote for player 1.

    Now, please refer to your beliefs and predetermined strategies to generate your final speech or action goals. Start with a symbol: "#FINAL:".

    # BELIEF:
    {belief}

    # STRATEGY:
    {strategy}

    """
        return ret

    def parse_llm_response(self, response_text, tag_name):

        index = response_text.find(tag_name)
        if index != -1:
            return response_text[index + len(tag_name):].strip()
        else:
            return response_text