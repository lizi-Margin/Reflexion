# Mind Games Challenge - Track 2 Submission

This repository contains our submission for the Mind Games Challenge Track 2 (Generalization Track).

## Requirements

- Python 3.13.7 (or Python 3.10+)
- TextArena, httpx, openai ... (see requirements.txt)
- We are using API keys for LLM providers, torch is not required.

## Installation

1. Clone the repository:
```bash
git clone --recursive https://github.com/lizi-Margin/Reflexion.git
cd Reflexion
git submodule update --init --recursive
```

2. Install dependencies:
- if you are using pip (recommended):
```bash
pip install -r requirements.txt
```
- if you are using mamba or micromamba:
```bash
mamba env create -f ./mamba_env.yml -n MindGameCorleone
```

## Running the Agent

### Single Entry Point (Competition Submission)

Use the main entry point script to run track2 small modle matches:

```bash
python run_competition.py --games <num_games>
```

**Arguments**:
- `--games`: Number of games to play per environment (default: 1)

**Example**:
```bash
python run_competition.py --games 3
```

### Offline Testing
#### Suggested Method

Test agents locally without connecting to the online arena:

```bash
python run_competition_test.py --games 100
```
This script will run self-play games locally.

#### Alternative Method
Edit `offline_play.py` to select the environment:
- `env_id = "Codenames-v0-train"; npc_num = 3` for Codenames (2v2)
- `env_id = "ColonelBlotto-v0-train"; npc_num = 1` for Colonel Blotto (1v1)
- `env_id = "ThreePlayerIPD-v0-train"; npc_num = 2` for Three Player IPD

```bash
python offline_play.py
```

## API Keys
All Agents use API keys for LLM providers (Already configured). If there are any network issues, please contact the team.

## Contact

For issues or questions, please contact the team sunhaocheng@bupt.edu.cn.
