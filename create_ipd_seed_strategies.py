"""
Create initial seed strategies for IPD agent using the global StrategyPoolManager format
"""
import json
from datetime import datetime


def create_seed_strategies(output_path="strategy_pool.json"):
    """Create seed strategy pool compatible with global StrategyPoolManager

    Args:
        output_path: Path to save the strategy pool (default: strategy_pool.json)
    """

    strategies = []

    # Conversation Strategies (formatted for global manager)
    strategies.extend([
        {
            "id": "ipd_conv_trust_builder",
            "game": "IPD",
            "strategy_type": "conversation",
            "type": "behavior",  # For global manager compatibility
            "role": "Player",
            "game_phase": "early_round",
            "strategy_text": "Build trust by promising cooperation and emphasizing mutual benefits. Example: 'I believe we can all benefit if we cooperate consistently.'",
            "context_features": {"phase": "early", "rank": 2, "gap": 0, "remaining": 7, "game": "IPD"},
            "usage_count": 0,
            "success_score": 0.0,
            "avg_performance": 0.5,
            "last_used": None,
            "created_at": datetime.now().isoformat()
        },
        {
            "id": "ipd_conv_warning",
            "game": "IPD",
            "strategy_type": "conversation",
            "type": "behavior",
            "role": "Player",
            "game_phase": "mid_round",
            "strategy_text": "Warn that defection will be punished. Example: 'I will cooperate, but if anyone defects against me, I will retaliate immediately.'",
            "context_features": {"phase": "mid", "rank": 1, "gap": 0, "remaining": 5, "game": "IPD"},
            "usage_count": 0,
            "success_score": 0.0,
            "avg_performance": 0.5,
            "last_used": None,
            "created_at": datetime.now().isoformat()
        },
        {
            "id": "ipd_conv_alliance",
            "game": "IPD",
            "strategy_type": "conversation",
            "type": "behavior",
            "role": "Player",
            "game_phase": "mid_round",
            "strategy_text": "Propose alliance against another player. Example: 'Player X keeps defecting. Let's cooperate with each other and defect against Player X.'",
            "context_features": {"phase": "mid", "rank": 2, "gap": -5, "remaining": 4, "game": "IPD"},
            "usage_count": 0,
            "success_score": 0.0,
            "avg_performance": 0.5,
            "last_used": None,
            "created_at": datetime.now().isoformat()
        },
        {
            "id": "ipd_conv_deceptive",
            "game": "IPD",
            "strategy_type": "conversation",
            "type": "behavior",
            "role": "Player",
            "game_phase": "late_round",
            "strategy_text": "Promise cooperation while planning to defect. Use when desperate. Example: 'I promise to cooperate with everyone.'",
            "context_features": {"phase": "late", "rank": 3, "gap": -10, "remaining": 1, "game": "IPD"},
            "usage_count": 0,
            "success_score": 0.0,
            "avg_performance": 0.5,
            "last_used": None,
            "created_at": datetime.now().isoformat()
        },
        {
            "id": "ipd_conv_neutral",
            "game": "IPD",
            "strategy_type": "conversation",
            "type": "behavior",
            "role": "Player",
            "game_phase": "early_round",
            "strategy_text": "Remain neutral and observe. Example: 'Let's see how this round plays out.'",
            "context_features": {"phase": "early", "rank": 1, "gap": 0, "remaining": 8, "game": "IPD"},
            "usage_count": 0,
            "success_score": 0.0,
            "avg_performance": 0.5,
            "last_used": None,
            "created_at": datetime.now().isoformat()
        }
    ])

    # Decision Strategies (formatted for global manager)
    strategies.extend([
        {
            "id": "ipd_dec_tit_for_tat",
            "game": "IPD",
            "strategy_type": "decision",
            "type": "behavior",
            "role": "Player",
            "game_phase": "mid_round",
            "strategy_text": "Mirror opponent's last action. If they cooperated, cooperate. If they defected, defect.",
            "context_features": {"phase": "mid", "rank": 2, "gap": 0, "remaining": 5, "game": "IPD"},
            "usage_count": 0,
            "success_score": 0.0,
            "avg_performance": 0.5,
            "last_used": None,
            "created_at": datetime.now().isoformat()
        },
        {
            "id": "ipd_dec_grim_trigger",
            "game": "IPD",
            "strategy_type": "decision",
            "type": "behavior",
            "role": "Player",
            "game_phase": "mid_round",
            "strategy_text": "Cooperate until opponent defects once, then defect forever against them.",
            "context_features": {"phase": "mid", "rank": 1, "gap": 0, "remaining": 4, "game": "IPD"},
            "usage_count": 0,
            "success_score": 0.0,
            "avg_performance": 0.5,
            "last_used": None,
            "created_at": datetime.now().isoformat()
        },
        {
            "id": "ipd_dec_always_cooperate",
            "game": "IPD",
            "strategy_type": "decision",
            "type": "behavior",
            "role": "Player",
            "game_phase": "early_round",
            "strategy_text": "Always cooperate to maximize collective benefit.",
            "context_features": {"phase": "early", "rank": 1, "gap": 2, "remaining": 8, "game": "IPD"},
            "usage_count": 0,
            "success_score": 0.0,
            "avg_performance": 0.5,
            "last_used": None,
            "created_at": datetime.now().isoformat()
        },
        {
            "id": "ipd_dec_pavlov",
            "game": "IPD",
            "strategy_type": "decision",
            "type": "behavior",
            "role": "Player",
            "game_phase": "mid_round",
            "strategy_text": "Win-stay, lose-shift: Repeat if you got R or T, switch if you got S or P.",
            "context_features": {"phase": "mid", "rank": 2, "gap": -2, "remaining": 5, "game": "IPD"},
            "usage_count": 0,
            "success_score": 0.0,
            "avg_performance": 0.5,
            "last_used": None,
            "created_at": datetime.now().isoformat()
        },
        {
            "id": "ipd_dec_desperate_defect",
            "game": "IPD",
            "strategy_type": "decision",
            "type": "behavior",
            "role": "Player",
            "game_phase": "late_round",
            "strategy_text": "Defect against all when far behind to maximize score gain.",
            "context_features": {"phase": "late", "rank": 3, "gap": -8, "remaining": 2, "game": "IPD"},
            "usage_count": 0,
            "success_score": 0.0,
            "avg_performance": 0.5,
            "last_used": None,
            "created_at": datetime.now().isoformat()
        },
        {
            "id": "ipd_dec_adaptive",
            "game": "IPD",
            "strategy_type": "decision",
            "type": "behavior",
            "role": "Player",
            "game_phase": "late_round",
            "strategy_text": "Adaptive: If leading, cooperate. If middle, use TFT. If last, defect more.",
            "context_features": {"phase": "late", "rank": 2, "gap": -3, "remaining": 2, "game": "IPD"},
            "usage_count": 0,
            "success_score": 0.0,
            "avg_performance": 0.5,
            "last_used": None,
            "created_at": datetime.now().isoformat()
        }
    ])

    pool = {
        "strategies": strategies,
        "metadata": {
            "version": "2.0",
            "total_strategies": len(strategies),
            "behavior_strategies": len(strategies),  # All IPD strategies are behavior type
            "language_strategies": 0,
            "last_updated": datetime.now().isoformat(),
            "training_sessions": 0,
            "games": ["IPD"]
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2, ensure_ascii=False)

    conv_count = sum(1 for s in strategies if s["strategy_type"] == "conversation")
    dec_count = sum(1 for s in strategies if s["strategy_type"] == "decision")

    print(f"✓ Created seed strategy pool at {output_path}")
    print(f"  - Total strategies: {len(strategies)}")
    print(f"  - Conversation: {conv_count}")
    print(f"  - Decision: {dec_count}")
    print(f"  - Format: Global StrategyPoolManager compatible")


if __name__ == "__main__":
    create_seed_strategies()
