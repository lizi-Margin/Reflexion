"""
Analyze IPD strategy pool
"""
import json


def analyze_pool():
    """Analyze strategy pool"""

    try:
        with open("strategy_pool_ipd.json", 'r', encoding='utf-8') as f:
            pool = json.load(f)
    except FileNotFoundError:
        print("Error: strategy_pool_ipd.json not found")
        return

    strategies = pool.get("strategies", [])
    metadata = pool.get("metadata", {})

    print(f"\n{'='*60}")
    print("IPD Strategy Pool Analysis")
    print(f"{'='*60}\n")

    print(f"Total strategies: {len(strategies)}")
    print(f"  - Conversation: {metadata.get('conversation_strategies', 0)}")
    print(f"  - Decision: {metadata.get('decision_strategies', 0)}")
    print(f"Last updated: {metadata.get('last_updated', 'Unknown')}\n")

    # Separate by type
    conv = [s for s in strategies if s["type"] == "conversation"]
    dec = [s for s in strategies if s["type"] == "decision"]

    # Top conversation strategies
    print(f"{'='*60}")
    print("Top Conversation Strategies (by performance)")
    print(f"{'='*60}\n")

    conv_sorted = sorted(conv, key=lambda x: x.get("performance", 0), reverse=True)
    for i, s in enumerate(conv_sorted[:5], 1):
        print(f"{i}. {s['id']}")
        print(f"   Performance: {s.get('performance', 0):.3f}")
        print(f"   Usage: {s.get('usage', 0)}")
        print(f"   Text: {s['text'][:80]}...")
        print()

    # Top decision strategies
    print(f"{'='*60}")
    print("Top Decision Strategies (by performance)")
    print(f"{'='*60}\n")

    dec_sorted = sorted(dec, key=lambda x: x.get("performance", 0), reverse=True)
    for i, s in enumerate(dec_sorted[:5], 1):
        print(f"{i}. {s['id']}")
        print(f"   Performance: {s.get('performance', 0):.3f}")
        print(f"   Usage: {s.get('usage', 0)}")
        print(f"   Text: {s['text'][:80]}...")
        print()

    # Most used
    print(f"{'='*60}")
    print("Most Used Strategies")
    print(f"{'='*60}\n")

    all_sorted = sorted(strategies, key=lambda x: x.get("usage", 0), reverse=True)
    for i, s in enumerate(all_sorted[:5], 1):
        print(f"{i}. {s['id']} ({s['type']})")
        print(f"   Usage: {s.get('usage', 0)}")
        print(f"   Performance: {s.get('performance', 0):.3f}")
        print()


if __name__ == "__main__":
    analyze_pool()
