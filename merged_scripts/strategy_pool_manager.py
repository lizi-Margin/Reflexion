"""
Strategy Pool Manager for managing and optimizing strategies in self-play training.
This module handles strategy evaluation, selection, and optimization based on performance data.
"""

import json
import os
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import math


@dataclass
class StrategyMetrics:
    """Metrics for tracking strategy performance"""
    strategy_id: str
    win_rate: float
    usage_count: int
    avg_performance: float
    success_score: float
    role_performance: Dict[str, float]  # Performance by role
    phase_performance: Dict[str, float]  # Performance by game phase
    context_success: Dict[str, float]  # Success in different contexts
    last_updated: str


@dataclass
class TrainingSession:
    """Training session data"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    games_played: int
    strategies_used: List[str]
    results: List[Dict]  # Game results
    performance_improvement: float


class StrategyPoolManager:
    """
    Advanced strategy pool manager for self-play training.
    Handles strategy optimization, selection, and performance tracking.
    """

    def __init__(self, strategy_pool_path: str = "strategy_pool.json"):
        self.strategy_pool_path = strategy_pool_path
        self.strategies = {}
        self.metrics = {}
        self.training_history = []
        self.current_session = None

        # Optimization parameters
        self.exploration_rate = 0.2  # Rate of trying new/underperforming strategies
        self.performance_window = 20  # Number of recent games to consider for performance
        self.min_usage_threshold = 5  # Minimum usage before strategy optimization

        # Load existing strategy pool
        self.load_strategy_pool()

    def load_strategy_pool(self):
        """Load strategy pool from JSON file"""
        try:
            if os.path.exists(self.strategy_pool_path):
                with open(self.strategy_pool_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.strategies = {s["id"]: s for s in data.get("strategies", [])}
                print(f"Loaded {len(self.strategies)} strategies from pool")
            else:
                print("No existing strategy pool found, creating empty pool")
                self.strategies = {}
        except Exception as e:
            print(f"Error loading strategy pool: {e}")
            self.strategies = {}

    def save_strategy_pool(self):
        """Save strategy pool to JSON file"""
        try:
            # Update metadata
            metadata = {
                "version": "2.0",
                "total_strategies": len(self.strategies),
                "behavior_strategies": sum(1 for s in self.strategies.values() if s["type"] == "behavior"),
                "language_strategies": sum(1 for s in self.strategies.values() if s["type"] == "language"),
                "last_updated": datetime.now().isoformat(),
                "training_sessions": len(self.training_history)
            }

            data = {
                "strategies": list(self.strategies.values()),
                "metadata": metadata,
                "training_history": [asdict(session) for session in self.training_history[-10:]]  # Last 10 sessions
            }

            with open(self.strategy_pool_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(self.strategies)} strategies to pool")
        except Exception as e:
            print(f"Error saving strategy pool: {e}")

    def calculate_strategy_metrics(self, strategy_id: str) -> StrategyMetrics:
        """Calculate comprehensive metrics for a strategy"""
        if strategy_id not in self.strategies:
            return None

        strategy = self.strategies[strategy_id]

        # Calculate win rate based on recent performance
        recent_performance = self._get_recent_performance(strategy_id)
        win_rate = np.mean(recent_performance) if recent_performance else strategy.get("avg_performance", 0.5)

        # Calculate role and phase performance
        role_performance = self._calculate_role_performance(strategy_id)
        phase_performance = self._calculate_phase_performance(strategy_id)
        context_success = self._calculate_context_performance(strategy_id)

        return StrategyMetrics(
            strategy_id=strategy_id,
            win_rate=win_rate,
            usage_count=strategy.get("usage_count", 0),
            avg_performance=strategy.get("avg_performance", 0.5),
            success_score=strategy.get("success_score", 0.0),
            role_performance=role_performance,
            phase_performance=phase_performance,
            context_success=context_success,
            last_updated=datetime.now().isoformat()
        )

    def _get_recent_performance(self, strategy_id: str) -> List[float]:
        """Get recent performance scores for a strategy"""
        # This would be populated from training logs
        # For now, return empty list - will be implemented in training script
        return []

    def _calculate_role_performance(self, strategy_id: str) -> Dict[str, float]:
        """Calculate performance by role"""
        strategy = self.strategies[strategy_id]
        role = strategy.get("role", "Unknown")

        # For now, return based on strategy's average performance
        # In full implementation, this would track actual role-specific performance
        return {
            role: strategy.get("avg_performance", 0.5),
            "overall": strategy.get("avg_performance", 0.5)
        }

    def _calculate_phase_performance(self, strategy_id: str) -> Dict[str, float]:
        """Calculate performance by game phase"""
        strategy = self.strategies[strategy_id]
        phase = strategy.get("game_phase", "day_speak")

        return {
            phase: strategy.get("avg_performance", 0.5),
            "overall": strategy.get("avg_performance", 0.5)
        }

    def _calculate_context_performance(self, strategy_id: str) -> Dict[str, float]:
        """Calculate performance in different contexts"""
        strategy = self.strategies[strategy_id]
        context_features = strategy.get("context_features", {})

        # Simple heuristic based on context features
        base_performance = strategy.get("avg_performance", 0.5)

        return {
            "low_urgency": base_performance * 1.0,
            "medium_urgency": base_performance * 1.1,
            "high_urgency": base_performance * 0.9,
            "few_players": base_performance * 0.95,
            "many_players": base_performance * 1.05
        }

    def select_strategies_for_training(self, num_strategies: int = 10) -> List[str]:
        """
        Select strategies for self-play training using intelligent selection.
        Balances exploration and exploitation.
        """
        if len(self.strategies) <= num_strategies:
            return list(self.strategies.keys())

        strategy_scores = []

        for strategy_id, strategy in self.strategies.items():
            score = self._calculate_selection_score(strategy_id)
            strategy_scores.append((strategy_id, score))

        # Sort by selection score
        strategy_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top strategies with some randomness for exploration
        selected = []
        remaining_strategies = [sid for sid, _ in strategy_scores[num_strategies:]]

        # Add top performers
        for strategy_id, _ in strategy_scores[:int(num_strategies * 0.7)]:
            selected.append(strategy_id)

        # Add some random strategies for exploration
        if remaining_strategies:
            exploration_count = num_strategies - len(selected)
            exploration_strategies = random.sample(
                remaining_strategies,
                min(exploration_count, len(remaining_strategies))
            )
            selected.extend(exploration_strategies)

        return selected[:num_strategies]

    def _calculate_selection_score(self, strategy_id: str) -> float:
        """Calculate selection score for a strategy"""
        if strategy_id not in self.strategies:
            return 0.0

        strategy = self.strategies[strategy_id]

        # Base score from average performance
        base_score = strategy.get("avg_performance", 0.5)

        # Usage bonus (encourage balanced usage)
        usage_count = strategy.get("usage_count", 0)
        usage_bonus = math.exp(-usage_count / 10.0) * 0.2

        # Recency bonus (encourage recently successful strategies)
        last_used = strategy.get("last_used")
        if last_used:
            last_used_time = datetime.fromisoformat(last_used.replace('Z', '+00:00'))
            days_since_used = (datetime.now() - last_used_time).days
            recency_bonus = max(0, 0.1 - days_since_used * 0.01)
        else:
            recency_bonus = 0.1  # Bonus for unused strategies

        # Diversity bonus (encourage underrepresented strategy types/roles)
        diversity_bonus = self._calculate_diversity_bonus(strategy_id)

        total_score = base_score + usage_bonus + recency_bonus + diversity_bonus
        return min(1.0, total_score)

    def _calculate_diversity_bonus(self, strategy_id: str) -> float:
        """Calculate diversity bonus for underrepresented strategies"""
        strategy = self.strategies[strategy_id]
        strategy_type = strategy.get("type", "behavior")
        role = strategy.get("role", "Unknown")

        # Count strategies of each type and role
        type_counts = defaultdict(int)
        role_counts = defaultdict(int)

        for s in self.strategies.values():
            type_counts[s.get("type", "behavior")] += 1
            role_counts[s.get("role", "Unknown")] += 1

        # Give bonus to underrepresented types and roles
        total_strategies = len(self.strategies)
        type_ratio = type_counts[strategy_type] / total_strategies
        role_ratio = role_counts[role] / total_strategies

        diversity_bonus = (1.0 - type_ratio) * 0.1 + (1.0 - role_ratio) * 0.1
        return diversity_bonus

    def optimize_strategy_pool(self, training_results: List[Dict]):
        """
        Optimize strategy pool based on training results.
        This includes:
        1. Updating performance metrics
        2. Removing underperforming strategies
        3. Creating strategy variants
        4. Balancing strategy diversity
        """
        print("Optimizing strategy pool based on training results...")

        # Update strategy metrics
        self._update_strategy_metrics(training_results)

        # Remove consistently underperforming strategies
        self._prune_underperforming_strategies()

        # Create variants of successful strategies
        self._create_strategy_variants()

        # Balance strategy pool diversity
        self._balance_strategy_diversity()

        # Save optimized pool
        self.save_strategy_pool()
        print("Strategy pool optimization completed.")

    def _update_strategy_metrics(self, training_results: List[Dict]):
        """Update strategy metrics based on training results"""
        strategy_updates = defaultdict(list)

        for result in training_results:
            strategies_used = result.get("strategies_used", [])
            game_won = result.get("victory", False)
            role = result.get("role", "Unknown")

            for strategy_id in strategies_used:
                score = 1.0 if game_won else -0.5
                if role in ["Detective", "Doctor"]:
                    score += 0.1 if game_won else -0.05  # Bonus for special roles

                strategy_updates[strategy_id].append(score)

        # Update strategy performance
        for strategy_id, scores in strategy_updates.items():
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]
                strategy["success_score"] += sum(scores)
                strategy["usage_count"] += len(scores)
                strategy["avg_performance"] = strategy["success_score"] / strategy["usage_count"]
                strategy["last_used"] = datetime.now().isoformat()

    def _prune_underperforming_strategies(self):
        """Remove consistently underperforming strategies"""
        strategies_to_remove = []

        for strategy_id, strategy in self.strategies.items():
            usage_count = strategy.get("usage_count", 0)
            avg_performance = strategy.get("avg_performance", 0.5)

            # Remove strategies with low performance and sufficient usage
            if usage_count >= self.min_usage_threshold and avg_performance < 0.3:
                strategies_to_remove.append(strategy_id)

        for strategy_id in strategies_to_remove:
            del self.strategies[strategy_id]
            print(f"Removed underperforming strategy: {strategy_id}")

    def _create_strategy_variants(self):
        """Create variants of successful strategies"""
        successful_strategies = [
            (sid, s) for sid, s in self.strategies.items()
            if s.get("avg_performance", 0.5) > 0.7 and s.get("usage_count", 0) >= 3
        ]

        for strategy_id, strategy in successful_strategies[:3]:  # Limit variants creation
            if random.random() < 0.3:  # 30% chance to create variant
                variant_id = self._create_strategy_variant(strategy)
                if variant_id:
                    print(f"Created strategy variant: {variant_id}")

    def _create_strategy_variant(self, parent_strategy: Dict) -> Optional[str]:
        """Create a variant of a successful strategy"""
        import uuid

        base_text = parent_strategy.get("strategy_text", "")

        # Simple variation: modify the strategy text slightly
        variations = [
            "Enhanced: " + base_text,
            "Adaptive: " + base_text,
            "Optimized: " + base_text,
        ]

        variant_text = random.choice(variations)

        new_strategy = {
            "id": f"{parent_strategy['type']}_{uuid.uuid4().hex[:8]}",
            "type": parent_strategy["type"],
            "role": parent_strategy["role"],
            "game_phase": parent_strategy["game_phase"],
            "strategy_text": variant_text,
            "context_features": parent_strategy.get("context_features", {}),
            "usage_count": 0,
            "success_score": 0.0,
            "avg_performance": 0.6,  # Slightly optimistic initial performance
            "related_behavior_id": parent_strategy.get("related_behavior_id"),
            "parent_strategy": parent_strategy["id"],
            "last_used": None,
            "created_at": datetime.now().isoformat()
        }

        self.strategies[new_strategy["id"]] = new_strategy
        return new_strategy["id"]

    def _balance_strategy_diversity(self):
        """Balance strategy pool to ensure good diversity"""
        # Count strategies by type and role
        type_counts = defaultdict(int)
        role_counts = defaultdict(int)

        for strategy in self.strategies.values():
            type_counts[strategy["type"]] += 1
            role_counts[strategy["role"]] += 1

        # Ensure minimum diversity
        min_per_type = 3
        min_per_role = 2

        # Add missing strategy types/roles if needed
        self._ensure_minimum_diversity(type_counts, role_counts, min_per_type, min_per_role)

    def _ensure_minimum_diversity(self, type_counts: Dict, role_counts: Dict,
                                 min_per_type: int, min_per_role: int):
        """Ensure minimum diversity in strategy pool"""
        # This would create basic strategies for missing types/roles
        # Implementation depends on specific game requirements
        pass

    def get_training_report(self) -> Dict:
        """Generate comprehensive training report"""
        if not self.strategies:
            return {"error": "No strategies in pool"}

        # Calculate overall metrics
        total_usage = sum(s.get("usage_count", 0) for s in self.strategies.values())
        avg_performance = np.mean([s.get("avg_performance", 0.5) for s in self.strategies.values()])

        # Best and worst strategies
        best_strategy = max(self.strategies.items(),
                          key=lambda x: x[1].get("avg_performance", 0.5))
        worst_strategy = min(self.strategies.items(),
                           key=lambda x: x[1].get("avg_performance", 0.5))

        # Strategy distribution
        type_distribution = defaultdict(int)
        role_distribution = defaultdict(int)

        for strategy in self.strategies.values():
            type_distribution[strategy.get("type", "unknown")] += 1
            role_distribution[strategy.get("role", "unknown")] += 1

        return {
            "total_strategies": len(self.strategies),
            "total_usage": total_usage,
            "average_performance": avg_performance,
            "best_strategy": {
                "id": best_strategy[0],
                "performance": best_strategy[1].get("avg_performance", 0.5),
                "usage": best_strategy[1].get("usage_count", 0)
            },
            "worst_strategy": {
                "id": worst_strategy[0],
                "performance": worst_strategy[1].get("avg_performance", 0.5),
                "usage": worst_strategy[1].get("usage_count", 0)
            },
            "type_distribution": dict(type_distribution),
            "role_distribution": dict(role_distribution),
            "training_sessions": len(self.training_history),
            "last_updated": datetime.now().isoformat()
        }

    def visualize_strategy_performance(self, save_path: str = "strategy_performance.png"):
        """Create visualization of strategy performance"""
        if not self.strategies:
            print("No strategies to visualize")
            return

        # Prepare data
        strategy_names = []
        performances = []
        usages = []
        types = []

        for strategy_id, strategy in self.strategies.items():
            strategy_names.append(strategy_id[:12] + "...")  # Truncate long names
            performances.append(strategy.get("avg_performance", 0.5))
            usages.append(strategy.get("usage_count", 0))
            types.append(strategy.get("type", "unknown"))

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Performance distribution
        ax1.hist(performances, bins=10, alpha=0.7, color='skyblue')
        ax1.set_title('Strategy Performance Distribution')
        ax1.set_xlabel('Average Performance')
        ax1.set_ylabel('Number of Strategies')

        # Usage vs Performance scatter
        colors = {'behavior': 'blue', 'language': 'red', 'unknown': 'gray'}
        scatter_colors = [colors.get(t, 'gray') for t in types]

        ax2.scatter(usages, performances, c=scatter_colors, alpha=0.6)
        ax2.set_xlabel('Usage Count')
        ax2.set_ylabel('Average Performance')
        ax2.set_title('Usage vs Performance')

        # Strategy type distribution
        type_counts = defaultdict(int)
        for t in types:
            type_counts[t] += 1

        ax3.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        ax3.set_title('Strategy Type Distribution')

        # Top performing strategies
        sorted_strategies = sorted(zip(strategy_names, performances), key=lambda x: x[1], reverse=True)[:10]
        top_names, top_performances = zip(*sorted_strategies)

        ax4.barh(top_names, top_performances, color='lightgreen')
        ax4.set_xlabel('Average Performance')
        ax4.set_title('Top 10 Performing Strategies')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Strategy performance visualization saved to {save_path}")