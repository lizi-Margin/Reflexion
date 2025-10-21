"""
Strategy Evaluation and Analysis Tools
This module provides comprehensive evaluation and analysis capabilities for strategy pools.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import itertools
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from strategy_pool_manager import StrategyPoolManager, StrategyMetrics


class StrategyEvaluator:
    """
    Comprehensive strategy evaluation and analysis system.
    Provides deep insights into strategy performance, patterns, and optimization opportunities.
    """

    def __init__(self, strategy_pool_path: str = "strategy_pool.json"):
        self.strategy_pool_path = strategy_pool_path
        self.strategy_manager = StrategyPoolManager(strategy_pool_path)
        self.evaluation_results = {}

    def comprehensive_evaluation(self) -> Dict:
        """Perform comprehensive evaluation of the strategy pool"""
        print("Starting comprehensive strategy evaluation...")

        evaluation = {
            "timestamp": datetime.now().isoformat(),
            "pool_overview": self._evaluate_pool_overview(),
            "performance_analysis": self._analyze_performance_distribution(),
            "diversity_analysis": self._analyze_strategy_diversity(),
            "role_effectiveness": self._analyze_role_effectiveness(),
            "phase_effectiveness": self._analyze_phase_effectiveness(),
            "context_analysis": self._analyze_context_effectiveness(),
            "strategy_interactions": self._analyze_strategy_interactions(),
            "optimization_recommendations": self._generate_optimization_recommendations(),
            "quality_metrics": self._calculate_quality_metrics()
        }

        self.evaluation_results = evaluation
        print("Comprehensive evaluation completed.")
        return evaluation

    def _evaluate_pool_overview(self) -> Dict:
        """Evaluate basic pool overview metrics"""
        strategies = self.strategy_manager.strategies

        # Basic counts
        total_strategies = len(strategies)
        behavior_count = sum(1 for s in strategies.values() if s.get("type") == "behavior")
        language_count = sum(1 for s in strategies.values() if s.get("type") == "language")

        # Usage statistics
        usages = [s.get("usage_count", 0) for s in strategies.values()]
        total_usage = sum(usages)
        avg_usage = np.mean(usages) if usages else 0
        usage_std = np.std(usages) if usages else 0

        # Performance statistics
        performances = [s.get("avg_performance", 0.5) for s in strategies.values()]
        avg_performance = np.mean(performances) if performances else 0.5
        performance_std = np.std(performances) if performances else 0

        # Age analysis
        now = datetime.now()
        ages = []
        for s in strategies.values():
            created_str = s.get("created_at", "")
            if created_str:
                try:
                    created = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                    age_days = (now - created).days
                    ages.append(age_days)
                except:
                    ages.append(0)
            else:
                ages.append(0)

        avg_age = np.mean(ages) if ages else 0

        return {
            "total_strategies": total_strategies,
            "behavior_strategies": behavior_count,
            "language_strategies": language_count,
            "total_usage": total_usage,
            "average_usage_per_strategy": avg_usage,
            "usage_distribution_std": usage_std,
            "average_performance": avg_performance,
            "performance_distribution_std": performance_std,
            "average_strategy_age_days": avg_age,
            "unused_strategies": sum(1 for u in usages if u == 0),
            "high_usage_strategies": sum(1 for u in usages if u >= 10),
            "high_performance_strategies": sum(1 for p in performances if p >= 0.7)
        }

    def _analyze_performance_distribution(self) -> Dict:
        """Analyze performance distribution across strategies"""
        strategies = self.strategy_manager.strategies

        # Performance by type
        type_performance = defaultdict(list)
        for s in strategies.values():
            perf = s.get("avg_performance", 0.5)
            type_performance[s.get("type", "unknown")].append(perf)

        # Performance by role
        role_performance = defaultdict(list)
        for s in strategies.values():
            perf = s.get("avg_performance", 0.5)
            role_performance[s.get("role", "Unknown")].append(perf)

        # Performance by usage
        usage_bins = {
            "unused": [],
            "low": [],  # 1-5 uses
            "medium": [],  # 6-20 uses
            "high": []  # 20+ uses
        }

        for s in strategies.values():
            usage = s.get("usage_count", 0)
            perf = s.get("avg_performance", 0.5)

            if usage == 0:
                usage_bins["unused"].append(perf)
            elif usage <= 5:
                usage_bins["low"].append(perf)
            elif usage <= 20:
                usage_bins["medium"].append(perf)
            else:
                usage_bins["high"].append(perf)

        # Calculate statistics for each category
        def calculate_stats(values):
            if not values:
                return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0}
            return {
                "count": len(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }

        return {
            "by_type": {k: calculate_stats(v) for k, v in type_performance.items()},
            "by_role": {k: calculate_stats(v) for k, v in role_performance.items()},
            "by_usage": {k: calculate_stats(v) for k, v in usage_bins.items()},
            "correlation_usage_performance": self._calculate_usage_performance_correlation()
        }

    def _calculate_usage_performance_correlation(self) -> Dict:
        """Calculate correlation between usage and performance"""
        strategies = self.strategy_manager.strategies

        usages = []
        performances = []

        for s in strategies.values():
            usage = s.get("usage_count", 0)
            perf = s.get("avg_performance", 0.5)
            if usage > 0:  # Only include used strategies
                usages.append(usage)
                performances.append(perf)

        if len(usages) < 2:
            return {"correlation": 0, "p_value": 1, "sample_size": 0}

        correlation, p_value = stats.pearsonr(usages, performances)

        return {
            "correlation": correlation,
            "p_value": p_value,
            "sample_size": len(usages),
            "interpretation": self._interpret_correlation(correlation)
        }

    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient"""
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            return "negligible"
        elif abs_corr < 0.3:
            return "weak"
        elif abs_corr < 0.5:
            return "moderate"
        elif abs_corr < 0.7:
            return "strong"
        else:
            return "very strong"

    def _analyze_strategy_diversity(self) -> Dict:
        """Analyze diversity in the strategy pool"""
        strategies = self.strategy_manager.strategies

        # Type diversity
        type_counts = Counter(s.get("type", "unknown") for s in strategies.values())
        type_entropy = self._calculate_entropy(list(type_counts.values()))

        # Role diversity
        role_counts = Counter(s.get("role", "Unknown") for s in strategies.values())
        role_entropy = self._calculate_entropy(list(role_counts.values()))

        # Phase diversity
        phase_counts = Counter(s.get("game_phase", "unknown") for s in strategies.values())
        phase_entropy = self._calculate_entropy(list(phase_counts.values()))

        # Text similarity diversity (simplified)
        strategy_texts = [s.get("strategy_text", "") for s in strategies.values()]
        text_diversity = self._calculate_text_diversity(strategy_texts)

        return {
            "type_diversity": {
                "counts": dict(type_counts),
                "entropy": type_entropy,
                "dominant_type": type_counts.most_common(1)[0] if type_counts else None
            },
            "role_diversity": {
                "counts": dict(role_counts),
                "entropy": role_entropy,
                "dominant_role": role_counts.most_common(1)[0] if role_counts else None
            },
            "phase_diversity": {
                "counts": dict(phase_counts),
                "entropy": phase_entropy,
                "dominant_phase": phase_counts.most_common(1)[0] if phase_counts else None
            },
            "text_diversity": text_diversity,
            "overall_diversity_score": (type_entropy + role_entropy + phase_entropy) / 3
        }

    def _calculate_entropy(self, counts: List[int]) -> float:
        """Calculate Shannon entropy"""
        if not counts or sum(counts) == 0:
            return 0

        total = sum(counts)
        probabilities = [c / total for c in counts if c > 0]
        entropy = -sum(p * np.log2(p) for p in probabilities)
        return entropy

    def _calculate_text_diversity(self, texts: List[str]) -> Dict:
        """Calculate text diversity metrics"""
        if not texts:
            return {"avg_length": 0, "unique_words": 0, "vocabulary_richness": 0}

        # Basic text statistics
        lengths = [len(text.split()) for text in texts]
        avg_length = np.mean(lengths)

        # Vocabulary analysis
        all_words = " ".join(texts).lower().split()
        unique_words = len(set(all_words))
        total_words = len(all_words)
        vocabulary_richness = unique_words / total_words if total_words > 0 else 0

        return {
            "avg_length": avg_length,
            "unique_words": unique_words,
            "total_words": total_words,
            "vocabulary_richness": vocabulary_richness
        }

    def _analyze_role_effectiveness(self) -> Dict:
        """Analyze strategy effectiveness by role"""
        strategies = self.strategy_manager.strategies

        role_analysis = {}

        for role in set(s.get("role", "Unknown") for s in strategies.values()):
            role_strategies = [s for s in strategies.values() if s.get("role") == role]

            if not role_strategies:
                continue

            performances = [s.get("avg_performance", 0.5) for s in role_strategies]
            usages = [s.get("usage_count", 0) for s in role_strategies]

            # Find best and worst strategies for this role
            best_strategy = max(role_strategies, key=lambda s: s.get("avg_performance", 0.5))
            worst_strategy = min(role_strategies, key=lambda s: s.get("avg_performance", 0.5))

            role_analysis[role] = {
                "strategy_count": len(role_strategies),
                "avg_performance": np.mean(performances),
                "performance_std": np.std(performances),
                "total_usage": sum(usages),
                "best_strategy": {
                    "id": best_strategy.get("id"),
                    "performance": best_strategy.get("avg_performance", 0.5),
                    "usage": best_strategy.get("usage_count", 0)
                },
                "worst_strategy": {
                    "id": worst_strategy.get("id"),
                    "performance": worst_strategy.get("avg_performance", 0.5),
                    "usage": worst_strategy.get("usage_count", 0)
                },
                "coverage": len([s for s in role_strategies if s.get("usage_count", 0) > 0]) / len(role_strategies)
            }

        return role_analysis

    def _analyze_phase_effectiveness(self) -> Dict:
        """Analyze strategy effectiveness by game phase"""
        strategies = self.strategy_manager.strategies

        phase_analysis = {}

        for phase in set(s.get("game_phase", "unknown") for s in strategies.values()):
            phase_strategies = [s for s in strategies.values() if s.get("game_phase") == phase]

            if not phase_strategies:
                continue

            performances = [s.get("avg_performance", 0.5) for s in phase_strategies]
            usages = [s.get("usage_count", 0) for s in phase_strategies]

            # Type distribution within phase
            type_counts = Counter(s.get("type", "unknown") for s in phase_strategies)

            phase_analysis[phase] = {
                "strategy_count": len(phase_strategies),
                "avg_performance": np.mean(performances),
                "performance_std": np.std(performances),
                "total_usage": sum(usages),
                "type_distribution": dict(type_counts),
                "most_used_type": type_counts.most_common(1)[0] if type_counts else None
            }

        return phase_analysis

    def _analyze_context_effectiveness(self) -> Dict:
        """Analyze strategy effectiveness in different contexts"""
        strategies = self.strategy_manager.strategies

        context_analysis = {
            "urgency_analysis": {},
            "player_count_analysis": {}
        }

        # Analyze urgency levels
        urgency_performance = defaultdict(list)
        player_count_performance = defaultdict(list)

        for s in strategies.values():
            perf = s.get("avg_performance", 0.5)
            usage = s.get("usage_count", 0)

            if usage == 0:
                continue  # Skip unused strategies

            context_features = s.get("context_features", {})
            urgency_levels = context_features.get("urgency_levels", [])
            min_players = context_features.get("min_alive_players", 0)
            max_players = context_features.get("max_alive_players", 10)

            # Categorize by urgency
            if not urgency_levels:
                urgency_category = "general"
            else:
                urgency_category = "multi_urgency" if len(urgency_levels) > 1 else urgency_levels[0]

            urgency_performance[urgency_category].append(perf)

            # Categorize by player count range
            if min_players <= 3 and max_players >= 5:
                player_category = "small_to_medium"
            elif min_players <= 5 and max_players >= 8:
                player_category = "medium_to_large"
            elif max_players <= 5:
                player_category = "small_games"
            elif min_players >= 6:
                player_category = "large_games"
            else:
                player_category = "medium_games"

            player_count_performance[player_category].append(perf)

        # Calculate statistics
        for category, performances in urgency_performance.items():
            if performances:
                context_analysis["urgency_analysis"][category] = {
                    "count": len(performances),
                    "avg_performance": np.mean(performances),
                    "std": np.std(performances)
                }

        for category, performances in player_count_performance.items():
            if performances:
                context_analysis["player_count_analysis"][category] = {
                    "count": len(performances),
                    "avg_performance": np.mean(performances),
                    "std": np.std(performances)
                }

        return context_analysis

    def _analyze_strategy_interactions(self) -> Dict:
        """Analyze potential interactions between strategies"""
        strategies = self.strategy_manager.strategies

        # Language strategies and their behavior strategies
        behavior_language_pairs = []
        language_performance_by_behavior = defaultdict(list)

        for s in strategies.values():
            if s.get("type") == "language":
                behavior_id = s.get("related_behavior_id")
                if behavior_id and behavior_id in strategies:
                    behavior_strategy = strategies[behavior_id]
                    pair_performance = (s.get("avg_performance", 0.5) + behavior_strategy.get("avg_performance", 0.5)) / 2
                    behavior_language_pairs.append({
                        "behavior_id": behavior_id,
                        "language_id": s.get("id"),
                        "combined_performance": pair_performance,
                        "behavior_performance": behavior_strategy.get("avg_performance", 0.5),
                        "language_performance": s.get("avg_performance", 0.5)
                    })

                    language_performance_by_behavior[behavior_id].append(s.get("avg_performance", 0.5))

        # Find best behavior-language combinations
        if behavior_language_pairs:
            best_pairs = sorted(behavior_language_pairs, key=lambda x: x["combined_performance"], reverse=True)[:5]
            worst_pairs = sorted(behavior_language_pairs, key=lambda x: x["combined_performance"])[:5]
        else:
            best_pairs = []
            worst_pairs = []

        return {
            "behavior_language_pairs": {
                "total_pairs": len(behavior_language_pairs),
                "best_combinations": best_pairs,
                "worst_combinations": worst_pairs,
                "avg_combined_performance": np.mean([p["combined_performance"] for p in behavior_language_pairs]) if behavior_language_pairs else 0
            },
            "language_strategy_coverage": {
                "with_behavior": len([s for s in strategies.values() if s.get("type") == "language" and s.get("related_behavior_id")]),
                "without_behavior": len([s for s in strategies.values() if s.get("type") == "language" and not s.get("related_behavior_id")])
            }
        }

    def _generate_optimization_recommendations(self) -> List[Dict]:
        """Generate specific optimization recommendations"""
        recommendations = []
        strategies = self.strategy_manager.strategies

        # Check for underperforming strategies
        underperforming = [s for s in strategies.values()
                          if s.get("usage_count", 0) >= 5 and s.get("avg_performance", 0.5) < 0.3]

        if underperforming:
            recommendations.append({
                "type": "remove_underperforming",
                "priority": "high",
                "description": f"Remove {len(underperforming)} consistently underperforming strategies",
                "affected_strategies": [s.get("id") for s in underperforming],
                "expected_impact": "Improved pool quality and faster convergence"
            })

        # Check for unused strategies
        unused = [s for s in strategies.values() if s.get("usage_count", 0) == 0]
        if len(unused) > len(strategies) * 0.2:  # More than 20% unused
            recommendations.append({
                "type": "remove_unused",
                "priority": "medium",
                "description": f"Consider removing {len(unused)} unused strategies",
                "affected_strategies": [s.get("id") for s in unused],
                "expected_impact": "Cleaner strategy pool and better focus"
            })

        # Check for role balance
        role_counts = Counter(s.get("role", "Unknown") for s in strategies.values())
        if len(role_counts) < 3:  # Poor role coverage
            recommendations.append({
                "type": "improve_role_diversity",
                "priority": "high",
                "description": "Add strategies for underrepresented roles",
                "missing_roles": [role for role in ["Mafia", "Detective", "Doctor", "A regular villager"]
                                if role not in role_counts],
                "expected_impact": "Better coverage of all game scenarios"
            })

        # Check for type balance
        type_counts = Counter(s.get("type", "unknown") for s in strategies.values())
        behavior_count = type_counts.get("behavior", 0)
        language_count = type_counts.get("language", 0)

        if behavior_count == 0 or language_count == 0:
            recommendations.append({
                "type": "balance_strategy_types",
                "priority": "high",
                "description": f"Missing {'behavior' if behavior_count == 0 else 'language'} strategies",
                "current_balance": dict(type_counts),
                "expected_impact": "More comprehensive strategy coverage"
            })

        # Check for overused strategies
        max_usage = max([s.get("usage_count", 0) for s in strategies.values()], default=0)
        avg_usage = np.mean([s.get("usage_count", 0) for s in strategies.values()])
        if max_usage > avg_usage * 5:  # Some strategies are overused
            overused = [s for s in strategies.values() if s.get("usage_count", 0) > avg_usage * 5]
            recommendations.append({
                "type": "reduce_overuse",
                "priority": "medium",
                "description": f"Reduce over-reliance on {len(overused)} strategies",
                "affected_strategies": [s.get("id") for s in overused],
                "expected_impact": "Better exploration and strategy diversity"
            })

        # Performance improvement suggestions
        high_performers = [s for s in strategies.values()
                          if s.get("avg_performance", 0.5) > 0.8 and s.get("usage_count", 0) >= 3]

        if high_performers:
            recommendations.append({
                "type": "create_variants",
                "priority": "medium",
                "description": f"Create variants of {len(high_performers)} high-performing strategies",
                "base_strategies": [s.get("id") for s in high_performers],
                "expected_impact": "Leverage successful patterns"
            })

        return recommendations

    def _calculate_quality_metrics(self) -> Dict:
        """Calculate overall quality metrics for the strategy pool"""
        strategies = self.strategy_manager.strategies

        if not strategies:
            return {"overall_score": 0, "quality_grade": "F"}

        # Component scores
        usage_balance_score = self._calculate_usage_balance_score()
        performance_score = np.mean([s.get("avg_performance", 0.5) for s in strategies.values()])
        diversity_score = self._calculate_diversity_score()
        coverage_score = self._calculate_coverage_score()

        # Weighted overall score
        weights = {
            "usage_balance": 0.2,
            "performance": 0.4,
            "diversity": 0.2,
            "coverage": 0.2
        }

        overall_score = (
            usage_balance_score * weights["usage_balance"] +
            performance_score * weights["performance"] +
            diversity_score * weights["diversity"] +
            coverage_score * weights["coverage"]
        )

        # Quality grade
        if overall_score >= 0.9:
            grade = "A+"
        elif overall_score >= 0.8:
            grade = "A"
        elif overall_score >= 0.7:
            grade = "B"
        elif overall_score >= 0.6:
            grade = "C"
        elif overall_score >= 0.5:
            grade = "D"
        else:
            grade = "F"

        return {
            "overall_score": overall_score,
            "quality_grade": grade,
            "component_scores": {
                "usage_balance": usage_balance_score,
                "performance": performance_score,
                "diversity": diversity_score,
                "coverage": coverage_score
            },
            "weights": weights
        }

    def _calculate_usage_balance_score(self) -> float:
        """Calculate how balanced the usage is across strategies"""
        strategies = self.strategy_manager.strategies
        usages = [s.get("usage_count", 0) for s in strategies.values()]

        if not usages or max(usages) == 0:
            return 0.5  # Neutral score for no usage

        # Calculate coefficient of variation (lower is more balanced)
        mean_usage = np.mean(usages)
        if mean_usage == 0:
            return 0.5

        cv = np.std(usages) / mean_usage
        # Convert CV to score (lower CV = higher score)
        balance_score = max(0, 1 - cv / 2)  # Normalize to 0-1 range
        return balance_score

    def _calculate_diversity_score(self) -> float:
        """Calculate diversity score across different dimensions"""
        diversity_analysis = self._analyze_strategy_diversity()

        # Normalize entropy scores (max possible entropy depends on number of categories)
        type_score = min(1, diversity_analysis["type_diversity"]["entropy"] / 2)
        role_score = min(1, diversity_analysis["role_diversity"]["entropy"] / 2.5)
        phase_score = min(1, diversity_analysis["phase_diversity"]["entropy"] / 1.5)
        text_score = min(1, diversity_analysis["text_diversity"]["vocabulary_richness"] * 10)

        return (type_score + role_score + phase_score + text_score) / 4

    def _calculate_coverage_score(self) -> float:
        """Calculate how well the strategy pool covers different scenarios"""
        strategies = self.strategy_manager.strategies

        # Check role coverage
        all_roles = {"Mafia", "Detective", "Doctor", "A regular villager"}
        covered_roles = set(s.get("role", "Unknown") for s in strategies.values())
        role_coverage = len(covered_roles & all_roles) / len(all_roles)

        # Check type coverage
        all_types = {"behavior", "language"}
        covered_types = set(s.get("type", "unknown") for s in strategies.values())
        type_coverage = len(covered_types & all_types) / len(all_types)

        # Check phase coverage
        all_phases = {"night", "day_speak", "day_vote"}
        covered_phases = set(s.get("game_phase", "unknown") for s in strategies.values())
        phase_coverage = len(covered_phases & all_phases) / len(all_phases)

        # Check usage coverage (percentage of strategies that have been used)
        used_strategies = len([s for s in strategies.values() if s.get("usage_count", 0) > 0])
        usage_coverage = used_strategies / len(strategies) if strategies else 0

        return (role_coverage + type_coverage + phase_coverage + usage_coverage) / 4

    def create_evaluation_dashboard(self, save_path: str = "strategy_evaluation_dashboard.png"):
        """Create comprehensive visualization dashboard"""
        if not self.evaluation_results:
            print("No evaluation results available. Run comprehensive_evaluation() first.")
            return

        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. Pool Overview (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        overview = self.evaluation_results["pool_overview"]
        ax1.pie([overview["behavior_strategies"], overview["language_strategies"]],
               labels=["Behavior", "Language"], autopct='%1.1f%%')
        ax1.set_title("Strategy Type Distribution")

        # 2. Performance Distribution (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        perf_analysis = self.evaluation_results["performance_analysis"]
        all_performances = []
        for role_data in perf_analysis["by_role"].values():
            all_performances.extend([role_data["mean"]] * role_data["count"])

        ax2.hist(all_performances, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title("Performance Distribution")
        ax2.set_xlabel("Average Performance")
        ax2.set_ylabel("Frequency")

        # 3. Usage vs Performance Scatter (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        strategies = self.strategy_manager.strategies
        usages = [s.get("usage_count", 0) for s in strategies.values()]
        performances = [s.get("avg_performance", 0.5) for s in strategies.values()]

        scatter = ax3.scatter(usages, performances, alpha=0.6, c=performances, cmap='RdYlGn')
        ax3.set_xlabel("Usage Count")
        ax3.set_ylabel("Average Performance")
        ax3.set_title("Usage vs Performance")
        plt.colorbar(scatter, ax=ax3)

        # 4. Role Effectiveness (top far right)
        ax4 = fig.add_subplot(gs[0, 3])
        role_effectiveness = self.evaluation_results["role_effectiveness"]
        if role_effectiveness:
            roles = list(role_effectiveness.keys())
            avg_perfs = [role_effectiveness[role]["avg_performance"] for role in roles]
            bars = ax4.bar(range(len(roles)), avg_perfs, color=['red', 'blue', 'green', 'orange'][:len(roles)])
            ax4.set_xticks(range(len(roles)))
            ax4.set_xticklabels(roles, rotation=45, ha='right')
            ax4.set_title("Performance by Role")
            ax4.set_ylabel("Average Performance")

        # 5. Phase Effectiveness (middle left)
        ax5 = fig.add_subplot(gs[1, 0])
        phase_analysis = self.evaluation_results["phase_effectiveness"]
        if phase_analysis:
            phases = list(phase_analysis.keys())
            phase_perfs = [phase_analysis[phase]["avg_performance"] for phase in phases]
            ax5.bar(range(len(phases)), phase_perfs, color='lightcoral')
            ax5.set_xticks(range(len(phases)))
            ax5.set_xticklabels(phases, rotation=45, ha='right')
            ax5.set_title("Performance by Phase")
            ax5.set_ylabel("Average Performance")

        # 6. Quality Metrics (middle middle)
        ax6 = fig.add_subplot(gs[1, 1])
        quality_metrics = self.evaluation_results["quality_metrics"]["component_scores"]
        components = list(quality_metrics.keys())
        scores = list(quality_metrics.values())

        colors = ['gold' if score >= 0.8 else 'lightgray' if score >= 0.6 else 'lightcoral' for score in scores]
        ax6.bar(components, scores, color=colors)
        ax6.set_title("Quality Component Scores")
        ax6.set_ylabel("Score")
        ax6.set_ylim(0, 1)
        plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')

        # 7. Overall Quality Score (middle right)
        ax7 = fig.add_subplot(gs[1, 2])
        overall_score = self.evaluation_results["quality_metrics"]["overall_score"]
        grade = self.evaluation_results["quality_metrics"]["quality_grade"]

        # Create gauge chart
        theta = np.linspace(0, np.pi, 100)
        r = 1
        ax7.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
        ax7.plot([-1, 1], [0, 0], 'k-', linewidth=1)

        # Draw needle
        needle_angle = np.pi * (1 - overall_score)
        ax7.plot([0, 0.8 * np.cos(needle_angle)], [0, 0.8 * np.sin(needle_angle)], 'r-', linewidth=3)
        ax7.set_xlim(-1.2, 1.2)
        ax7.set_ylim(-0.2, 1.2)
        ax7.set_aspect('equal')
        ax7.axis('off')
        ax7.set_title(f"Overall Quality: {grade}\nScore: {overall_score:.3f}")

        # 8. Strategy Age Distribution (middle far right)
        ax8 = fig.add_subplot(gs[1, 3])
        ages = []
        for s in strategies.values():
            created_str = s.get("created_at", "")
            if created_str:
                try:
                    created = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                    age_days = (datetime.now() - created).days
                    ages.append(age_days)
                except:
                    pass

        if ages:
            ax8.hist(ages, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            ax8.set_title("Strategy Age Distribution")
            ax8.set_xlabel("Age (days)")
            ax8.set_ylabel("Number of Strategies")

        # 9. Recommendations Summary (bottom panels)
        ax9 = fig.add_subplot(gs[2:, :])
        ax9.axis('off')

        recommendations = self.evaluation_results.get("optimization_recommendations", [])
        if recommendations:
            rec_text = "OPTIMIZATION RECOMMENDATIONS:\n\n"
            for i, rec in enumerate(recommendations[:10], 1):  # Show top 10
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec["priority"], "⚪")
                rec_text += f"{i}. {priority_emoji} {rec['description']} (Priority: {rec['priority']})\n"
                rec_text += f"   Expected Impact: {rec['expected_impact']}\n\n"

            ax9.text(0.05, 0.95, rec_text, transform=ax9.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        else:
            ax9.text(0.5, 0.5, "No optimization recommendations needed!",
                    transform=ax9.transAxes, fontsize=14, ha='center', va='center')

        plt.suptitle("Strategy Pool Evaluation Dashboard", fontsize=16, fontweight='bold')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Evaluation dashboard saved to: {save_path}")

    def save_evaluation_report(self, save_path: str = None):
        """Save comprehensive evaluation report to file"""
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"strategy_evaluation_report_{timestamp}.json"

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False, default=str)

        print(f"Evaluation report saved to: {save_path}")
        return save_path


def main():
    """Main function to run strategy evaluation"""
    import argparse

    parser = argparse.ArgumentParser(description="Strategy Pool Evaluation")
    parser.add_argument("--strategy-pool", type=str, default="strategy_pool.json", help="Strategy pool file")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--create-dashboard", action="store_true", help="Create visual dashboard")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize evaluator
    evaluator = StrategyEvaluator(args.strategy_pool)

    # Run comprehensive evaluation
    print("Starting comprehensive strategy evaluation...")
    evaluation_results = evaluator.comprehensive_evaluation()

    # Save report
    report_path = os.path.join(args.output_dir, "comprehensive_evaluation.json")
    evaluator.save_evaluation_report(report_path)

    # Create dashboard if requested
    if args.create_dashboard:
        dashboard_path = os.path.join(args.output_dir, "evaluation_dashboard.png")
        evaluator.create_evaluation_dashboard(dashboard_path)

    # Print summary
    quality_metrics = evaluation_results["quality_metrics"]
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Overall Quality Score: {quality_metrics['overall_score']:.3f}")
    print(f"Quality Grade: {quality_metrics['quality_grade']}")
    print(f"Total Strategies: {evaluation_results['pool_overview']['total_strategies']}")
    print(f"Average Performance: {evaluation_results['pool_overview']['average_performance']:.3f}")
    print(f"Optimization Recommendations: {len(evaluation_results['optimization_recommendations'])}")

    print("\nComponent Scores:")
    for component, score in quality_metrics['component_scores'].items():
        print(f"  {component}: {score:.3f}")

    if evaluation_results['optimization_recommendations']:
        print("\nTop Recommendations:")
        for rec in evaluation_results['optimization_recommendations'][:3]:
            print(f"  - {rec['description']} (Priority: {rec['priority']})")


if __name__ == "__main__":
    main()