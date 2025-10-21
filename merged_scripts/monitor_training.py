"""
Real-time Training Monitoring Dashboard
This script provides a web-based dashboard for monitoring self-play training progress.
"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    print("Streamlit not available. Install with: pip install streamlit plotly")
    STREAMLIT_AVAILABLE = False

from src.strategy_pool_manager import StrategyPoolManager
from src.strategy_evaluator import StrategyEvaluator


class TrainingMonitor:
    """
    Real-time training monitoring system with web dashboard.
    """

    def __init__(self, strategy_pool_path: str = "strategy_pool.json", log_dir: str = "self_play_logs"):
        self.strategy_pool_path = strategy_pool_path
        self.log_dir = log_dir
        self.strategy_manager = StrategyPoolManager(strategy_pool_path)
        self.evaluator = StrategyEvaluator(strategy_pool_path)

        # Monitoring state
        self.last_update = datetime.now()
        self.update_interval = 30  # seconds
        self.auto_refresh = True

    def load_training_data(self) -> Dict:
        """Load current training data and statistics"""
        # Load strategy pool
        self.strategy_manager.load_strategy_pool()

        # Load training logs if available
        training_logs = []
        log_files = []

        if os.path.exists(self.log_dir):
            log_files = [f for f in os.listdir(self.log_dir) if f.endswith('.json')]

        for log_file in log_files[:10]:  # Load last 10 log files
            try:
                with open(os.path.join(self.log_dir, log_file), 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    training_logs.append(log_data)
            except Exception as e:
                print(f"Error loading log file {log_file}: {e}")

        # Extract training statistics
        total_games = sum(log.get('games_played', 0) for log in training_logs)
        total_strategies = len(self.strategy_manager.strategies)

        return {
            "strategies": self.strategy_manager.strategies,
            "training_logs": training_logs,
            "total_games": total_games,
            "total_strategies": total_strategies,
            "last_updated": datetime.now().isoformat()
        }

    def create_streamlit_dashboard(self):
        """Create Streamlit dashboard for training monitoring"""
        if not STREAMLIT_AVAILABLE:
            print("Streamlit not available. Cannot create dashboard.")
            return

        st.set_page_config(
            page_title="Strategy Pool Training Monitor",
            page_icon="🎮",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🎮 Strategy Pool Training Monitor")
        st.markdown("---")

        # Sidebar configuration
        st.sidebar.header("Configuration")
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 120, 30)

        # Load data
        if auto_refresh:
            # Auto-refresh placeholder
            placeholder = st.empty()

            while auto_refresh:
                with placeholder.container():
                    self._render_dashboard_content()

                time.sleep(refresh_interval)
        else:
            self._render_dashboard_content()

    def _render_dashboard_content(self):
        """Render the main dashboard content"""
        # Load fresh data
        data = self.load_training_data()

        # Header metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Total Strategies",
                value=data["total_strategies"],
                delta=self._calculate_strategy_delta(data)
            )

        with col2:
            st.metric(
                label="Total Games Played",
                value=data["total_games"],
                delta=self._calculate_games_delta(data)
            )

        with col3:
            avg_performance = self._calculate_avg_performance(data["strategies"])
            st.metric(
                label="Average Performance",
                value=f"{avg_performance:.3f}",
                delta=f"{avg_performance - 0.5:.3f}"
            )

        with col4:
            active_strategies = len([s for s in data["strategies"].values() if s.get("usage_count", 0) > 0])
            st.metric(
                label="Active Strategies",
                value=active_strategies,
                delta=f"{active_strategies}/{data['total_strategies']}"
            )

        st.markdown("---")

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "🎯 Strategy Performance", "📈 Training Progress",
            "🔍 Analysis", "⚙️ Strategy Details"
        ])

        with tab1:
            self._render_overview_tab(data)

        with tab2:
            self._render_strategy_performance_tab(data)

        with tab3:
            self._render_training_progress_tab(data)

        with tab4:
            self._render_analysis_tab(data)

        with tab5:
            self._render_strategy_details_tab(data)

    def _render_overview_tab(self, data: Dict):
        """Render overview tab"""
        st.header("📊 Strategy Pool Overview")

        # Strategy distribution charts
        col1, col2 = st.columns(2)

        with col1:
            # Strategy type distribution
            strategies = data["strategies"]
            type_counts = {}
            for s in strategies.values():
                strategy_type = s.get("type", "unknown")
                type_counts[strategy_type] = type_counts.get(strategy_type, 0) + 1

            if type_counts:
                fig_type = px.pie(
                    values=list(type_counts.values()),
                    names=list(type_counts.keys()),
                    title="Strategy Type Distribution"
                )
                st.plotly_chart(fig_type, use_container_width=True)

        with col2:
            # Role distribution
            role_counts = {}
            for s in strategies.values():
                role = s.get("role", "Unknown")
                role_counts[role] = role_counts.get(role, 0) + 1

            if role_counts:
                fig_role = px.pie(
                    values=list(role_counts.values()),
                    names=list(role_counts.keys()),
                    title="Role Distribution"
                )
                st.plotly_chart(fig_role, use_container_width=True)

        # Performance distribution
        st.subheader("Performance Distribution")

        performances = [s.get("avg_performance", 0.5) for s in strategies.values()]
        usages = [s.get("usage_count", 0) for s in strategies.values()]

        col1, col2 = st.columns(2)

        with col1:
            fig_perf = px.histogram(
                x=performances,
                nbins=20,
                title="Performance Distribution",
                labels={"x": "Average Performance", "y": "Count"}
            )
            st.plotly_chart(fig_perf, use_container_width=True)

        with col2:
            # Usage vs Performance scatter
            fig_scatter = px.scatter(
                x=usages,
                y=performances,
                title="Usage vs Performance",
                labels={"x": "Usage Count", "y": "Average Performance"},
                color=performances,
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    def _render_strategy_performance_tab(self, data: Dict):
        """Render strategy performance tab"""
        st.header("🎯 Strategy Performance Analysis")

        strategies = data["strategies"]

        # Performance table
        st.subheader("Strategy Performance Rankings")

        # Create DataFrame for easier manipulation
        strategy_data = []
        for strategy_id, strategy in strategies.items():
            strategy_data.append({
                "ID": strategy_id[:15] + "...",
                "Type": strategy.get("type", "unknown"),
                "Role": strategy.get("role", "Unknown"),
                "Performance": strategy.get("avg_performance", 0.5),
                "Usage": strategy.get("usage_count", 0),
                "Success Score": strategy.get("success_score", 0.0),
                "Last Used": strategy.get("last_used", "Never")
            })

        if strategy_data:
            df = pd.DataFrame(strategy_data)

            # Sort by performance
            df_sorted = df.sort_values("Performance", ascending=False)

            # Display metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Best Strategy", df_sorted.iloc[0]["ID"], f"{df_sorted.iloc[0]['Performance']:.3f}")

            with col2:
                worst_idx = len(df_sorted) - 1
                st.metric("Worst Strategy", df_sorted.iloc[worst_idx]["ID"], f"{df_sorted.iloc[worst_idx]['Performance']:.3f}")

            with col3:
                median_perf = df_sorted["Performance"].median()
                st.metric("Median Performance", f"{median_perf:.3f}")

            # Performance table
            st.dataframe(df_sorted, use_container_width=True)

            # Performance by role
            st.subheader("Performance by Role")
            role_performance = df.groupby("Role")["Performance"].agg(["mean", "std", "count"]).reset_index()

            fig_role_perf = px.bar(
                role_performance,
                x="Role",
                y="mean",
                error_y="std" if "std" in role_performance.columns else None,
                title="Average Performance by Role",
                labels={"mean": "Average Performance", "Role": "Role"}
            )
            st.plotly_chart(fig_role_perf, use_container_width=True)

    def _render_training_progress_tab(self, data: Dict):
        """Render training progress tab"""
        st.header("📈 Training Progress")

        training_logs = data["training_logs"]

        if not training_logs:
            st.info("No training logs found. Start training to see progress.")
            return

        # Create timeline data
        timeline_data = []
        cumulative_games = 0

        for log in sorted(training_logs, key=lambda x: x.get("start_time", "")):
            start_time = log.get("start_time", "")
            games_played = log.get("games_played", 0)

            if start_time:
                try:
                    # Parse timestamp
                    if "T" in start_time:
                        timestamp = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

                    cumulative_games += games_played
                    timeline_data.append({
                        "timestamp": timestamp,
                        "games_played": games_played,
                        "cumulative_games": cumulative_games,
                        "session_id": log.get("session_id", "unknown")
                    })
                except:
                    continue

        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)

            # Cumulative games over time
            fig_cumulative = px.line(
                df_timeline,
                x="timestamp",
                y="cumulative_games",
                title="Cumulative Games Played Over Time",
                labels={"timestamp": "Time", "cumulative_games": "Total Games"}
            )
            st.plotly_chart(fig_cumulative, use_container_width=True)

            # Games per session
            fig_session = px.bar(
                df_timeline,
                x="session_id",
                y="games_played",
                title="Games Played per Session",
                labels={"session_id": "Session", "games_played": "Games"}
            )
            fig_session.update_xaxes(tickangle=45)
            st.plotly_chart(fig_session, use_container_width=True)

        # Strategy evolution
        st.subheader("Strategy Pool Evolution")

        strategies = data["strategies"]

        # Strategy creation timeline
        creation_data = []
        for strategy_id, strategy in strategies.items():
            created_at = strategy.get("created_at", "")
            if created_at:
                try:
                    if "T" in created_at:
                        timestamp = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")

                    creation_data.append({
                        "timestamp": timestamp,
                        "strategy_id": strategy_id,
                        "type": strategy.get("type", "unknown"),
                        "role": strategy.get("role", "Unknown")
                    })
                except:
                    continue

        if creation_data:
            df_creation = pd.DataFrame(creation_data)
            df_creation = df_creation.sort_values("timestamp")

            # Strategies created over time
            df_creation["cumulative_count"] = range(1, len(df_creation) + 1)

            fig_evolution = px.line(
                df_creation,
                x="timestamp",
                y="cumulative_count",
                title="Strategy Pool Growth Over Time",
                labels={"timestamp": "Time", "cumulative_count": "Total Strategies"}
            )
            st.plotly_chart(fig_evolution, use_container_width=True)

            # Strategy creation by type
            creation_by_type = df_creation.groupby(["timestamp", "type"]).size().reset_index(name="count")

            fig_type_creation = px.scatter(
                creation_by_type,
                x="timestamp",
                y="count",
                color="type",
                size="count",
                title="Strategy Creation by Type Over Time",
                labels={"timestamp": "Time", "count": "Number Created"}
            )
            st.plotly_chart(fig_type_creation, use_container_width=True)

    def _render_analysis_tab(self, data: Dict):
        """Render analysis tab"""
        st.header("🔍 Strategy Analysis")

        # Run comprehensive evaluation
        if st.button("Run Comprehensive Analysis"):
            with st.spinner("Analyzing strategy pool..."):
                evaluation_results = self.evaluator.comprehensive_evaluation()

                # Display key findings
                col1, col2 = st.columns(2)

                with col1:
                    quality_metrics = evaluation_results["quality_metrics"]
                    st.metric(
                        "Overall Quality Score",
                        f"{quality_metrics['overall_score']:.3f}",
                        f"Grade: {quality_metrics['quality_grade']}"
                    )

                with col2:
                    recommendations = evaluation_results.get("optimization_recommendations", [])
                    st.metric(
                        "Optimization Recommendations",
                        len(recommendations),
                        "Areas for improvement"
                    )

                # Component scores
                st.subheader("Quality Component Scores")
                component_scores = quality_metrics["component_scores"]

                fig_components = go.Figure()

                categories = list(component_scores.keys())
                scores = list(component_scores.values())

                fig_components.add_trace(go.Bar(
                    x=categories,
                    y=scores,
                    marker_color=['green' if s >= 0.8 else 'orange' if s >= 0.6 else 'red' for s in scores]
                ))

                fig_components.update_layout(
                    title="Quality Component Scores",
                    xaxis_title="Component",
                    yaxis_title="Score",
                    yaxis=dict(range=[0, 1])
                )

                st.plotly_chart(fig_components, use_container_width=True)

                # Recommendations
                if recommendations:
                    st.subheader("Optimization Recommendations")

                    for i, rec in enumerate(recommendations[:5], 1):
                        priority_color = {
                            "high": "🔴",
                            "medium": "🟡",
                            "low": "🟢"
                        }.get(rec["priority"], "⚪")

                        st.markdown(f"**{i}. {priority_color} {rec['description']}**")
                        st.markdown(f"- *Priority*: {rec['priority']}")
                        st.markdown(f"- *Expected Impact*: {rec['expected_impact']}")
                        st.markdown("---")

        # Strategy interaction analysis
        st.subheader("Strategy Interaction Analysis")

        strategies = data["strategies"]
        behavior_strategies = [s for s in strategies.values() if s.get("type") == "behavior"]
        language_strategies = [s for s in strategies.values() if s.get("type") == "language"]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Behavior Strategies", len(behavior_strategies))
            st.metric("Language Strategies", len(language_strategies))

        with col2:
            # Calculate language strategies with behavior links
            linked_languages = len([s for s in language_strategies if s.get("related_behavior_id")])
            st.metric("Linked Language Strategies", linked_languages)
            if language_strategies:
                st.metric("Link Coverage", f"{linked_languages/len(language_strategies)*100:.1f}%")

    def _render_strategy_details_tab(self, data: Dict):
        """Render strategy details tab"""
        st.header("⚙️ Strategy Details")

        strategies = data["strategies"]

        if not strategies:
            st.info("No strategies found in the pool.")
            return

        # Strategy selector
        strategy_options = [f"{sid[:20]}... ({s.get('type', 'unknown')})" for sid, s in strategies.items()]
        selected_option = st.selectbox("Select Strategy to View", strategy_options)

        # Get selected strategy
        selected_index = strategy_options.index(selected_option)
        selected_strategy_id = list(strategies.keys())[selected_index]
        selected_strategy = strategies[selected_strategy_id]

        # Display strategy details
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Basic Information")
            st.text_input("Strategy ID", selected_strategy_id, disabled=True)
            st.text_input("Type", selected_strategy.get("type", "unknown"), disabled=True)
            st.text_input("Role", selected_strategy.get("role", "Unknown"), disabled=True)
            st.text_input("Game Phase", selected_strategy.get("game_phase", "unknown"), disabled=True)

            # Performance metrics
            st.subheader("Performance Metrics")
            st.metric("Average Performance", f"{selected_strategy.get('avg_performance', 0.5):.3f}")
            st.metric("Usage Count", selected_strategy.get("usage_count", 0))
            st.metric("Success Score", f"{selected_strategy.get('success_score', 0.0):.2f}")

        with col2:
            st.subheader("Strategy Content")
            strategy_text = selected_strategy.get("strategy_text", "No content available")
            st.text_area("Strategy Text", strategy_text, height=300, disabled=True)

            st.subheader("Context Features")
            context_features = selected_strategy.get("context_features", {})
            if context_features:
                st.json(context_features)
            else:
                st.info("No context features defined.")

        # Strategy history
        st.subheader("Strategy History")

        history_data = {
            "Created At": selected_strategy.get("created_at", "Unknown"),
            "Last Used": selected_strategy.get("last_used", "Never"),
            "Related Behavior ID": selected_strategy.get("related_behavior_id", "None"),
            "Parent Strategy": selected_strategy.get("parent_strategy", "None")
        }

        for key, value in history_data.items():
            st.text_input(key, str(value), disabled=True)

    def _calculate_strategy_delta(self, data: Dict) -> str:
        """Calculate strategy count delta (placeholder)"""
        # In a real implementation, this would compare with previous data
        return "0"

    def _calculate_games_delta(self, data: Dict) -> str:
        """Calculate games delta (placeholder)"""
        # In a real implementation, this would compare with previous data
        return "0"

    def _calculate_avg_performance(self, strategies: Dict) -> float:
        """Calculate average performance across all strategies"""
        if not strategies:
            return 0.0

        performances = [s.get("avg_performance", 0.5) for s in strategies.values()]
        return sum(performances) / len(performances)


def main():
    """Main function to run the training monitor"""
    import argparse

    parser = argparse.ArgumentParser(description="Strategy Pool Training Monitor")
    parser.add_argument("--strategy-pool", type=str, default="strategy_pool.json", help="Strategy pool file")
    parser.add_argument("--log-dir", type=str, default="self_play_logs", help="Training log directory")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port")

    args = parser.parse_args()

    if not STREAMLIT_AVAILABLE:
        print("Streamlit is required for the monitoring dashboard.")
        print("Install with: pip install streamlit plotly")
        return

    # Initialize monitor
    monitor = TrainingMonitor(args.strategy_pool, args.log_dir)

    # Run Streamlit app
    import subprocess
    import sys

    script_path = os.path.abspath(__file__)

    # Run streamlit with this script
    cmd = [
        sys.executable, "-m", "streamlit", "run", script_path,
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting Streamlit: {e}")
    except KeyboardInterrupt:
        print("\nMonitoring dashboard stopped.")


if __name__ == "__main__":
    # Check if running as Streamlit app
    if "--" not in sys.argv and "streamlit" not in sys.argv[0]:
        # Running directly - start monitoring
        monitor = TrainingMonitor()
        monitor.create_streamlit_dashboard()
    else:
        # Running via streamlit run command
        main()