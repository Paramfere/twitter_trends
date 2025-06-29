#!/usr/bin/env python3
"""
Velocity & Momentum Tracking Report Generator - Tracks topic movement and changes over time.
Focuses on velocity analysis, rank changes, and momentum patterns.
"""

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VelocityReportGenerator:
    """Generates velocity and momentum tracking reports from trending analysis."""
    
    def __init__(self):
        """Initialize the velocity report generator."""
        self.velocity_state_file = Path("data/velocity_state.json")
        self.session_history = self._load_session_history()
        
    def _load_session_history(self) -> Dict[str, Any]:
        """Load session history for velocity tracking."""
        if self.velocity_state_file.exists():
            try:
                with open(self.velocity_state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning("Could not read velocity state file, starting fresh.")
                return {'sessions': [], 'last_update': None}
        return {'sessions': [], 'last_update': None}
    
    def _save_session_history(self, session_data: Dict[str, Any]):
        """Save current session to history."""
        # Keep only last 5 sessions for velocity tracking
        if len(self.session_history['sessions']) >= 5:
            self.session_history['sessions'] = self.session_history['sessions'][-4:]
        
        self.session_history['sessions'].append(session_data)
        self.session_history['last_update'] = datetime.now().isoformat()
        
        self.velocity_state_file.parent.mkdir(exist_ok=True)
        with open(self.velocity_state_file, 'w', encoding='utf-8') as f:
            json.dump(self.session_history, f, indent=2)
    
    def generate_velocity_report(self, csv_path: str, session_id: str) -> Dict[str, Any]:
        """Generate a comprehensive velocity and momentum report."""
        start_time = time.time()
        logger.info(f"Generating velocity report for session {session_id}")
        
        # Load current session data
        current_df = pd.read_csv(csv_path)
        current_session = self._prepare_session_data(current_df, session_id, csv_path)
        
        # Calculate velocity metrics
        velocity_analysis = self._calculate_velocity_metrics(current_session)
        
        # Generate report structure
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'session_overview': {
                'current_session': session_id,
                'total_topics': len(current_df),
                'historical_sessions': len(self.session_history['sessions']),
                'time_since_last': self._calculate_time_since_last()
            },
            'velocity_analysis': velocity_analysis,
            'strategic_insights': self._generate_strategic_insights(velocity_analysis),
            'processing_time': round(time.time() - start_time, 1)
        }
        
        # Save current session to history
        self._save_session_history(current_session)
        
        return report
    
    def _prepare_session_data(self, df: pd.DataFrame, session_id: str, csv_path: str) -> Dict[str, Any]:
        """Prepare session data for velocity tracking."""
        # Create topic rankings based on tweet volume
        df_sorted = df.sort_values('tweet_volume', ascending=False).reset_index(drop=True)
        df_sorted['rank'] = df_sorted.index + 1
        
        topics = {}
        for _, row in df_sorted.iterrows():
            topics[row['topic']] = {
                'volume': int(row['tweet_volume']),
                'rank': int(row['rank']),
                'region': row['region'],
                'category': row['category'],
                'significance': int(row['significance_score'])
            }
        
        return {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'csv_path': csv_path,
            'total_topics': len(df),
            'topics': topics
        }
    
    def _calculate_velocity_metrics(self, current_session: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate all velocity metrics."""
        if len(self.session_history['sessions']) == 0:
            # First session - everything is new
            return self._handle_first_session(current_session)
        
        # Get the most recent previous session
        previous_session = self.session_history['sessions'][-1]
        
        current_topics = current_session['topics']
        previous_topics = previous_session['topics']
        
        # Calculate different types of changes
        new_topics = self._find_new_topics(current_topics, previous_topics)
        volume_surges = self._find_volume_surges(current_topics, previous_topics)
        volume_drops = self._find_volume_drops(current_topics, previous_topics)
        rank_climbers = self._find_rank_climbers(current_topics, previous_topics)
        
        return {
            'new_topics': new_topics,
            'volume_surges': volume_surges,
            'volume_drops': volume_drops,
            'rank_climbers': rank_climbers,
            'comparison_session': previous_session['session_id']
        }
    
    def _handle_first_session(self, current_session: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the case where this is the first session."""
        # All topics are "new" in the first session
        new_topics = []
        for topic, data in current_session['topics'].items():
            new_topics.append({
                'topic': topic,
                'region': data['region'],
                'volume': data['volume'],
                'rank': data['rank']
            })
        
        # Sort by volume descending
        new_topics.sort(key=lambda x: x['volume'], reverse=True)
        
        return {
            'new_topics': new_topics[:20],  # Top 20 new topics
            'volume_surges': [],
            'volume_drops': [],
            'rank_climbers': [],
            'comparison_session': None
        }
    
    def _find_new_topics(self, current_topics: Dict, previous_topics: Dict) -> List[Dict]:
        """Find topics that are new in the current session."""
        new_topics = []
        
        for topic, data in current_topics.items():
            if topic not in previous_topics:
                new_topics.append({
                    'topic': topic,
                    'region': data['region'],
                    'volume': data['volume'],
                    'rank': data['rank']
                })
        
        # Sort by volume descending
        new_topics.sort(key=lambda x: x['volume'], reverse=True)
        return new_topics[:20]  # Top 20 new topics
    
    def _find_volume_surges(self, current_topics: Dict, previous_topics: Dict, min_surge_percent: float = 200) -> List[Dict]:
        """Find topics with significant volume increases."""
        surges = []
        
        for topic, current_data in current_topics.items():
            if topic in previous_topics:
                previous_volume = previous_topics[topic]['volume']
                current_volume = current_data['volume']
                
                if previous_volume > 0:
                    surge_percent = ((current_volume - previous_volume) / previous_volume) * 100
                    
                    if surge_percent >= min_surge_percent:
                        surges.append({
                            'topic': topic,
                            'region': current_data['region'],
                            'volume': current_volume,
                            'surge_percent': round(surge_percent),
                            'previous_volume': previous_volume
                        })
        
        # Sort by surge percentage descending
        surges.sort(key=lambda x: x['surge_percent'], reverse=True)
        return surges[:20]  # Top 20 surges
    
    def _find_volume_drops(self, current_topics: Dict, previous_topics: Dict, min_drop_percent: float = 50) -> List[Dict]:
        """Find topics with significant volume decreases."""
        drops = []
        
        for topic, previous_data in previous_topics.items():
            if topic in current_topics:
                previous_volume = previous_data['volume']
                current_volume = current_topics[topic]['volume']
                
                if previous_volume > 0:
                    drop_percent = ((previous_volume - current_volume) / previous_volume) * 100
                    
                    if drop_percent >= min_drop_percent:
                        drops.append({
                            'topic': topic,
                            'region': current_topics[topic]['region'],
                            'volume': current_volume,
                            'drop_percent': round(drop_percent),
                            'previous_volume': previous_volume
                        })
        
        # Sort by drop percentage descending
        drops.sort(key=lambda x: x['drop_percent'], reverse=True)
        return drops[:20]  # Top 20 drops
    
    def _find_rank_climbers(self, current_topics: Dict, previous_topics: Dict, min_climb: int = 5) -> List[Dict]:
        """Find topics that climbed significantly in rankings."""
        climbers = []
        
        for topic, current_data in current_topics.items():
            if topic in previous_topics:
                previous_rank = previous_topics[topic]['rank']
                current_rank = current_data['rank']
                
                # Rank climb is when current rank is lower number (better position)
                rank_change = previous_rank - current_rank
                
                if rank_change >= min_climb:
                    climbers.append({
                        'topic': topic,
                        'region': current_data['region'],
                        'current_rank': current_rank,
                        'rank_change': rank_change,
                        'previous_rank': previous_rank,
                        'volume': current_data['volume']
                    })
        
        # Sort by rank change descending
        climbers.sort(key=lambda x: x['rank_change'], reverse=True)
        return climbers[:20]  # Top 20 climbers
    
    def _calculate_time_since_last(self) -> str:
        """Calculate time since last session."""
        if len(self.session_history['sessions']) == 0:
            return "N/A (First session)"
        
        last_session = self.session_history['sessions'][-1]
        last_time = datetime.fromisoformat(last_session['timestamp'])
        current_time = datetime.now()
        
        time_diff = current_time - last_time
        
        if time_diff.total_seconds() < 3600:  # Less than 1 hour
            minutes = time_diff.total_seconds() / 60
            return f"{minutes:.1f} minutes"
        elif time_diff.total_seconds() < 86400:  # Less than 1 day
            hours = time_diff.total_seconds() / 3600
            return f"{hours:.1f} hours"
        else:
            days = time_diff.days
            return f"{days} days"
    
    def _generate_strategic_insights(self, velocity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategic insights from velocity data."""
        return {
            'active_tracking': len(self.session_history['sessions']),
            'velocity_window': "Last session comparison",
            'new_topic_rate': len(velocity_analysis['new_topics']),
            'high_velocity_topics': len(velocity_analysis['volume_surges'])
        }
    
    def save_velocity_report(self, report: Dict[str, Any], session_dir: Path) -> str:
        """Save the velocity report to a markdown file."""
        reports_dir = session_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        report_path = reports_dir / f"velocity_report_{timestamp}.md"
        
        # Generate markdown content
        markdown_content = self._generate_markdown_report(report)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Velocity report saved to {report_path}")
        return str(report_path)
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate formatted markdown report."""
        markdown = f"""# 🚀 VELOCITY & MOMENTUM TRACKING REPORT
*Generated: {report['timestamp']}*

## 📊 Session Overview
- **Current Session**: {report['session_overview']['current_session']}
- **Total Topics**: {report['session_overview']['total_topics']}
- **Historical Sessions**: {report['session_overview']['historical_sessions']}
- **Time Since Last**: {report['session_overview']['time_since_last']}

## ⚡ VELOCITY ANALYSIS
"""

        # New Topics Section
        new_topics = report['velocity_analysis']['new_topics']
        markdown += f"""### 🆕 New Topics ({len(new_topics)})
*Topics appearing for the first time*
"""
        if new_topics:
            for topic in new_topics[:10]:  # Show top 10
                markdown += f"- **{topic['topic']}** ({topic['region']}) - {topic['volume']:,} tweets\n"
        else:
            markdown += "*No new topics detected.*\n"

        # Volume Surges Section
        volume_surges = report['velocity_analysis']['volume_surges']
        markdown += f"""
### 🚀 Volume Surges ({len(volume_surges)})
*Topics with >200% volume increase*
"""
        if volume_surges:
            for surge in volume_surges[:10]:  # Show top 10
                markdown += f"- **{surge['topic']}** ({surge['region']}) - {surge['volume']:,} tweets (+{surge['surge_percent']}%)\n"
        else:
            markdown += "*No significant volume surges detected.*\n"

        # Volume Drops Section
        volume_drops = report['velocity_analysis']['volume_drops']
        markdown += f"""
### 📉 Volume Drops ({len(volume_drops)})
*Topics with >50% volume decrease*
"""
        if volume_drops:
            for drop in volume_drops[:10]:  # Show top 10
                markdown += f"- **{drop['topic']}** ({drop['region']}) - {drop['volume']:,} tweets (-{drop['drop_percent']}%)\n"
        else:
            markdown += "*No significant volume drops detected.*\n"

        # Rank Climbers Section
        rank_climbers = report['velocity_analysis']['rank_climbers']
        markdown += f"""
### ⬆️ Rank Climbers ({len(rank_climbers)})
*Topics climbing 5+ positions*
"""
        if rank_climbers:
            for climber in rank_climbers[:10]:  # Show top 10
                markdown += f"- **{climber['topic']}** ({climber['region']}) - Rank {climber['current_rank']} (↑{climber['rank_change']} positions)\n"
        else:
            markdown += "*No significant rank climbers detected.*\n"

        # Strategic Insights Section
        insights = report['strategic_insights']
        markdown += f"""
## 🎯 STRATEGIC INSIGHTS
- **Active Tracking**: {insights['active_tracking']} sessions in memory
- **Velocity Window**: {insights['velocity_window']}
- **New Topic Rate**: {insights['new_topic_rate']} per session
- **High Velocity Topics**: {insights['high_velocity_topics']} surging

---
*Velocity analysis completed in {report['processing_time']}s*
"""

        return markdown

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate velocity and momentum tracking report")
    parser.add_argument("--session-id", required=True, help="Session ID to analyze (e.g., '021')")
    args = parser.parse_args()
    
    # Find the analysis CSV for the session
    session_path = Path(f"data/session_{args.session_id}")
    try:
        analysis_csv = next((session_path / "analysis").glob("trending_analysis_*.csv"))
    except StopIteration:
        logger.error(f"No analysis CSV found for session {args.session_id}")
        sys.exit(1)
    
    # Generate velocity report
    generator = VelocityReportGenerator()
    report = generator.generate_velocity_report(str(analysis_csv), args.session_id)
    
    # Save report
    report_path = generator.save_velocity_report(report, session_path)
    
    print(f"✅ Velocity Report generated successfully!")
    print(f"📄 Saved to: {report_path}")
    print(f"🆕 New topics: {len(report['velocity_analysis']['new_topics'])}")
    print(f"🚀 Volume surges: {len(report['velocity_analysis']['volume_surges'])}")
    print(f"📉 Volume drops: {len(report['velocity_analysis']['volume_drops'])}")
    print(f"⬆️ Rank climbers: {len(report['velocity_analysis']['rank_climbers'])}")
