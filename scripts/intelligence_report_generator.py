#!/usr/bin/env python3
"""
Generates a comprehensive Marketing Intelligence Report using a structured AI prompt.
This script gathers data from a session, prepares it into a specific JSON format,
and uses the OpenAI API to generate a detailed markdown report.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import openai
    openai_available = True
except ImportError:
    openai_available = False
    
class IntelligenceReportGenerator:
    """Generates a marketing intelligence report from session data."""

    def __init__(self):
        """Initializes the generator and the OpenAI client."""
        self.state_file = Path("data/intelligence_state.json")
        self.previous_state = self._load_previous_state()
        
        if not openai_available:
            raise ImportError("OpenAI library not found. Please run 'pip install openai'.")
            
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        self.client = openai.OpenAI(api_key=api_key)
        logger.info("IntelligenceReportGenerator initialized successfully.")

    def _load_state_from_session(self, session_path: str) -> Dict[str, Any]:
        """Loads a state object from a given session's analysis file."""
        try:
            analysis_dir = Path(session_path) / "analysis"
            # Find the first csv file in the analysis directory
            analysis_file = next(analysis_dir.glob('*.csv'), None)
            if not analysis_file:
                logger.warning(f"No analysis CSV found in {session_path}. Cannot build state.")
                return {}

            df = pd.read_csv(analysis_file)
            state_data = {
                'timestamp': datetime.fromtimestamp(analysis_file.stat().st_mtime).isoformat(),
                'topics': {
                    row['topic']: {
                        'volume': int(row['tweet_volume']),
                        'category': row['category']
                    } for _, row in df.iterrows()
                },
                'category_volumes': df.groupby('category')['tweet_volume'].sum().to_dict()
            }
            logger.info(f"Successfully built baseline state from {analysis_file}")
            return state_data
        except Exception as e:
            logger.error(f"Failed to build state from session '{session_path}': {e}")
            return {}

    def _load_previous_state(self) -> Dict[str, Any]:
        """Loads the state from the most recent session for change detection."""
        # Find all session directories
        data_dir = Path("data")
        session_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("session_")]
        
        if len(session_dirs) < 1:
            logger.info("No previous sessions found, starting fresh.")
            return {}
        
        # Sort by session number/name to get sessions in order (handle numeric sorting)
        def extract_session_number(session_name):
            try:
                # Extract number from session_XXX format
                return int(session_name.split('_')[1]) if '_' in session_name else 0
            except (ValueError, IndexError):
                return 0
        
        session_dirs.sort(key=lambda x: extract_session_number(x.name))
        
        # Find the most recent session that's not the current one
        most_recent = None
        for session_dir in reversed(session_dirs):
            if not hasattr(self, 'current_session_name') or session_dir.name != self.current_session_name:
                most_recent = session_dir
                break
        
        if not most_recent:
            logger.info("No suitable previous session found for comparison.")
            return {}
        
        logger.info(f"Auto-detecting session {most_recent.name} as baseline for comparison.")
        
        # Try to load state from the most recent session
        try:
            analysis_files = list((most_recent / "analysis").glob("trending_analysis_*.csv"))
            if analysis_files:
                # Use the most recent analysis file
                return self._load_state_from_session(str(most_recent))
        except Exception as e:
            logger.warning(f"Could not load state from most recent session {most_recent.name}: {e}")
        
        # Fallback to old state file method
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning("Could not read state file, starting fresh.")
                return {}
        return {}

    def _save_current_state(self, df: pd.DataFrame):
        """Saves the current state for the next run."""
        state_data = {
            'timestamp': datetime.now().isoformat(),
            'topics': {
                row['topic']: {
                    'volume': int(row['tweet_volume']),
                    'category': row['category']
                } for _, row in df.iterrows()
            },
            'category_volumes': df.groupby('category')['tweet_volume'].sum().to_dict()
        }
        self.state_file.parent.mkdir(exist_ok=True)
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2)

    def _prepare_data_for_ai(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepares all data structures required by the AI prompt."""
        changes = self._detect_changes(df)
        
        # 1. Summary Metrics
        summary_metrics = {
            'total_topics': len(df),
            'tech_share': f"{len(df[df['category'] == 'Technology']) / len(df) * 100:.1f}%",
            'avg_volume': df['tweet_volume'].mean(),
            'avg_significance': df['significance_score'].mean()
        }
        
        # 2. Category Insights
        category_stats = df.groupby('category')['tweet_volume'].agg(['count', 'sum']).reset_index()
        total_volume = df['tweet_volume'].sum()
        category_stats['percentage'] = (category_stats['sum'] / total_volume) * 100
        category_insights = category_stats.rename(columns={'count': 'topic_count', 'sum': 'total_volume'}).to_dict('records')

        # 3. Trend Table - Limit to top 20 most significant topics to save tokens
        trend_table = df.nlargest(20, 'significance_score').to_dict('records')

        return {
            "summary_metrics": summary_metrics,
            "category_insights": category_insights,
            "changes": changes,
            "trend_table": trend_table
        }

    def _detect_changes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detects changes from the previous state, including category momentum."""
        changes = {
            'new_topics': [],
            'volume_surges': [],
            'declining_topics': [],
            'rising_categories': []
        }
        
        # Load previous state fresh for each detection
        previous_state = self._load_previous_state()

        if not previous_state.get('topics'):
            logger.info("No previous state found. All topics will be considered new.")
            for _, row in df.iterrows():
                changes['new_topics'].append({
                    'topic': row['topic'],
                    'volume': row['tweet_volume'],
                    'region': row.get('region', 'N/A'),
                    'significance': row.get('significance_score', 0)
                })
            return changes
            
        prev_topics = previous_state.get('topics', {})
        current_topics_map = {row['topic']: row for _, row in df.iterrows()}
        
        # Logic to find changes
        for topic, current_row in current_topics_map.items():
            current_volume = current_row['tweet_volume']
            if topic not in prev_topics:
                changes['new_topics'].append({
                    'topic': topic, 
                    'volume': current_volume, 
                    'region': current_row.get('region', 'N/A'),
                    'significance': current_row.get('significance_score', 0)
                })
            elif current_volume > prev_topics[topic]['volume'] * 1.5: # Lowering threshold to 50%
                prev_volume = prev_topics[topic]['volume']
                if prev_volume > 0:  # Avoid division by zero
                    surge_percent = ((current_volume - prev_volume) / prev_volume) * 100
                else:
                    surge_percent = float('inf')  # Infinite growth from zero
                changes['volume_surges'].append({
                    'topic': topic,
                    'surge_percent': surge_percent,
                    'previous_volume': prev_volume,
                    'current_volume': current_volume
                })

        for topic, prev_data in prev_topics.items():
            if topic in current_topics_map and current_topics_map[topic]['tweet_volume'] < prev_data['volume'] * 0.7: # Lowering threshold
                prev_volume = prev_data['volume']
                current_volume = current_topics_map[topic]['tweet_volume']
                if prev_volume > 0:  # Avoid division by zero
                    decline_percent = (prev_volume - current_volume) / prev_volume * 100
                else:
                    decline_percent = 0  # No decline if previous was zero
                changes['declining_topics'].append({
                    'topic': topic, 
                    'decline_percent': decline_percent
                })
        
        prev_category_volumes = previous_state.get('category_volumes', {})
        if prev_category_volumes:
            current_category_volumes = df.groupby('category')['tweet_volume'].sum()
            for category, current_vol in current_category_volumes.items():
                if category in prev_category_volumes and isinstance(prev_category_volumes[category], (int, float)):
                    prev_vol = prev_category_volumes[category]
                    if current_vol > prev_vol * 1.2:  # 20% growth
                        if prev_vol > 0:  # Avoid division by zero
                            growth_percent = ((current_vol - prev_vol) / prev_vol) * 100
                        else:
                            growth_percent = float('inf')  # Infinite growth from zero
                        changes['rising_categories'].append({
                            'category': category,
                            'growth_percent': growth_percent,
                            'current_volume': f"{current_vol:,.0f}"
                        })

        logger.info(f"Detected changes: New Topics={len(changes['new_topics'])}, Surges={len(changes['volume_surges'])}, Declines={len(changes['declining_topics'])}, Rising Categories={len(changes['rising_categories'])}")
        return changes

    def generate_report(self, csv_path: str) -> Optional[str]:
        """Orchestrates the report generation process."""
        start_time = time.time()
        logger.info(f"Starting intelligence report generation for {csv_path}")

        try:
            # Determine current session from path to avoid using it as baseline
            current_session_dir = self._find_session_dir(Path(csv_path))
            if current_session_dir:
                self.current_session_name = current_session_dir.name
            else:
                self.current_session_name = None
                
            df = pd.read_csv(csv_path)
            # Placeholder for velocity if not present
            if 'velocity_6h_live_per_hr' not in df.columns:
                df['velocity_6h_live_per_hr'] = 'N/A'

            ai_data = self._prepare_data_for_ai(df)
            
            system_prompt = """
You are a world-class marketing intelligence engine. Your analysis must be sharp, predictive, and actionable.
You will be given JSON data with four keys: 'summary_metrics', 'category_insights', 'changes', and 'trend_table'.
Your task is to produce a **Markdown** report matching the following structure precisely.
Use your expertise to interpret the data and fill in the logical placeholders (e.g., assess urgency, define content angles, determine market implications).

---

# 📊 Marketing Intelligence Report – {CURRENT_TIMESTAMP}

## 🚨 CRITICAL MOMENTUM ALERTS
*Based on the 'changes' data, generate the following sections. If a section has no data (e.g., no new_topics), omit that specific subsection.*

### 🆕 NEW BREAKOUT TOPICS (LAST 30 MIN) - TOP PRIORITY
🎯 **These topics just started trending and represent the highest opportunity for viral reach**
*(For each topic in `changes.new_topics`, assess its viral potential, urgency, and define a clear action plan.)*
**{topic}** ({region}) - {Viral Potential Assessment: e.g., 🚀 EXTREME VIRAL POTENTIAL}
• **Volume**: {volume} tweets (NEW BREAKOUT)
• **Urgency**: {Urgency Assessment: e.g., 🚨 CRITICAL - Post within 15 minutes}
• **Action**: {Action-based Directive: e.g., Create immediate reaction content}
• **Content Hook**: {Specific, Engaging Hook: e.g., Breaking: X is exploding...}
• **Window**: {Actionable Time Window: e.g., ⚡ ACT NOW - Peak viral window}

### 📈 VOLUME SURGES (CRITICAL MOMENTUM) - IMMEDIATE PIVOT REQUIRED
🚨 **These topics are experiencing massive growth - pivot resources immediately**
*(For each topic in `changes.volume_surges`, describe the surge and prescribe a strategy.)*

### 📊 Category Momentum Shifts
*(For each category in `changes.rising_categories`)*
**{category}** - +{growth_percent:.0f}% growth
• **Current Volume**: {current_volume} tweets
• **Strategy**: Scale up {category} content production immediately.

## 1️⃣ CURRENT TOP PERFORMERS
### 🔥 Highest Volume Right Now  
*(List the top 3 topics from `trend_table` by tweet_volume)*
**{i}. {topic}** ({region})
• **Volume**: {tweet_volume} tweets
• **Momentum**: {Assess Momentum: e.g., ⭐ NEW BREAKOUT or 📊 STABLE}
• **Action Window**: {Define Action Window: e.g., ⚡ NOW - Peak viral moment}

## 2️⃣ WEB3 & CRYPTO PULSE
### 💰 Active Crypto/Tech Topics  
*(List top 3 topics from `trend_table` where category is 'Technology' or subcategory is crypto-related)*
**{topic}** - {tweet_volume} tweets
• **Market Context**: {Provide Market Context: e.g., AI sector attention - potential AI token correlation}

### 🔗 Crypto Relations to General Trends  
*(Analyze the entire trend_table to find deeper connections between crypto/tech and mainstream trends. Provide a bulleted list of 3-5 specific, data-driven insights. For each, mention the pattern and the opportunity.)*
• **{Pattern Name e.g., K-pop Fan Economy}:** {Observation e.g., X trending events with Y total volume} - {Opportunity e.g., prime for fan tokens & NFT drops}.
• **{Pattern Name e.g., Brand Collaboration Trend}:** {Observation e.g., X partnership topics trending} - {Opportunity e.g., leverage for Web3 brand partnerships}.
• **{Pattern Name e.g., Transparency Movement}:** {Observation e.g., X volume on 'truth/behind' topics} - {Opportunity e.g., perfect for blockchain transparency narratives}.
• **{Pattern Name e.g., AI-Crypto Convergence}:** {Observation e.g., X AI topics trending} - {Opportunity e.g., capitalize on AI agent tokens}.
• **{Pattern Name e.g., Viral Tech Gap}:** {Observation e.g., X of Y viral topics are tech-related} - {Opportunity e.g., massive opportunity for crypto content to break into mainstream}.

## 3️⃣ IMMEDIATE MARKETING ACTIONS (Next 30 Min)
### ⏰ Time-Sensitive Opportunities  
*(List top 3 most urgent topics from the data)*
**{topic}**
• **Action**: {Specific Marketing Action}
• **Deadline**: {Actionable Deadline}
• **Content Angle**: {Creative Content Angle}

## 4️⃣ COMPREHENSIVE MARKET ANALYSIS
### 🌍 Regional Performance Breakdown  
*(For each region)*
**{region} Region**
• **Topics**: {count} trending topics
• **Total Volume**: {sum_volume} tweets
• **Top Category**: {top_category}
• **Strategy**: {Provide a concise, data-driven strategy for the region}

### 📊 Category Performance Insights  
*(For each category in `category_insights`)*
**{category}** ({topic_count} topics)
• **Volume**: {total_volume} tweets ({percentage:.1f}%)
• **Opportunity**: {Define the strategic opportunity for this category}

### 📈 Market Dynamics & Patterns  
*(Provide 3-4 bulleted insights about the overall market dynamics.)*

### 🎯 Strategic Positioning (Next Cycle)
*(Provide 3-4 strategic directives for the next analysis cycle.)*

## ✨ ADDITIONAL INSIGHTS
*(Optional sections if data supports)*
### 🌱 Emerging Micro-Trends  
• List any niche topics with high velocity but low absolute volume—opportunity for thought leadership.  
### 🔗 Hashtag Co-occurrence Network  
• Describe 2–3 key hashtag bridges between categories.  
### 😊 Sentiment Deep-Dive  
• Highlight any categories or regions where sentiment skews unusually positive/negative.  
"""
            
            user_prompt = f"""
Here is the data for your analysis:
```json
{json.dumps(ai_data, indent=2)}
```Please generate the report. Make sure you:
- Use the exact headings, numbering, and emojis as specified in the system prompt.
- Populate only the sections for which there is relevant data (e.g., skip empty alerts).
- Output pure Markdown without any additional explanatory prose outside the defined structure.
- Replace {{CURRENT_TIMESTAMP}} with the current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
            )
            
            report_content = response.choices[0].message.content
            
            # Save the report
            session_dir = self._find_session_dir(Path(csv_path))
            if not session_dir:
                logger.error(f"Could not determine session directory for '{csv_path}'. Cannot save report.")
                return None
            report_path = self._save_report(report_content, session_dir)
            
            # Save state for next run
            self._save_current_state(df)
            
            logger.info(f"Report generation complete in {time.time() - start_time:.1f}s")
            return report_path

        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            return None

    def _find_session_dir(self, path: Path) -> Optional[Path]:
        """Finds the session directory from a given path."""
        current = path.parent
        while current != current.parent: # Stop at the root
            if current.name.startswith("session_"):
                return current
            current = current.parent
        return None

    def _save_report(self, report_content: str, session_dir: Path) -> str:
        """Saves the generated report to the session's reports directory."""
        report_dir = session_dir / "reports"
        report_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        report_path = report_dir / f"intelligence_report_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logger.info(f"Report saved to {report_path}")
        return str(report_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a marketing intelligence report for a given session.")
    parser.add_argument("--session-id", required=True, help="The ID of the session to analyze (e.g., '001').")
    parser.add_argument("--baseline-session", help="Optional. The session ID to use as a baseline for comparison.")
    args = parser.parse_args()

    generator = IntelligenceReportGenerator()

    # If a baseline session is provided, override the default previous_state.
    if args.baseline_session:
        logger.info(f"Using session {args.baseline_session} as a custom baseline.")
        baseline_path = f"data/session_{args.baseline_session}"
        baseline_state = generator._load_state_from_session(baseline_path)
        if baseline_state:
            generator.previous_state = baseline_state
        else:
            logger.error(f"Could not load baseline state from session {args.baseline_session}. Aborting.")
            sys.exit(1)

    # Find the analysis CSV for the target session
    session_path = Path(f"data/session_{args.session_id}")
    try:
        analysis_csv = next((session_path / "analysis").glob("trending_analysis_*.csv"))
    except StopIteration:
        logger.error(f"No analysis CSV found for session {args.session_id} in {session_path / 'analysis'}")
        sys.exit(1)

    report_path = generator.generate_report(str(analysis_csv))
    if report_path:
        print(f"Report generation process completed. Path: {report_path}")
    else:
        print("Report generation failed. Check logs for details.")
