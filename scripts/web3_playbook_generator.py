#!/usr/bin/env python3
"""
Web3 Playbook Generator - Converts trending topics analysis into actionable posting recommendations.
Focuses on tech and Web3 opportunities with specific content angles and timing recommendations.
"""

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Web3PlaybookGenerator:
    """Generates actionable Web3 and tech posting playbooks from trending analysis."""
    
    def __init__(self):
        """Initialize the playbook generator."""
        self.web3_subcategories = ['Defi', 'Nft', 'Blockchain', 'Metaverse', 'Crypto Trading', 'Web3 Social', 'Gamefi', 'Dao']
        self.tech_categories = ['Technology']
        
    def generate_playbook(self, csv_path: str, baseline_csv_path: str = None) -> Dict[str, Any]:
        """Generate a comprehensive Web3 posting playbook."""
        logger.info(f"Generating Web3 playbook from {csv_path}")
        
        # Load current data
        df = pd.read_csv(csv_path)
        
        # Load baseline for velocity calculation if provided
        baseline_df = None
        if baseline_csv_path and Path(baseline_csv_path).exists():
            baseline_df = pd.read_csv(baseline_csv_path)
            logger.info(f"Using baseline data from {baseline_csv_path}")
        
        # Generate all playbook sections
        playbook = {
            'timestamp': datetime.now().isoformat(),
            'total_topics': len(df),
            'tech_topics': len(df[df['category'] == 'Technology']),
            'web3_breakdown': self._analyze_web3_breakdown(df),
            'priority_posting_list': self._generate_priority_posting_list(df),
            'significance_volume_matrix': self._create_significance_volume_matrix(df),
            'content_angle_suggestions': self._generate_content_angles(df),
            'regional_insights': self._analyze_regional_performance(df),
            'velocity_analysis': self._calculate_velocity(df, baseline_df) if baseline_df is not None else None,
            'partnership_signals': self._identify_partnership_opportunities(df),
            'engagement_experiments': self._suggest_engagement_experiments(df),
            'hashtag_bridges': self._analyze_hashtag_bridges(df),
            'market_timing': self._analyze_market_timing(df)
        }
        
        return playbook
    
    def _analyze_web3_breakdown(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Web3 topic distribution and performance."""
        tech_df = df[df['category'] == 'Technology'].copy()
        
        # Web3 subcategory analysis
        web3_stats = {}
        for subcat in self.web3_subcategories:
            subcat_df = tech_df[tech_df['subcategory'] == subcat]
            if len(subcat_df) > 0:
                web3_stats[subcat] = {
                    'topic_count': len(subcat_df),
                    'total_volume': int(subcat_df['tweet_volume'].sum()),
                    'avg_volume': int(subcat_df['tweet_volume'].mean()),
                    'avg_significance': round(subcat_df['significance_score'].mean(), 1),
                    'top_topic': subcat_df.nlargest(1, 'tweet_volume')['topic'].iloc[0] if len(subcat_df) > 0 else None,
                    'sentiment_mix': subcat_df['sentiment'].value_counts().to_dict()
                }
        
        return {
            'total_tech_volume': int(tech_df['tweet_volume'].sum()),
            'tech_share_of_total': round(len(tech_df) / len(df) * 100, 1),
            'web3_subcategories': web3_stats,
            'top_web3_topics': tech_df.nlargest(5, 'tweet_volume')[['topic', 'subcategory', 'tweet_volume', 'significance_score']].to_dict('records')
        }
    
    def _generate_priority_posting_list(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate top priority posting opportunities."""
        # Filter for high-impact tech/Web3 topics
        tech_df = df[df['category'] == 'Technology'].copy()
        
        if len(tech_df) == 0:
            return []
        
        # Create priority score: significance * volume * sentiment boost
        tech_df['sentiment_multiplier'] = tech_df['sentiment'].map({'positive': 1.2, 'neutral': 1.0, 'negative': 0.8})
        tech_df['priority_score'] = tech_df['significance_score'] * np.log10(tech_df['tweet_volume'] + 1) * tech_df['sentiment_multiplier']
        
        # Get top priority topics per region
        priority_topics = []
        for region in df['region'].unique():
            region_tech = tech_df[tech_df['region'] == region]
            if len(region_tech) > 0:
                top_topics = region_tech.nlargest(min(3, len(region_tech)), 'priority_score')
                for _, topic in top_topics.iterrows():
                    priority_topics.append({
                        'topic': topic['topic'],
                        'region': topic['region'],
                        'subcategory': topic['subcategory'],
                        'volume': int(topic['tweet_volume']),
                        'significance': int(topic['significance_score']),
                        'sentiment': topic['sentiment'],
                        'priority_score': round(topic['priority_score'], 2),
                        'urgency': self._assess_urgency(topic['tweet_volume'], topic['significance_score']),
                        'posting_window': self._determine_posting_window(topic['tweet_volume'], topic['sentiment'])
                    })
        
        return sorted(priority_topics, key=lambda x: x['priority_score'], reverse=True)[:10]
    
    def _create_significance_volume_matrix(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Create the significance vs volume quadrant matrix."""
        tech_df = df[df['category'] == 'Technology'].copy()
        
        if len(tech_df) == 0:
            return {'must_post_now': [], 'watch_for_angle': [], 'niche_leadership': [], 'ignore_for_now': []}
        
        # Define thresholds
        volume_threshold = tech_df['tweet_volume'].median()
        significance_threshold = tech_df['significance_score'].median()
        
        quadrants = {
            'must_post_now': [],      # High Volume, High Significance
            'watch_for_angle': [],    # High Volume, Low Significance  
            'niche_leadership': [],   # Low Volume, High Significance
            'ignore_for_now': []      # Low Volume, Low Significance
        }
        
        for _, topic in tech_df.iterrows():
            topic_data = {
                'topic': topic['topic'],
                'volume': int(topic['tweet_volume']),
                'significance': int(topic['significance_score']),
                'subcategory': topic['subcategory'],
                'sentiment': topic['sentiment']
            }
            
            if topic['tweet_volume'] >= volume_threshold and topic['significance_score'] >= significance_threshold:
                quadrants['must_post_now'].append(topic_data)
            elif topic['tweet_volume'] >= volume_threshold and topic['significance_score'] < significance_threshold:
                quadrants['watch_for_angle'].append(topic_data)
            elif topic['tweet_volume'] < volume_threshold and topic['significance_score'] >= significance_threshold:
                quadrants['niche_leadership'].append(topic_data)
            else:
                quadrants['ignore_for_now'].append(topic_data)
        
        return quadrants
    
    def _generate_content_angles(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate specific content angle suggestions for top topics."""
        tech_df = df[df['category'] == 'Technology'].copy()
        
        if len(tech_df) == 0:
            return []
        
        top_topics = tech_df.nlargest(min(8, len(tech_df)), 'tweet_volume')
        
        content_angles = []
        for _, topic in top_topics.iterrows():
            angles = self._create_content_angles_for_topic(topic['topic'], topic['subcategory'], topic['sentiment'])
            content_angles.append({
                'topic': topic['topic'],
                'subcategory': topic['subcategory'],
                'volume': int(topic['tweet_volume']),
                'angles': angles
            })
        
        return content_angles
    
    def _create_content_angles_for_topic(self, topic: str, subcategory: str, sentiment: str) -> List[str]:
        """Create specific content angles based on topic and subcategory."""
        angles = []
        
        if subcategory == 'Defi':
            angles = [
                f"Thread: How {topic} is revolutionizing traditional finance",
                f"Explainer: Why {topic} matters for the future of money",
                f"Poll: Would you try {topic} for your next financial transaction?"
            ]
        elif subcategory == 'Nft':
            angles = [
                f"Analysis: What {topic} tells us about digital ownership trends",
                f"Thread: The technology behind {topic} explained simply",
                f"Opinion: Why {topic} represents the next evolution of collectibles"
            ]
        elif subcategory == 'Ai':
            angles = [
                f"Deep dive: How {topic} is changing the AI landscape",
                f"Thread: What {topic} means for the future of work",
                f"Explainer: The technology powering {topic}"
            ]
        elif subcategory == 'Blockchain':
            angles = [
                f"Technical breakdown: How {topic} improves blockchain scalability",
                f"Thread: Why {topic} matters for Web3 adoption",
                f"Analysis: {topic}'s impact on decentralization"
            ]
        elif subcategory == 'Metaverse':
            angles = [
                f"Vision: How {topic} is building the metaverse",
                f"Thread: What {topic} means for virtual experiences",
                f"Analysis: The tech stack behind {topic}"
            ]
        else:
            angles = [
                f"Thread: Breaking down {topic} and its implications",
                f"Analysis: Why {topic} is trending in tech circles",
                f"Explainer: What you need to know about {topic}"
            ]
        
        # Adjust tone based on sentiment
        if sentiment == 'negative':
            angles = [angle.replace('Why', 'The controversy around').replace('How', 'Addressing concerns about') for angle in angles]
        elif sentiment == 'positive':
            angles = [angle.replace('Analysis:', 'Celebrating:').replace('Thread:', 'Excited about:') for angle in angles]
        
        return angles
    
    def _analyze_regional_performance(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze tech performance by region."""
        regional_analysis = {}
        
        for region in df['region'].unique():
            region_df = df[df['region'] == region]
            tech_df = region_df[region_df['category'] == 'Technology']
            
            regional_analysis[region] = {
                'total_topics': len(region_df),
                'tech_topics': len(tech_df),
                'tech_share': round(len(tech_df) / len(region_df) * 100, 1) if len(region_df) > 0 else 0,
                'total_tech_volume': int(tech_df['tweet_volume'].sum()) if len(tech_df) > 0 else 0,
                'avg_significance': round(tech_df['significance_score'].mean(), 1) if len(tech_df) > 0 else 0,
                'top_subcategory': tech_df['subcategory'].mode().iloc[0] if len(tech_df) > 0 else None,
                'sentiment_distribution': tech_df['sentiment'].value_counts().to_dict() if len(tech_df) > 0 else {}
            }
        
        return regional_analysis
    
    def _calculate_velocity(self, current_df: pd.DataFrame, baseline_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate velocity metrics by comparing current vs baseline."""
        velocity_analysis = {
            'rising_topics': [],
            'declining_topics': [],
            'new_breakouts': [],
            'category_momentum': {}
        }
        
        # Create topic volume maps
        current_volumes = dict(zip(current_df['topic'], current_df['tweet_volume']))
        baseline_volumes = dict(zip(baseline_df['topic'], baseline_df['tweet_volume']))
        
        # Analyze topic velocity
        for topic, current_vol in current_volumes.items():
            if topic in baseline_volumes:
                baseline_vol = baseline_volumes[topic]
                velocity = ((current_vol - baseline_vol) / baseline_vol) * 100 if baseline_vol > 0 else 0
                
                if velocity > 50:  # 50% growth
                    velocity_analysis['rising_topics'].append({
                        'topic': topic,
                        'velocity': round(velocity, 1),
                        'current_volume': current_vol,
                        'baseline_volume': baseline_vol
                    })
                elif velocity < -30:  # 30% decline
                    velocity_analysis['declining_topics'].append({
                        'topic': topic,
                        'velocity': round(velocity, 1),
                        'current_volume': current_vol,
                        'baseline_volume': baseline_vol
                    })
            else:
                # New topic
                velocity_analysis['new_breakouts'].append({
                    'topic': topic,
                    'volume': current_vol,
                    'category': current_df[current_df['topic'] == topic]['category'].iloc[0]
                })
        
        return velocity_analysis
    
    def _identify_partnership_opportunities(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify potential partnership opportunities from cross-category trends."""
        partnerships = []
        
        # Look for non-tech topics with high volume that could have Web3 connections
        non_tech_df = df[df['category'] != 'Technology']
        high_volume_non_tech = non_tech_df[non_tech_df['tweet_volume'] > 100000]
        
        for _, topic in high_volume_non_tech.iterrows():
            # Check for Web3 partnership potential
            if topic['category'] in ['Entertainment', 'Sports', 'Culture/Social']:
                partnerships.append({
                    'topic': topic['topic'],
                    'category': topic['category'],
                    'volume': int(topic['tweet_volume']),
                    'partnership_type': self._suggest_partnership_type(topic['category'], topic['topic']),
                    'web3_angle': self._suggest_web3_angle(topic['category'], topic['topic'])
                })
        
        return partnerships[:5]  # Top 5 opportunities
    
    def _suggest_partnership_type(self, category: str, topic: str) -> str:
        """Suggest partnership type based on category and topic."""
        if category == 'Entertainment':
            return 'NFT Collection or Fan Token'
        elif category == 'Sports':
            return 'Fan Engagement Platform or Digital Collectibles'
        elif category == 'Culture/Social':
            return 'Community Token or Social Platform Integration'
        else:
            return 'Brand Collaboration or Sponsored Content'
    
    def _suggest_web3_angle(self, category: str, topic: str) -> str:
        """Suggest Web3 angle for partnership."""
        if 'fan' in topic.lower():
            return 'Fan token ecosystem with voting rights and exclusive content'
        elif 'concert' in topic.lower() or 'festival' in topic.lower():
            return 'NFT tickets with proof of attendance and collectible value'
        elif 'sport' in topic.lower():
            return 'Digital trading cards and fantasy sports integration'
        else:
            return 'Community-driven platform with token incentives'
    
    def _suggest_engagement_experiments(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest A/B testing experiments for niche but rising topics."""
        tech_df = df[df['category'] == 'Technology']
        
        if len(tech_df) == 0:
            return []
        
        # Find niche topics (lower volume but high significance)
        niche_topics = tech_df[
            (tech_df['tweet_volume'] < tech_df['tweet_volume'].median()) & 
            (tech_df['significance_score'] >= 7)
        ]
        
        experiments = []
        for _, topic in niche_topics.head(3).iterrows():
            experiments.append({
                'topic': topic['topic'],
                'subcategory': topic['subcategory'],
                'volume': int(topic['tweet_volume']),
                'significance': int(topic['significance_score']),
                'experiment_type': 'Niche Audience Test',
                'hypothesis': f"Tech-savvy audience will engage highly with {topic['subcategory']} content",
                'test_content': f"Educational thread about {topic['topic']} targeting crypto enthusiasts",
                'success_metrics': 'Engagement rate >5%, tech community mentions, follow-up questions'
            })
        
        return experiments
    
    def _analyze_hashtag_bridges(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze hashtag connections between tech and mainstream topics."""
        # This is a simplified version - in reality you'd need hashtag co-occurrence data
        bridges = []
        
        tech_topics = df[df['category'] == 'Technology']['topic'].tolist()
        non_tech_topics = df[df['category'] != 'Technology']['topic'].tolist()
        
        # Look for potential bridges (simplified heuristic)
        for tech_topic in tech_topics[:3]:
            for non_tech_topic in non_tech_topics[:5]:
                if any(word in tech_topic.lower() for word in ['ai', 'crypto', 'nft', 'blockchain']):
                    bridges.append({
                        'tech_topic': tech_topic,
                        'mainstream_topic': non_tech_topic,
                        'bridge_opportunity': f"Connect {tech_topic} to {non_tech_topic} audience",
                        'content_strategy': f"Show how {tech_topic} enhances {non_tech_topic} experience"
                    })
        
        return bridges[:3]
    
    def _analyze_market_timing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze optimal timing for different types of content."""
        tech_df = df[df['category'] == 'Technology']
        
        timing_analysis = {
            'immediate_opportunities': [],
            'watch_and_wait': [],
            'long_term_positioning': []
        }
        
        for _, topic in tech_df.iterrows():
            urgency = self._assess_urgency(topic['tweet_volume'], topic['significance_score'])
            
            timing_data = {
                'topic': topic['topic'],
                'subcategory': topic['subcategory'],
                'volume': int(topic['tweet_volume']),
                'significance': int(topic['significance_score'])
            }
            
            if urgency == 'Critical':
                timing_analysis['immediate_opportunities'].append(timing_data)
            elif urgency == 'High':
                timing_analysis['watch_and_wait'].append(timing_data)
            else:
                timing_analysis['long_term_positioning'].append(timing_data)
        
        return timing_analysis
    
    def _assess_urgency(self, volume: int, significance: int) -> str:
        """Assess posting urgency based on volume and significance."""
        if volume > 500000 and significance >= 8:
            return 'Critical'
        elif volume > 200000 and significance >= 6:
            return 'High'
        elif volume > 50000 and significance >= 5:
            return 'Medium'
        else:
            return 'Low'
    
    def _determine_posting_window(self, volume: int, sentiment: str) -> str:
        """Determine optimal posting window."""
        if volume > 1000000:
            return '⚡ NOW - Peak viral moment'
        elif volume > 500000:
            return '🚨 Within 2 hours - High momentum'
        elif volume > 200000:
            return '⏰ Within 6 hours - Good engagement window'
        elif sentiment == 'positive':
            return '📈 Within 24 hours - Positive sentiment window'
        else:
            return '🔍 Monitor and time carefully'
    
    def save_playbook(self, playbook: Dict[str, Any], session_dir: Path) -> str:
        """Save the playbook to a markdown file."""
        reports_dir = session_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        playbook_path = reports_dir / f"web3_playbook_{timestamp}.md"
        
        # Generate markdown report
        markdown_content = self._generate_markdown_report(playbook)
        
        with open(playbook_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Web3 playbook saved to {playbook_path}")
        return str(playbook_path)
    
    def _generate_markdown_report(self, playbook: Dict[str, Any]) -> str:
        """Generate a formatted markdown report from the playbook data."""
        report = f"""# 🚀 Web3 & Tech Posting Playbook
*Generated: {playbook['timestamp']}*

## 📊 Executive Summary
- **Total Topics Analyzed**: {playbook['total_topics']}
- **Tech/Web3 Topics**: {playbook['tech_topics']}
- **Tech Share**: {playbook['web3_breakdown']['tech_share_of_total']}%
- **Total Tech Volume**: {playbook['web3_breakdown']['total_tech_volume']:,} tweets

## 🎯 Priority Posting List (Next 4 Hours)
"""
        
        if playbook['priority_posting_list']:
            for i, topic in enumerate(playbook['priority_posting_list'][:5], 1):
                report += f"""
### {i}. {topic['topic']} ({topic['region']})
- **Category**: {topic['subcategory']}
- **Volume**: {topic['volume']:,} tweets
- **Significance**: {topic['significance']}/10
- **Sentiment**: {topic['sentiment']} 
- **Urgency**: {topic['urgency']}
- **Posting Window**: {topic['posting_window']}
"""
        else:
            report += "\n*No high-priority tech topics detected in current data.*\n"

        report += f"""
## 🎨 Content Angle Suggestions
"""
        
        if playbook['content_angle_suggestions']:
            for content in playbook['content_angle_suggestions'][:3]:
                report += f"""
### {content['topic']} ({content['subcategory']})
**Volume**: {content['volume']:,} tweets
"""
                for angle in content['angles']:
                    report += f"- {angle}\n"
        else:
            report += "\n*No tech content angles available.*\n"
        
        report += f"""
## 📈 Significance vs Volume Matrix

### 🚨 Must Post Now (High Volume + High Significance)
"""
        if playbook['significance_volume_matrix']['must_post_now']:
            for topic in playbook['significance_volume_matrix']['must_post_now'][:3]:
                report += f"- **{topic['topic']}** - {topic['volume']:,} tweets, {topic['significance']}/10 significance\n"
        else:
            report += "*No critical posting opportunities detected.*\n"
        
        report += f"""
### 🔍 Niche Leadership Opportunities (High Significance + Lower Volume)
"""
        if playbook['significance_volume_matrix']['niche_leadership']:
            for topic in playbook['significance_volume_matrix']['niche_leadership'][:3]:
                report += f"- **{topic['topic']}** - {topic['volume']:,} tweets, {topic['significance']}/10 significance\n"
        else:
            report += "*No niche leadership opportunities detected.*\n"
        
        if playbook['velocity_analysis']:
            report += f"""
## 🚀 Velocity Analysis
### Rising Topics (+50% growth)
"""
            if playbook['velocity_analysis']['rising_topics']:
                for topic in playbook['velocity_analysis']['rising_topics'][:3]:
                    report += f"- **{topic['topic']}** - {topic['velocity']}% growth ({topic['current_volume']:,} tweets)\n"
            else:
                report += "*No rising topics detected.*\n"
            
            report += f"""
### New Breakouts
"""
            if playbook['velocity_analysis']['new_breakouts']:
                for topic in playbook['velocity_analysis']['new_breakouts'][:3]:
                    report += f"- **{topic['topic']}** - {topic['volume']:,} tweets (NEW)\n"
            else:
                report += "*No new breakout topics detected.*\n"
        
        report += f"""
## 🤝 Partnership Opportunities
"""
        if playbook['partnership_signals']:
            for partnership in playbook['partnership_signals'][:3]:
                report += f"""
### {partnership['topic']} ({partnership['category']})
- **Volume**: {partnership['volume']:,} tweets
- **Partnership Type**: {partnership['partnership_type']}
- **Web3 Angle**: {partnership['web3_angle']}
"""
        else:
            report += "\n*No partnership opportunities detected.*\n"
        
        report += f"""
## 🧪 Engagement Experiments
"""
        if playbook['engagement_experiments']:
            for experiment in playbook['engagement_experiments']:
                report += f"""
### {experiment['topic']} Test
- **Hypothesis**: {experiment['hypothesis']}
- **Test Content**: {experiment['test_content']}
- **Success Metrics**: {experiment['success_metrics']}
"""
        else:
            report += "\n*No engagement experiments suggested.*\n"
        
        report += f"""
## 🌍 Regional Performance
"""
        for region, data in playbook['regional_insights'].items():
            report += f"""
### {region}
- **Tech Topics**: {data['tech_topics']} ({data['tech_share']}% of total)
- **Total Tech Volume**: {data['total_tech_volume']:,} tweets
- **Top Subcategory**: {data['top_subcategory']}
"""
        
        return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Web3 posting playbook from trending analysis")
    parser.add_argument("--session-id", required=True, help="Session ID to analyze (e.g., '021')")
    parser.add_argument("--baseline-session", help="Optional baseline session for velocity analysis")
    args = parser.parse_args()
    
    # Find the analysis CSV for the session
    session_path = Path(f"data/session_{args.session_id}")
    try:
        analysis_csv = next((session_path / "analysis").glob("trending_analysis_*.csv"))
    except StopIteration:
        logger.error(f"No analysis CSV found for session {args.session_id}")
        sys.exit(1)
    
    # Find baseline CSV if specified
    baseline_csv = None
    if args.baseline_session:
        baseline_path = Path(f"data/session_{args.baseline_session}")
        try:
            baseline_csv = next((baseline_path / "analysis").glob("trending_analysis_*.csv"))
        except StopIteration:
            logger.warning(f"No baseline CSV found for session {args.baseline_session}")
    
    # Generate playbook
    generator = Web3PlaybookGenerator()
    playbook = generator.generate_playbook(str(analysis_csv), str(baseline_csv) if baseline_csv else None)
    
    # Save playbook
    playbook_path = generator.save_playbook(playbook, session_path)
    
    print(f"✅ Web3 Playbook generated successfully!")
    print(f"📄 Saved to: {playbook_path}")
    print(f"🎯 Priority topics: {len(playbook['priority_posting_list'])}")
    print(f"🤝 Partnership opportunities: {len(playbook['partnership_signals'])}") 