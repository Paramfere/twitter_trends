#!/usr/bin/env python3

"""
Twitter Topics Fetcher - Fetch trending topics using Apify actors.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fetchers.twitter_topic_seeder import fetch_topics_multi
from fetchers.rule_categorizer import RuleBasedCategorizer
from scripts.session_manager import SessionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_rule_based_summary(df: pd.DataFrame, categorizer: RuleBasedCategorizer) -> pd.DataFrame:
    """Create enhanced summary with rule-based categorization."""
    
    # Group by region for processing
    enhanced_rows = []
    
    for region in df['region'].unique():
        region_data = df[df['region'] == region].copy()
        region_data = region_data.sort_values(by='tweet_volume', ascending=False).reset_index(drop=True)
        
        # Prepare topics for analysis
        topics_for_analysis = []
        for _, row in region_data.iterrows():
            topics_for_analysis.append({
                'topic': row['topic'],
                'tweet_volume': row['tweet_volume'],
                'region': region,
                'url': row.get('url', ''),
                'fetched_at': row.get('fetched_at', '')
            })
        
        # Get rule-based analysis
        analysis_results = categorizer.categorize_topics(topics_for_analysis)
        
        # Create enhanced rows
        for idx, (_, row) in enumerate(region_data.iterrows()):
            topic_name = row['topic']
            analysis = analysis_results.get(topic_name)
            
            if analysis:
                enhanced_row = {
                    'region': region,
                    'rank': idx + 1,
                    'topic': topic_name,
                    'tweet_volume': row['tweet_volume'],
                    'category': analysis.category,
                    'subcategory': analysis.subcategory,
                    'significance_score': analysis.significance_score,
                    'sentiment': analysis.sentiment,
                    'context': analysis.context,
                    'trending_reason': analysis.trending_reason,
                    'confidence': round(analysis.confidence, 2),
                    'url': row.get('url', ''),
                    'window': row.get('window', 'Live'),
                    'fetched_at': row.get('fetched_at', datetime.now().isoformat())
                }
            else:
                # Fallback for missing analysis
                enhanced_row = {
                    'region': region,
                    'rank': idx + 1,
                    'topic': topic_name,
                    'tweet_volume': row['tweet_volume'],
                    'category': 'Unknown',
                    'subcategory': 'Uncategorized',
                    'significance_score': 1,
                    'sentiment': 'neutral',
                    'context': f"Topic with {row['tweet_volume']:,} tweets",
                    'trending_reason': 'General social media engagement',
                    'confidence': 0.0,
                    'url': row.get('url', ''),
                    'window': row.get('window', 'Live'),
                    'fetched_at': row.get('fetched_at', datetime.now().isoformat())
                }
            
            enhanced_rows.append(enhanced_row)
    
    return pd.DataFrame(enhanced_rows)

def generate_analysis_report(df: pd.DataFrame, timestamp: str) -> str:
    """Generate a comprehensive text report of the analysis."""
    
    total_topics = len(df)
    total_volume = df['tweet_volume'].sum()
    regions = df['region'].unique()
    
    # Category breakdown
    category_stats = df['category'].value_counts()
    sentiment_stats = df['sentiment'].value_counts()
    
    # Top topics by region
    report_lines = [
        "=" * 80,
        f"TRENDING TOPICS ANALYSIS REPORT - {timestamp}",
        "=" * 80,
        "",
        "📊 OVERVIEW:",
        f"  • Total Topics Analyzed: {total_topics:,}",
        f"  • Total Tweet Volume: {total_volume:,}",
        f"  • Regions Covered: {', '.join(regions)}",
        f"  • Average Significance Score: {df['significance_score'].mean():.1f}/10",
        "",
        "📈 CATEGORY BREAKDOWN:",
    ]
    
    for category, count in category_stats.head(10).items():
        percentage = (count / total_topics) * 100
        avg_volume = df[df['category'] == category]['tweet_volume'].mean()
        report_lines.append(f"  • {category}: {count} topics ({percentage:.1f}%) - Avg: {avg_volume:,.0f} tweets")
    
    report_lines.extend([
        "",
        "💭 SENTIMENT ANALYSIS:",
    ])
    
    for sentiment, count in sentiment_stats.items():
        percentage = (count / total_topics) * 100
        report_lines.append(f"  • {sentiment.title()}: {count} topics ({percentage:.1f}%)")
    
    # Regional highlights
    for region in regions:
        region_df = df[df['region'] == region]
        top_topics = region_df.head(5)
        
        report_lines.extend([
            "",
            f"🌍 {region} REGION HIGHLIGHTS:",
        ])
        
        for _, topic in top_topics.iterrows():
            report_lines.append(
                f"  • #{topic['rank']}: {topic['topic']} "
                f"({topic['tweet_volume']:,} tweets, {topic['category']}, Score: {topic['significance_score']}/10)"
            )
    
    # High significance topics
    high_sig_topics = df[df['significance_score'] >= 7].sort_values('significance_score', ascending=False)
    
    if not high_sig_topics.empty:
        report_lines.extend([
            "",
            "⭐ HIGH SIGNIFICANCE TOPICS (Score ≥ 7):",
        ])
        
        for _, topic in high_sig_topics.head(10).iterrows():
            report_lines.append(
                f"  • {topic['topic']} ({topic['region']}) - "
                f"Score: {topic['significance_score']}/10, {topic['tweet_volume']:,} tweets"
            )
            report_lines.append(f"    {topic['context']}")
    
    report_lines.extend([
        "",
        "🔍 ANALYSIS NOTES:",
        "  • Categories determined using rule-based keyword matching",
        "  • Significance scores based on tweet volume and category importance",
        "  • Sentiment analysis uses keyword patterns and context clues",
        "  • Confidence scores indicate categorization reliability",
        "",
        f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80
    ])
    
    return "\n".join(report_lines)

def main():
    """Main execution function."""
    # argparse moved to top-level so we can check flags early
    parser = argparse.ArgumentParser(description="Fetch trending topics and generate reports")
    parser.add_argument("--with-content-analysis", action="store_true",
                        help="Include content analysis scraping (slow & extra API calls).")
    parsed_args = parser.parse_args()

    try:
        # Initialize session manager
        session_manager = SessionManager()
        session_name, session_dir = session_manager.create_new_session()

        logger.info(f"🚀 Starting Twitter topics fetch – {session_name}")
        logger.info(f"📂 Session directory: {session_dir}")

        categorizer = RuleBasedCategorizer()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        raw_file = session_dir / "raw_data" / f"trending_topics_{timestamp}.csv"
        analysis_file = session_dir / "analysis" / f"trending_analysis_{timestamp}.csv"
        report_file = session_dir / "analysis" / f"trending_report_{timestamp}.txt"

        logger.info("📡 Fetching trending topics…")
        all_data = fetch_topics_multi()
        if not all_data:
            logger.error("❌ No trending topics fetched")
            return

        topics_df = pd.DataFrame(all_data)
        topics_df.to_csv(raw_file, index=False)
        logger.info(f"💾 Raw data saved: {raw_file}")

        logger.info("🏷️ Categorizing topics…")
        analysis_df = create_rule_based_summary(topics_df, categorizer)
        analysis_df.to_csv(analysis_file, index=False)
        logger.info(f"💾 Analysis saved: {analysis_file}")

        logger.info("📊 Generating plain-text analysis report…")
        report_content = generate_analysis_report(analysis_df, timestamp)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        # -------- Optional heavy/paid reports below --------
        logger.info("📋 Generating supplemental reports…")

        # 1. Intelligence Report
        try:
            from scripts.intelligence_report_generator import IntelligenceReportGenerator
            logger.info("🧠 Intelligence Report…")
            intel_path = IntelligenceReportGenerator().generate_report(str(analysis_file))
            logger.info(f"✅ Intelligence Report: {intel_path}")
        except Exception as e:
            logger.warning(f"⚠️ Intelligence Report failed: {e}")

        # 2. Velocity Report
        try:
            from scripts.velocity_report_generator import VelocityReportGenerator
            logger.info("🚀 Velocity Report…")
            vel_gen = VelocityReportGenerator()
            vel_report = vel_gen.generate_velocity_report(str(analysis_file), session_name)
            vel_path = vel_gen.save_velocity_report(vel_report, session_dir)
            logger.info(f"✅ Velocity Report: {vel_path}")
        except Exception as e:
            logger.warning(f"⚠️ Velocity Report failed: {e}")

        # 3. Web3 Playbook
        try:
            from scripts.web3_playbook_generator import Web3PlaybookGenerator
            logger.info("🌐 Web3 Playbook…")
            web3_gen = Web3PlaybookGenerator()
            playbook = web3_gen.generate_playbook(str(analysis_file))
            playbook_path = web3_gen.save_playbook(playbook, session_dir)
            logger.info(f"✅ Web3 Playbook: {playbook_path}")
        except Exception as e:
            logger.warning(f"⚠️ Web3 Playbook failed: {e}")

        # 4. Content Analysis – ONLY if flag supplied
        if parsed_args.with_content_analysis:
            try:
                from scripts.content_analysis_engine import AntiGamingContentEngine
                logger.info("📱 Content Analysis (Kaito)…")
                report_path = AntiGamingContentEngine().run_content_analysis(session_dir)
                logger.info(f"✅ Content Analysis: {report_path}")
            except Exception as e:
                logger.warning(f"⚠️ Content Analysis failed: {e}")
        else:
            logger.info("ℹ️ Skipping Content Analysis (use --with-content-analysis to enable)")

        logger.info("🎉 Fetch session complete")

    except Exception as exc:
        logger.error(f"❌ Fatal error: {exc}")
        raise

if __name__ == "__main__":
    main() 