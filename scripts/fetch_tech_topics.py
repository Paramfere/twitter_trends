#!/usr/bin/env python3
# mypy: ignore-errors
# ruff: noqa

"""
Technology and Web3 Topics Fetcher - Fetch only tech and web3 related trending topics.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import argparse
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fetchers.twitter_topic_seeder import fetch_topics_multi
from fetchers.rule_categorizer import RuleBasedCategorizer
from scripts.session_manager import SessionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_tech_web3_topics(df: pd.DataFrame, categorizer: RuleBasedCategorizer) -> pd.DataFrame:
    """Filter dataframe to keep only Technology and Web3 related topics.
    
    Criteria:
    1. Category must be "Technology" OR
    2. `web3_relevance` is flagged (low/medium/high).  
    This excludes generic political/news topics that only have a crypto influencer connection.
    """
    
    if df.empty:
        return df.copy()
    
    # Normalise web3_relevance values to lowercase strings for safety
    df['web3_relevance'] = df['web3_relevance'].astype(str).str.lower()
    
    tech_mask = df['category'] == 'Technology'
    web3_mask = df['web3_relevance'] != 'none'
    
    filtered_df = df.loc[tech_mask | web3_mask].copy()
    
    # Remove obvious non-tech categories that slipped through (safety net)
    disallowed_categories = [
        'Politics', 'Sports', 'Entertainment', 'News/Events', 'Global/Places',
        'Daily/Lifestyle', 'Culture/Social'
    ]
    filtered_df = filtered_df[~filtered_df['category'].isin(disallowed_categories)].copy()
    
    # Drop duplicates, sort, and reset index for readability
    filtered_df.drop_duplicates(subset=['topic'], inplace=True)
    filtered_df.sort_values(by='significance_score', ascending=False, inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)
    
    return filtered_df

def generate_tech_report(df: pd.DataFrame, timestamp: str) -> str:
    """Generate a tech-focused report."""
    
    total_topics = len(df)
    total_volume = df['tweet_volume'].sum()
    
    # Tech subcategory breakdown
    subcategory_stats = df['subcategory'].value_counts()
    
    # Web3 relevance breakdown
    web3_relevance_stats = df['web3_relevance'].value_counts()
    
    report_lines = [
        "=" * 80,
        f"TECHNOLOGY & WEB3 TOPICS REPORT - {timestamp}",
        "=" * 80,
        "",
        "📊 OVERVIEW:",
        f"  • Total Tech/Web3 Topics: {total_topics:,}",
        f"  • Total Tweet Volume: {total_volume:,}",
        f"  • Average Significance Score: {df['significance_score'].mean():.1f}/10",
        "",
        "🔧 TECHNOLOGY SUBCATEGORIES:",
    ]
    
    for subcategory, count in subcategory_stats.items():
        percentage = (count / total_topics) * 100
        report_lines.append(f"  • {subcategory}: {count} topics ({percentage:.1f}%)")
    
    report_lines.extend([
        "",
        "🌐 WEB3 RELEVANCE:",
    ])
    
    for relevance, count in web3_relevance_stats.items():
        if isinstance(relevance, str):
            relevance_title = relevance.title()
        else:
            relevance_title = str(relevance)
        percentage = (count / total_topics) * 100
        report_lines.append(f"  • {relevance_title}: {count} topics ({percentage:.1f}%)")
    
    # Top tech topics
    top_topics = df.head(10)
    
    report_lines.extend([
        "",
        "🔝 TOP TECHNOLOGY & WEB3 TOPICS:",
    ])
    
    for _, topic in top_topics.iterrows():
        report_lines.append(
            f"  • {topic['topic']} ({topic['region']}) - "
            f"Score: {topic['significance_score']}/10, {topic['tweet_volume']:,} tweets"
        )
        report_lines.append(f"    {topic['context']}")
        report_lines.append(f"    Web3 Relevance: {topic['web3_relevance']}, Crypto Connection: {topic['crypto_connection']}")
    
    report_lines.extend([
        "",
        "🔍 ANALYSIS NOTES:",
        "  • Only Technology category and Web3-related topics are included",
        "  • Topics sorted by significance score",
        "  • Web3 relevance indicates blockchain/crypto relationship strength",
        "",
        f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80
    ])
    
    return "\n".join(report_lines)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Fetch only technology and web3 related trending topics")
    parser.add_argument("--with-content-analysis", action="store_true",
                        help="Include content analysis scraping (slow & extra API calls).")
    parsed_args = parser.parse_args()

    try:
        # Initialize session manager
        session_manager = SessionManager()
        session_name, session_dir = session_manager.create_new_session()

        logger.info(f"🚀 Starting Technology & Web3 topics fetch – {session_name}")
        logger.info(f"📂 Session directory: {session_dir}")

        categorizer = RuleBasedCategorizer()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        raw_file = session_dir / "raw_data" / f"{timestamp}_trending_topics.csv"
        tech_file = session_dir / "analysis" / f"{timestamp}_tech_topics.csv"
        
        # Ensure directories exist
        raw_file.parent.mkdir(exist_ok=True, parents=True)
        tech_file.parent.mkdir(exist_ok=True, parents=True)

        # Fetch trending topics
        logger.info("📡 Fetching trending topics…")
        raw_topics = fetch_topics_multi()
        raw_df = pd.DataFrame(raw_topics)
        
        # Save raw data
        raw_df.to_csv(raw_file, index=False)
        logger.info(f"💾 Raw data saved: {raw_file}")

        # Categorize topics
        logger.info("🏷️ Categorizing topics…")
        enhanced_df = pd.DataFrame()
        
        for region in raw_df['region'].unique():
            region_data = raw_df[raw_df['region'] == region].copy()
            topics_for_analysis = []
            
            for _, row in region_data.iterrows():
                topics_for_analysis.append({
                    'topic': row['topic'],
                    'tweet_volume': row['tweet_volume'],
                    'region': region,
                    'url': row.get('url', ''),
                    'fetched_at': row.get('fetched_at', '')
                })
            
            analysis_results = categorizer.categorize_topics(topics_for_analysis)
            
            for _, row in region_data.iterrows():
                topic_name = row['topic']
                analysis = analysis_results.get(topic_name)  # type: ignore[arg-type]
                
                if analysis:
                    enhanced_row = {
                        'region': region,
                        'topic': topic_name,
                        'tweet_volume': row['tweet_volume'],
                        'category': analysis.category,
                        'subcategory': analysis.subcategory,
                        'significance_score': analysis.significance_score,
                        'sentiment': analysis.sentiment,
                        'context': analysis.context,
                        'trending_reason': analysis.trending_reason,
                        'confidence': round(analysis.confidence, 2),
                        'web3_relevance': analysis.web3_relevance,
                        'crypto_connection': analysis.crypto_connection,
                        'tech_relationship': analysis.tech_relationship,
                        'url': row.get('url', ''),
                        'fetched_at': row.get('fetched_at', datetime.now().isoformat())
                    }
                    
                    enhanced_df = pd.concat([enhanced_df, pd.DataFrame([enhanced_row])], ignore_index=True)
        
        # Filter for tech and web3 topics
        tech_df = filter_tech_web3_topics(enhanced_df, categorizer)
        
        # Save tech topics
        tech_df.to_csv(tech_file, index=False)
        logger.info(f"💾 Tech topics saved: {tech_file}")
        
        # Save copies to latest directory for quick access
        try:
            import shutil
            latest_dir = Path("data/latest")
            latest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tech_file, latest_dir / "tech_topics.csv")
            logger.info("📂 Latest tech topics copied to data/latest/tech_topics.csv")
        except Exception as e:
            logger.warning(f"⚠️ Could not copy tech topics to latest: {e}")
        
        # Generate tech report
        report_text = generate_tech_report(tech_df, timestamp)
        report_file = session_dir / "analysis" / f"{timestamp}_tech_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        logger.info(f"📊 Tech report generated: {report_file}")
        
        # Copy latest report
        try:
            import shutil
            latest_dir = Path("data/latest")
            latest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(report_file, latest_dir / "tech_report.txt")
            logger.info("📂 Latest tech report copied to data/latest/tech_report.txt")
        except Exception as e:
            logger.warning(f"⚠️ Could not copy tech report to latest: {e}")
        
        # Run content analysis if requested
        if parsed_args.with_content_analysis:
            logger.info("📱 Running content analysis on tech topics...")
            try:
                from scripts.content_analysis_engine import AntiGamingContentEngine
                engine = AntiGamingContentEngine()
                report_path = engine.run_tech_content_analysis(session_dir, tech_df)
                if report_path:
                    logger.info(f"✅ Tech Content Analysis: {report_path}")
                else:
                    logger.warning("⚠️ Tech content analysis completed but no report was generated")
            except Exception as e:
                logger.error(f"❌ Error in tech content analysis: {e}")
        
        logger.info("🎉 Tech topics fetch complete")
        
    except Exception as e:
        logger.error(f"Error in tech topics fetch: {e}")
        raise

if __name__ == "__main__":
    main() 