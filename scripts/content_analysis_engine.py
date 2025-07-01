#!/usr/bin/env python3

"""
Advanced Content Analysis Engine - Anti-Gaming Tweet Scraper
Implements sophisticated quality metrics to filter out manipulated content
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import pandas as pd
import json
import re
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required packages
try:
    from apify_client import ApifyClient
    from dotenv import load_dotenv
    load_dotenv()
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Run: pip install apify-client python-dotenv")
    sys.exit(1)

def get_logger():
    """Get project logger."""
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

logger = get_logger()

class AntiGamingContentEngine:
    """Advanced content analysis engine with anti-gaming metrics."""
    
    HIGH_ENGAGEMENT_THRESHOLD = 5000  # Reduced from 8000 to 5000
    MAX_TOPICS = 5  # only analyse top 5 trending topics
    TWEETS_PER_TOPIC = 10  # fetch max 10 tweets per topic per API call
    # Additional hard filters
    MIN_AUTHOR_FOLLOWERS = 10_000
    MIN_FOLLOWER_RATIO = 2.0  # followers / following
    MIN_BOOKMARKS = 300  # Reduced from 500 to 300
    
    def __init__(self):
        """Initialize the Anti-Gaming Content Analysis Engine."""
        self.client = ApifyClient(os.getenv("APIFY_TOKEN"))
        self.total_tweets_limit = 10  # Total tweets across all trends
        
        # Crypto keywords for automatic inclusion
        self.crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency', 
            'blockchain', 'defi', 'nft', 'web3', 'dao', 'dapp', 'altcoin',
            'solana', 'cardano', 'polygon', 'chainlink', 'uniswap', 'binance'
        ]
        
        # Quality thresholds for general content
        self.quality_thresholds = {
            'min_followers': 10000,
            'min_engagement_per_tweet': 100,
            'min_follower_ratio': 0.5,  # followers / following ratio
            'min_account_age_days': 180,  # 6 months
            'max_following_count': 50000,  # Avoid follow-for-follow accounts
        }
    
    def is_crypto_related(self, topic: str) -> bool:
        """Check if a topic is crypto-related."""
        topic_lower = topic.lower()
        return any(keyword in topic_lower for keyword in self.crypto_keywords)
    
    def calculate_quality_score(self, author_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate quality score for an author to detect gaming."""
        metrics = {}
        score = 0.0
        
        try:
            followers = author_data.get('followers', 0)
            following = author_data.get('following', 0)
            verified = author_data.get('isVerified', False)
            blue_verified = author_data.get('isBlueVerified', False)
            created_at = author_data.get('createdAt', '')
            
            # 1. Follower-to-Following Ratio (30% of score)
            if following > 0:
                follower_ratio = followers / following
                metrics['follower_ratio'] = round(follower_ratio, 3)
                if follower_ratio >= 2.0:
                    score += 30
                elif follower_ratio >= 1.0:
                    score += 20
                elif follower_ratio >= 0.5:
                    score += 10
            else:
                metrics['follower_ratio'] = float('inf')
                score += 25  # No following is actually good
            
            # 2. Account Age (20% of score)
            if created_at:
                try:
                    account_date = datetime.strptime(created_at.split()[1:4], '%b %d %Y')
                    days_old = (datetime.now() - account_date).days
                    metrics['account_age_days'] = days_old
                    if days_old >= 365:
                        score += 20
                    elif days_old >= 180:
                        score += 15
                    elif days_old >= 90:
                        score += 10
                except:
                    metrics['account_age_days'] = 0
            
            # 3. Verification Status (20% of score)
            if verified:
                score += 20
                metrics['verification_type'] = 'legacy_verified'
            elif blue_verified:
                score += 15
                metrics['verification_type'] = 'blue_verified'
            else:
                metrics['verification_type'] = 'unverified'
            
            # 4. Follower Count Legitimacy (15% of score)
            if 1000 <= followers <= 1000000:
                score += 15
            elif followers > 1000000:
                score += 10  # Very high might be celebrity/brand
            elif followers >= 500:
                score += 8
            
            # 5. Following Count Reasonableness (15% of score)
            if following <= 5000:
                score += 15
            elif following <= 10000:
                score += 10
            elif following <= 25000:
                score += 5
            
            metrics['quality_score'] = round(score, 2)
            metrics['followers'] = followers
            metrics['following'] = following
            
        except Exception as e:
            logger.warning(f"Error calculating quality score: {e}")
            metrics['quality_score'] = 0.0
        
        return score, metrics
    
    def save_kaito_data(self, items: List[Dict], topic: str, content_type: str) -> None:
        """Save raw Kaito data and summary."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Save raw data
        raw_file = data_dir / f"kaito_raw_data_{timestamp}.json"
        with open(raw_file, 'w') as f:
            json.dump(items, f, indent=2)
        
        # Create summary
        summary_data = []
        for item in items:
            if isinstance(item, dict):
                summary = {
                    'tweet_id': item.get('id'),
                    'text': item.get('text', '')[:100] + '...',  # First 100 chars
                    'engagement': item.get('likeCount', 0) + item.get('retweetCount', 0),
                    'author': item.get('author', {}).get('userName', ''),
                    'quality_score': item.get('author_quality_score', 0)
                }
                summary_data.append(summary)
        
        # Save summary
        summary_file = data_dir / f"kaito_summary_{timestamp}.csv"
        pd.DataFrame(summary_data).to_csv(summary_file, index=False)
        
        logger.info(f"Saved Kaito data for {topic}")

    def _process_tweets(self, items: List[Dict], topic: str, content_type: str) -> List[Dict[str, Any]]:
        """Process raw Kaito tweets and calculate quality metrics."""
        processed_tweets = []
        
        for item in items:
            if not isinstance(item, dict):
                continue
                
            # Extract author data with Kaito's field names
            author_data = item.get('author', {})
            
            # Calculate quality score using correct Kaito field mapping
            quality_score, quality_metrics = self.calculate_quality_score({
                'userName': author_data.get('userName', ''),  # Kaito uses camelCase
                'name': author_data.get('name', ''),
                'followers': author_data.get('followers', 0),  # Direct field
                'following': author_data.get('following', 0),  # Direct field
                'isVerified': author_data.get('isVerified', False),
                'isBlueVerified': author_data.get('isBlueVerified', False),
                'createdAt': author_data.get('createdAt', '')
            })
            
            # Process media data
            media_types = []
            media_count = 0
            if 'extendedEntities' in item and 'media' in item['extendedEntities']:
                media = item['extendedEntities']['media']
                media_count = len(media)
                media_types = [m.get('type', '') for m in media if 'type' in m]
            
            # Create processed tweet object
            processed_tweet = {
                'trend_topic': topic,
                'content_type': content_type,
                'tweet_id': item.get('id', ''),
                'tweet_url': item.get('url', ''),
                'text': item.get('text', ''),
                'created_at': item.get('createdAt', ''),
                'language': item.get('language', ''),
                'like_count': item.get('likeCount', 0),
                'retweet_count': item.get('retweetCount', 0),
                'reply_count': item.get('replyCount', 0),
                'quote_count': item.get('quoteCount', 0),
                'bookmark_count': item.get('bookmarkCount', 0),
                'view_count': item.get('viewCount', 0),
                'author_username': author_data.get('userName', ''),
                'author_name': author_data.get('name', ''),
                'author_followers': author_data.get('followers', 0),
                'author_following': author_data.get('following', 0),
                'author_verified': author_data.get('isVerified', False),
                'author_blue_verified': author_data.get('isBlueVerified', False),
                'author_created_at': author_data.get('createdAt', ''),
                'author_quality_score': quality_score,
                'has_media': bool(media_count),
                'media_count': media_count,
                'media_types': media_types,
                'scraped_at': datetime.now().isoformat(),
                'is_crypto_related': self.is_crypto_related(topic),
                'quality_metrics': quality_metrics
            }
            
            processed_tweets.append(processed_tweet)
        
        logger.info(f"Processed {len(processed_tweets)}/{len(items)} tweets for topic '{topic}'")
        return processed_tweets

    def scrape_crypto_tweets(self, topic: str, session_dir: Path) -> List[Dict]:
        """Scrape crypto-related tweets for a given topic."""
        logger.info(f"🔗 Scraping crypto tweets for: {topic}")
        
        # Use Apify token from environment
        apify_token = os.getenv("APIFY_TOKEN")
        if not apify_token:
            logger.warning("No APIFY_TOKEN found in environment")
            return []
            
        logger.info(f"Using Apify token: {apify_token}")
        
        # Set up Apify client
        client = ApifyClient(apify_token)
        
        # Configure the actor input
        run_input = {
            "searchTerms": [topic],
            "lang": "en",
            "maxItems": self.TWEETS_PER_TOPIC,
            "maxItemsPerPage": self.TWEETS_PER_TOPIC,
            "queryType": "Top",
            "filter:has_engagement": True,
            "min_retweets": 3000,
            "min_faves": 5000,
            "min_bookmarks": 500,
            "filter:blue_verified": False,
            "filter:nativeretweets": False,
            "include:nativeretweets": False,
            "filter:replies": False,
            "filter:quote": False,
        }
        
        # Run the actor and wait for it to finish
        start_time = time.time()
        run = client.actor("kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest").call(run_input=run_input, timeout_secs=300)
        
        logger.info(f"Got Apify run response: {run}")
        
        # Fetch and process the results
        items = client.dataset(run["defaultDatasetId"]).list_items().items
        logger.info(f"Got {len(items)} items from Kaito API (requested {self.TWEETS_PER_TOPIC})")
        
        # Process tweets
        processed_tweets = []
        for i, tweet in enumerate(items):
            # Add the search term to each tweet
            tweet['search_term'] = topic
            processed_tweets.append(tweet)
            
        logger.info(f"Processed {len(processed_tweets)}/{len(items)} tweets for topic '{topic}'")
        
        end_time = time.time()
        logger.info(f"Processed {len(processed_tweets)} crypto tweets for {topic}")
        logger.info(f"Crypto scraping completed in {end_time - start_time:.1f}s")
        
        return processed_tweets

    def scrape_general_tweets(self, topic: str, session_dir: Path) -> List[Dict[str, Any]]:
        """Scrape general tweets with more selective filtering."""
        logger.info(f"🔗 Scraping general tweets for: {topic}")
        
        # Use Apify token from environment
        apify_token = os.getenv("APIFY_TOKEN")
        if not apify_token:
            logger.warning("No APIFY_TOKEN found in environment")
            return []
            
        logger.info(f"Using Apify token: {apify_token}")
        
        # Set up Apify client
        client = ApifyClient(apify_token)
        
        # Configure the actor input
        run_input = {
            "searchTerms": [topic],
            "lang": "en",
            "maxItems": self.TWEETS_PER_TOPIC,
            "maxItemsPerPage": self.TWEETS_PER_TOPIC,
            "queryType": "Top",
            "filter:has_engagement": True,
            "min_retweets": 3000,
            "min_faves": 5000,
            "min_bookmarks": 500,
            "filter:blue_verified": False,
            "filter:nativeretweets": False,
            "include:nativeretweets": False,
            "filter:replies": False,
            "filter:quote": False,
        }
        
        # Run the actor and wait for it to finish
        start_time = time.time()
        run = client.actor("kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest").call(run_input=run_input, timeout_secs=300)
        
        logger.info(f"Got Apify run response: {run}")
        
        # Fetch and process the results
        items = client.dataset(run["defaultDatasetId"]).list_items().items
        logger.info(f"Got {len(items)} items from Kaito API (requested {self.TWEETS_PER_TOPIC})")
        
        # Process tweets
        processed_tweets = []
        for i, tweet in enumerate(items):
            # Add the search term to each tweet
            tweet['search_term'] = topic
            processed_tweets.append(tweet)
            
        logger.info(f"Processed {len(processed_tweets)}/{len(items)} tweets for topic '{topic}'")
        
        end_time = time.time()
        logger.info(f"Processed {len(processed_tweets)} general tweets for {topic}")
        logger.info(f"General scraping completed in {end_time - start_time:.1f}s")
        
        return processed_tweets
    
    def identify_priority_trends(self, analysis_csv_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Separate crypto and general trends based on content."""
        try:
            df = pd.read_csv(analysis_csv_path)
            
            # Focus only on the top N topics by significance score (fallback to volume)
            sort_col = "significance_score" if "significance_score" in df.columns else "tweet_volume"
            df = df.sort_values(sort_col, ascending=False).head(self.MAX_TOPICS).reset_index(drop=True)
            
            crypto_trends = []
            general_trends = []
            
            for _, trend in df.iterrows():
                topic = str(trend['topic'])
                trend_data = {
                    'topic': topic,
                    'region': str(trend['region']),
                    'significance_score': float(trend['significance_score']),
                    'tweet_volume': int(trend['tweet_volume']),
                    'category': str(trend['category']),
                    'sentiment': str(trend['sentiment'])
                }
                
                if self.is_crypto_related(topic):
                    crypto_trends.append(trend_data)
                else:
                    general_trends.append(trend_data)
            
            # Sort crypto trends by tweet volume (take top 3)
            crypto_trends.sort(key=lambda x: x['tweet_volume'], reverse=True)
            crypto_trends = crypto_trends[:3]  # Limit to 3 crypto trends
            
            # Sort general trends by significance score (take top 3)
            general_trends = [t for t in general_trends if t['significance_score'] >= 6]
            general_trends.sort(key=lambda x: x['significance_score'], reverse=True)
            general_trends = general_trends[:3]  # Limit to 3 general trends
            
            logger.info(f"Identified {len(crypto_trends)} crypto trends and {len(general_trends)} high-quality general trends")
            return crypto_trends, general_trends
            
        except Exception as e:
            logger.error(f"Error identifying trends: {e}")
            return [], []
    
    def analyze_content_quality(self, tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content quality and gaming indicators."""
        if not tweets:
            return {}
        
        total_tweets = len(tweets)
        crypto_tweets = [t for t in tweets if t['is_crypto_related']]
        general_tweets = [t for t in tweets if not t['is_crypto_related']]
        
        # Quality score analysis
        quality_scores = [t['author_quality_score'] for t in tweets]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Gaming detection metrics
        suspicious_patterns = {
            'low_quality_accounts': len([t for t in tweets if t['author_quality_score'] < 30]),
            'high_following_accounts': len([t for t in tweets if t['author_following'] > 5000]),
            'new_accounts': len([t for t in tweets if t['quality_metrics'].get('account_age_days', 0) < 90]),
            'unverified_high_engagement': len([
                t for t in tweets 
                if not t['author_verified'] and not t['author_blue_verified'] 
                and (t['like_count'] + t['retweet_count']) > 1000
            ])
        }
        
        # Engagement analysis
        total_engagement = sum(t['like_count'] + t['retweet_count'] for t in tweets)
        avg_engagement = total_engagement / total_tweets if total_tweets > 0 else 0
        
        return {
            'total_tweets': total_tweets,
            'crypto_tweets': len(crypto_tweets),
            'general_tweets': len(general_tweets),
            'avg_quality_score': round(avg_quality, 2),
            'total_engagement': total_engagement,
            'avg_engagement_per_tweet': round(avg_engagement, 2),
            'gaming_indicators': suspicious_patterns,
            'quality_distribution': {
                'high_quality': len([t for t in tweets if t['author_quality_score'] >= 70]),
                'medium_quality': len([t for t in tweets if 40 <= t['author_quality_score'] < 70]),
                'low_quality': len([t for t in tweets if t['author_quality_score'] < 40])
            }
        }
    
    def save_tweets_data(self, all_tweets: List[Dict[str, Any]], session_dir: Path) -> str:
        """Save processed tweets data to files."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create kaito_data directory in raw_data
        kaito_dir = session_dir / "raw_data" / "kaito_data"
        kaito_dir.mkdir(exist_ok=True, parents=True)
        
        # Save raw data to JSON
        raw_file = kaito_dir / f"kaito_raw_data_{timestamp}.json"
        with open(raw_file, 'w') as f:
            json.dump(all_tweets, f, indent=2, default=str)
        
        logger.info(f"💾 Saved raw Kaito data to {raw_file}")
        logger.info(f"📊 Total raw items collected: {len(all_tweets)}")
        
        # Save simplified version to CSV
        if all_tweets:
            simple_data = []
            for item in all_tweets:
                if isinstance(item, dict):
                    followers = item.get('author_followers', 0)
                    following = item.get('author_following', 0) or 1  # avoid div by zero
                    follower_ratio = round(followers / following, 2) if following else float('inf')
                    simple_data.append({
                        'trend_topic': item.get('trend_topic', ''),
                        'content_type': item.get('content_type', ''),
                        'tweet_id': item.get('tweet_id', ''),
                        'tweet_url': item.get('tweet_url', ''),
                        'created_at': item.get('created_at', ''),
                        'author_username': item.get('author_username', ''),
                        'author_followers': followers,
                        'author_following': item.get('author_following', 0),
                        'follower_ratio': follower_ratio,
                        'author_verified': item.get('author_verified', False),
                        'author_blue_verified': item.get('author_blue_verified', False),
                        'like_count': item.get('like_count', 0),
                        'retweet_count': item.get('retweet_count', 0),
                        'reply_count': item.get('reply_count', 0),
                        'quote_count': item.get('quote_count', 0),
                        'bookmark_count': item.get('bookmark_count', 0),
                        'view_count': item.get('view_count', 0),
                        'engagement_total': item.get('like_count', 0) + item.get('retweet_count', 0),
                        'author_quality_score': round(item.get('author_quality_score', 0), 1),
                        'text_preview': (item.get('text', '')[:100] + '...') if len(item.get('text', '')) > 100 else item.get('text', '')
                    })
            
            csv_file = kaito_dir / f"kaito_summary_{timestamp}.csv"
            pd.DataFrame(simple_data).to_csv(csv_file, index=False)
            logger.info(f"📋 Saved summary to {csv_file}")
        
        return str(raw_file)
    
    def generate_content_report(self, session_dir: Path, all_tweets: List[Dict[str, Any]]) -> str:
        """Generate comprehensive anti-gaming content analysis report."""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        report_path = session_dir / f"content_analysis_report_{timestamp}.md"
        
        # Analyze content quality
        quality_analysis = self.analyze_content_quality(all_tweets)
        
        # Group tweets by topic
        topics = {}
        for tweet in all_tweets:
            topic = tweet['trend_topic']
            if topic not in topics:
                topics[topic] = []
            topics[topic].append(tweet)
        
        # Generate report
        report_content = f"""# 🛡️ Anti-Gaming Content Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Executive Summary
- **Total Tweets Analyzed**: {quality_analysis['total_tweets']}
- **Crypto-Related Tweets**: {quality_analysis['crypto_tweets']}
- **General Tweets**: {quality_analysis['general_tweets']}
- **Average Quality Score**: {quality_analysis['avg_quality_score']}/100
- **Total Engagement**: {quality_analysis['total_engagement']:,}
- **Average Engagement per Tweet**: {quality_analysis['avg_engagement_per_tweet']:.1f}

## 🚨 Gaming Detection Results

### Quality Distribution
- **High Quality (70-100)**: {quality_analysis['quality_distribution']['high_quality']} tweets
- **Medium Quality (40-69)**: {quality_analysis['quality_distribution']['medium_quality']} tweets
- **Low Quality (0-39)**: {quality_analysis['quality_distribution']['low_quality']} tweets

### Suspicious Patterns Detected
- **Low Quality Accounts**: {quality_analysis['gaming_indicators']['low_quality_accounts']} tweets
- **High Following Accounts**: {quality_analysis['gaming_indicators']['high_following_accounts']} tweets
- **New Accounts (<90 days)**: {quality_analysis['gaming_indicators']['new_accounts']} tweets
- **Unverified High Engagement**: {quality_analysis['gaming_indicators']['unverified_high_engagement']} tweets

## 📈 Content Analysis by Topic

"""
        
        for topic, topic_tweets in topics.items():
            crypto_indicator = "🔗" if topic_tweets[0]['is_crypto_related'] else "📊"
            content_type = topic_tweets[0]['content_type']
            
            avg_quality = sum(t['author_quality_score'] for t in topic_tweets) / len(topic_tweets)
            total_engagement = sum(t['like_count'] + t['retweet_count'] for t in topic_tweets)
            
            report_content += f"""
### {crypto_indicator} {topic} ({content_type.title()})
- **Tweets**: {len(topic_tweets)}
- **Avg Quality Score**: {avg_quality:.1f}/100
- **Total Engagement**: {total_engagement:,}
- **Top Tweet**: {max(topic_tweets, key=lambda x: x['like_count'] + x['retweet_count'])['like_count'] + max(topic_tweets, key=lambda x: x['like_count'] + x['retweet_count'])['retweet_count']:,} engagements

"""
        
        report_content += f"""
## 🔍 Quality Metrics Explained

### Quality Score Components (0-100):
- **Follower-to-Following Ratio** (30%): Higher ratios indicate organic growth
- **Account Age** (20%): Older accounts are less likely to be fake
- **Verification Status** (20%): Verified accounts have higher credibility
- **Follower Count Legitimacy** (15% of score)
- **Following Count Reasonableness** (15% of score)

### Content Strategy:
- **Crypto Content**: All crypto-related trends included regardless of quality
- **General Content**: Only high-quality accounts (50+ quality score) included
- **Gaming Protection**: Multiple metrics used to detect manipulated engagement

---
*Report generated by Anti-Gaming Content Analysis Engine*
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Content analysis report saved to: {report_path}")
        return str(report_path)
    
    def run_content_analysis(self, session_dir: Path) -> Optional[str]:
        """Run complete content analysis pipeline."""
        try:
            # Create raw_data and kaito_data directories
            raw_data_dir = session_dir / "raw_data"
            raw_data_dir.mkdir(exist_ok=True, parents=True)
            
            kaito_dir = raw_data_dir / "kaito_data"
            kaito_dir.mkdir(exist_ok=True, parents=True)
            
            # Prefer rich analysis CSV (has category/subcategory). Fallback to raw trending if missing.
            analysis_files = list((session_dir / "analysis").glob("trending_analysis_*.csv"))
            if analysis_files:
                analysis_file = max(analysis_files, key=lambda f: f.stat().st_mtime)
                logger.info(f"🔍 Using analysis file for topic selection: {analysis_file.name}")
                df = pd.read_csv(analysis_file)
            else:
                trending_files = list(raw_data_dir.glob("trending_topics_*.csv"))
                if not trending_files:
                    logger.error("No trending topics or analysis file found")
                    return None
                trending_file = max(trending_files, key=lambda f: f.stat().st_mtime)
                logger.info(f"🔍 Falling back to raw trending file: {trending_file.name}")
                df = pd.read_csv(trending_file)

            # Determine base selection of top N by significance (or volume)
            base_sort_col = "significance_score" if "significance_score" in df.columns else "tweet_volume"
            top_df = df.sort_values(base_sort_col, ascending=False).head(self.MAX_TOPICS)

            # Always include crypto or technology topics even if they are outside the top N
            tech_crypto_mask = (
                (df.get("category", "") == "Technology") |
                df["topic"].astype(str).apply(self.is_crypto_related)
            )
            tech_crypto_df = df[tech_crypto_mask]

            # Combine and remove duplicates
            combined_df = pd.concat([top_df, tech_crypto_df])
            combined_df = combined_df.drop_duplicates(subset="topic")
            combined_df = combined_df.reset_index(drop=True)
            
            # Limit to MAX_TOPICS total
            if len(combined_df) > self.MAX_TOPICS:
                logger.info(f"Limiting from {len(combined_df)} to {self.MAX_TOPICS} topics")
                combined_df = combined_df.head(self.MAX_TOPICS)

            logger.info(
                f"🔗 Selected {len(combined_df)} topics for content analysis"
            )
            
            # Track expected vs actual tweet counts - now expecting up to 50 tweets per topic
            expected_tweet_count = len(combined_df) * 50
            logger.info(f"Expected tweet count: up to {expected_tweet_count} ({len(combined_df)} topics × 50 tweets max per topic)")

            all_tweets_raw: List[Dict] = []
            all_tweets_processed: List[Dict] = []
            processed_topics: Set[str] = set()
            
            for _, row in combined_df.iterrows():
                topic = str(row['topic'])  # Convert to string explicitly
                
                # Skip duplicate topics
                if topic in processed_topics:
                    logger.info(f"Skipping duplicate topic: {topic}")
                    continue
                    
                processed_topics.add(topic)
                
                is_crypto = self.is_crypto_related(topic) or row['crypto_connection'] == 'direct'
                if is_crypto:
                    tweets_raw = self.scrape_crypto_tweets(topic, session_dir)
                    content_type = "Crypto"
                else:
                    tweets_raw = self.scrape_general_tweets(topic, session_dir)
                    content_type = "Tech"

                # Extend raw list
                all_tweets_raw.extend(tweets_raw)

                # Convert to standardized structure for downstream processing
                processed = self._process_tweets(tweets_raw, topic, content_type)
                all_tweets_processed.extend(processed)
            
            logger.info(f"Total raw tweets collected: {len(all_tweets_raw)} (from {len(processed_topics)} topics)")
            
            # Save full raw data
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            kaito_dir = session_dir / "raw_data" / "kaito_data"
            kaito_dir.mkdir(exist_ok=True, parents=True)
            raw_full_file = kaito_dir / f"kaito_raw_full_{timestamp}.json"
            with open(raw_full_file, 'w') as f:
                json.dump(all_tweets_raw, f, indent=2, default=str)
            logger.info(f"Saved full raw Kaito data to {raw_full_file} (total {len(all_tweets_raw)} tweets)")
            
            # Rank processed tweets by engagement (likes + retweets)
            ranked_tweets = sorted(all_tweets_processed, key=lambda t: (
                t.get('like_count', 0) + t.get('retweet_count', 0)
            ), reverse=True)
            
            # Keep only tweets passing engagement threshold
            high_quality = [
                t for t in ranked_tweets
                if (t.get('like_count', 0) + t.get('retweet_count', 0)) >= self.HIGH_ENGAGEMENT_THRESHOLD
            ]
            
            if not high_quality:
                logger.warning("No tweets collected")
                return None

            # Save all tweets and summary
            output_file = self.save_tweets_data(high_quality, session_dir)
            logger.info(f"Saved {len(high_quality)} tweets to {output_file}")
            
            # Generate content report
            report_file = self.generate_content_report(session_dir, high_quality)
            logger.info(f"Generated content report: {report_file}")
            
            return report_file
            
        except Exception as e:
            logger.error(f"Error in content analysis: {e}")
            return None

    def run_tech_content_analysis(self, session_dir: Path, tech_df: pd.DataFrame) -> Optional[str]:
        """Run content analysis specifically for tech and web3 topics.
        
        Args:
            session_dir: Path to the session directory
            tech_df: DataFrame containing tech and web3 topics
            
        Returns:
            Path to the generated report or None if failed
        """
        try:
            # Create raw_data and kaito_data directories
            raw_data_dir = session_dir / "raw_data"
            raw_data_dir.mkdir(exist_ok=True, parents=True)
            
            kaito_dir = raw_data_dir / "kaito_data"
            kaito_dir.mkdir(exist_ok=True, parents=True)
            
            # Select top tech topics for content analysis
            top_tech_df = tech_df.sort_values('significance_score', ascending=False).head(self.MAX_TOPICS)

            logger.info(
                f"🔗 Selected {len(top_tech_df)} tech topics for content analysis"
            )
            
            # Track expected vs actual tweet counts
            expected_tweet_count = len(top_tech_df) * 50
            logger.info(f"Expected tweet count: up to {expected_tweet_count} ({len(top_tech_df)} topics × 50 tweets max per topic)")

            all_tweets_raw: List[Dict] = []
            all_tweets_processed: List[Dict] = []
            processed_topics: Set[str] = set()
            
            for _, row in top_tech_df.iterrows():
                topic = str(row['topic'])  # Convert to string explicitly
                
                # Skip duplicate topics
                if topic in processed_topics:
                    logger.info(f"Skipping duplicate topic: {topic}")
                    continue
                    
                processed_topics.add(topic)
                
                is_crypto = self.is_crypto_related(topic) or row['crypto_connection'] == 'direct'
                if is_crypto:
                    tweets_raw = self.scrape_crypto_tweets(topic, session_dir)
                    content_type = "Crypto"
                else:
                    tweets_raw = self.scrape_general_tweets(topic, session_dir)
                    content_type = "Tech"

                # Extend raw list
                all_tweets_raw.extend(tweets_raw)

                # Convert to standardized structure for downstream processing
                processed = self._process_tweets(tweets_raw, topic, content_type)
                all_tweets_processed.extend(processed)
            
            logger.info(f"Total raw tweets collected: {len(all_tweets_raw)} (from {len(processed_topics)} topics)")
            
            # Save full raw data
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            kaito_dir = session_dir / "raw_data" / "kaito_data"
            kaito_dir.mkdir(exist_ok=True, parents=True)
            raw_full_file = kaito_dir / f"kaito_tech_raw_full_{timestamp}.json"
            with open(raw_full_file, 'w') as f:
                json.dump(all_tweets_raw, f, indent=2, default=str)
            logger.info(f"Saved full raw Kaito data to {raw_full_file} (total {len(all_tweets_raw)} tweets)")
            
            # Rank processed tweets by engagement (likes + retweets)
            ranked_tweets = sorted(all_tweets_processed, key=lambda t: (
                t.get('like_count', 0) + t.get('retweet_count', 0)
            ), reverse=True)
            
            # Use all tweets for analysis, no filtering or limiting
            high_quality = ranked_tweets
            
            if not high_quality:
                logger.warning("No tweets collected")
                return None

            # Save all tweets and summary
            output_file = self.save_tech_tweets_data(high_quality, session_dir, timestamp)
            logger.info(f"Saved {len(high_quality)} tweets to {output_file}")
            
            # Generate tech content report
            report_file = self.generate_tech_content_report(session_dir, high_quality, timestamp)
            if report_file:
                logger.info(f"Generated tech content report: {report_file}")
                return str(report_file)
            return None
            
        except Exception as e:
            logger.error(f"Error in tech content analysis: {e}")
            return None
            
    def save_tech_tweets_data(self, tweets: List[Dict], session_dir: Path, timestamp: str) -> Path:
        """Save tech tweets data to files."""
        # Save the processed data
        kaito_dir = session_dir / "raw_data" / "kaito_data"
        kaito_dir.mkdir(exist_ok=True, parents=True)
        
        # Save full data
        output_file = kaito_dir / f"kaito_tech_data_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(tweets, f, indent=2, default=str)
        
        # Create summary CSV
        summary_rows = []
        for tweet in tweets:
            tweet_text = tweet.get('text', '').replace('\n', ' ').replace('\r', '')
            summary_rows.append({
                'trend_topic': tweet.get('trend_topic', ''),
                'author_username': tweet.get('author_username', ''),
                'author_name': tweet.get('author_name', ''),
                'author_followers': tweet.get('author_followers', tweet.get('author_followers_count', 0)),
                'author_following': tweet.get('author_following', tweet.get('author_following_count', 0)),
                'tweet_text': tweet_text[:100] + ('...' if len(tweet_text) > 100 else ''),
                'like_count': tweet.get('like_count', 0),
                'retweet_count': tweet.get('retweet_count', 0),
                'reply_count': tweet.get('reply_count', 0),
                'bookmark_count': tweet.get('bookmark_count', 0),
                'created_at': tweet.get('created_at', ''),
                'lang': tweet.get('language', tweet.get('lang', '')),
                'quality_score': tweet.get('author_quality_score', None)
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_file = kaito_dir / f"kaito_tech_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"📊 Total raw items collected: {len(tweets)}")
        logger.info(f"📋 Saved summary to {summary_file}")
        
        return output_file
        
    def generate_tech_content_report(self, session_dir: Path, tweets: List[Dict], timestamp: str) -> Optional[Path]:
        """Generate a tech-focused content analysis report."""
        if not tweets:
            return None
            
        # Collect quality scores directly from processed tweet objects
        quality_scores = [tweet.get('author_quality_score', 0) for tweet in tweets if isinstance(tweet.get('author_quality_score', 0), (int, float))]
        
        if not quality_scores:
            logger.warning("No valid quality scores calculated")
            return None
            
        high_quality = sum(1 for score in quality_scores if score >= 70)
        medium_quality = sum(1 for score in quality_scores if 40 <= score < 70)
        low_quality = sum(1 for score in quality_scores if score < 40)
        
        # Calculate engagement metrics
        total_engagement = sum(tweet.get('like_count', 0) + tweet.get('retweet_count', 0) for tweet in tweets)
        avg_engagement = total_engagement / len(tweets) if tweets else 0
        
        # Group by topics
        topic_tweets = {}
        for tweet in tweets:
            topic = tweet.get('search_term', 'Unknown')
            if topic not in topic_tweets:
                topic_tweets[topic] = []
            topic_tweets[topic].append(tweet)
        
        # Generate report content
        report_lines = [
            "# 🛡️ Tech & Web3 Content Analysis Report",
            f"Generated: {timestamp.replace('_', ' ')}",
            "",
            "## 📊 Executive Summary",
            f"- **Total Tweets Analyzed**: {len(tweets)}",
            f"- **Crypto-Related Tweets**: {sum(1 for tweet in tweets if self.is_crypto_related(tweet.get('search_term', '')))}",
            f"- **Tech-Related Tweets**: {len(tweets) - sum(1 for tweet in tweets if self.is_crypto_related(tweet.get('search_term', '')))}",
            f"- **Average Quality Score**: {sum(quality_scores)/len(quality_scores):.2f}/100",
            f"- **Total Engagement**: {total_engagement:,}",
            f"- **Average Engagement per Tweet**: {avg_engagement:.1f}",
            "",
            "## 🚨 Gaming Detection Results",
            "",
            "### Quality Distribution",
            f"- **High Quality (70-100)**: {high_quality} tweets",
            f"- **Medium Quality (40-69)**: {medium_quality} tweets",
            f"- **Low Quality (0-39)**: {low_quality} tweets",
            "",
            "### Suspicious Patterns Detected",
            f"- **Low Quality Accounts**: {sum(1 for tweet in tweets if isinstance(tweet.get('author_quality_score', 0), (int, float)) and tweet.get('author_quality_score', 0) < 30)} tweets",
            f"- **High Following Accounts**: {sum(1 for tweet in tweets if tweet.get('author_following', 0) > 5000)} tweets",
            f"- **New Accounts (<90 days)**: {sum(1 for tweet in tweets if self.is_new_account(tweet))} tweets",
            f"- **Unverified High Engagement**: {sum(1 for tweet in tweets if not tweet.get('author_is_blue_verified', False) and (tweet.get('like_count', 0) + tweet.get('retweet_count', 0)) > 10000)} tweets",
            "",
            "## 📈 Content Analysis by Topic",
            "",
        ]
        
        # Add topic-specific sections
        for topic, topic_tweet_list in topic_tweets.items():
            # Skip if no tweets
            if not topic_tweet_list:
                continue
                
            # Calculate topic metrics
            topic_quality_scores = [tweet.get('author_quality_score', 0) for tweet in topic_tweet_list if isinstance(tweet.get('author_quality_score', 0), (int, float))]
            
            if not topic_quality_scores:
                continue
                
            topic_quality = sum(topic_quality_scores) / len(topic_quality_scores)
            topic_engagement = sum(tweet.get('like_count', 0) + tweet.get('retweet_count', 0) for tweet in topic_tweet_list)
            
            # Find top tweet by engagement
            top_tweet = max(topic_tweet_list, key=lambda t: t.get('like_count', 0) + t.get('retweet_count', 0))
            top_engagement = top_tweet.get('like_count', 0) + top_tweet.get('retweet_count', 0)
            
            # Determine if crypto or general
            topic_type = "Crypto" if self.is_crypto_related(topic) else "Tech"
            
            report_lines.extend([
                f"### 📊 {topic} ({topic_type})",
                f"- **Tweets**: {len(topic_tweet_list)}",
                f"- **Avg Quality Score**: {topic_quality:.1f}/100",
                f"- **Total Engagement**: {topic_engagement:,}",
                f"- **Top Tweet**: {top_engagement:,} engagements",
                "",
            ])
        
        # Add explanation section
        report_lines.extend([
            "## 🔍 Quality Metrics Explained",
            "",
            "### Quality Score Components (0-100):",
            "- **Follower-to-Following Ratio** (30%): Higher ratios indicate organic growth",
            "- **Account Age** (20%): Older accounts are less likely to be fake",
            "- **Verification Status** (20%): Verified accounts have higher credibility",
            "- **Follower Count Legitimacy** (15% of score)",
            "- **Following Count Reasonableness** (15% of score)",
            "",
            "### Content Strategy:",
            "- **Crypto Content**: All crypto-related trends included regardless of quality",
            "- **Tech Content**: Focus on emerging technologies and industry trends",
            "- **Gaming Protection**: Multiple metrics used to detect manipulated engagement",
            "",
            "---",
            "*Report generated by Anti-Gaming Content Analysis Engine*",
            ""
        ])
        
        # Write report to file
        report_text = "\n".join(report_lines)
        report_file = session_dir / f"tech_content_analysis_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report_text)
            
        return report_file

    def is_new_account(self, tweet: Dict) -> bool:
        """Check if an account is less than 90 days old."""
        try:
            created_at_str = tweet.get('author_created_at', '')
            if not created_at_str:
                return False
                
            # Parse the created_at date
            if isinstance(created_at_str, str):
                # Try different date formats
                for fmt in ('%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d %H:%M:%S%z', '%a %b %d %H:%M:%S %z %Y'):
                    try:
                        created_at = datetime.strptime(created_at_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # If none of the formats match
                    return False
            else:
                # Already a datetime object
                created_at = created_at_str
                
            # Calculate account age in days
            now = datetime.now(timezone.utc) if created_at.tzinfo else datetime.now()
            account_age_days = (now - created_at).days
            
            return account_age_days < 90
        except Exception:
            return False

def main():
    """Main function for standalone execution."""
    if len(sys.argv) != 2:
        print("Usage: python content_analysis_engine.py <session_directory>")
        sys.exit(1)
    
    session_dir = Path(sys.argv[1])
    if not session_dir.exists():
        print(f"Session directory not found: {session_dir}")
        sys.exit(1)
    
    engine = AntiGamingContentEngine()
    report_path = engine.run_content_analysis(session_dir)
    
    if report_path:
        print(f"✅ Content analysis completed: {report_path}")
    else:
        print("❌ Content analysis failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 