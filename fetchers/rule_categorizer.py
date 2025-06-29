"""Rule-based topic categorization for trending topics analysis."""

import logging
import time
from typing import Dict, List, Set
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TopicAnalysis:
    """Analysis result for a single topic."""
    category: str
    subcategory: str
    significance_score: int  # 1-10
    sentiment: str  # positive/negative/neutral
    context: str
    trending_reason: str
    confidence: float = 0.0  # 0.0-1.0
    web3_relevance: str = "none"  # none/low/medium/high
    crypto_connection: str = "none"  # none/indirect/direct
    tech_relationship: str = "standard"  # standard/emerging/disruptive


class RuleBasedCategorizer:
    """Fast rule-based categorization using keyword matching and patterns."""
    
    def __init__(self):
        """Initialize the rule-based categorizer."""
        self.logger = logging.getLogger(__name__)
        self._setup_rules()
    
    def _setup_rules(self):
        """Initialize all categorization rules and keyword mappings."""
        
        # Entertainment keywords
        self.ENTERTAINMENT_KEYWORDS = {
            'music': ['#sb19', 'concert', 'album', 'song', 'band', 'artist', 'music', 'tour', 'performance'],
            'tv_shows': ['fanfest', 'episode', 'season', 'series', 'show', 'drama', 'reality'],
            'celebrities': ['taylor', 'swift', 'beyonce', 'drake', 'ariana', 'justin', 'selena'],
            'movies': ['movie', 'film', 'cinema', 'trailer', 'premiere', 'oscar', 'box office'],
            'gaming': ['game', 'gaming', 'esports', 'tournament', 'stream', 'twitch']
        }
        
        # Politics keywords
        self.POLITICS_KEYWORDS = {
            'government': ['senate', 'congress', 'parliament', 'government', 'minister', 'president'],
            'politicians': ['trump', 'biden', 'pelosi', 'republicans', 'democrats', 'senator'],
            'policies': ['medicaid', 'obamacare', 'healthcare', 'immigration', 'tax', 'policy'],
            'military': ['idf', 'military', 'defense', 'army', 'navy', 'forces', 'troops'],
            'elections': ['election', 'vote', 'campaign', 'ballot', 'primary', 'candidate']
        }
        
        # Sports keywords
        self.SPORTS_KEYWORDS = {
            'formula1': ['#formula1', '#austriagp', 'ferrari', 'sainz', 'lando', 'lewis', 'verstappen'],
            'general': ['championship', 'game', 'match', 'victory', 'win', 'loss', 'playoffs'],
            'teams': ['lakers', 'warriors', 'yankees', 'patriots', 'chelsea', 'madrid'],
            'athletes': ['lebron', 'messi', 'ronaldo', 'serena', 'federer', 'hamilton']
        }
        
        # Technology keywords
        self.TECHNOLOGY_KEYWORDS = {
            'crypto': ['bitcoin', 'btc', 'ethereum', 'crypto', 'blockchain', 'binance', 'coinbase'],
            'ai': ['chatgpt', 'ai', 'claude', 'openai', 'machine learning', 'neural'],
            'companies': ['apple', 'google', 'microsoft', 'tesla', 'meta', 'amazon', 'netflix'],
            'products': ['iphone', 'android', 'windows', 'ios', 'app', 'software', 'update']
        }
        
        # Web3/Crypto specific keywords
        self.WEB3_KEYWORDS = {
            'defi': ['defi', 'uniswap', 'compound', 'aave', 'yield farming', 'liquidity', 'protocol'],
            'nft': ['nft', 'opensea', 'bored ape', 'pfp', 'mint', 'collection', 'rare'],
            'blockchain': ['blockchain', 'ethereum', 'solana', 'polygon', 'avalanche', 'layer2'],
            'metaverse': ['metaverse', 'vr', 'ar', 'virtual world', 'sandbox', 'decentraland'],
            'crypto_trading': ['binance', 'coinbase', 'ftx', 'trading', 'pump', 'dump', 'hodl'],
            'web3_social': ['lens protocol', 'farcaster', 'friend.tech', 'social token'],
            'gamefi': ['gamefi', 'play to earn', 'p2e', 'gaming token', 'nft game'],
            'dao': ['dao', 'governance', 'voting', 'proposal', 'community token']
        }
        
        # Crypto influence indicators (people/entities that move crypto markets)
        self.CRYPTO_INFLUENCERS = {
            'high_influence': ['elon musk', 'tesla', 'michael saylor', 'cathie wood'],
            'medium_influence': ['jack dorsey', 'tim cook', 'mark cuban', 'gary vee'],
            'institutions': ['goldman sachs', 'jpmorgan', 'blackrock', 'fidelity'],
            'exchanges': ['binance', 'coinbase', 'kraken', 'ftx', 'okx']
        }
        
        # News/Events keywords
        self.NEWS_KEYWORDS = {
            'breaking': ['breaking', 'urgent', 'alert', 'update', 'announcement', 'report'],
            'disasters': ['earthquake', 'flood', 'fire', 'accident', 'emergency', 'disaster'],
            'international': ['ukraine', 'russia', 'china', 'israel', 'palestine', 'war', 'conflict']
        }
        
        # Geographic locations
        self.COUNTRIES = {
            'london', 'japan', 'singapore', 'thailand', 'europe', 'asia', 'ukraine', 
            'spain', 'taiwan', 'haiti', 'istanbul', 'united states', 'usa'
        }
        
        self.CITIES = {
            'london', 'tokyo', 'singapore', 'bangkok', 'madrid', 'taipei', 
            'istanbul', 'new york', 'los angeles', 'chicago'
        }
        
        # Sentiment keywords
        self.POSITIVE_KEYWORDS = {
            'celebration', 'fanfest', 'concert', 'festival', 'birthday', 'anniversary',
            'love', 'blessed', 'amazing', 'beautiful', 'fantastic', 'incredible',
            'victory', 'champion', 'success', 'achievement', 'breakthrough',
            'launch', 'premiere', 'debut', 'opening', 'unveiling', 'funday'
        }
        
        self.NEGATIVE_KEYWORDS = {
            'war', 'attack', 'violence', 'conflict', 'crisis', 'scandal',
            'crash', 'fire', 'flood', 'earthquake', 'accident', 'emergency',
            'outrage', 'controversy', 'backlash', 'criticism', 'protest',
            'issue', 'problem', 'failure', 'concern', 'worry', 'death'
        }
        
        # Category sentiment defaults
        self.CATEGORY_SENTIMENT_DEFAULTS = {
            'Entertainment': 'positive',
            'Sports': 'positive', 
            'Politics': 'neutral',
            'News/Events': 'neutral',
            'Technology': 'neutral',
            'Culture/Social': 'positive',
            'Business/Finance': 'neutral',
            'Global/Places': 'neutral',
            'Daily/Lifestyle': 'positive'
        }
    
    def categorize_topics(self, topics: List[Dict]) -> Dict[str, TopicAnalysis]:
        """Categorize a list of topics using rule-based analysis.
        
        Args:
            topics: List of topic dicts with 'topic', 'tweet_volume', 'region' keys
            
        Returns:
            Dict mapping topic names to TopicAnalysis objects
        """
        if not topics:
            return {}
        
        start_time = time.time()
        self.logger.info(f"Categorizing {len(topics)} topics using rule-based analysis...")
        
        results = {}
        for topic_data in topics:
            topic_name = topic_data['topic']
            analysis = self._analyze_single_topic(topic_data)
            results[topic_name] = analysis
        
        elapsed = time.time() - start_time
        self.logger.info(f"Rule-based categorization completed in {elapsed:.1f}s")
        
        return results
    
    def _analyze_single_topic(self, topic_data: Dict) -> TopicAnalysis:
        """Analyze a single topic and return categorization."""
        topic = topic_data['topic']
        tweet_volume = topic_data.get('tweet_volume', 0)
        region = topic_data.get('region', 'Unknown')
        
        topic_lower = topic.lower()
        
        # Determine category and subcategory
        category, subcategory, confidence = self._categorize_topic(topic_lower, tweet_volume)
        
        # Determine sentiment
        sentiment = self._determine_sentiment(topic_lower, category, tweet_volume)
        
        # Calculate significance score
        significance_score = self._calculate_significance(topic_lower, category, tweet_volume, region)
        
        # Generate context and trending reason
        context = self._generate_context(topic, category, subcategory, tweet_volume)
        trending_reason = self._generate_trending_reason(topic, category, tweet_volume, region)
        
        # Analyze Web3/crypto relevance
        web3_relevance = self._analyze_web3_relevance(topic_lower, category)
        crypto_connection = self._analyze_crypto_connection(topic_lower, tweet_volume)
        tech_relationship = self._analyze_tech_relationship(topic_lower, category, tweet_volume)
        
        return TopicAnalysis(
            category=category,
            subcategory=subcategory,
            significance_score=significance_score,
            sentiment=sentiment,
            context=context,
            trending_reason=trending_reason,
            confidence=confidence,
            web3_relevance=web3_relevance,
            crypto_connection=crypto_connection,
            tech_relationship=tech_relationship
        )
    
    def _categorize_topic(self, topic_lower: str, tweet_volume: int) -> tuple:
        """Determine category and subcategory for a topic."""
        
        # Check hashtags first
        if topic_lower.startswith('#'):
            if any(word in topic_lower for word in ['fanfest', 'concert', 'tour']):
                return 'Entertainment', 'Fan Events', 0.9
            elif any(word in topic_lower for word in ['formula1', 'gp', 'race']):
                return 'Sports', 'Formula 1', 0.9
            elif any(word in topic_lower for word in ['ufc', 'fight']):
                return 'Sports', 'Combat Sports', 0.9
            elif any(word in topic_lower for word in ['sunday', 'motivation', 'vibes']):
                return 'Culture/Social', 'Lifestyle', 0.8
            else:
                return 'Culture/Social', 'Hashtags', 0.6
        
        # Check entertainment keywords
        for subcat, keywords in self.ENTERTAINMENT_KEYWORDS.items():
            if any(keyword in topic_lower for keyword in keywords):
                return 'Entertainment', subcat.replace('_', ' ').title(), 0.8
        
        # Check politics keywords
        for subcat, keywords in self.POLITICS_KEYWORDS.items():
            if any(keyword in topic_lower for keyword in keywords):
                return 'Politics', subcat.replace('_', ' ').title(), 0.8
        
        # Check sports keywords
        for subcat, keywords in self.SPORTS_KEYWORDS.items():
            if any(keyword in topic_lower for keyword in keywords):
                return 'Sports', subcat.replace('_', ' ').title(), 0.8
        
        # Check technology keywords
        for subcat, keywords in self.TECHNOLOGY_KEYWORDS.items():
            if any(keyword in topic_lower for keyword in keywords):
                return 'Technology', subcat.replace('_', ' ').title(), 0.8
        
        # Check web3 keywords
        for subcat, keywords in self.WEB3_KEYWORDS.items():
            if any(keyword in topic_lower for keyword in keywords):
                return 'Technology', subcat.replace('_', ' ').title(), 0.8
        
        # Check news keywords
        for subcat, keywords in self.NEWS_KEYWORDS.items():
            if any(keyword in topic_lower for keyword in keywords):
                return 'News/Events', subcat.replace('_', ' ').title(), 0.7
        
        # Check geographic locations
        if any(location in topic_lower for location in self.COUNTRIES | self.CITIES):
            return 'Global/Places', 'Geographic', 0.7
        
        # Check for names (likely people)
        if topic_lower.istitle() and len(topic_lower.split()) <= 2:
            return 'Culture/Social', 'People', 0.5
        
        # Default categorization
        if tweet_volume > 100000:
            return 'News/Events', 'Viral Content', 0.3
        else:
            return 'Culture/Social', 'General Discussion', 0.3
    
    def _determine_sentiment(self, topic_lower: str, category: str, tweet_volume: int) -> str:
        """Determine sentiment for a topic."""
        
        # Check for explicit positive keywords
        if any(keyword in topic_lower for keyword in self.POSITIVE_KEYWORDS):
            return 'positive'
        
        # Check for explicit negative keywords
        if any(keyword in topic_lower for keyword in self.NEGATIVE_KEYWORDS):
            return 'negative'
        
        # High volume entertainment topics are usually positive
        if category == 'Entertainment' and tweet_volume > 200000:
            return 'positive'
        
        # High volume political topics are often controversial (neutral)
        if category == 'Politics' and tweet_volume > 100000:
            return 'neutral'
        
        # Use category defaults
        return self.CATEGORY_SENTIMENT_DEFAULTS.get(category, 'neutral')
    
    def _calculate_significance(self, topic_lower: str, category: str, tweet_volume: int, region: str) -> int:
        """Calculate significance score 1-10."""
        
        base_score = 1
        
        # Volume-based scoring
        if tweet_volume > 1000000:
            base_score = 9
        elif tweet_volume > 500000:
            base_score = 7
        elif tweet_volume > 200000:
            base_score = 5
        elif tweet_volume > 50000:
            base_score = 3
        else:
            base_score = 1
        
        # Category modifiers
        if category == 'Politics' and tweet_volume > 100000:
            base_score += 1
        elif category == 'News/Events' and tweet_volume > 200000:
            base_score += 2
        elif category == 'Entertainment' and tweet_volume > 500000:
            base_score += 1
        
        # Cap at 10
        return min(base_score, 10)
    
    def _generate_context(self, topic: str, category: str, subcategory: str, tweet_volume: int) -> str:
        """Generate contextual explanation for the topic."""
        
        if category == 'Entertainment':
            if 'fanfest' in topic.lower():
                return f"Fan event or festival related to entertainment content with {tweet_volume:,} tweets of engagement"
            elif subcategory == 'Music':
                return f"Music-related content trending with significant fan engagement ({tweet_volume:,} tweets)"
            else:
                return f"Entertainment content in the {subcategory.lower()} category generating buzz"
        
        elif category == 'Politics':
            if subcategory == 'Military':
                return f"Military or defense-related topic with {tweet_volume:,} tweets, likely due to current events"
            elif subcategory == 'Healthcare':
                return f"Healthcare policy discussion trending with {tweet_volume:,} tweets of public engagement"
            else:
                return f"Political topic in {subcategory.lower()} generating public discussion"
        
        elif category == 'Sports':
            if 'formula1' in topic.lower() or 'gp' in topic.lower():
                return f"Formula 1 racing content with {tweet_volume:,} tweets from fans and motorsport enthusiasts"
            else:
                return f"Sports-related content in {subcategory.lower()} with active fan engagement"
        
        elif category == 'Technology':
            if subcategory == 'Crypto':
                return f"Cryptocurrency or blockchain-related topic with {tweet_volume:,} tweets from investors and enthusiasts"
            elif subcategory == 'Ai':
                return f"Artificial intelligence or tech innovation topic generating {tweet_volume:,} tweets"
            else:
                return f"Technology topic in {subcategory.lower()} trending among tech communities"
        
        elif category == 'Culture/Social':
            if topic.startswith('#'):
                return f"Social media hashtag trending with {tweet_volume:,} tweets across various communities"
            else:
                return f"Cultural or social topic generating {tweet_volume:,} tweets of community engagement"
        
        else:
            return f"Trending topic in {category.lower()} with {tweet_volume:,} tweets of public interest"
    
    def _generate_trending_reason(self, topic: str, category: str, tweet_volume: int, region: str) -> str:
        """Generate reason why this topic is trending."""
        
        if tweet_volume > 1000000:
            return "Viral content with massive global engagement across social media platforms"
        elif tweet_volume > 500000:
            return "Major event or announcement driving significant social media discussion"
        elif tweet_volume > 200000:
            return "Popular topic generating substantial engagement within interested communities"
        elif tweet_volume > 50000:
            return "Emerging trend gaining traction among specific audience segments"
        else:
            return "Niche topic trending within specialized communities or regions"
    
    def _analyze_web3_relevance(self, topic_lower: str, category: str) -> str:
        """Analyze Web3/crypto relevance for a topic."""
        if category == 'Technology':
            for subcat, keywords in self.WEB3_KEYWORDS.items():
                if any(keyword in topic_lower for keyword in keywords):
                    return subcat.replace('_', ' ').title()
        return "none"
    
    def _analyze_crypto_connection(self, topic_lower: str, tweet_volume: int) -> str:
        """Analyze crypto connection for a topic."""
        if tweet_volume > 100000:
            return "direct"
        elif tweet_volume > 50000:
            return "indirect"
        else:
            return "none"
    
    def _analyze_tech_relationship(self, topic_lower: str, category: str, tweet_volume: int) -> str:
        """Analyze tech relationship for a topic."""
        if category == 'Technology':
            if tweet_volume > 100000:
                return "emerging"
            elif tweet_volume > 50000:
                return "standard"
            else:
                return "disruptive"
        return "standard" 