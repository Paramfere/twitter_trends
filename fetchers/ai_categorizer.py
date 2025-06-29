"""AI-powered topic categorization using OpenAI and Claude APIs."""

import logging
import os
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class TopicAnalysis:
    """Analysis result for a trending topic."""
    category: str
    subcategory: str
    significance_score: int  # 1-10 scale
    sentiment: str  # positive, negative, neutral
    context: str
    trending_reason: str

def get_logger():
    """Get configured logger."""
    return logging.getLogger(__name__)

class AITopicCategorizer:
    """Intelligent topic categorization using AI APIs."""
    
    def __init__(self, preferred_api: str = "openai"):
        """Initialize the categorizer.
        
        Args:
            preferred_api: "openai" or "claude" (falls back to other if primary fails)
        """
        self.preferred_api = preferred_api
        self.logger = get_logger()
        
        # Initialize API clients
        self._init_openai()
        self._init_claude()
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.openai_available = True
            self.logger.info("OpenAI client initialized")
        except ImportError:
            self.logger.warning("OpenAI library not available")
            self.openai_available = False
        except Exception as e:
            self.logger.warning(f"OpenAI initialization failed: {e}")
            self.openai_available = False
    
    def _init_claude(self):
        """Initialize Claude client."""
        try:
            import anthropic
            self.claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.claude_available = True
            self.logger.info("Claude client initialized")
        except ImportError:
            self.logger.warning("Anthropic library not available")
            self.claude_available = False
        except Exception as e:
            self.logger.warning(f"Claude initialization failed: {e}")
            self.claude_available = False
    
    def _get_categorization_prompt(self, topics: List[Dict]) -> str:
        """Generate the AI prompt for topic categorization."""
        topics_text = ""
        for i, topic in enumerate(topics, 1):
            url_info = f" (URL: {topic.get('url', 'N/A')})" if topic.get('url') else ""
            topics_text += f"{i}. {topic['topic']} - {topic['tweet_volume']:,} tweets from {topic['region']}{url_info}\n"
        
        return f"""You are an expert social media analyst with deep knowledge of current events, pop culture, politics, and trending phenomena. Analyze these {len(topics)} trending Twitter topics and provide detailed, specific insights.

TRENDING TOPICS TO ANALYZE:
{topics_text}

For each topic, provide analysis in this exact JSON format:
{{
  "topic_name": {{
    "category": "main category",
    "subcategory": "specific subcategory", 
    "significance_score": 1-10,
    "sentiment": "positive/negative/neutral",
    "context": "detailed explanation of what this topic is about and why it matters",
    "trending_reason": "specific reason why this is trending right now"
  }}
}}

ANALYSIS GUIDELINES:
- **Be SPECIFIC and DETAILED** - avoid generic explanations
- **Research knowledge** - use your training data to identify what these topics likely refer to
- **Context matters** - consider the region (US vs Singapore trends differ)
- **Current events** - think about what might be happening now that would cause these trends
- **Significance scoring**: 
  • 9-10: Major global events, breaking news, major political developments
  • 7-8: Significant cultural events, major entertainment news, important policy changes
  • 5-6: Popular culture moments, viral content, regional news
  • 3-4: Niche interests, smaller events, fan communities
  • 1-2: Very specific or unclear topics

CATEGORIES (choose the most specific):
- Politics (elections, policies, government, politicians, military, healthcare policy)
- Sports (specific sports, athletes, competitions, teams, Formula 1, etc.)
- Entertainment (movies, music, celebrities, TV shows, gaming, fan events)
- Technology (AI, crypto, apps, companies, tech innovations, blockchain)
- News/Events (breaking news, current events, disasters, major announcements)
- Culture/Social (social movements, viral content, memes, hashtags, trends)
- Business/Finance (markets, companies, economy, stocks, corporate news)
- Global/Places (countries, cities, international affairs, travel, locations)
- Daily/Lifestyle (everyday topics, holidays, routine activities, weather)

EXAMPLE of GOOD analysis:
"Bitcoin": {{
  "category": "Technology",
  "subcategory": "Cryptocurrency",
  "significance_score": 6,
  "sentiment": "neutral",
  "context": "Bitcoin is experiencing price volatility amid regulatory discussions in the US and institutional adoption news",
  "trending_reason": "Recent SEC announcements and major investment firm adoption driving market speculation"
}}

AVOID generic responses like "trending topic" or "high engagement" - be specific!

Return ONLY the JSON response, no other text."""

    def _parse_ai_response(self, response_text: str) -> Dict[str, TopicAnalysis]:
        """Parse AI response into TopicAnalysis objects."""
        try:
            # Clean the response (remove markdown code blocks if present)
            clean_response = response_text.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            
            data = json.loads(clean_response.strip())
            
            results = {}
            for topic_name, analysis in data.items():
                results[topic_name] = TopicAnalysis(
                    category=analysis.get("category", "Other"),
                    subcategory=analysis.get("subcategory", "General"),
                    significance_score=int(analysis.get("significance_score", 5)),
                    sentiment=analysis.get("sentiment", "neutral"),
                    context=analysis.get("context", ""),
                    trending_reason=analysis.get("trending_reason", "")
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to parse AI response: {e}")
            self.logger.debug(f"Raw response: {response_text}")
            return {}
    
    def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI API."""
        if not self.openai_available:
            return None
            
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert social media analyst. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            return None
    
    def _call_claude(self, prompt: str) -> Optional[str]:
        """Call Claude API."""
        if not self.claude_available:
            return None
            
        try:
            response = self.claude_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
            
        except Exception as e:
            self.logger.error(f"Claude API call failed: {e}")
            return None
    
    def categorize_topics(self, topics: List[Dict]) -> Dict[str, TopicAnalysis]:
        """Categorize a list of topics using AI.
        
        Args:
            topics: List of topic dicts with 'topic', 'tweet_volume', 'region' keys
            
        Returns:
            Dict mapping topic names to TopicAnalysis objects
        """
        if not topics:
            return {}
        
        start_time = time.time()
        self.logger.info(f"Categorizing {len(topics)} topics using AI...")
        
        # Process topics in smaller batches for better AI focus
        batch_size = 8  # Smaller batches for more detailed analysis
        all_results = {}
        
        for i in range(0, len(topics), batch_size):
            batch = topics[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(topics) + batch_size - 1) // batch_size
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} topics)")
            
            prompt = self._get_categorization_prompt(batch)
            
            # Try preferred API first, then fallback
            apis_to_try = [self.preferred_api]
            if self.preferred_api == "openai" and self.claude_available:
                apis_to_try.append("claude")
            elif self.preferred_api == "claude" and self.openai_available:
                apis_to_try.append("openai")
            
            response_text = None
            for api in apis_to_try:
                self.logger.info(f"Trying {api} API for batch {batch_num}...")
                
                if api == "openai":
                    response_text = self._call_openai(prompt)
                else:
                    response_text = self._call_claude(prompt)
                
                if response_text:
                    self.logger.info(f"Successfully got response from {api} for batch {batch_num}")
                    break
            
            if not response_text:
                self.logger.error(f"All AI APIs failed for batch {batch_num}, using fallback")
                batch_results = self._fallback_categorization(batch)
            else:
                # Parse the response
                batch_results = self._parse_ai_response(response_text)
                
                # Fill in any missing topics with fallback
                for topic in batch:
                    topic_name = topic['topic']
                    if topic_name not in batch_results:
                        self.logger.warning(f"AI didn't categorize '{topic_name}', using fallback")
                        fallback = self._fallback_single_topic(topic)
                        batch_results[topic_name] = fallback
            
            all_results.update(batch_results)
            
            # Add a small delay between batches to avoid rate limiting
            if i + batch_size < len(topics):
                time.sleep(1)
        
        elapsed = time.time() - start_time
        self.logger.info(f"AI categorization completed in {elapsed:.1f}s")
        
        return all_results
    
    def _fallback_categorization(self, topics: List[Dict]) -> Dict[str, TopicAnalysis]:
        """Fallback to rule-based categorization if AI fails."""
        results = {}
        for topic in topics:
            results[topic['topic']] = self._fallback_single_topic(topic)
        return results
    
    def _fallback_single_topic(self, topic: Dict) -> TopicAnalysis:
        """Rule-based categorization for a single topic."""
        topic_name = topic['topic']
        topic_lower = topic_name.lower()
        volume = topic['tweet_volume']
        
        # Simple rule-based categorization (same as before)
        if any(word in topic_lower for word in ['medicaid', 'republican', 'democrat', 'senator', 'politics']):
            category, subcategory = "Politics", "Government"
        elif any(word in topic_lower for word in ['gp', 'formula', 'ferrari', 'sports', 'football']):
            category, subcategory = "Sports", "Racing/General"
        elif any(word in topic_lower for word in ['bitcoin', 'ethereum', 'crypto', 'ai', 'tech']):
            category, subcategory = "Technology", "Crypto/AI"
        elif topic_name.startswith('#'):
            category, subcategory = "Culture/Social", "Hashtag"
        elif volume >= 500000:
            category, subcategory = "News/Events", "Viral"
        else:
            category, subcategory = "Other", "General"
        
        return TopicAnalysis(
            category=category,
            subcategory=subcategory,
            significance_score=min(10, max(1, volume // 100000 + 1)),
            sentiment="neutral",
            context=f"Trending topic: {topic_name}",
            trending_reason="High engagement on social media"
        ) 