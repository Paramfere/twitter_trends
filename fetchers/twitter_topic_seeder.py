from __future__ import annotations
import os, logging
from datetime import datetime, timezone
from apify_client import ApifyClient
import re  # Import re for parsing
from typing import List, Dict
from dotenv import load_dotenv
import time

load_dotenv()  # Load environment variables from .env file

# Karamelo actor country codes (from documentation)
KARAMELO_REGION_CODES = {
    "US": "2",   # United States  
    "SG": "20",  # Singapore
}

def parse_tweet_volume(volume_str: str) -> int:
    """Convert tweet volume string (e.g., '35.8k', '94,633Tweets') to integer."""
    if not volume_str:
        return 0
        
    try:
        # Remove 'Tweets' suffix and commas
        volume_str = volume_str.replace("Tweets", "").replace(",", "").strip()
        
        # Handle k/m suffixes
        volume_str_lower = volume_str.lower()
        if volume_str_lower.endswith("k"):
            return int(float(volume_str_lower[:-1]) * 1000)
        if volume_str_lower.endswith("m"):
            return int(float(volume_str_lower[:-1]) * 1000000)
            
        return int(float(volume_str))
    except (ValueError, TypeError):
        return 0

def fetch_topics_multi(limit: int = 999) -> List[Dict]:
    """Fetch trending topics using Karamelo actor."""
    client = ApifyClient(os.getenv("APIFY_TOKEN"))
    all_topics = []
    now = datetime.now(timezone.utc)
    
    for region_key, country_code in KARAMELO_REGION_CODES.items():
        for attempt in range(3): # Try up to 3 times
            try:
                # Use Karamelo actor with country code
                payload = {
                    "country": country_code,
                    "proxyOptions": { "useApifyProxy": True }
                }
                logging.info(f"Requesting topics for {region_key} (Country code: {country_code}) - Attempt {attempt + 1}")
                run = client.actor("karamelo/twitter-trends-scraper").call(
                    run_input=payload, timeout_secs=120
                )
                items = client.dataset(run["defaultDatasetId"]).list_items().items
                
                # Save ALL items without limiting
                for row in items:
                    all_topics.append({
                        "region": region_key,
                        "window": row.get("timePeriod", "unknown"),
                        "topic": row.get("trend", ""),
                        "tweet_volume": parse_tweet_volume(row.get("volume", "0")),
                        "url": None,  # Karamelo actor doesn't provide URLs
                        "fetched_at": now,
                    })
                    
                logging.info(f"Saved {len(items)} topics for {region_key}")
                break # Success, exit the retry loop
            except Exception as e:
                logging.error(f"Karamelo seeder {region_key} failed on attempt {attempt + 1}: {e}")
                if attempt < 2: # If not the last attempt
                    wait_time = 2 ** attempt # Exponential back-off: 1s, 2s
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Karamelo seeder {region_key} failed after 3 attempts.")
    
    return all_topics 