from dotenv import load_dotenv
from fetchers.twitter_topic_seeder import fetch_topics_multi

load_dotenv()

def test_multi_seeder():
    rows = fetch_topics_multi(limit=3)
    assert rows, "Seeder returned no rows"
    first = rows[0]
    # check required keys and URL format
    for key in ["region", "window", "topic", "url", "fetched_at"]:
        assert key in first, f"{key} missing"
    assert first["url"].startswith("http"), "URL looks invalid" 