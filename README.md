# 🚀 Twitter Trends Intelligence Pipeline

A comprehensive real-time Twitter trending analysis system that provides tactical posting recommendations, momentum tracking, and AI-powered strategic insights for social media marketing and Web3 opportunities.

## 📊 Overview

This system fetches trending topics from Twitter (US & Singapore), analyzes them using rule-based categorization and AI, then generates four types of actionable reports:

1. **📈 Trending Analysis Report** - Category breakdowns, sentiment analysis, regional highlights
2. **🚀 Velocity & Momentum Report** - Real-time change detection between sessions
3. **🎯 Web3 Playbook Report** - Tactical posting recommendations and partnership opportunities  
4. **🧠 AI Intelligence Report** - Strategic insights and immediate marketing actions

## ✨ Key Features

- **Real-time Data Collection** from Twitter trending APIs
- **Automatic Session Management** with timestamped data storage
- **Session-to-Session Comparison** for momentum tracking
- **AI-Powered Analysis** using OpenAI GPT-4 for strategic insights
- **Web3/Crypto Focus** with specialized tech topic analysis
- **Regional Analysis** (US & Singapore markets)
- **Actionable Timing** with specific posting windows (15 min, 30 min, 1 hour)
- **Volume Surge Detection** with percentage change tracking
- **Rank Movement Analysis** for trending position changes

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- OpenAI API key
- Apify API key (for Twitter data)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/twitter_trends.git
cd twitter_trends
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Environment setup:**
Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
APIFY_API_TOKEN=your_apify_api_token_here
```

## 🚀 Quick Start

### Basic Usage

1. **Fetch trending data:**
```bash
python scripts/fetch_topics.py
```

2. **Generate all reports for the latest session:**
```bash
# Replace 'XXX' with your session number (e.g., 024)
python scripts/velocity_report_generator.py --session-id XXX
python scripts/web3_playbook_generator.py --session-id XXX  
python scripts/intelligence_report_generator.py --session-id XXX
```

### Optional: High-Engagement Content Analysis  
Scrape the **5 most significant trending topics** and capture **up to 10 high-engagement tweets** per topic (server-side minimum 500 likes & retweets, 50 bookmarks).  Offline filters tighten this further to likes+RT ≥ 1000 and strong author metrics.

1. **One-off run on an existing session:**
   ```bash
   python scripts/content_analysis_engine.py data/session_043
   ```

2. **Include during data collection:**
   ```bash
   python scripts/fetch_topics.py --with-content-analysis
   ```

Outputs are written under the session directory:
```
raw_data/kaito_data/
  ├── kaito_raw_full_<ts>.json      # full API response (≤50 tweets)
  ├── kaito_raw_data_<ts>.json      # high-quality subset
  ├── kaito_summary_<ts>.csv        # rich CSV summary
content_analysis_report_<ts>.md     # markdown insight report
```

The engine never writes to the project-root `data/` folder and therefore remains Git-ignored.

### Complete Pipeline Example

```bash
# Step 1: Fetch fresh data (creates new session)
python scripts/fetch_topics.py

# Step 2: Generate momentum tracking
python scripts/velocity_report_generator.py --session-id 025

# Step 3: Generate posting recommendations  
python scripts/web3_playbook_generator.py --session-id 025

# Step 4: Generate AI strategic analysis
python scripts/intelligence_report_generator.py --session-id 025
```

### 🚀 One-shot full pipeline

For maximum convenience run the entire workflow (fetch → reports → optional tweet scraping) with a single command:

```bash
# lightweight – no tweet scraping
python scripts/full_pipeline.py

# include high-engagement tweet collection & filtering
python scripts/full_pipeline.py --with-content-analysis
```

The wrapper delegates all heavy-lifting to `fetch_topics.py` while providing a memorable entry-point so you don't have to recall multiple commands.

## 📊 Report Types

### 1. 📈 Trending Analysis Report (Auto-generated)
**File:** `data/session_XXX/analysis/trending_report_*.txt`

- **Overview:** Total topics, tweet volume, regions, significance scores
- **Category Breakdown:** Percentages and averages by category
- **Sentiment Analysis:** Positive/neutral/negative distribution  
- **Regional Highlights:** Top 5 topics per region with scores
- **High Significance Topics:** Detailed descriptions of top performers

### 2. 🚀 Velocity & Momentum Report  
**File:** `data/session_XXX/reports/velocity_report_*.md`

- **New Topics:** First-time trending topics with volume
- **Volume Surges:** Topics with >200% volume increase
- **Volume Drops:** Topics with >50% volume decrease
- **Rank Climbers:** Topics climbing 5+ positions
- **Time Tracking:** Minutes since last session

### 3. 🎯 Web3 Playbook Report
**File:** `data/session_XXX/reports/web3_playbook_*.md`

- **Priority Posting List:** 6 topics with urgency levels and timing windows
- **Content Angle Suggestions:** Specific post ideas and hooks
- **Volume vs Significance Matrix:** Strategic quadrant analysis
- **Partnership Opportunities:** Web3 collaboration ideas with angles
- **Regional Performance:** Tech topic breakdown by region

### 4. 🧠 AI Intelligence Report
**File:** `data/session_XXX/reports/intelligence_report_*.md`

- **Critical Momentum Alerts:** Breakout topics with action plans
- **Volume Surge Analysis:** Growth patterns with strategies
- **Top Performers:** Current highest volume with action windows
- **Web3 & Crypto Pulse:** Market context and correlations
- **Immediate Marketing Actions:** 30-minute tactical opportunities
- **Regional & Category Performance:** Strategic insights and positioning

## 📁 Project Structure

```
twitter_trends/
├── data/                          # All session data
│   ├── session_XXX/              # Individual session folders
│   │   ├── raw_data/             # Original trending data
│   │   ├── analysis/             # Categorized and analyzed data
│   │   └── reports/              # Generated reports
│   ├── session_base/             # Baseline reference session
│   ├── intelligence_state.json   # AI analysis state
│   ├── session_state.json        # Session management state
│   └── velocity_state.json       # Velocity tracking state
├── scripts/                       # Main execution scripts
│   ├── fetch_topics.py           # Data collection
│   ├── velocity_report_generator.py  # Momentum tracking
│   ├── web3_playbook_generator.py    # Posting recommendations
│   ├── intelligence_report_generator.py  # AI analysis
│   ├── session_manager.py        # Session management
│   └── browse_sessions.py        # Session exploration
├── fetchers/                      # Data processing modules
│   ├── twitter_topic_seeder.py   # Twitter API interface
│   ├── rule_categorizer.py       # Topic categorization
│   └── ai_categorizer.py         # AI-powered categorization
├── tests/                         # Test suite
└── requirements.txt               # Dependencies
```

## 🔧 Configuration

### Regions
Currently configured for:
- **US** (Country code: 2)
- **Singapore** (Country code: 20)

### Categories
- Culture/Social
- News/Events  
- Technology (with Web3 subcategories)
- Sports
- Entertainment
- Politics
- Global/Places

### Web3 Subcategories
- AI/ML
- Blockchain
- DeFi
- NFT
- Metaverse
- Gaming
- Products
- Other

## 📈 Sample Output

### Velocity Report Example
```markdown
# 🚀 VELOCITY & MOMENTUM TRACKING REPORT

## 📊 Session Overview
- **Current Session**: 024
- **Total Topics**: 173
- **Time Since Last**: 9.8 minutes

## ⚡ VELOCITY ANALYSIS
### 🆕 New Topics (20)
- **Leyva** (US) - 112,000 tweets
- **Billionaires** (US) - 99,000 tweets

### 🚀 Volume Surges (0)
*No significant volume surges detected.*

### ⬆️ Rank Climbers (11)
- **ALL RISE** (US) - Rank 107 (↑37 positions)
```

### Intelligence Report Example
```markdown
# 📊 Marketing Intelligence Report

## 🚨 CRITICAL MOMENTUM ALERTS
### 🆕 NEW BREAKOUT TOPICS - TOP PRIORITY
**Leyva** (US) - 🚀 EXTREME VIRAL POTENTIAL
- **Volume**: 112,000 tweets (NEW BREAKOUT)
- **Urgency**: 🚨 CRITICAL - Post within 15 minutes
- **Action**: Create immediate reaction content
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Requirements

See `requirements.txt` for full dependency list. Key packages:
- `openai>=1.93.0` - AI analysis
- `pandas` - Data processing
- `httpx` - API requests
- `python-dotenv` - Environment management
- `pathlib` - File system operations

## 🔒 Security

- Store API keys in `.env` file (never commit)
- Add `.env` to `.gitignore`
- Use environment variables for sensitive data

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🚨 Disclaimer

This tool is for educational and research purposes. Ensure compliance with Twitter's Terms of Service and API usage policies. The authors are not responsible for any misuse of this software.

## 🙋‍♂️ Support

For questions, issues, or feature requests:
1. Check existing [Issues](https://github.com/yourusername/twitter_trends/issues)
2. Create a new issue with detailed information
3. Include session logs and error messages when reporting bugs

## 🎯 Roadmap

- [ ] Additional regional support
- [ ] Real-time dashboard
- [ ] Webhook integrations
- [ ] Advanced AI models
- [ ] Custom categorization rules
- [ ] API endpoint exposure
- [ ] Docker containerization

---

**Built with ❤️ for social media intelligence and Web3 marketing** 