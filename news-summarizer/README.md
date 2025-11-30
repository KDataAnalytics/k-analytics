![Python](https://img.shields.io/badge/python-3.12-blue)
![Scraping](https://img.shields.io/badge/scraping-requests%20%2B%20BeautifulSoup4-brightgreen)
![AI](https://img.shields.io/badge/AI-OpenAI%20Gemini%20Grok-orange)

### [Daily AI News Brief](news-summarizer/)
  - **What it does:** Scrapes live headlines (Google News, BBC, etc.)
  - **Features:** One-line switch between news source and model: OpenAI, Gemini, and Grok backends
  - ### Tech Stack
- **Web scraping:** `requests` + **BeautifulSoup4** — industry-standard combo
- **LLM backends:** OpenAI GPT-3.5-turbo, Google Gemini 2.5-flash, xAI Grok-4  
  → Switch models with one line: `BACKEND = "grok"`
- **Pure Python** — no external services, runs anywhere
- **Demo Output:** [Sample Output](news_summarizer_output.txt)
- **Run:** `python news_summarizer_master.py`
- **Coming Next:** Ready for daily email automation

