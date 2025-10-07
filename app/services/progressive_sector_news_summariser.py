from datetime import date, datetime, timedelta
import sys
import time

import google.generativeai as genai
import re
from newspaper import Article
from nltk.tokenize import sent_tokenize
from pymongo import MongoClient, errors

from app.config import settings

# Config
genai.configure(api_key=settings.gemini_api_key)
model = genai.GenerativeModel("gemini-2.5-flash-lite-preview-06-17", generation_config=genai.GenerationConfig(
        temperature=0,
        top_p=1
    ))

client = MongoClient(
    settings.mongo_uri,
    tls=True,
    tlsAllowInvalidCertificates=True,
    tlsAllowInvalidHostnames=True,
    serverSelectionTimeoutMS=10000,
    connectTimeoutMS=10000
)
db = client["sector_news"]
articles_col = db["sector_articles"]
prog_col  = db["progressive_sector_news"]

prog_col.create_index([("sector", 1)], unique=True)

today_str = date.today().strftime("%Y-%m-%d")
seven_days_ago = (date.today() - timedelta(days=6)).strftime("%Y-%m-%d")

# --- Article Extraction Helper ---
def clean_text(text):
    ad_patterns = [
        r'advertisement', r'sponsored content', r'subscribe now',
        r'sign up for our newsletter', r'related articles', r'\[.?ad.?\]', 
        r'share this article', r'follow us on'
    ]
    cleaned_text = text
    for pattern in ad_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = '\n'.join(line.strip() for line in cleaned_text.split('\n') if line.strip())
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    return cleaned_text

def extract_article_content(url):
    try:
        article = Article(url)
        article.download()
        time.sleep(2)
        article.parse()
        sentences = sent_tokenize(article.text)
        processed_text = '\n'.join(sentences)
        final_text = clean_text(processed_text)
        return {
            'title': article.title,
            'text': final_text,
            'authors': article.authors,
            'date': article.publish_date
        }
    except Exception as e:
        return None

# --- Main Logic ---
sectors = articles_col.distinct("sector")
if not sectors:
    print("No sectors found in sector_articles collection.")
    sys.exit(0)

for sector in sectors:
    # Fetch articles from last 7 days for the sector
    cursor = articles_col.find(
        {"sector": sector},
        {"_id": 0, "link": 1, "Date": 1}
    )

    if not cursor:
        print(f"No articles found for sector {sector} in last 7 days. Skipping.")
        continue

    # Build up articles content
    prompt_content = ""
    for doc in cursor:
        url = doc['link']
        print(url)
        article = extract_article_content(url)
        print(f"Extracted article: {article['title'] if article else 'N/A'}")
        if article and article['text']:
            prompt_content += (
                f"Title: {article['title']}\nDate: {doc['Date']}\n\n"
                f"{article['text']}\nEOF\n"
            )
    if not prompt_content.strip():
        print(f"No valid content extracted for sector {sector}. Skipping.")
        continue

    # SYSTEM PROMPT
    system_prompt = f"""
You are a senior financial analyst with the task of synthesizing a sector news summary.
Integrate the most pertinent information, distinguishing between factual news and analysts' opinions.
Please structure the output exactly using the format below. Keep language analytical, concise, factual.
Do not add extra commentary or headings. Give more weight to the latest news, but consider historical context.
Refer to all articles provided in the prompt for context and continuity.
*Give more importance to the latest news, but also consider the historical context.*

Structure your output exactly as shown below. Use concise, analytical language and avoid any additional headings or disclaimers.

# Summary Period: [Start Date - End Date]
# Sector: {sector}

## 1. Ongoing Sector Trends & Recap
- Summarize major trends from past news.

## 2. Notable Sector-Specific Developments
- Summarize important sector news.

## 3. General Industry/Market Progress
- Summarize industry-wide changes.

## 4. Market & Analyst Sentiment Overview
- Capture sentiment and consensus.

## 5. Impact on Investment Considerations
- Optional: Implications for investors.
"""

    # Prepend system instructions to prompt (Gemini pip package does not support system_instruction arg)
    full_prompt = system_prompt + "\n\n" + prompt_content

    # Call Gemini (with retries)
    progressive_summary = None
    for attempt in range(1, 4):
        try:
            resp = model.generate_content(full_prompt)
            # time.sleep(2)
            progressive_summary = resp.text.strip()
            break
        except Exception as e:
            print(f"[Attempt {attempt}] Gemini error for sector '{sector}': {e}")
            time.sleep(2)
    if not progressive_summary:
        print(f" Failed to generate progressive summary for {sector}. Skipping.")
        continue

    # Upsert into MongoDB
    try:
        prog_col.update_one(
            {"sector": sector},
            {
                "$set": {
                    "sector": sector,
                    "date_updated": today_str,
                    "progressive_summary": progressive_summary
                }
            },
            upsert=True
        )
        print(f" Progressive summary updated for sector '{sector}'.")
    except Exception as e:
        print(f"[ERROR] MongoDB upsert failed for sector '{sector}': {e}")

print(" All sectoral progressive summaries processed.")
client.close()
