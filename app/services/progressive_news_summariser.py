"""
**************************************************************************
 * Project: SmartNest News Summarizer
 * Author: SmartNest Systems
 * 
 * ©️ [2025] [SmartNest International]. All rights reserved.
 * 
 * This software and associated documentation files (the “Software”) are 
 * the proprietary and confidential property of the author(s) and/or 
 * their affiliated institution(s), and are protected by applicable 
 * intellectual property laws and international treaties.
 * 
 * Unauthorized copying, distribution, modification, or use of this 
 * software, in whole or in part, without the express written permission 
 * of the copyright holder is strictly prohibited.
 * 
 * The Software is provided "AS IS", without warranty of any kind, express 
 * or implied, including but not limited to the warranties of 
 * merchantability, fitness for a particular purpose and noninfringement. 
 * In no event shall the authors or copyright holders be liable for any 
 * claim, damages or other liability, whether in an action of contract, 
 * tort or otherwise, arising from, out of or in connection with the 
 * Software or the use or other dealings in the Software.
 
 **************************************************************************
 """

import ssl
import re
import time
import nltk
import pandas as pd
from newspaper import Article
from nltk.tokenize import sent_tokenize
import google.generativeai as genai
import os
import sys
from datetime import date

from pymongo.mongo_client import MongoClient  # Updated import for clarity

from app.config import settings


ticker = sys.argv[1]  # The ticker will be passed from reasoning.py
# ticker = "HDFCBANK"
t = ticker
ticker = ticker + ".NS"  # Append ".NS" for Indian stocks if needed

# SSL handling
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# else do the generation as usual


# Download tokenizer
nltk.download('punkt', quiet=True)

# MongoDB connection
client = MongoClient(settings.mongo_uri, tls=True, tlsAllowInvalidCertificates=True)

try:
    client.admin.command('ping')
    # print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")

db = client['stock_news']
articles = db['stock_articles']
collection = db['progressive_news']


# Configure Gemini
genai.configure(api_key=settings.gemini_api_key)


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
    except:
        return None 


# CSV Inputs
# csv_file_path = f'{t}_news_today.csv'
# csv_file_path2 = f'{t}_NSE_news_today.csv'

# df = pd.read_csv(csv_file_path)
# df2 = pd.read_csv(csv_file_path2)

# daily_summary2 = "No NSE market news available for this stock today."
# daily_summary = "No market news available for this stock today."


# # ===== NSE Market News Summary Generation =====
# unique_stocks2 = df2["Ticker"].unique()
# for stock_ticker in unique_stocks2:
#     df_filtered = df2[df2["Ticker"].str.upper() == stock_ticker.upper()]
#     articles_content = []

#     for _, row in df_filtered.iterrows():
#         url = row["Link"]
#         result = extract_article_content(url)
#         if result:
#             article_text = f"Title: {result['title']}\nAuthors: {', '.join(result['authors']) or 'Unknown'}\nDate: {result['date'] or 'Unknown'}\n\n{result['text']}\nEOF"
#             articles_content.append(article_text)
#     combined_content = "\n\n".join(articles_content)

#     analyst_summary = f"""
# Mimic the role of an experienced financial analyst and distill the stock's daily market related news and generate a concise daily news summary.
# Please structure the output exactly using the format provided below. 
# Keep the language analytical, concise, and factual. Do not add extra commentary or headings.

# 📅 Date: [DD MMM YYYY]
# 🏢 Company: {ticker}
# 1. Stock-Specific News
# [Summarize key news directly impacting the stock price: earnings, market sentiment, product updates, etc.]

# 2. Market Reaction & Trading Insights
# [Include price change, volume spikes, and any notable market impact]

# 3. Stock Price Dynamics
# [If applicable, note any significant price movement trends, volatility, etc.]
# """

#     combined_content += "\n\n" + analyst_summary

#     try:
#         response2 = model.generate_content(combined_content)
#         daily_summary2 = response2.text
#     except Exception as e:
#         print(f"Error generating Gemini NSE summary: {e}")
#         continue

# # ===== Generic Company News Summary Generation =====
# unique_stocks = df["Ticker"].unique()

# for stock_ticker in unique_stocks:
#     df_filtered = df[df["Ticker"].str.upper() == stock_ticker.upper()]
#     articles_content = []

#     for _, row in df_filtered.iterrows():
#         url = row["Link"]
#         result = extract_article_content(url)
#         if result:
#             article_text = f"Title: {result['title']}\nAuthors: {', '.join(result['authors']) or 'Unknown'}\nDate: {result['date'] or 'Unknown'}\n\n{result['text']}\nEOF"
#             articles_content.append(article_text)

#     combined_content = "\n\n".join(articles_content)

#     analyst_summary = f"""
# Mimic the role of an experienced financial analyst and distill the company's generic daily news and generate a concise daily news summary.
# Please structure the output exactly using the format provided below. 
# Keep the language analytical, concise, and factual. Do not add extra commentary or headings.

# 📅 Date: [DD MMM YYYY]
# 🏢 Company: {ticker}
# 1. Key Company Announcements
# [Summarize any important announcements that are not stock-specific.]

# 2. Strategic Developments
# [Include new business strategies, innovations, product launches.]

# 3. Corporate Sentiment
# [General sentiment around the company’s recent actions.]
# """

#     combined_content += "\n\n" + analyst_summary

#     try:
#         response = model.generate_content(combined_content)
#         daily_summary = response.text
#     except Exception as e:
#         print(f"Error generating Gemini generic summary: {e}")
#         continue

# ===== Progressive Summary Generation =====
progressive_news_doc = collection.find_one({"stock": ticker})
progressive_summary = progressive_news_doc.get('progressive_summary', "") if progressive_news_doc else "No progressive summary available for this stock."

system_prompt = f"""
Mimic the role of a financial analyst with the task of synthesizing a full-fledged news summary.
Integrate the most pertinent information, distinguishing factual news and analysts' opinions.
Please structure the output exactly using the format provided below. 
Keep the language analytical, concise, and factual. Do not add extra commentary or headings.
Refer to the summaries provided in the prompt for context and continuity.
*Give more importance to the latest news, but also consider the historical context.*
If no news is available, use web search to fill in the gaps.
"""

prompt = f"""

Structure your output exactly as shown below. Use concise, analytical language and avoid any additional headings or disclaimers.

# Summary Period: [Start Date - End Date]
# Company: {ticker}

## 1. Ongoing Trends & Recap of Previous News
- Interpret EMA 10, 20, 50 for short-to-medium term trends.
- Discuss EMA 100 and EMA 200 for long-term trend alignment or divergence.
- Use MACD line, signal line, and histogram to judge momentum shifts.
- Analyze Supertrend direction to confirm bullish/bearish bias.

## 2. Stock-Specific Developments
- Assess RSI to determine overbought (>70) or oversold (<30) conditions.
- Note significant volume surges or drops and their context.
- Discuss VWAP relative to price for intraday strength or weakness.

## 3. General Company Progress
- Report any detected candlestick patterns (Hammer, Doji, Bullish Engulfing, Bearish Engulfing).
- Describe what each observed pattern implies for short-term price action.

## 4. Market & Analyst Sentiment Overview
- Highlight if multiple indicators align for a strong signal (e.g., EMA alignment + bullish MACD + bullish candle).
- If signals are conflicting (e.g., RSI overbought but bullish MACD), mention this and advise caution.

## 5. Impact on Investment Considerations
- Identify significant support/resistance and potential entry levels based on EMAs, Supertrend, or historical price action.
- Discuss how current price action relates to these levels.
- Provide an actionable summary of the technical outlook, including price levels to watch.

"""
# --- Fetch all docs for the stock ---
cursor = articles.find({"Ticker": t}, {"_id": 0, "Link": 1, "Date": 1})
# --- Store the results ---

for doc in cursor:
    url = doc['Link']
    article = extract_article_content(url)
    if article and article['text']:
        prompt += f"Title: {article['title']}\nDate: {doc['Date']}\n\n{article['text']}\nEOF\n"

try:
    model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=system_prompt, generation_config=genai.GenerationConfig(
        temperature=0,
        top_p=1
    ))
    response_progressive = model.generate_content(prompt)
    progressive_summary = response_progressive.text
    print(progressive_summary)

    collection.update_one(
        {"stock": ticker},
        {
            "$set": {
                "progressive_summary": progressive_summary,
                "date": date.today().strftime("%Y-%m-%d")
            }
        },
        upsert=True
    )

except Exception as e:
    print(f"Error generating progressive summary: {e}")
client.close()
