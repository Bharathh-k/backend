import json
import os
import time
from datetime import date, datetime

import google.generativeai as genai
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from pymongo.mongo_client import MongoClient

from app.config import settings

# ========== HEADERS ==========
headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9"
}

# ========== URLS ==========
te_indicators_url = "https://tradingeconomics.com/india/indicators"
te_gdp_url        = "https://tradingeconomics.com/india/gdp-growth"
yahoo_oil_url     = "https://finance.yahoo.com/quote/BZ=F?p=BZ=F"
wise_url          = "https://wise.com/us/currency-converter/usd-to-inr-rate"


# MongoDB connection
client = MongoClient(settings.mongo_uri, tls=True, tlsAllowInvalidCertificates=True)

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")

db = client['macro']
collection = db['macro']

# Ensure DB and collection creation by inserting a dummy document if collection is empty
if collection.count_documents({}) == 0:
    collection.insert_one({"initialized": True, "timestamp": time.time()})
    print("Initialized 'macro' database and 'macro' collection with dummy document.")

# ========== SCRAPER FUNCTIONS ==========

def get_gdp_yoy():
    try:
        res  = requests.get(te_gdp_url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        table = soup.find("table", class_="table")
        for row in table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) >= 2:
                label = cols[0].get_text(strip=True)
                if "YoY" in label or "Annual" in label:
                    return cols[1].get_text(strip=True)
        return "N/A"
    except Exception as e:
        return f"Error: {e}"

def get_te_indicator(indicator_keyword):
    try:
        res  = requests.get(te_indicators_url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        table = soup.find("table", class_="table")
        for row in table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) >= 2:
                label = cols[0].get_text(strip=True)
                value = cols[1].get_text(strip=True)
                if indicator_keyword.lower() in label.lower():
                    return value
        return "N/A"
    except Exception as e:
        return f"Error: {e}"

def get_usd_inr():
    try:
        res  = requests.get(wise_url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        rate_div = soup.find("span", class_="text-success")
        if rate_div:
            return rate_div.get_text(strip=True) + " INR"
        return "N/A"
    except Exception as e:
        return f"Error: {e}"

def get_brent_crude_price():
    try:
        # Brent crude oil ticker symbol on Yahoo Finance
        brent = yf.Ticker("BZ=F")
        
        # Get the latest market data
        data = brent.history(period="1d")
        if not data.empty:
            price = data["Close"].iloc[-1]
            return f"{price:.2f} USD/barrel"
        return "N/A"
    except Exception as e:
        return f"Error: {e}"

# ========== ADDITIONAL INDICATOR FUNCTIONS ==========
def get_additional_indicators():
    additional = {}
    additional["Consumer Confidence Index"] = get_te_indicator("Consumer Confidence")
    additional["PMI Manufacturing"] = get_te_indicator("Manufacturing PMI")
    additional["Fiscal Deficit (% of GDP)"] = get_te_indicator("Government Budget")
    additional["Current Account Balance"] = get_te_indicator("Current Account")
    return additional

# ========== GEMINI AI SUMMARY ==========
def generate_economic_summary(indicators):
    genai.configure(api_key=settings.gemini_api_key)
    system_instruction = """You are a seasoned financial analyst specializing in macroeconomic strategy. Based on the latest macroeconomic indicators, generate a structured summary of the Indian economy. Analyze recent economic conditions and trends across inflation, growth, interest rates, currency, and trade.
                            Highlight both opportunities and systemic risks using only the available data. Avoid speculative or generic commentary."""
    model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=system_instruction, generation_config=genai.GenerationConfig(
        temperature=0,
        top_p=1
    ))
    prompt = f"""
Structure your output exactly as shown below. Use concise, analytical language and avoid any additional headings or disclaimers.

 Date: {datetime.now().strftime('%d %b %Y')}
 Region: India

1. Growth & Output
[Summarize GDP growth trends and economic momentum.]
[e.g., GDP YoY growth at 7.4% signals robust post-pandemic recovery.]

2. Inflation & Monetary Policy
[Summarize CPI inflation trend and RBI repo stance. Discuss potential impact on rates or liquidity.]
[e.g., CPI remains at 3.16%, well within RBI's target; repo at 5.5% suggests accommodative stance.]

3. Labour & Employment
[Summarize unemployment trends and overall labour market health.]
[e.g., Unemployment at 7.9% indicates persistent job market challenges.]

4. Currency & External Sector
[Summarize USD/INR trend, Brent crude prices, current account status, and global trade dynamics.]
[e.g., INR stable at x/USD; moderate oil prices support macroeconomic stability.]

5. Business & Consumer Sentiment
[Summarize PMI trends, consumer confidence, and business outlook signals.]
[e.g., Manufacturing PMI at 57.6 shows expansion; consumer confidence at 95.5 reflects optimism.]

6. Fiscal & Budget Position
[Summarize fiscal deficit status and government spending trajectory.]
[e.g., Fiscal deficit at 4.8% of GDP shows fiscal consolidation progress.]

7. Market Risks & Outlook
[Summarize near-term risks and macro opportunities across sectors. No disclaimers.]
[e.g., Global uncertainty and oil price volatility remain key risks; domestic consumption drives growth.]

8. Final Take
[Provide a macro sentiment explanation and a 3–6 month outlook.]
[e.g., Outlook; strong growth momentum offset by external headwinds.]

MACROECONOMIC INDICATORS (JSON):
{json.dumps(indicators, indent=2)}
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {e}"

# ========== MAIN RUNNER ==========
def run_accurate_macro_scraper():
    core = {
        "GDP Growth Rate (YoY)"   : get_gdp_yoy(),
        "Inflation Rate (CPI)"    : get_te_indicator("Inflation Rate"),
        "Unemployment Rate"       : get_te_indicator("Unemployment Rate"),
        "Interest Rate (Repo)"    : get_te_indicator("Interest Rate"),
        "Exchange Rate USD/INR"   : get_usd_inr(),
        "Crude Oil Price (Brent)" : get_brent_crude_price()
    }
    core.update(get_additional_indicators())
    summary = generate_economic_summary(core)
    return summary

# ========== SECTOR-LEVEL ANALYSIS & FILE SAVING ==========

def analyze_sector_with_macro(macro_report, sector):
    genai.configure(api_key=settings.gemini_api_key)
    system_instruction = "You are a seasoned financial analyst specializing in sectoral macroeconomic insights."
    model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=system_instruction, generation_config=genai.GenerationConfig(
        temperature=0,
        top_p=1
    ))

    prompt = f"""
Based on the following macroeconomic summary of India:
{macro_report}

Perform a focused analysis of the implications for the "{sector}" sector in India.

Instructions:

Identify which macroeconomic indicators (e.g., GDP growth, inflation, interest rates, exchange rate, etc.) are favorable or unfavorable for the given sector.

Explain how each relevant indicator could influence business performance, investment prospects, or sector-specific demand/supply dynamics.

Keep the language concise, analytical, and investor-friendly.

Do not add general disclaimers or suggest that “more analysis is needed.”

End with a 1–2 line professional outlook and reasoning based on the data.

Your analysis should help an investor or business decision-maker quickly understand how current macroeconomic conditions affect the "{sector}" sector.
...
"""

    try:
        response = model.generate_content(prompt)
        
        # Prepare document filter and update
        doc_filter = {"sector": sector}
        doc_update = {
            "$set": {
                "analysis": response.text,
                "last_updated": str(date.today())
            }
        }
        
        # Upsert sector analysis into MongoDB
        collection.update_one(doc_filter, doc_update, upsert=True)
        
        print(f" Saved analysis for {sector} in MongoDB collection '{collection.name}'")
    except Exception as e:
        print(f" Error generating analysis for {sector}: {e}")

# ========== EXECUTION ==========

if __name__ == "__main__":
    macro_summary = run_accurate_macro_scraper()

    if macro_summary.startswith("Error"):
        print(macro_summary)
    else:

        sectors = [
            'Commodities', 'Financial Services', 'Information Technology',
            'Diversified', 'Industrials', 'Services', 'Energy', 'Healthcare',
            'Consumer Discretionary', 'Fast Moving Consumer Goods',
            'Utilities', 'Telecommunication'
        ]

        for sector in sectors:
            analyze_sector_with_macro(macro_summary, sector)
