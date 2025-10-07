import requests
from bs4 import BeautifulSoup
import re
from datetime import date
from pymongo import MongoClient, errors
import sys

ticker = sys.argv[1]
# ticker = "HDFCBANK"
query = f"{ticker} nse"
url = f"https://www.google.com/search?q={query}&tbm=nws"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")
time_pattern = re.compile(r"\b\d+\s+(minutes?|mins?|hours?)\s+ago\b", re.IGNORECASE)
article_blocks = soup.find_all("a", href=True)
today = date.today().strftime("%Y-%m-%d")

# MongoDB setup
client = MongoClient(
    "mongodb+srv://smartnest26:smartnest26@cluster0.ceho9t7.mongodb.net/"
    "?retryWrites=true&w=majority",
    tls=True,
    tlsAllowInvalidCertificates=True,
    tlsAllowInvalidHostnames=True,
    serverSelectionTimeoutMS=10000,
    connectTimeoutMS=10000
)
db = client["stock_news"]
collection = db["stock_articles"]

# Ensure compound unique index exists (Ticker+Link+Date)
collection.create_index([("Ticker", 1), ("Link", 1)], unique=True)

inserted, skipped = 0, 0

for tag in article_blocks:
    parent_text = tag.find_parent().get_text(" ", strip=True)
    time_match = time_pattern.search(parent_text)
    if time_match:
        title = tag.get_text(strip=True)
        link = tag["href"]
        if title and "google.com" not in link:
            doc = {
                "Ticker": ticker,
                "Link": link,
                "Date": today
            }
            try:
                collection.insert_one(doc)
                inserted += 1
            except errors.DuplicateKeyError:
                skipped += 1

print(f" Inserted {inserted} new news links for {ticker} on {today}.")
if skipped:
    print(f" Skipped {skipped} duplicate links.")

