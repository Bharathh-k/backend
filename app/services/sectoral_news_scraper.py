import os
import requests
from bs4 import BeautifulSoup
from datetime import date
import time
import re
from urllib.parse import urlparse, parse_qs
from pymongo import MongoClient, errors

# ------------------------------
# MongoDB setup
# ------------------------------
URI = (
    "mongodb+srv://smartnest26:smartnest26"
    "@cluster0.ceho9t7.mongodb.net/"
    "?retryWrites=true&w=majority"
)
client = MongoClient(URI, tls=True, tlsAllowInvalidCertificates=True)
db = client["sector_news"]
collection = db["sector_articles"]
# Ensure uniqueness per (sector, link)
collection.create_index([("sector", 1), ("link", 1)], unique=True)

# ------------------------------
# Sector → search query map
# ------------------------------
SECTOR_KEYWORDS = {
    "Minerals & Mining": "Minerals and Mining sector India",
    "Capital Markets": "Capital Markets sector India",
    "IT - Software": "IT Software sector India",
    "Diversified": "Diversified business sector India",
    "Finance": "Finance sector India",
    "Construction": "Construction sector India",
    "Commercial Services & Supplies": "Commercial services and supplies India",
    "Oil": "Oil sector India",
    "Pharmaceuticals & Biotechnology": "Pharmaceuticals and biotechnology India",
    "Electrical Equipment": "Electrical equipment sector India",
    "Chemicals & Petrochemicals": "Chemicals and petrochemicals sector India",
    "Textiles & Apparels": "Textiles and apparels India",
    "Beverages": "Beverages sector India",
    "Retailing": "Retail sector India",
    "Paper, Forest & Jute Products": "Paper forest jute products India",
    "Cement & Cement Products": "Cement sector India",
    "Transport Services": "Transport services India",
    "Agricultural, Commercial & Construction Vehicles": "Agricultural and construction vehicles India",
    "Power": "Power sector India",
    "Metals & Minerals Trading": "Metals and minerals trading India",
    "Transport Infrastructure": "Transport infrastructure India",
    "Food Products": "Food products sector India",
    "Consumer Durables": "Consumer durables sector India",
    "Industrial Products": "Industrial products sector India",
    "IT - Services": "IT services sector India",
    "Leisure Services": "Leisure services sector India",
    "Gas": "Gas sector India",
    "Industrial Manufacturing": "Industrial manufacturing India",
    "Healthcare Services": "Healthcare services sector India",
    "Realty": "Real estate sector India",
    "Agricultural Food & other Products": "Agricultural food products India",
    "Fertilizers & Agrochemicals": "Fertilizers and agrochemicals India",
    "Financial Technology (Fintech)": "Fintech sector India",
    "Telecom -  Equipment & Accessories": "Telecom equipment sector India",
    "Auto Components": "Auto components sector India",
    "Household Products": "Household products India",
    "Ferrous Metals": "Ferrous metals sector India",
    "Consumable Fuels": "Consumable fuels sector India",
    "Aerospace & Defense": "Aerospace and defense India",
    "Other Consumer Services": "Consumer services sector India",
    "Automobiles": "Automobile sector India",
    "Banks": "Banking sector India",
    "Other Utilities": "Utility sector India",
    "Entertainment": "Entertainment sector India",
    "Personal Products": "Personal products sector India",
    "Non - Ferrous Metals": "Non-ferrous metals sector India",
    "Telecom - Services": "Telecom services India",
    "Petroleum Products": "Petroleum products India",
    "IT - Hardware": "IT hardware sector India",
    "Media": "Media sector India",
    "Engineering Services": "Engineering services India",
    "Other Construction Materials": "Construction materials sector India",
    "Insurance": "Insurance sector India",
    "Diversified FMCG": "Diversified FMCG sector India",
    "Cigarettes & Tobacco Products": "Cigarettes and tobacco India",
    "Printing & Publication": "Printing and publication India",
    "Healthcare Equipment & Supplies": "Healthcare equipment supplies India",
    "Diversified Metals": "Diversified metals sector India"
}

# ------------------------------
# Scraping parameters
# ------------------------------
TIME_PATTERN = re.compile(r"(?:\d+\s*(?:minutes?|hours?)\s*ago)", re.IGNORECASE)
HEADERS = {"User-Agent": "Mozilla/5.0"}
TODAY = date.today().strftime("%Y-%m-%d")

# ------------------------------
# Main scraping loop
# ------------------------------
for idx, (sector, query) in enumerate(SECTOR_KEYWORDS.items(), start=1):
    pulled_count = 0
    saved_count = 0

    search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}&tbm=nws"

    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=10)
        html = resp.text.lower()
        if "unusual traffic" in html:
            print(" Google blocked your request. Exiting.")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        anchors = soup.find_all("a", href=True)

        for a in anchors:
            parent_text = a.find_parent().get_text(" ", strip=True)
            match = TIME_PATTERN.search(parent_text)
            if not match:
                continue

            title = a.get_text(strip=True)
            raw_href = a["href"]
            # Extract the real URL
            link = (
                parse_qs(urlparse(raw_href).query).get("q", [""])[0]
                if "/url?q=" in raw_href else raw_href
            )
            if not title or "google.com" in link:
                continue

            pulled_count += 1

            doc = {
                "sector": sector,
                "link": link,
                "Date": TODAY  # always use lowercase 'date'
            }
            try:
                collection.insert_one(doc)
                saved_count += 1
            except errors.DuplicateKeyError:
                # Duplicate: already saved for this sector+link
                pass

        print(
            f"[{idx}/{len(SECTOR_KEYWORDS)}]  {sector}: "
            f"Pulled {pulled_count}, Saved {saved_count} unique"
        )

    except Exception as e:
        print(f"[{idx}/{len(SECTOR_KEYWORDS)}]  Error for {sector}: {e}")

print("✅ Sectoral news scraping complete.")
client.close()