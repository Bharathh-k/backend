# analysis_functions.py

import subprocess
import pandas as pd
from multiprocessing import Pipe # Import Pipe here
from pymongo import MongoClient, ASCENDING
import requests
from bs4 import BeautifulSoup
from datetime import date
import ast
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
import random
import sys

from app.config import settings


def run_sector_hierarchy_script(ticker, conn):
    df = pd.read_csv(settings.get_data_path("sector_hierarchy.csv"))
    if ticker in df['SYMBOL'].values:
        conn.close()  # Close the connection if ticker already exists
        return
    try:
        url = f"https://www.screener.in/company/{ticker}/"
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")

        # Get ALL <a> tags from breadcrumb area
        breadcrumb_div = soup.find("div", class_="flex flex-space-between")
        if not breadcrumb_div:
            sector = "N/A"
        a_tags = breadcrumb_div.find_all("a")
        sectors = [a.text.strip() for a in a_tags if a.get("href", "").startswith("/market/")]
        if sectors:
            sector = " > ".join(sectors)
        else:
            sector = "N/A"
    except Exception as e:
        print(f"[{ticker}] Error: {e}")
        sector = "N/A"
    finally:
        # Create a new DataFrame with the ticker and sector
        new_row = pd.DataFrame({"SYMBOL": [ticker], "Sector": [sector]})
        # Append the new row to the existing DataFrame
        df = pd.concat([df, new_row], ignore_index=True)

        # Split into up to 4 pieces, filling missing with NaN
        split_sectors = df['Sector'].str.split(" > ", expand=True).reindex(columns=range(4))

        # Name them Level 1…Level 4
        split_sectors.columns = [f'Level {i+1}' for i in range(4)]

        # Overwrite your existing columns in-place
        df[[f'Level {i+1}' for i in range(4)]] = split_sectors

        # Now save
        df.to_csv(settings.get_data_path("sector_hierarchy.csv"), index=False)
    conn.close()  # Close the connection after processing
    return
    

def run_fundamental_script(ticker, conn):
    try:
        # MongoDB setup
        client = MongoClient(settings.mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
        db = client['funda']
        collection = db['funda']

        # Check if summary exists in DB
        existing_doc = collection.find_one({"ticker": ticker})

        # Scrape np_quarterly first to compare
        def scrape_np_quarterly(soup):
            np_rows = soup.find_all('tr', class_="strong")
            if len(np_rows) < 3:
                return []
            soup_quarterly = BeautifulSoup(str(np_rows[2]), 'html.parser')
            np_quarterly = [td.text.strip() for td in soup_quarterly.find_all('td') if
                            td.text.strip() != '' and (
                                td.text.strip().isdigit() or
                                td.text.strip()[0] == '-' or
                                ',' in td.text.strip() or
                                '.' in td.text.strip())]
            return np_quarterly

        # Fetch consolidated page first
        url = f"https://www.screener.in/company/{ticker}/consolidated/"
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        np_quarterly_scraped = scrape_np_quarterly(soup)

        # fallback if np_quarterly empty
        if not np_quarterly_scraped:
            url = f"https://www.screener.in/company/{ticker}/"
            page = requests.get(url)
            soup = BeautifulSoup(page.content, 'html.parser')
            np_quarterly_scraped = scrape_np_quarterly(soup)
            

        # Load CSV and get np_quarterly value for ticker
        csv_path = settings.get_data_path("stock(11-06-2025).csv")
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            df = pd.DataFrame()  # empty if file doesn't exist

        if not df.empty and 'SYMBOL' in df.columns and 'np_quarterly' in df.columns:
            row = df[df['SYMBOL'] == ticker]
            if not row.empty:
                np_quarterly_csv = row.iloc[0]['np_quarterly']
                try:
                    np_quarterly_csv = ast.literal_eval(np_quarterly_csv)
                except Exception:
                    pass
            else:
                np_quarterly_csv = None
        else:
            np_quarterly_csv = None

        name = soup.find_all(class_="name")
        list_name = [j.get_text().strip() for j in name]
        cleaned_list = ["SYMBOL"] + list_name

        values = soup.find_all(class_="number")
        list_values = [ticker] + [j.get_text() for j in values]

        # Fix High/Low formatting like original code
        if len(list_values) > 4:
            high_low = list_values[3] + "/" + list_values[4]
            list_values[3] = high_low
            for i in range(4, 10):
                if i + 1 < len(list_values):
                    list_values[i] = list_values[i + 1]
            if len(list_values) > 10:
                list_values.pop()

        stock_info = dict(zip(cleaned_list, list_values))

        # Net profit quarterly and yearly
        np = soup.find_all('tr', class_="strong")
        if len(np) > 2:
            soup1 = BeautifulSoup(str(np[2]), 'html.parser')
            np_quarterly = [td.text for td in soup1.find_all('td') if (td.text.isdigit() or td.text[0] == '-' or ',' in td.text or '.' in td.text)]
            stock_info["np_quarterly"] = np_quarterly
        if len(np) > 5:
            soup2 = BeautifulSoup(str(np[5]), 'html.parser')
            np_yearly = [td.text for td in soup2.find_all('td') if (td.text.isdigit() or td.text[0] == '-' or ',' in td.text or '.' in td.text)]
            stock_info["np_yearly"] = np_yearly
            stock_info["np_yearly"].pop()

        # Compounded growths
        cg = soup.find_all('table', class_="ranges-table")
        if len(cg) >= 2:
            def extract_ttm(html):
                soup_cg = BeautifulSoup(html, 'html.parser')
                table_rows = soup_cg.find_all('tr')
                for row in table_rows:
                    cells = row.find_all('td')
                    for cell in cells:
                        if cell.text.strip() == 'TTM:':
                            return cell.find_next('td').text.strip()
                return None

            sales_growth_ttm = extract_ttm(str(cg[0]))
            profit_growth_ttm = extract_ttm(str(cg[1]))
            stock_info["Compounded Sales Growth"] = sales_growth_ttm
            stock_info["Compounded Profit Growth"] = profit_growth_ttm

        # Extract CWIP, Cash Operating Activities, Debtor Days from 'stripe' rows
        stripe_rows = soup.find_all('tr', class_="stripe")
        soup_stripe = BeautifulSoup(str(stripe_rows), 'html.parser')
        rows = soup_stripe.find_all('tr')

        for label in ['CWIP', 'Cash from Operating Activity', 'Debtor Days']:
            for row in rows:
                if label in row.get_text():
                    values = [cell.get_text().strip() for cell in row.find_all('td')[1:]]
                    key = label if label != 'Cash from Operating Activity' else 'Cash from Operating Activities'
                    stock_info[key] = values
                    break

        # Shareholding patterns
        shrp = soup.find_all('tr')
        soup_shrp = BeautifulSoup(str(shrp), 'html.parser')
        new_shrp = str(soup_shrp).replace('[', '').replace(']', '').split(', ')

        html_snippets = [s for s in new_shrp if any(k in s.lower() for k in ['promoter', 'fii', 'dii', 'public'])]

        promoters, fiis, diis, public = [], [], [], []
        for snippet in html_snippets:
            sub_soup = BeautifulSoup(snippet, 'html.parser')
            percentages = [td.get_text(strip=True) for td in sub_soup.find_all('td') if '%' in td.get_text(strip=True)]
            s_lower = snippet.lower()
            if 'promoters' in s_lower:
                promoters.extend(percentages)
            elif 'fiis' in s_lower:
                fiis.extend(percentages)
            elif 'diis' in s_lower:
                diis.extend(percentages)
            elif 'public' in s_lower:
                public.extend(percentages)

        stock_info["Promoter holding"] = promoters
        stock_info["FIIs"] = fiis
        stock_info["DIIs"] = diis
        stock_info["Public"] = public

        # Read existing CSV to update or create new list of rows
        try:
            existing_df = pd.read_csv(csv_path)
            # Check if ticker exists; if yes, update, else append
            if ticker in existing_df['SYMBOL'].values:
                idx = existing_df.index[existing_df['SYMBOL'] == ticker][0]
                for col in stock_info:
                    value = stock_info[col]
                    # Convert numeric strings to float to avoid dtype warning
                    if isinstance(value, str):
                        try:
                            value = float(value.replace(',', ''))
                        except ValueError:
                            pass  # keep original if not a number
                    existing_df.at[idx, col] = value
            else:
                existing_df = pd.concat([existing_df, pd.DataFrame([stock_info])], ignore_index=True)
        except FileNotFoundError:
            existing_df = pd.DataFrame([stock_info])

        # Save updated CSV
        existing_df.to_csv(csv_path, index=False)

        # If summary exists and np_quarterly matches, return existing summary
        if existing_doc and (np_quarterly_scraped == np_quarterly_csv):
            summary = existing_doc.get("summary", "No summary available")
            conn.send(f"{summary}")
            conn.close()
            return

        # ELSE: generate new summary
        # Run your summary script
        result = subprocess.run(
            ['python', '-m', 'app.services.fundamental', ticker],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(settings.project_root.parent)
        )
        db = client['funda']
        collection = db['funda']
        # Get summary output
        summary_output = result.stdout

        # Update summary in MongoDB
        collection.update_one(
            {"ticker": ticker},
            {"$set": {"summary": summary_output}},
            upsert=True
        )

        # Send summary back via pipe
        conn.send(f"{summary_output}")
        conn.close()

    except FileNotFoundError:
        error_msg = "Error: 'fundamental.py' not found. Check the script path."
        print(error_msg)
        conn.send(error_msg)
        conn.close()
    except subprocess.CalledProcessError as e:
        error_msg = f"Error executing fundamental.py:\n{e.stderr}"
        print(error_msg)
        conn.send(error_msg)
        conn.close()
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        conn.send(error_msg)
        conn.close()


def run_technical_script(ticker, conn): # conn is one end of the pipe
    """Runs the technical analysis script and sends its output via a pipe."""
    try:
        result = subprocess.run(
            ['python', '-m', 'app.services.technical', ticker],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(settings.project_root.parent)
        )
        output = f"{result.stdout}\n"
        conn.send(output) # Send the result through the pipe
    except FileNotFoundError:
        error_msg = "Error: 'technical.py' not found. Check the script path."
        print(error_msg)
        conn.send(error_msg)
    except subprocess.CalledProcessError as e:
        error_msg = f"Error executing technical.py:\n{e.stderr}"
        print(error_msg)
        conn.send(error_msg)
    finally:
        conn.close() # Close the connection when done


def fetch_sector_analysis(ticker, conn):
    """
    Fetches the sector analysis for a given ticker from MongoDB and sends its output via a pipe.
    
    Args:
        ticker (str): The stock ticker symbol.
        conn: One end of a multiprocessing Pipe (for sending results).
        collection: MongoDB collection object (e.g., db['macro']).
    """
    try:
        # MongoDB connection
        client = MongoClient(settings.mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
        db = client['macro']
        collection = db['macro']
        
        # Load sector hierarchy CSV to find sector for ticker
        sector_hierarchy = pd.read_csv(settings.get_data_path("sector_hierarchy.csv"))
        
        sector_series = sector_hierarchy[sector_hierarchy['SYMBOL'] == ticker]['Level 1']
        
        if sector_series.empty:
            message = f"No sector information found for ticker: {ticker}"
            conn.send(message)
            return
        
        sector = sector_series.iloc[0]
        
        # Query MongoDB for the sector analysis document
        doc = collection.find_one({"sector": sector})
        
        if not doc or "analysis" not in doc:
            message = f"No analysis found in DB for sector: {sector}"
            conn.send(message)
            return
        
        analysis_text = doc["analysis"]
        output = f"Sector Macro Analysis ({sector}):\n{analysis_text}\n"
        conn.send(output)
        
    except FileNotFoundError:
        # This may not occur anymore unless sector_hierarchy.csv is missing
        message = f"Sector hierarchy CSV file not found."
        conn.send(message)
    except Exception as e:
        message = f"An unexpected error occurred: {e}"
        conn.send(message)
    finally:
        conn.close()


def run_news_script(ticker, conn):
    """
    1) Run NSE-specific scraper
    2) Run general-news scraper
    3) Feed both outputs into the progressive summariser
    4) Return the summarised text
    """
    try:
        # 1 & 2: scrape raw feeds (run in parallel to cut latency)
        def _run_scraper(script_module: str) -> str:
            return subprocess.run(
                ["python", "-m", f"app.services.{script_module}", ticker],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(settings.project_root.parent)
            ).stdout

        with ThreadPoolExecutor(max_workers=2) as pool:
            nse_future = pool.submit(_run_scraper, "news_scrap_nse")
            general_future = pool.submit(_run_scraper, "news_scrap_general")
            nse = nse_future.result()
            general = general_future.result()

        # 3: summarise
        try:
            result = subprocess.run(
                ["python", "-m", "app.services.progressive_news_summariser", ticker],
                capture_output=True,
                text=True,
                check=True,
                cwd=str(settings.project_root.parent)
            )
        except subprocess.CalledProcessError as e:
            # e.stderr will contain the Python traceback or error msg
            conn.send(f"❗ News pipeline error:\n{e.stderr}")


        output = f"News Analysis:\n{result.stdout}\n"
        conn.send(output) # Send the result through the pipe

    except Exception as e:
        conn.send(f"❗ News pipeline error: {e}")
    finally:
        conn.close()

def run_sector_news_script(ticker, conn): # conn is one end of the pipe
    """Fetches the sector news analysis for a given ticker and sends its output via a pipe."""
    try:
        # MongoDB connection
        client = MongoClient(settings.mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
        db = client['sector_news']
        collection = db['progressive_sector_news']
        sector_hierarchy = pd.read_csv(settings.get_data_path("sector_hierarchy.csv"))
        
        sector_series = sector_hierarchy[sector_hierarchy['SYMBOL'] == ticker]['Level 3']
        
        if sector_series.empty:
            message = f"No sector information found for ticker: {ticker}"
            conn.send(message)
            return
            
        sector = sector_series.iloc[0]
        
        # Query MongoDB for the sector analysis document
        doc = collection.find_one({"sector": sector})
        
        if not doc or "progressive_summary" not in doc:
            message = f"No analysis found in DB for sector: {sector}"
            conn.send(message)
            return
        
        analysis_text = doc["progressive_summary"]
        output = f"Sector News Analysis ({sector}):\n{analysis_text}\n"
        conn.send(output)
        
    except FileNotFoundError:
        # This may not occur anymore unless sector_hierarchy.csv is missing
        message = f"Sector hierarchy CSV file not found."
        conn.send(message)
    except Exception as e:
        message = f"An unexpected error occurred: {e}"
        conn.send(message)
    finally:
        conn.close()

GEMINI_API_KEY = settings.gemini_api_key
MODEL_NAME     = "gemini-2.5-flash"
DB_ELI5        = "ELI5" 

def _take_text_from_response(resp) -> str:
    """
    Robustly extract text from google.generativeai response.
    Works across variants: resp.text or candidates[0].content.parts[*].text
    """
    try:
        if hasattr(resp, "text") and resp.text:
            return resp.text

        cands = getattr(resp, "candidates", None)
        if not cands:
            return ""
        content = getattr(cands[0], "content", None)
        if not content:
            return ""
        parts = getattr(content, "parts", None)
        if not parts:
            return ""
        texts = []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                texts.append(t)
        return "\n".join(texts).strip()
    except Exception:
        return ""


def _build_bucket_prompt(label: str, T: str, today_str: str, raw: str) -> str:
    """
    Construct a concise prompt for ELI5 bucket summaries.
    """
    raw = (raw or "").strip()
    if len(raw) > 6000:
        raw = raw[:6000] + "\n…"

    return (
        f"You are an expert Indian-equities explainer. Date: {today_str}. Ticker: {T}.\n"
        f"Bucket: {label}.\n\n"
        "SOURCE (raw, unstructured):\n"
        f"{raw}\n\n"
        "TASK:\n"
        "- Write a simple, accurate ELI5-style summary for retail investors in India.\n"
        "- Be objective, specific, and avoid generic filler.\n"
        "- Max ~12 bullet points or short paragraphs; keep under 2400 chars.\n"
        "- If data is incomplete, say what's missing briefly, but still summarize what's available.\n"
        "- No headings besides the content itself. No disclaimers.\n"
    )


def _fallback_bucket_eli5(title: str, raw: str) -> str:
    """
    Minimal fallback if the LLM call fails.
    """
    raw = (raw or "").strip()
    sample = raw[:400].replace("\n", " ")
    if not sample:
        sample = "No input data available today."
    return (
        f"{title} — quick take:\n"
        f"- Detailed summary unavailable right now. Showing a brief note.\n"
        f"- Snapshot: {sample}\n"
    )


def _gen_one_bucket_worker(args):
    """
    Builds prompt, calls Gemini with retries/backoff, applies fallbacks,
    and returns (out_field, coll_name, text, debug_log).
    """
    (key, label, out_field, coll_name, raw, T, today_str, model_name, gemini_key) = args
    logs = [f"🧱 [{out_field}] start (has_data={bool(raw)})"]

    def _try_generate(genai, model, prompt):
        resp = model.generate_content(prompt)
        txt = _take_text_from_response(resp).strip()
        if not txt:
            raise RuntimeError("Empty response text")
        return txt

    try:
        import google.generativeai as genai

        if not raw:
            txt = _fallback_bucket_eli5(f"{label} (plain words)", "Data not available today.")
        else:
            # tiny jitter so all threads don’t hit at once
            sleep(random.uniform(0.05, 0.25))

            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(
                model_name,
                generation_config=genai.GenerationConfig(temperature=0, top_p=1)
            )

            full_prompt = _build_bucket_prompt(label, T, today_str, raw)
            logs.append(f"📝 [{out_field}] prompt chars={len(full_prompt)}")

            attempts = [
                full_prompt,
                _build_bucket_prompt(label, T, today_str, (raw or "")[:1500]),
                _build_bucket_prompt(label, T, today_str, (raw or "")[:800]),
            ]

            txt, last_err = None, None
            for i, p in enumerate(attempts, start=1):
                try:
                    txt = _try_generate(genai, model, p)
                    break
                except Exception as e:
                    last_err = e
                    logs.append(f"⚠️ [{out_field}] attempt {i} failed: {e}")
                    sleep((0.4 * (2 ** (i - 1))) + random.uniform(0, 0.2))

            if txt is None:
                logs.append(f"⚠️ [{out_field}] using fallback after retries: {last_err}")
                txt = _fallback_bucket_eli5(f"{label} (plain words)", raw)

    except Exception as e:
        logs.append(f"❌ [{out_field}] worker error: {e}")
        txt = _fallback_bucket_eli5(f"{label} (plain words)", raw or "Data not available today.")

    if len(txt) > 2400:
        txt = txt[:2400] + "\n…"

    logs.append(f"✅ [{out_field}] done, len={len(txt)}")
    return (out_field, coll_name, txt, "\n".join(logs))


def generate_eli5_per_bucket_parallel(
    ticker: str,
    buckets: dict,
    mongo_uri: str,
    save: bool = True,
    max_workers: int = 5,
    use_threads: bool = True,  # threads are default (kept for API symmetry)
) -> dict:
    """
    Threaded ELI5 generation for the 5 buckets.
    Expects `buckets` keys: 'fundamental', 'technical', 'macro', 'news', 'sector_news'.
    Saves into DB `ELI5`: eli5_funda, eli5_technical, eli5_macro, eli5_news, eli5_sector.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_KEY_HERE":
        raise RuntimeError("❌ GEMINI_API_KEY not set.")
    if not mongo_uri or mongo_uri == "YOUR_MONGO_URI_HERE":
        raise RuntimeError("❌ MONGO_URI not set.")

    T = ticker.strip().upper()
    today_str = date.today().isoformat()

    sections = {
        "fundamental": ("Company basics",  "funda_eli5",     "eli5_funda"),
        "technical":   ("Price trend",     "technical_eli5", "eli5_technical"),
        "macro":       ("Macro",           "macro_eli5",     "eli5_macro"),
        "news":        ("News",            "news_eli5",      "eli5_news"),
        "sector_news": ("Sector news",     "sector_eli5",    "eli5_sector"),
    }

    # map for nicer fallbacks if a thread fails
    out_to_label = {out_field: label for _, (label, out_field, _) in sections.items()}

    # Build per-bucket task args
    tasks = []
    for key, (label, out_field, coll_name) in sections.items():
        raw = buckets.get(key)
        tasks.append((key, label, out_field, coll_name, raw, T, today_str, MODEL_NAME, GEMINI_API_KEY))

    results = {}
    logs_to_print = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_meta = {}
        for args in tasks:
            fut = pool.submit(_gen_one_bucket_worker, args)
            fut_meta[fut] = (args[2], args[3])  # (out_field, coll_name)

        for fut in as_completed(fut_meta):
            out_field, coll_name = fut_meta[fut]
            try:
                of, cn, txt, dbg = fut.result()
                results[of] = txt
                logs_to_print.append(dbg)
            except Exception as e:
                # never let one failure kill the whole run
                label = out_to_label.get(out_field, out_field)
                results[out_field] = _fallback_bucket_eli5(f"{label} (plain words)", "Data not available today.")
                logs_to_print.append(f"❌ [{out_field}] thread error: {e}")

    # Print gathered logs (cleaner than interleaved prints)
    # print("\n".join(logs_to_print))

    if save:
        client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
        try:
            for _, (label, out_field, coll_name) in sections.items():
                payload = {"stock": T, "date": today_str, "eli5_text": results.get(out_field, "")}
                client[DB_ELI5][coll_name].update_one({"stock": T}, {"$set": payload}, upsert=True)
                print(f"💾 Saved {out_field} into {DB_ELI5}.{coll_name}")
        finally:
            client.close()

    return results

def ensure_eli5_indexes(mongo_uri: str):
    """
    Idempotently ensure indexes for ELI5 collections.

    - Skips creation if an index with the same key pattern already exists (regardless of name).
    - Creates indexes without explicit names to avoid 'IndexOptionsConflict' (Mongo will use 'field_1').
    - If a non-unique 'stock' index already exists, we warn and skip changing it to unique (to avoid data errors).
    """
    client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
    try:
        db = client[DB_ELI5]
        colls = ["eli5_funda", "eli5_technical", "eli5_macro", "eli5_news", "eli5_sector"]

        for coll_name in colls:
            col = db[coll_name]
            info = col.index_information()  # {name: {'key': [('field', 1)], 'unique': bool, ...}}

            # Helpers to check existence by key pattern
            def _has_index(keys):
                for spec in info.values():
                    if spec.get("key") == keys:
                        return True
                return False

            def _get_index(keys):
                for name, spec in info.items():
                    if spec.get("key") == keys:
                        return name, spec
                return None, None

            # Ensure unique index on 'stock'
            stock_keys = [("stock", ASCENDING)]
            if not _has_index(stock_keys):
                try:
                    # No explicit name -> avoids name clashes if an identically named index already exists
                    col.create_index(stock_keys, unique=True)
                    print(f"[eli5] Created unique index on 'stock' for {coll_name}")
                except Exception as e:
                    print(f"[eli5][warn] creating unique 'stock' index on {coll_name}: {e}")
            else:
                name, spec = _get_index(stock_keys)
                if not spec.get("unique", False):
                    # Don’t auto-drop/convert to unique (could fail with duplicates) – just warn.
                    print(f"[eli5][warn] '{coll_name}.{name}' on 'stock' exists but is NOT unique. "
                          "Consider cleaning and recreating as unique if required.")

            # Ensure (non-unique) index on 'date'
            date_keys = [("date", ASCENDING)]
            if not _has_index(date_keys):
                try:
                    col.create_index(date_keys)  # no name -> will be 'date_1'
                    print(f"[eli5] Created index on 'date' for {coll_name}")
                except Exception as e:
                    print(f"[eli5][warn] creating 'date' index on {coll_name}: {e}")

    finally:
        client.close()

# --- Reddit + FinBERT sentiment (self-contained) -----------------

def reddit_finbert_sentiment_pct(ticker: str):
    """
    Returns (positive_pct, negative_pct, neutral_pct) for the given ticker.
    NOTE: heavy deps are imported inside so they load only in the child process.
    """
    import re, time
    from datetime import datetime, timedelta, timezone

    import praw
    from praw.models import MoreComments
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    FINBERT_MODEL = "ProsusAI/finbert"
    URL_RE = re.compile(r"https?://\S+")
    WHITESPACE_RE = re.compile(r"\s+")
    TICKER_RE_TEMPLATE = r"(?i)(?:\b{t}\b|\${t}\b)"  # TSLA or $TSLA

    HOURS = 504                # lookback window (3 weeks)
    POST_LIMIT = 60
    COMMENT_LIMIT = 50
    MAX_ITEMS = 500

    def clean_text(text: str) -> str:
        if not text:
            return ""
        text = URL_RE.sub("", text)
        text = WHITESPACE_RE.sub(" ", text)
        return text.strip()

    def within_hours(utc_ts: float, hours: int) -> bool:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return datetime.fromtimestamp(utc_ts, tz=timezone.utc) >= cutoff

    # ---------- cached reddit client ----------
    if not hasattr(reddit_finbert_sentiment_pct, "_reddit"):
        client_id = settings.reddit_client_id
        client_secret = settings.reddit_client_secret
        user_agent = settings.reddit_user_agent
        reddit_finbert_sentiment_pct._reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            ratelimit_seconds=5,
        )
    reddit = reddit_finbert_sentiment_pct._reddit

    # ---------- cached finbert ----------
    want_device = "cuda" if torch.cuda.is_available() else "cpu"
    cached = getattr(reddit_finbert_sentiment_pct, "_finbert", None)
    if (cached is None) or (cached[2] != want_device):
        tok = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        mdl.to(want_device)
        mdl.eval()
        reddit_finbert_sentiment_pct._finbert = (tok, mdl, want_device)
    tokenizer, model, device_used = reddit_finbert_sentiment_pct._finbert

    # ---------- gather reddit rows ----------
    try:
        ticker_clean = ticker.strip().upper()
        ticker_re = re.compile(TICKER_RE_TEMPLATE.format(t=re.escape(ticker_clean)))
        rows = []

        sr = reddit.subreddit("all")
        time_filter = "month" if HOURS <= 31 * 24 else "year"
        query = f'"{ticker_clean}" OR "${ticker_clean}"'

        for submission in sr.search(query=query, sort="new", time_filter=time_filter, limit=POST_LIMIT):
            try:
                if not within_hours(submission.created_utc, HOURS):
                    continue
                title = submission.title or ""
                selftext = submission.selftext or ""
                combined = f"{title}\n\n{selftext}"
                if not ticker_re.search(combined):
                    continue
                rows.append({
                    "kind": "post",
                    "subreddit": str(submission.subreddit),
                    "id": submission.id,
                    "permalink": f"https://www.reddit.com{submission.permalink}",
                    "author": str(submission.author) if submission.author else "[deleted]",
                    "created_utc": submission.created_utc,
                    "text": clean_text(combined),
                })

                # comments
                try:
                    submission.comments.replace_more(limit=0)
                    count = 0
                    for c in submission.comments.list():
                        if isinstance(c, MoreComments):
                            continue
                        if not within_hours(c.created_utc, HOURS):
                            continue
                        body = c.body or ""
                        if not body:
                            continue
                        if ticker_re.search(body) or ticker_re.search(getattr(c, "parent_id", "")):
                            rows.append({
                                "kind": "comment",
                                "subreddit": str(submission.subreddit),
                                "id": c.id,
                                "permalink": f"https://www.reddit.com{c.permalink}",
                                "author": str(c.author) if c.author else "[deleted]",
                                "created_utc": c.created_utc,
                                "text": clean_text(body),
                            })
                            count += 1
                            if count >= COMMENT_LIMIT:
                                break
                except praw.exceptions.RedditAPIException:
                    time.sleep(2)

            except Exception:
                continue

        # de-dup, sort, cap
        seen = set()
        uniq = []
        for r in rows:
            key = (r["kind"], r["id"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        uniq.sort(key=lambda r: r["created_utc"], reverse=True)
        if not uniq:
            return (0.0, 0.0, 0.0)
        uniq = uniq[:MAX_ITEMS]

        # ---------- finbert inference ----------
        @torch.no_grad()
        def finbert_probs(texts, tok, mdl, dev, batch_size=16, max_length=256):
            out = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(dev)
                logits = mdl(**enc).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()  # [neutral, positive, negative]
                for row in probs:
                    out.append({"neutral": float(row[0]), "positive": float(row[1]), "negative": float(row[2])})
            return out

        texts = [r["text"] for r in uniq]
        prob_rows = finbert_probs(texts, tokenizer, model, device_used)

        # ---------- simple average aggregation ----------
        if not prob_rows:
            return (0.0, 0.0, 0.0)
        sums = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for d in prob_rows:
            for k in sums:
                sums[k] += d[k]
        n = max(1, len(prob_rows))
        avg = {k: (sums[k] / n) for k in sums}
        total = sum(avg.values()) or 1.0
        pos = round(100.0 * (avg["positive"] / total), 1)
        neg = round(100.0 * (avg["negative"] / total), 1)
        neu = round(100.0 * (avg["neutral"] / total), 1)
        return (pos, neg, neu)

    except Exception:
        return (0.0, 0.0, 0.0)

def run_reddit_sentiment_script(ticker: str, conn) -> None:
    """
    Multiprocessing wrapper: compute Reddit sentiment, save to MongoDB, and send a tiny dict.
    No printing/logging here.
    """
    from datetime import date
    from pymongo import MongoClient
    import os
    import os, warnings
    # 1) Stop transformers from importing torchvision at all (kills the beta warnings)
    os.environ["TRANSFORMERS_NO_TORCHVISION_IMPORT"] = "1"

    # 2) Targeted silencing in case anything still slips through
    warnings.filterwarnings(
        "ignore",
        message=".*torchvision\\.datapoints and torchvision\\.transforms\\.v2 namespaces are still Beta.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*TypedStorage is deprecated.*",
        category=UserWarning,
    )


    mongo_uri = settings.mongo_uri
    DB_NAME = "alt_sentiment"
    COLL_NAME = "reddit_finbert"

    payload = {
        "positive_pct": 0.0,
        "negative_pct": 0.0,
        "neutral_pct":  0.0,
    }

    try:
        pos, neg, neu = reddit_finbert_sentiment_pct(ticker)
        payload.update({
            "positive_pct": pos,
            "negative_pct": neg,
            "neutral_pct":  neu,
        })

        # Save to Mongo (upsert by stock+date)
        client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
        try:
            today = date.today().strftime("%Y-%m-%d")
            doc = {
                "stock": ticker.strip().upper(),
                "date":  today,
                **payload
            }
            client[DB_NAME][COLL_NAME].update_one(
                {"stock": doc["stock"], "date": doc["date"]},
                {"$set": doc},
                upsert=True
            )
        finally:
            client.close()

        # Send back (for ELI5 bucket usage if needed)
        conn.send(payload)

    except Exception as e:
        # On any failure, save zeros for today so the record still exists (silent)
        try:
            client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
            try:
                today = date.today().strftime("%Y-%m-%d")
                client[DB_NAME][COLL_NAME].update_one(
                    {"stock": ticker.strip().upper(), "date": today},
                    {"$set": {
                        "stock": ticker.strip().upper(),
                        "date":  today,
                        "positive_pct": 0.0,
                        "negative_pct": 0.0,
                        "neutral_pct":  0.0,
                        "error": str(e),
                    }},
                    upsert=True
                )
            finally:
                client.close()
        except:
            pass  # remain silent

        conn.send(payload)  # still send zeros

    finally:
        conn.close()
