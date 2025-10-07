from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
from datetime import date
from multiprocessing import Pipe, Process
from typing import Optional
import pandas as pd
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import threading
import time
from .config import settings
from .core.analysis_functions import generate_eli5_per_bucket_parallel, ensure_eli5_indexes
from .core.analysis_functions import (
    run_fundamental_script,
    run_technical_script,
    fetch_sector_analysis,
    run_news_script,
    run_sector_news_script,
    run_sector_hierarchy_script,
    run_reddit_sentiment_script
)
from .core.generate_signal import (
    generate_stock_signal,
    extract_and_save_final_recommendation,
)  # custom helpers
from fastapi.testclient import TestClient
from markdown import markdown
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xhtml2pdf import pisa
settings.ensure_directories()

app = FastAPI()

# Allow all origins for dev (limit in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB URI (replace with env var for security in production)
MONGO_URI = settings.mongo_uri
client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)

try:
    ensure_eli5_indexes(MONGO_URI)
except Exception as _e:
    print(f"[warn] ensure_eli5_indexes: {_e}")


def _normalize_ticker(raw: str) -> str:
    return raw.strip().upper()


_PDF_FONT_NAME = "SummaryBodyFont"
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/NotoSans-Regular.ttf",
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arialuni.ttf"),
]
_FONT_READY = False
_FONT_AVAILABLE = False


def _ensure_pdf_font() -> Optional[str]:
    global _FONT_READY, _FONT_AVAILABLE
    if _FONT_READY:
        return _PDF_FONT_NAME if _FONT_AVAILABLE else None

    for path in _FONT_CANDIDATES:
        if not path or not os.path.exists(path):
            continue
        try:
            pdfmetrics.registerFont(TTFont(_PDF_FONT_NAME, path))
            _FONT_AVAILABLE = True
            break
        except Exception as font_err:
            print(f"[warn] Failed to register PDF font {path}: {font_err}")

    _FONT_READY = True
    return _PDF_FONT_NAME if _FONT_AVAILABLE else None


def _render_markdown_to_pdf(markdown_text: str, title: str) -> BytesIO:
    """Render markdown content to a PDF stream."""
    font_name = _ensure_pdf_font()
    font_stack = f"{font_name}, Helvetica, Arial, sans-serif" if font_name else "Helvetica, Arial, sans-serif"
    source_markdown = markdown_text or ""
    if not font_name:
        source_markdown = source_markdown.replace("₹", "INR ")
    html_body = markdown(source_markdown, extensions=["extra", "sane_lists"])
    html_template = f"""
    <html>
      <head>
        <meta charset='utf-8' />
        <style>
          body {{ font-family: {font_stack}; color: #111; line-height: 1.4; font-size: 12pt; }}
          h1, h2, h3, h4 {{ font-family: {font_stack}; color: #0f3c8c; margin-top: 1.6em; }}
          table {{ width: 100%; border-collapse: collapse; margin: 1em 0; font-size: 11pt; }}
          th, td {{ border: 1px solid #d0d7de; padding: 6px; }}
          pre {{ background: #f6f8fa; padding: 12px; overflow-x: auto; }}
          code {{ font-family: 'Courier New', monospace; background: #f6f8fa; padding: 2px 4px; }}
          ul, ol {{ margin-left: 1.4em; }}
          blockquote {{ border-left: 3px solid #d0d7de; margin: 1em 0; padding-left: 12px; color: #4a5568; }}
        </style>
      </head>
      <body>
        <h1>{title} – Stock Summary</h1>
        {html_body}
      </body>
    </html>
    """

    pdf_stream = BytesIO()
    result = pisa.CreatePDF(html_template, dest=pdf_stream, encoding='utf-8')
    if result.err:
        raise ValueError("PDF rendering failed")

    pdf_stream.seek(0)
    return pdf_stream


_analysis_condition = threading.Condition()
_analysis_in_progress = set()


def _analysis_try_begin(ticker: str) -> bool:
    normalized = _normalize_ticker(ticker)
    with _analysis_condition:
        if normalized in _analysis_in_progress:
            return False
        _analysis_in_progress.add(normalized)
        return True


def _analysis_wait_for_completion(ticker: str, timeout: float = 600.0) -> bool:
    normalized = _normalize_ticker(ticker)
    deadline = time.monotonic() + timeout
    with _analysis_condition:
        while normalized in _analysis_in_progress:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _analysis_condition.wait(timeout=remaining)
    return True


def _analysis_end(ticker: str) -> None:
    normalized = _normalize_ticker(ticker)
    with _analysis_condition:
        _analysis_in_progress.discard(normalized)
        _analysis_condition.notify_all()


@app.post("/run-analysis/{ticker}")
def run_analysis(ticker: str):
    ticker = _normalize_ticker(ticker)

    while not _analysis_try_begin(ticker):
        if not _analysis_wait_for_completion(ticker):
            raise HTTPException(status_code=409, detail=f" Analysis already running for {ticker}. Please retry shortly.")

    try:
        today_str = date.today().strftime("%Y-%m-%d")

        stored = None
        client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
        try:
            db = client["stock_signal"]
            collection = db["json"]
            existing_data = collection.find_one({"stock": ticker})

            if existing_data and existing_data.get("date") == today_str:
                return {
                    "message": f" Report for {ticker} already exists for today. No new analysis triggered.",
                    "ticker": ticker,
                    "data": existing_data.get("extracted_json", "No structured output found.")
                }

            print(f"\n Starting analysis for {ticker} using Pipes...")
            # Step 2: Set up Pipes for all subprocesses
            parent_conn_f, child_conn_f = Pipe()
            parent_conn_t, child_conn_t = Pipe()
            parent_conn_s, child_conn_s = Pipe()
            parent_conn_n, child_conn_n = Pipe()
            parent_conn_se, child_conn_se = Pipe()
            parent_conn_h, child_conn_h = Pipe()
            parent_conn_r, child_conn_r = Pipe()

            # Step 3: Spawn subprocesses
            p1 = Process(target=run_sector_hierarchy_script, args=(ticker, child_conn_h))
            p2 = Process(target=run_fundamental_script, args=(ticker, child_conn_f))
            p3 = Process(target=run_technical_script, args=(ticker, child_conn_t))
            p4 = Process(target=fetch_sector_analysis, args=(ticker, child_conn_s))
            p5 = Process(target=run_news_script, args=(ticker, child_conn_n))
            p6 = Process(target=run_sector_news_script, args=(ticker, child_conn_se))
            p7 = Process(target=run_reddit_sentiment_script, args=(ticker, child_conn_r))

            p1.start()
            child_conn_h.close()
            p1.join()
            print(" Sector hierarchy complete")
            parent_conn_h.close()

            # Start the rest in parallel
            p2.start()
            p3.start()
            p4.start()
            p5.start()
            p6.start()
            p7.start()

            child_conn_f.close()
            child_conn_t.close()
            child_conn_s.close()
            child_conn_n.close()
            child_conn_se.close()
            child_conn_r.close()

            p2.join(); print(" Fundamental complete")
            p3.join(); print(" Technical complete")
            p4.join(); print(" Sector analysis complete")
            p5.join(); print(" News complete")
            p6.join(); print(" Sector news complete")
            p7.join(); print(" Reddit sentiment complete")

            # Step 4: Collect results
            fundamental_result = parent_conn_f.recv()
            technical_result = parent_conn_t.recv()
            sector_result = parent_conn_s.recv()
            news_result = parent_conn_n.recv()
            sector_news_result = parent_conn_se.recv()
            reddit_result = parent_conn_r.recv()

            parent_conn_f.close()
            parent_conn_t.close()
            parent_conn_s.close()
            parent_conn_n.close()
            parent_conn_se.close()
            parent_conn_r.close()

            # Build buckets for ELI5 generator
            buckets = {
                "fundamental": fundamental_result,
                "technical":   technical_result,
                "macro":       sector_result,      
                "news":        news_result,
                "sector_news": sector_news_result,
                "reddit":      reddit_result
            }

            # Run ELI5 (threaded) in parallel with final signal generation
            with ThreadPoolExecutor(max_workers=2) as pool:
                eli5_future = pool.submit(
                    generate_eli5_per_bucket_parallel,
                    ticker,
                    buckets,
                    MONGO_URI,
                    True,          # save to DB
                    5              # max_workers inside the ELI5 generator
                )

                # Step 5: Generate final signal and save (runs concurrently with ELI5)
                signal = generate_stock_signal(
                    news=news_result,
                    sector=sector_news_result,
                    macro=sector_result,
                    fundamental=fundamental_result,
                    technical=technical_result,
                    ticker=ticker,
                )
                extract_and_save_final_recommendation(signal, ticker)

                # Ensure ELI5 finished (and saved) before we return
                _eli5_out = eli5_future.result()

            stored = collection.find_one({"stock": ticker})
        finally:
            client.close()

        if not stored or "extracted_json" not in stored:
            raise HTTPException(status_code=500, detail="Structured JSON output missing from DB")

        return {
            "message": f" Analysis complete and saved for {ticker}",
            "ticker": ticker,
            "data": stored["extracted_json"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f" Analysis failed: {e}")
    finally:
        _analysis_end(ticker)


    #define other endpoints as needed


from fastapi.testclient import TestClient
import os

from fastapi.testclient import TestClient

#actual code to run , replace when once frontend done 
@app.get("/get-insight/{ticker}")
def get_insight(ticker: str):
    ticker = _normalize_ticker(ticker)
    db = client["stock_signal"]
    collection = db["json"]

    try:
        existing = collection.find_one({"stock": ticker})
        today = date.today().strftime("%Y-%m-%d")

        if existing and existing.get("date") == today:
            return {
                "ticker": ticker,
                "data": existing.get("extracted_json", "No structured insight found")
            }

        # Ticker is missing or stale → trigger internal regeneration regardless of RUN_MAIN
        local_client = TestClient(app)

        print(f"⏳ No recent report found for {ticker}. Triggering analysis...")
        response = local_client.post(f"/run-analysis/{ticker}")

        if response.status_code != 200:
            raise Exception(f"Analysis failed: {response.text}")

        result = response.json()
        return {
            "ticker": ticker,
            "data": result.get("data", "No structured insight returned")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch insight: {e}") 


"""@ app.get("/get-insight/{ticker}")
def get_insight(ticker: str):
    ticker = ticker.strip().upper()
    db = client["stock_signal"]
    collection = db["json"]

    try:
        existing = collection.find_one({"stock": ticker})

        # Disable date checking: just return any existing doc
        if existing:
            return {
                "ticker": ticker,
                "data": existing.get("extracted_json", "No structured insight found")
            }

        # If not found, still try to run analysis (optional; you can remove this too)
        # from main import app
        # local_client = TestClient(app)
        # print(f"⏳ No recent report found for {ticker}. Triggering analysis...")
        # response = local_client.post(f"/run-analysis/{ticker}")

        # if response.status_code != 200:
        #     raise Exception(f"Analysis failed: {response.text}")

        # result = response.json()
        # return {
        #     "ticker": ticker,
        #     "data": result.get("data", "No structured insight returned")
        # }

        # If you don't want to trigger new analysis at all, just return not found
        raise HTTPException(status_code=404, detail="No structured insight found for this ticker.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch insight: {e}")

 """

    
@app.get("/tickers")
def get_all_tickers():
    try:
        db = client["stock_signal"]
        collection = db["json"]
        tickers = collection.find({}, {"stock": 1, "_id": 0})
        return {"tickers": [t["stock"] for t in tickers]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch tickers: {e}")
    

@app.get("/get-final-recommendation/{ticker}")
def get_final_recommendation(ticker: str):
    ticker = ticker.strip().upper()
    db = client["stock_signal"]
    collection = db["final_recommendation"]

    try:
        doc = collection.find_one({"stock": ticker})
        if not doc:
            raise HTTPException(status_code=404, detail="Final recommendation not found.")
        
        return {"ticker": ticker, "recommendation": doc.get("final_recommendation", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")


@app.get("/get-stock-summary/{ticker}")
def get_stock_summary(ticker: str):
    ticker = _normalize_ticker(ticker)
    db = client["stock_signal"]
    collection = db["stock_signal_summary"]

    try:
        doc = collection.find_one({"stock": ticker})
        if not doc or not doc.get("signal"):
            raise HTTPException(status_code=404, detail="Stock summary not found.")

        return {"ticker": ticker, "summary_markdown": doc.get("signal", "")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")


@app.get("/export-stock-summary/{ticker}")
def export_stock_summary(ticker: str):
    ticker = _normalize_ticker(ticker)
    db = client["stock_signal"]
    collection = db["stock_signal_summary"]

    try:
        doc = collection.find_one({"stock": ticker})
        if not doc or not doc.get("signal"):
            raise HTTPException(status_code=404, detail="Stock summary not found.")

        try:
            pdf_stream = _render_markdown_to_pdf(doc.get("signal", ""), ticker)
        except ValueError as err:
            raise HTTPException(status_code=500, detail=f"Unable to generate PDF: {err}")

        headers = {
            "Content-Disposition": f'inline; filename="{ticker}_stock_summary.pdf"'
        }
        return StreamingResponse(pdf_stream, media_type="application/pdf", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")


@app.get("/get-technical-analysis/{ticker}")
def get_technical_analysis(ticker: str):
    ticker = ticker.strip().upper()
    db = client["tech"]
    collection = db["tech"]

    try:
        doc = collection.find_one({"stock": ticker})
        if not doc:
            raise HTTPException(status_code=404, detail="Technical analysis not found.")

        return {"ticker": ticker, "technical_analysis": doc.get("summary", "")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    

@app.get("/get-fundamental-analysis/{ticker}")
def get_fundamental_analysis(ticker: str):
    ticker = ticker.strip().upper()
    db = client["funda"]
    collection = db["funda"]

    try:
        doc = collection.find_one({"ticker": ticker})
        if not doc:
            raise HTTPException(status_code=404, detail="Fundamental analysis not found.")

        return {"ticker": ticker, "summary": doc.get("summary", "")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    
#error to get sector analysis if the 3rd level not there
@app.get("/get-sector-analysis/{ticker}")
def get_sector_analysis(ticker: str):
    ticker = ticker.strip().upper()
    # 1. Load ticker-sector mapping (for demo; in prod, load once or use Mongo)
    df = pd.read_csv(settings.get_data_path("sector_hierarchy.csv"))
    row = df[df["SYMBOL"] == ticker]
    if row.empty:
        raise HTTPException(status_code=404, detail="Ticker not found in sector mapping.")

    sector = row["Level 3"].values[0]

    # 2. Query the MongoDB for sector summary
    db = client["sector_news"]
    collection = db["progressive_sector_news"]
    doc = collection.find_one({"sector": sector})

    if not doc:
        raise HTTPException(status_code=404, detail="Sector analysis not found.")

    return {
        "ticker": ticker,
        "sector": sector,
        "sector_analysis": doc.get("progressive_summary", "")
    }


@app.get("/get-stock-news/{ticker}")
def get_stock_news(ticker: str, request: Request):
    try:
        ticker = ticker.strip().upper()
        if not ticker.endswith(".NS"):
            ticker = ticker + ".NS"

        print(f"🔍 Fetching stock news for: {ticker}")

        db = client["stock_news"]
        collection = db["progressive_news"]

        doc = collection.find_one({"stock": ticker})
        if not doc:
            print(f"⚠️ No news found for {ticker}")
            raise HTTPException(status_code=404, detail="News not found for this stock.")

        return {
            "ticker": ticker,
            "news": doc.get("progressive_summary", "No summary found")
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")


@app.get("/get-macro-economic-analysis/{ticker}")
def get_macro_economic_analysis(ticker: str):
    import pandas as pd  # Local import to keep everything inside

    ticker = ticker.strip().upper()

    # Step 1: Load sector_hierarchy.csv and find Level 1 sector for the ticker
    df = pd.read_csv(settings.get_data_path("sector_hierarchy.csv"))
    row = df[df["SYMBOL"] == ticker]
    if row.empty:
        raise HTTPException(status_code=404, detail="Sector not found for ticker.")

    sector = row["Level 1"].values[0]

    # Step 2: Fetch macroeconomic analysis for the sector from MongoDB
    db = client["macro"]
    collection = db["macro"]

    try:
        doc = collection.find_one({"sector": sector})
        if not doc:
            raise HTTPException(status_code=404, detail="Macro economic analysis not found for sector.")

        return {
            "ticker": ticker,
            "sector": sector,
            "macro_economic_analysis": doc.get("analysis", "")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")


def _fetch_eli5_summary(collection_name: str, ticker: str):
    normalized = _normalize_ticker(ticker)

    try:
        db = client["ELI5"]
        collection = db[collection_name]
        doc = collection.find_one({"stock": normalized})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch ELI5 summary: {e}")

    if not doc:
        raise HTTPException(status_code=404, detail=f"ELI5 summary not found for {normalized}.")

    return {
        "ticker": normalized,
        "eli5_text": doc.get("eli5_text", ""),
        "date": doc.get("date"),
    }


@app.get("/get-eli5-fundamental/{ticker}")
def get_eli5_fundamental(ticker: str):
    return _fetch_eli5_summary("eli5_funda", ticker)


@app.get("/get-eli5-technical/{ticker}")
def get_eli5_technical(ticker: str):
    return _fetch_eli5_summary("eli5_technical", ticker)


@app.get("/get-eli5-sector/{ticker}")
def get_eli5_sector(ticker: str):
    return _fetch_eli5_summary("eli5_sector", ticker)


@app.get("/get-eli5-news/{ticker}")
def get_eli5_news(ticker: str):
    return _fetch_eli5_summary("eli5_news", ticker)


def _candidate_tickers(ticker: str):
    base = _normalize_ticker(ticker)
    candidates = {base}
    if base.endswith(".NS"):
        candidates.add(base[:-3])
    else:
        candidates.add(f"{base}.NS")
    return candidates


@app.get("/get-news-articles/{ticker}")
def get_news_articles(ticker: str, limit: int = Query(10, ge=1, le=100)):
    candidates = _candidate_tickers(ticker)
    collection = client["stock_news"]["stock_articles"]

    try:
        docs = []
        for candidate in candidates:
            cursor = collection.find({"Ticker": candidate}, {"_id": 0}).sort("Date", -1).limit(limit)
            docs.extend(list(cursor))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch news articles: {e}")

    seen_links = set()
    articles = []
    for doc in sorted(docs, key=lambda d: str(d.get("Date", "")), reverse=True):
        link = doc.get("Link") or doc.get("link")
        if not link or link in seen_links:
            continue
        seen_links.add(link)
        articles.append({
            "title": doc.get("Title") or doc.get("title") or "",
            "summary": doc.get("Summary") or doc.get("summary") or "",
            "link": link,
            "timestamp": doc.get("Date") or doc.get("timestamp"),
            "source": doc.get("Source") or doc.get("source") or "",
        })

    if not articles:
        raise HTTPException(status_code=404, detail=f"No news articles found for {ticker}.")

    return {
        "ticker": _normalize_ticker(ticker),
        "articles": articles[:limit],
    }


def _fetch_latest_reddit_doc(ticker: str):
    collection = client["alt_sentiment"]["reddit_finbert"]
    return collection.find_one({"stock": ticker}, sort=[("date", -1)])


def _classify_sentiment(positive: float, negative: float, neutral: float) -> str:
    strongest = max(positive, negative, neutral)
    if strongest == positive:
        return "very positive" if positive >= 65 else "positive"
    if strongest == negative:
        return "very negative" if negative >= 65 else "negative"
    return "neutral"


def _build_reddit_payload(ticker: str, hours: int, doc: dict):
    if not doc:
        raise HTTPException(status_code=404, detail=f"Reddit sentiment not found for {ticker}.")

    positive = float(doc.get("positive_pct", 0.0))
    negative = float(doc.get("negative_pct", 0.0))
    neutral = float(doc.get("neutral_pct", 0.0))

    sentiment = {
        "simple_avg": {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
        },
        "time_weighted": {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
        },
        "counts": {
            "positive": int(doc.get("positive_count", 0)),
            "negative": int(doc.get("negative_count", 0)),
            "neutral": int(doc.get("neutral_count", 0)),
        },
        "total_items": int(doc.get("total_items", doc.get("items_processed", 0))),
        "overall_sentiment": doc.get("overall_sentiment") or _classify_sentiment(positive, negative, neutral),
        "examples": {
            "most_positive": doc.get("most_positive", []),
            "most_negative": doc.get("most_negative", []),
        },
        "generated_at": doc.get("generated_at") or doc.get("date"),
        "ticker": ticker,
        "analysis_window_hours": int(doc.get("analysis_window_hours", hours)),
        "success": True,
    }

    return {
        "ticker": ticker,
        "analysis_type": doc.get("analysis_type", "reddit_finbert"),
        "window_hours": int(doc.get("analysis_window_hours", hours)),
        "sentiment_data": sentiment,
    }


@app.get("/get-reddit-sentiment/{ticker}")
def get_reddit_sentiment(ticker: str, hours: int = Query(168, ge=1, le=720)):
    normalized = _normalize_ticker(ticker)
    try:
        doc = _fetch_latest_reddit_doc(normalized)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Reddit sentiment: {e}")

    return _build_reddit_payload(normalized, hours, doc)


@app.get("/get-reddit-sentiment-mongo/{ticker}")
def get_reddit_sentiment_mongo(ticker: str, hours: int = Query(168, ge=1, le=720)):
    normalized = _normalize_ticker(ticker)
    try:
        doc = _fetch_latest_reddit_doc(normalized)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Reddit sentiment: {e}")

    return _build_reddit_payload(normalized, hours, doc)


@app.post("/refresh-reddit-sentiment/{ticker}")
def refresh_reddit_sentiment(ticker: str, hours: int = Query(168, ge=1, le=720)):
    normalized = _normalize_ticker(ticker)

    parent_conn, child_conn = Pipe()
    process = Process(target=run_reddit_sentiment_script, args=(normalized, child_conn))

    try:
        process.start()
        child_conn.close()
        process.join(timeout=300)

        if process.is_alive():
            process.terminate()
            process.join()

        if parent_conn.poll():
            _ = parent_conn.recv()
    finally:
        parent_conn.close()

    try:
        doc = _fetch_latest_reddit_doc(normalized)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh Reddit sentiment: {e}")

    return _build_reddit_payload(normalized, hours, doc)
