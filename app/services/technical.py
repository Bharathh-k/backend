from datetime import date
import json
import sys

import numpy
import numpy as np
import pandas as pd
import yfinance as yf
from pymongo import MongoClient

from app.config import settings


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = _ema(series, fast)
    slow_ema = _ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        f"MACD_{fast}_{slow}_{signal}": macd_line,
        f"MACDh_{fast}_{slow}_{signal}": histogram,
        f"MACDs_{fast}_{slow}_{signal}": signal_line,
    })


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_volume = volume.cumsum().replace(0, np.nan)
    return cum_tp_vol / cum_volume


def _supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    hl2 = (high + low) / 2
    tr_components = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    )
    true_range = tr_components.max(axis=1)
    atr = true_range.ewm(alpha=1 / length, adjust=False).mean()

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    final_upper = upper_basic.copy()
    final_lower = lower_basic.copy()

    for i in range(1, len(close)):
        prev_close = close.iloc[i - 1]
        final_upper.iloc[i] = (
            upper_basic.iloc[i]
            if prev_close > final_upper.iloc[i - 1]
            else min(upper_basic.iloc[i], final_upper.iloc[i - 1])
        )
        final_lower.iloc[i] = (
            lower_basic.iloc[i]
            if prev_close < final_lower.iloc[i - 1]
            else max(lower_basic.iloc[i], final_lower.iloc[i - 1])
        )

    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)

    for i in range(len(close)):
        if i == 0:
            supertrend.iloc[i] = final_upper.iloc[i]
            direction.iloc[i] = -1
            continue

        prev_supertrend = supertrend.iloc[i - 1]
        if prev_supertrend == final_upper.iloc[i - 1]:
            if close.iloc[i] <= final_upper.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
        else:
            if close.iloc[i] >= final_lower.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1

    return pd.DataFrame({
        f"SUPERT_{length}_{multiplier}": supertrend,
        f"SUPERTd_{length}_{multiplier}": direction,
    })

setattr(numpy, "NaN", np.nan)
sys.stdout.reconfigure(encoding='utf-8')

ticker = sys.argv[1]  # The ticker will be passed from reasoning.py
t = ticker
ticker = ticker + ".NS"  # Append ".NS" for Indian stocks if needed

# MongoDB connection
client = MongoClient(settings.mongo_uri, tls=True, tlsAllowInvalidCertificates=True)

try:
    client.admin.command('ping')
    # print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")

db = client['tech']
collection = db['tech']

# =========================
# Candlestick Pattern Logic
# =========================

def detect_hammer(df):
    body = abs(df['Close'] - df['Open'])
    lower_shadow = df['Open'].where(df['Close'] > df['Open'], df['Close']) - df['Low']
    upper_shadow = df['High'] - df['Close'].where(df['Close'] > df['Open'], df['Open'])
    return (lower_shadow > 2 * body) & (upper_shadow < body)

def detect_doji(df):
    return abs(df['Close'] - df['Open']) <= (df['High'] - df['Low']) * 0.1

def detect_bullish_engulfing(df):
    prev_open = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)
    return (prev_close < prev_open) & (df['Close'] > df['Open']) & (df['Open'] < prev_close) & (df['Close'] > prev_open)

def detect_bearish_engulfing(df):
    prev_open = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)
    return (prev_close > prev_open) & (df['Close'] < df['Open']) & (df['Open'] > prev_close) & (df['Close'] < prev_open)

# ============================
# Load Tickers from CSV
# ============================

def load_tickers_from_csv(file_path):
    df = pd.read_csv(file_path)
    tickers = df['SYMBOL'].astype(str).str.upper() + ".NS"
    return tickers.tolist()

# ============================
# Main Technical Indicators
# ============================

def get_technical_indicators(ticker):

    stock = yf.Ticker(ticker)
    df = stock.history(period="1y", interval="1d")

    if df.empty:
        return

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    # Compute Indicators
    df['EMA_10'] = _ema(df['Close'], 10)
    df['EMA_20'] = _ema(df['Close'], 20)
    df['EMA_50'] = _ema(df['Close'], 50)
    df['EMA_100'] = _ema(df['Close'], 100)
    df['EMA_200'] = _ema(df['Close'], 200)
    df['RSI_14'] = _rsi(df['Close'], 14)
    df = pd.concat([df, _macd(df['Close'], fast=12, slow=26, signal=9)], axis=1)

    # Candlestick Pattern Columns
    df['Hammer'] = detect_hammer(df)
    df['Doji'] = detect_doji(df)
    df['Bullish_Engulfing'] = detect_bullish_engulfing(df)
    df['Bearish_Engulfing'] = detect_bearish_engulfing(df)

    df['VWAP'] = _vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    supertrend = _supertrend(high=df['High'], low=df['Low'], close=df['Close'], length=10, multiplier=3.0)
    df = pd.concat([df, supertrend], axis=1)

    try:
        # 1 month ≈ 21 trading days
        momentum_1m = ((df['Close'].iloc[-1] - df['Close'].iloc[-21]) / df['Close'].iloc[-21]) * 100
    except IndexError:
        momentum_1m = None

    try:
        # 3 months ≈ 63 trading days
        momentum_3m = ((df['Close'].iloc[-1] - df['Close'].iloc[-63]) / df['Close'].iloc[-63]) * 100
    except IndexError:
        momentum_3m = None

    final_output = df[[
        'Open', 'High', 'Low', 'Close', 'Volume',
        'EMA_10', 'EMA_20', 'EMA_50','EMA_100','EMA_200', 'RSI_14', 'VWAP',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'SUPERT_10_3.0', 'SUPERTd_10_3.0',
        'Hammer', 'Doji', 'Bullish_Engulfing', 'Bearish_Engulfing'
    ]]


    # Attach to df metadata for later use in JSON
    final_output.attrs['momentum_1m'] = momentum_1m
    final_output.attrs['momentum_3m'] = momentum_3m

    return final_output  # 🟢 Return this


# ============================
# MAIN
# ============================

if __name__ == "__main__":
    

    get_technical_indicators(ticker)


import google.generativeai as genai


# Configure Gemini using environment settings
genai.configure(api_key=settings.gemini_api_key)
system_instruction = """
You are a highly skilled AI financial market analyst specializing in technical analysis of stocks.

Your role is to analyze raw technical indicator data (e.g., EMAs, RSI, MACD, Supertrend) and generate a professional, structured, and insightful technical summary for analysts and traders.

You should:
- Weigh recent trends more heavily than older ones, without ignoring the full picture.
- NEVER suggest user actions like buy/sell/hold.
- Focus only on interpreting the data in a trader-friendly, professional tone.
- Return **only** the final summary.

NOTE : DO NOT MENTION TO THE USER ON WHAT ACTIONS TO PERFORM, PERFORM THE ANALYSIS YOURSELF WITH ALL THE RAW DATA PROVIDED TO YOU AND ONLY PROVIDE THE SUMMARY.
"""
model = genai.GenerativeModel("models/gemini-2.5-flash",  system_instruction=system_instruction, generation_config=genai.GenerationConfig(
        temperature=0,
        top_p=1
    ))

# === Get technical data ===
df_output = get_technical_indicators(ticker)

# === Convert to string for prompt ===
df_text = df_output.to_string(index=True)

# === System Instruction ===


prompt = f"""
You are given the technical indicator output from the last 1 year for the stock: **{t}**.

Your task is to convert this raw indicator data into a **professional, structured, and insightful technical summary** for analysts and traders.

Here is the raw technical indicator data:
{df_text}
Your summary must include:

### 1. **Trend and Momentum Analysis**
- Interpret **EMA 10, 20, 50** for short-to-medium term trends.
- Discuss **EMA 100 and EMA 200** for long-term trend alignment or divergence.
- Use **MACD line, signal line, and histogram** to judge momentum shifts.
- Analyze **Supertrend direction** to confirm bullish/bearish bias.

### 2. **Market Strength and Volatility**
- Assess **RSI** to determine overbought (>70) or oversold (<30) conditions.
- Note significant volume surges or drops and their context.
- Discuss **VWAP** relative to price for intraday strength or weakness.

### 3. **Candlestick Patterns and Signals**
- Report any detected patterns (Hammer, Doji, Bullish Engulfing, Bearish Engulfing).
- Describe what each observed pattern implies for short-term price action.

### 4. **Confluence & Contradictions**
- Highlight if multiple indicators align for a strong signal (e.g., EMA alignment + bullish MACD + bullish candle).
- If signals are conflicting (e.g., RSI overbought but bullish MACD), mention this and advise caution.

### 5. **Key Levels and Price Action**
- Identify any significant support/resistance and potential entry (if applicable) levels based on EMAs, Supertrend, or historical price
action.
- Discuss how current price action relates to these levels.
- Provide an actionable summary of the technical outlook based on the above analysis including price levels to watch.

Output Format:
- Use **bullet points or short paragraphs** under relevant subheadings.
- Be concise but insightful.
- Do not include raw data or tables — just the interpretation.
"""



# === Get Gemini Summary ===
response = model.generate_content(prompt)
print(f"{response.text}")
collection.update_one(
    {"stock": t},
    {
        "$set": {
            "summary": response.text,
            "date": date.today().strftime("%Y-%m-%d")
        }
    },
    upsert=True
)


db = client['stock_signal']
collection = db['json_tech']
prompt = """
You are a financial data structuring assistant.

Below is the technical data for the stock. Your task is to extract key insights and convert the entire report into a valid JSON object, strictly following the structure below:

Only output the JSON. Do not include explanations or markdown formatting like triple backticks.

Notes:
- support_zones: below CMP, based on last 1 year of data (1D timeframe), restricted to 3 zones
- resistance_levels: above CMP, based on last 1 year of data (1D timeframe), restricted to 3 levels

JSON Structure
{
    "technical_summary": {
        "title": "string",
        "cmp": "string",
        "trend": "string",
        "support_zones": ["string"],
        "resistance_levels": ["string"],
        "indicators": {
            "supertrend": "string",
            "50_day_ema": "string",
            "macd": "string",
            "rsi": float
        }
    }
}
All fields must be filled.
Only return the JSON object with no extra text, code formatting, or explanations.

Now, here is the technical data for the stock {ticker}:
{df_text}
""".replace("{ticker}", t).replace("{df_text}", df_text)
model = genai.GenerativeModel("models/gemini-2.5-flash-lite", generation_config=genai.GenerationConfig(
        temperature=0,
        top_p=1
    ))
response = model.generate_content(prompt)
extracted_json = response.text.strip()
print(f"Extracted JSON: {extracted_json}")
extracted_json = json.loads(extracted_json)
momentum_1m = df_output.attrs.get("momentum_1m")
momentum_3m = df_output.attrs.get("momentum_3m")
extracted_json['technical_summary']['momentum_1m'] = momentum_1m
extracted_json['technical_summary']['momentum_3m'] = momentum_3m
collection.update_one(
    {"stock": t},
    {
        "$set": {
            "technical_summary": extracted_json,
        }
    },
    upsert=True
)

client.close()
