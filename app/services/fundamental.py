#BHARGAV ILLEGAL SCRAPING STANDARDISATION
import ast
import json
import sys
from datetime import date
from pathlib import Path

import google.generativeai as genai
import pandas as pd
import yfinance as yf
from pymongo import MongoClient

from app.config import settings

ticker = sys.argv[1]  # The ticker will be passed from reasoning.py
# ticker = "SAMHI"
t = ticker
ticker = ticker + ".NS"  # Append ".NS" for Indian stocks if needed
# Load your Screener-exported CSV
df = pd.read_csv(settings.get_data_path("stock(11-06-2025).csv"))

# ---- Conversion Functions ----

# Convert a single number from crores → million/billion string
def convert_single_crore(value):
    try:
        # Remove commas from the string
        value_cleaned = str(value).replace(",", "")
        num = float(value_cleaned) * 10  # 1 crore = 10 million
        if num >= 1000:
            return f"{num / 1000:.2f} billion"
        else:
            return f"{num:.2f} million"
    except:
        return str(value)


# Convert a list of crores (e.g., ["2", "4", "5"]) → ["20.00 million", "40.00 million", ...]
def convert_list_of_crores(val):
    try:
        items = ast.literal_eval(val)
        if isinstance(items, list):
            converted = []
            for x in items:
                num = float(x) * 10
                if num >= 1000:
                    converted.append(f"{num / 1000:.2f} billion")
                else:
                    converted.append(f"{num:.2f} million")
            return str(converted)
        return val
    except:
        return val

# ---- Columns to Convert ----

# Columns where values are lists of crores
list_crore_columns = [
    "np_quarterly",
    "np_yearly",
    "CWIP",
    "Cash from Operating Activities"
]

# Single-value columns in crores
single_crore_columns = [
    "Market Cap"
]

# ---- Apply Conversions ----

# Convert list-based columns
for col in list_crore_columns:
    if col in df.columns:
        df[col] = df[col].apply(convert_list_of_crores)

# Convert single crore value columns
for col in single_crore_columns:
    if col in df.columns:
        df[col] = df[col].apply(convert_single_crore)




# Select the stocks with the same Sector as the ticker from sector_hierarchy.csv

sector_hierarchy = pd.read_csv(settings.get_data_path("sector_hierarchy.csv"))
sector = sector_hierarchy[sector_hierarchy['SYMBOL'] == ticker[:-3]]['Sector'].to_list()[0]
sector_stocks = sector_hierarchy[sector_hierarchy['Sector'] == sector]['SYMBOL']
df_original = pd.read_csv(settings.get_data_path("stock(11-06-2025).csv"))

# Ensure 'Market Cap' is a numeric column (float)
df_original["Market Cap"] = pd.to_numeric(df_original["Market Cap"], errors='coerce')

# Fill any NaN values that might result from coercion (optional)
df_original["Market Cap"] = df_original["Market Cap"].fillna(0)
top_stocks = df_original[df_original['SYMBOL'].isin(sector_stocks)].nlargest(7, 'Market Cap')


# SUMMARIZATION USING GEMINI
# Configure Gemini
genai.configure(api_key=settings.gemini_api_key)

# Load standardized Screener dataset
screener_df = df  # Ensure df is preloaded with your Screener CSV

# Format large numbers
def preprocess_number(number):
    try:
        number = float(number)
        if number >= 1e9:
            return f"{number / 1e9:.2f} billion"
        elif number >= 1e6:
            return f"{number / 1e6:.2f} million"
        elif number >= 1e3:
            return f"{number / 1e3:.2f} thousand"
        else:
            return str(number)
    except:
        return "N/A"

def standardize_data(df):
    return df.apply(lambda col: col.map(preprocess_number) if pd.api.types.is_numeric_dtype(col) else col)

# Fetch Yahoo Finance data
def fetch_and_standardize_financial_data(ticker):
    stock = yf.Ticker(ticker)

    financials = stock.quarterly_financials
    balance_sheet = stock.quarterly_balance_sheet
    cash_flow = stock.quarterly_cashflow

    if cash_flow.empty:
        cash_flow = stock.cashflow

    financials_standardized = standardize_data(financials)
    balance_sheet_standardized = standardize_data(balance_sheet)
    cash_flow_standardized = standardize_data(cash_flow)

    return financials_standardized, balance_sheet_standardized, cash_flow_standardized

# Combine Yahoo + Screener data
def prepare_text_for_summary(ticker, financials, balance_sheet, cash_flow, screener_row):
    def get(row, df):
        try:
            value = df.loc[row].iloc[0]
            return value if pd.notna(value) else None
        except:
            return None

    def add_metric(label, row, df):
        val = get(row, df)
        return f"- {label}: {val}\n" if val is not None else ""

    # Columns to exclude from the summary
    excluded_columns = [
        'Market cap', 'Current Price', 'High / Low'
    ]

    # Remove the excluded columns
    screener_row_filtered = screener_row.drop(columns=excluded_columns, errors='ignore')

    text = f"Financial Summary for {ticker} (latest available quarter):\n\n"

    text += "🔹 Profitability:\n"
    text += add_metric("Net Income", "Net Income", financials)
    text += add_metric("EBITDA", "EBITDA", financials)
    text += add_metric("EBIT", "EBIT", financials)
    text += add_metric("Gross Profit", "Gross Profit", financials)
    text += add_metric("Operating Income", "Operating Income", financials)
    text += add_metric("Net Interest Income", "Net Interest Income", financials)
    text += add_metric("Basic EPS", "Basic EPS", financials)
    text += add_metric("Diluted EPS", "Diluted EPS", financials)
    text += "\n"

    text += "🔹 Revenue:\n"
    text += add_metric("Total Revenue", "Total Revenue", financials)
    text += add_metric("Operating Revenue", "Operating Revenue", financials)
    text += add_metric("Cost of Revenue", "Cost Of Revenue", financials)
    text += add_metric("Reconciled Cost Of Revenue", "Reconciled Cost Of Revenue", financials)
    text += "\n"

    text += "🔹 Expenses:\n"
    text += add_metric("Total Expenses", "Total Expenses", financials)
    text += add_metric("Operating Expense", "Operating Expense", financials)
    text += add_metric("Other Operating Expenses", "Other Operating Expenses", financials)
    text += add_metric("Interest Expense", "Interest Expense", financials)
    text += add_metric("Tax Provision", "Tax Provision", financials)
    text += "\n"

    text += "🔹 Cash Flow (Fallback to Annual if quarterly unavailable):\n"
    text += add_metric("Operating Cash Flow", "Operating Cash Flow", cash_flow)
    text += add_metric("Free Cash Flow", "Free Cash Flow", cash_flow)
    text += add_metric("Capital Expenditure", "Capital Expenditure", cash_flow)
    text += add_metric("Depreciation and Amortization", "Depreciation And Amortization", cash_flow)
    text += "\n"

    text += "🔹 Balance Sheet:\n"
    text += add_metric("Total Debt", "Total Debt", balance_sheet)
    text += add_metric("Net Debt", "Net Debt", balance_sheet)
    text += add_metric("Tangible Book Value", "Tangible Book Value", balance_sheet)
    text += add_metric("Cash & Cash Equivalents", "Cash And Cash Equivalents", balance_sheet)
    text += add_metric("Ordinary Shares Number", "Ordinary Shares Number", balance_sheet)
    text += "\n"

    text += "🔹 Screener Insights: (values in list are in the past to present order)\n"
    screener_columns = [
        'Market Cap', 'Current Price', 'High / Low', 'Stock P/E',
        'Book Value', 'Dividend Yield', 'ROCE', 'ROE', 'Face Value',
        'Compounded Sales Growth',
        'Compounded Profit Growth', 'CWIP', 'Cash from Operating Activities',
        'Debtor Days'
    ]
    for col in screener_columns:
        if col in screener_row_filtered and pd.notna(screener_row_filtered[col]):
            text += f"- {col}: {screener_row_filtered[col]}\n"

    return text


def extract_full_or_single(val):
    """
    Return full list if string is a list, else return as-is.
    """
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return parsed  # Return full list
    except:
        pass
    return val

def format_dataframe_for_prompt(df):
    """
    Returns a multi-line string where each row in the DataFrame is converted
    to a formatted string with all column values, preserving lists.
    """
    output_lines = []
    for _, row in df.iterrows():
        ticker = row.get("SYMBOL", "N/A")
        parts = [f"Ticker: {ticker}"]
        for col in df.columns:
            if col == "SYMBOL":
                continue
            val = extract_full_or_single(row[col])
            parts.append(f"{col}: {val}")
        output_lines.append(" | ".join(parts))
    return "\n".join(output_lines)


# Assuming your DataFrame is named top_stocks_df
peer_comparison_string = format_dataframe_for_prompt(top_stocks)


# Gemini summary generation
def summarize_with_gemini(prompt_text):
    system_instruction = """
        You are a seasoned financial analyst.

        Your task is to analyze structured company fundamentals (e.g., ROE, ROCE, PE, debt, profit trends, etc.) provided from Yahoo Finance and Screener.

        You will receive raw numerical data with the latest values appearing last (i.e., chronological order). Based on this data:

        - Identify trends over time
        - Highlight key strengths and risks
        - Give a professional assessment of financial health and valuation
        - Conclude with a one-word tone: POSITIVE, NEUTRAL, or NEGATIVE — backed with a reason

        You must always follow a strict structure. Never include raw data or extra commentary.
"""
    model = genai.GenerativeModel(model_name="models/gemini-2.5-flash", system_instruction=system_instruction, generation_config=genai.GenerationConfig(
        temperature=0,
        top_p=1
    ))
    today = date.today().strftime("%d %b %Y")
    prompt_context = f"""
Date: {today}
Company: {ticker}

## 1. Profitability Metrics
- Summarize trends in Net Profit, ROE, ROCE, and other profitability ratios.
- Comment on stability, improvement, or decline over time.
- Example: ROCE has steadily improved from 12% to 18% over 4 years, indicating enhanced capital efficiency.

## 2. Valuation Overview
- Interpret P/E, P/B, and other valuation multiples.
- Compare historical vs current levels to gauge attractiveness.
- Example: P/E has dropped from 32x to 22x over the last 3 years, making the stock relatively more attractive.

## 3. Return & Efficiency
- Cover asset turnover, EPS growth, and capital efficiency trends.
- Highlight compounding ability and operational efficiency.
- Example: EPS has grown consistently at 18% CAGR over the last 5 years.

## 4. Financial Health
- Evaluate debt levels, interest coverage, and cash flows.
- Highlight any notable improvement or deterioration.
- Example: Debt-to-equity reduced from 1.1x to 0.3x; cash flows remain consistently positive.

## 5. Key Strengths
- List 2–4 strong fundamental points.
- Example:
  - High ROCE with improving trend
  - Debt-free status
  - Strong free cash flow generation

## 6. Peer Comparison
- Compare performance against peers using provided peer data.
- Mention key competitive advantages or gaps.
- Example:
  - Highest ROE among peers at 18%
  - P/B ratio below sector average, indicating better valuation
  - Revenue growth rate above sector average

## 7. Risks & Concerns
- List 2–4 red flags or watchpoints.
- Example:
  - Recent margin contraction
  - Declining interest coverage ratio

## 8. Overall Outlook
- Categorize as POSITIVE, NEUTRAL, or NEGATIVE (choose only one).
- Provide a detailed explanation after the classification as to why this categorization has been made.

---

Raw Financial Data:
{prompt_text}

Peer Comparison: (all values in the list are in the past to present order) (net profit values are in crores)
{peer_comparison_string}
"""

    response = model.generate_content(prompt_context)

    return response.text.strip()




# Master function
def generate_full_summary(ticker):
    financials, balance_sheet, cash_flow = fetch_and_standardize_financial_data(ticker)
    symbol = ticker.replace(".NS", "")
    screener_row = screener_df[screener_df["SYMBOL"] == symbol].iloc[0] if symbol in screener_df["SYMBOL"].values else {}
    screener_row = screener_row.rename({
        "Compounded Sales Growth": "Compounded Sales Growth (TTM)",
        "Compounded Profit Growth": "Compounded Profit Growth (TTM)"
    })
    combined_text = prepare_text_for_summary(ticker, financials, balance_sheet, cash_flow, screener_row)
    summary = summarize_with_gemini(combined_text)
    print(summary)

generate_full_summary(ticker)
