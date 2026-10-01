import os
import json
import logging
from datetime import date

log = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"


def _call_gemini(system_prompt, user_prompt):
    """Gemini is disabled — no longer required for now. Returns "" instead
    of calling out; every caller (generate_summary/generate_recommendations/
    generate_followup) already treats a parse failure
    here as "no AI data" and degrades to its own fallback, so nothing else
    needs to change. To re-enable, restore the google-genai call here."""
    return ""


SYSTEM_PROMPT = """You are an expert financial analyst and portfolio manager specializing in mining, commodities, and equity markets.
You analyze quantitative financial data and generate concise, professional investment analysis.
Always respond with valid JSON only. No preamble, no markdown, no explanation outside the JSON."""


def build_stock_prompt(stock):
    ticker    = stock.get("ticker")
    annual    = stock.get("annualIncome", [])
    quarterly = stock.get("quarterlyIncome", [])
    balance   = stock.get("annualBalance", [])
    cashflow  = stock.get("annualCashflow", [])
    changes   = stock.get("annualChanges", {})
    q_changes = stock.get("quarterlyChanges", {})
    cal       = stock.get("calendar", {})
    ev        = stock.get("events", {})
    today     = date.today().strftime("%Y-%m-%d")

    return f"""Today is {today}. Analyze this stock and return a JSON object with EXACTLY these fields:

{{
  "highlight": "2-3 sentence investment thesis",
  "marketPosition": "1 sentence competitive position",
  "majorGrowthProjects": ["catalyst 1", "catalyst 2"],
  "shareholderReturns": "1 sentence on dividends and buybacks",
  "balanceSheetSummary": "1 sentence on financial health",
  "riskFactors": ["risk 1", "risk 2"],
  "analystSentiment": "bullish OR neutral OR bearish",
  "shortWarning": {{
    "shouldShort": true or false,
    "reason": "explanation or null",
    "confidence": "high OR medium OR low"
  }},
  "eventImpacts": [
    {{
      "event": "event name",
      "date": "YYYY-MM-DD",
      "startWatchingDate": "YYYY-MM-DD (exact date to start paying attention, calculated from today {today})",
      "expectedImpact": "Detailed price impact: what direction, estimated % move range, why",
      "epsEstimate": "exact EPS estimate if available, e.g. $0.78 avg, range $0.65-$0.86",
      "revenueEstimate": "exact revenue estimate if available",
      "whereToMonitor": ["source 1 e.g. Yahoo Finance earnings page", "source 2 e.g. SEC EDGAR", "source 3 e.g. company IR page"],
      "whatToWatchFor": ["specific metric 1 to compare vs estimate", "specific metric 2"],
      "actionPlan": [
        {{"date": "YYYY-MM-DD", "action": "what to do on this specific date"}},
        {{"date": "YYYY-MM-DD", "action": "what to do on this specific date"}},
        {{"date": "YYYY-MM-DD", "action": "what to do on this specific date"}}
      ],
      "strategy": "overall strategy summary"
    }}
  ]
}}

FINANCIAL DATA:
Ticker: {ticker}
Sector: {stock.get("sector")} | Industry: {stock.get("industry")}
Price: ${stock.get("price")} | Market Cap: ${stock.get("marketCap")}
Annual Profit Margin: {stock.get("annualProfitMargin")} | Quarterly Margin: {stock.get("quarterlyProfitMargin")}
Net Cash: ${stock.get("netCash")} | Latest FCF: ${stock.get("latestFCF")} | CapEx: ${stock.get("latestCapex")}

Latest Annual Income:
{json.dumps(annual[0] if annual else {}, indent=2)}

Latest Quarterly Income:
{json.dumps(quarterly[0] if quarterly else {}, indent=2)}

Balance Sheet:
{json.dumps(balance[0] if balance else {}, indent=2)}

Annual Cash Flow:
{json.dumps(cashflow[0] if cashflow else {}, indent=2)}

Annual Changes YoY:
{json.dumps(changes, indent=2)}

Quarterly Changes QoQ/YoY:
{json.dumps(q_changes, indent=2)}

Upcoming Events:
Earnings Date: {cal.get("earningsDateStart")}
EPS Estimate High/Low/Avg: {cal.get("earningsEPSHigh")} / {cal.get("earningsEPSLow")} / {cal.get("earningsEPSAvg")}
Revenue Estimate High/Low/Avg: {cal.get("revenueEstimateHigh")} / {cal.get("revenueEstimateLow")} / {cal.get("revenueEstimateAvg")}
Ex-Dividend Date: {cal.get("exDividendDate")}
Dividend Pay Date: {cal.get("dividendPayDate")}
Earnings Call: {ev.get("earningsCallStart")}
Dividend Yield: {ev.get("dividendYield")}
Earnings Growth YoY: {ev.get("earningsGrowthYoY")}
Earnings Quarterly Growth: {ev.get("earningsQuarterlyGrowth")}
Last Dividend: ${ev.get("lastDividendValue")} on {ev.get("lastDividendDate")}
"""


def build_recommendations_prompt(watchlist, macro=None, dream_candidates=None):
    today = date.today().strftime("%Y-%m-%d")

    context = []
    for s in watchlist:
        context.append({
            "ticker":             s.get("ticker"),
            "sector":             s.get("sector"),
            "industry":           s.get("industry"),
            "price":              s.get("price"),
            "rsi14":              s.get("rsi14"),
            "annualProfitMargin": s.get("annualProfitMargin"),
            "netCash":            s.get("netCash"),
            "latestFCF":          s.get("latestFCF"),
            "marketCap":          s.get("marketCap"),
            "analystSentiment":   s.get("aiSummary", {}).get("analystSentiment"),
            "change7d":           s.get("change7d"),
            "change30d":          s.get("change30d"),
        })

    macro_block = "No macro data available."
    if macro:
        indicators = macro.get("indicators", {})
        lines = [f"- {k}: {v.get('value') if isinstance(v, dict) else v}" for k, v in list(indicators.items())[:8]]
        macro_block = "\n".join(lines)
        if macro.get("aiSummary"):
            macro_block += f"\nMacro Summary: {macro['aiSummary'][:400]}"

    dream_block = "No dream candidates available."
    if dream_candidates:
        top = sorted(dream_candidates, key=lambda x: x.get("score", 0), reverse=True)[:10]
        dream_block = "\n".join([
            f"- {d['ticker']} | score:{d.get('score')} | rsi:{d.get('rsi14')} | sector:{d.get('sector')} | {d.get('change7d', 0):+.1f}% 7d"
            for d in top
        ])

    return f"""Today is {today}. You are a portfolio advisor. Based on the user's watchlist, macro environment, and pre-screened dream candidates, recommend 5-10 stocks to consider buying.
Prioritize strong fundamentals, reasonable valuation, and growth catalysts. Ensure sector diversification relative to existing holdings.

CURRENT WATCHLIST:
{json.dumps(context, indent=2)}

MACRO ENVIRONMENT:
{macro_block}

PRE-SCREENED DREAM CANDIDATES (ranked by score):
{dream_block}

Return a JSON object with EXACTLY this structure:
{{
  "recommendations": [
    {{
      "ticker": "TICKER",
      "companyName": "Full Company Name",
      "reason": "2-3 sentence explanation of why to buy",
      "sector": "sector name",
      "catalysts": ["catalyst 1", "catalyst 2"],
      "riskLevel": "low OR medium OR high",
      "timeHorizon": "short-term OR medium-term OR long-term"
    }}
  ],
  "marketContext": "1-2 sentence summary of current macro conditions influencing recommendations",
  "generatedAt": "{today}"
}}"""


def generate_recommendations(watchlist, macro=None, dream_candidates=None):
    try:
        log.info(f"AI generating recommendations | macro:{bool(macro)} | dream_candidates:{len(dream_candidates) if dream_candidates else 0}")
        raw    = _call_gemini(SYSTEM_PROMPT, build_recommendations_prompt(watchlist, macro, dream_candidates))
        result = json.loads(raw)
        log.info(f"AI recommendations OK: {len(result.get('recommendations', []))} stocks")
        return result
    except Exception as e:
        log.error(f"AI recommendations FAIL: {e}")
        return {"recommendations": [], "marketContext": "Unavailable", "generatedAt": ""}


def generate_summary(stock):
    ticker = stock.get("ticker")
    try:
        log.info(f"AI generating summary: {ticker}")
        raw     = _call_gemini(SYSTEM_PROMPT, build_stock_prompt(stock))
        summary = json.loads(raw)
        log.info(f"AI OK: {ticker} | sentiment:{summary.get('analystSentiment')} | short:{summary.get('shortWarning', {}).get('shouldShort')}")
        return summary
    except json.JSONDecodeError as e:
        log.error(f"AI JSON parse FAIL: {ticker} | {e}")
        return default_summary(ticker)
    except Exception as e:
        log.error(f"AI FAIL: {ticker} | {e}")
        return default_summary(ticker)


def generate_followup(stock, event_name):
    ticker = stock.get("ticker")
    today  = date.today().strftime("%Y-%m-%d")
    ai     = stock.get("aiSummary", {})
    event  = next((e for e in ai.get("eventImpacts", []) if e.get("event") == event_name), {})

    prompt = f"""Today is {today}. The user just clicked to get a follow-up plan for {ticker} regarding: {event_name}.

Event details: {json.dumps(event, indent=2)}
Stock context: Ticker={ticker}, Price=${stock.get('price')}, Sentiment={ai.get('analystSentiment')}
Latest quarterly changes: {json.dumps(stock.get('quarterlyChanges', {}), indent=2)}

Generate a detailed step-by-step follow-up plan. Return JSON with EXACTLY:
{{
  "summary": "1-2 sentence current situation summary",
  "steps": [
    {{
      "date": "YYYY-MM-DD",
      "title": "short action title",
      "instructions": "detailed instructions what to do, what to look for, what numbers to check",
      "fetchRequired": true or false,
      "sources": ["where to check"]
    }}
  ],
  "redFlags": ["warning sign 1 to watch for", "warning sign 2"],
  "greenFlags": ["positive signal 1", "positive signal 2"],
  "decisionPoint": "date and condition that triggers buy/sell/hold decision"
}}"""

    try:
        log.info(f"AI followup: {ticker} | {event_name}")
        raw    = _call_gemini(SYSTEM_PROMPT, prompt)
        result = json.loads(raw)
        log.info(f"AI followup OK: {ticker}")
        return result
    except Exception as e:
        log.error(f"AI followup FAIL: {ticker} | {e}")
        return {"error": "Follow-up generation failed"}


def default_summary(ticker):
    return {
        "highlight":            f"Analysis unavailable for {ticker}",
        "marketPosition":       "N/A",
        "majorGrowthProjects":  [],
        "shareholderReturns":   "N/A",
        "balanceSheetSummary":  "N/A",
        "riskFactors":          [],
        "analystSentiment":     "neutral",
        "shortWarning":         {"shouldShort": False, "reason": None, "confidence": "low"},
        "eventImpacts":         []
    }
