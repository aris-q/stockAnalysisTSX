import requests
import yfinance as yf
import logging
import math
import time
from datetime import datetime, timezone, timedelta

log = logging.getLogger(__name__)

def fetch_live_prices(tickers):
    """Fetch a fresh market price for each ticker. Returns {ticker: price}.
    Tickers that fail to fetch are omitted so callers can tell fresh from stale."""
    prices = {}
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            price = None
            try:
                price = stock.fast_info.last_price
                if price is not None and math.isnan(float(price)):
                    price = None
            except Exception:
                pass
            if not price:
                info = stock.info
                price = info.get("currentPrice") or info.get("regularMarketPrice")
            if not price:
                hist = stock.history(period="5d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            if price and price > 0:
                prices[t] = round(float(price), 4)
        except Exception as e:
            log.warning(f"fetch_live_prices FAIL: {t} | {e}")
    log.info(f"fetch_live_prices: {len(prices)}/{len(tickers)} fetched")
    return prices


def fetch_prev_closes(tickers):
    """Fetch yesterday's official close for each ticker (daily-drop stop checks).
    Tickers that fail are omitted so callers can tell fresh from missing."""
    closes = {}
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            pc = None
            try:
                pc = stock.fast_info.previous_close
                if pc is not None and math.isnan(float(pc)):
                    pc = None
            except Exception:
                pass
            if not pc:
                hist = stock.history(period="5d")
                if len(hist) >= 2:
                    pc = float(hist["Close"].iloc[-2])
            if pc and pc > 0:
                closes[t] = round(float(pc), 4)
        except Exception as e:
            log.warning(f"fetch_prev_closes FAIL: {t} | {e}")
    log.info(f"fetch_prev_closes: {len(closes)}/{len(tickers)} fetched")
    return closes


def fetch_index_trend_signal(ticker="SPY", period_days=200):
    """Trend/regime filter: is `ticker` currently trading above its own
    N-day simple moving average? A well-known, simple way to sidestep the
    worst of a sustained downtrend (distinct from the VIX gate, which reacts
    to short-term fear spikes, and the macro gate, which reacts to
    structural signals like yield-curve inversion)."""
    try:
        hist = yf.Ticker(ticker).history(period=f"{period_days + 30}d")
        if len(hist) < period_days:
            return {"available": False}
        close = hist["Close"]
        sma = close.rolling(period_days).mean().iloc[-1]
        last = float(close.iloc[-1])
        if sma is None or math.isnan(float(sma)):
            return {"available": False}
        sma = float(sma)
        return {
            "available": True, "ticker": ticker,
            "lastClose": round(last, 2), "sma": round(sma, 2),
            "aboveSma": last >= sma,
            "pctFromSma": round((last - sma) / sma * 100, 2),
        }
    except Exception as e:
        log.warning(f"fetch_index_trend_signal FAIL: {ticker} | {e}")
        return {"available": False}


def fetch_daily_gainers(include_cdn=True):
    us_gainers = fetch_us_gainers()
    cdn_gainers = fetch_cdn_gainers() if include_cdn else []
    return {"us": us_gainers, "cdn": cdn_gainers}


def fetch_us_gainers():
    try:
        screener = yf.screen("day_gainers")
        quotes = screener.get("quotes", [])
        gainers = [
            {
                "ticker": q.get("symbol"),
                "percentGain": round(q.get("regularMarketChangePercent", 0), 2),
                "price": q.get("regularMarketPrice"),
                "volume": q.get("regularMarketVolume"),
            }
            for q in quotes[:10]
        ]
        log.info(f"US gainers fetched: {len(gainers)}")
        return gainers
    except Exception as e:
        log.error(f"US gainers FAIL: {e}")
        return []


def fetch_cdn_gainers():
    try:
        r = requests.post(
            'https://app-money.tmx.com/graphql',
            headers={
                'User-Agent': 'Mozilla/5.0',
                'Content-Type': 'application/json',
                'Origin': 'https://money.tmx.com',
                'Referer': 'https://money.tmx.com/'
            },
            json={'query': '{ getMarketMovers(statExchange: "TSX", sortOrder: "DESC", limit: 100) { symbol percentChange price volume } }'},
            timeout=10
        )
        data = r.json()
        movers = data.get('data', {}).get('getMarketMovers', [])
        gainers = [
            {
                "ticker": f"{m['symbol']}.TO",
                "percentGain": round(m['percentChange'], 2),
                "price": m['price'],
                "volume": m['volume'],
            }
            for m in movers
            if m.get('percentChange', 0) > 0
        ]
        gainers.sort(key=lambda x: x['percentGain'], reverse=True)
        top = gainers[:10]
        log.info(f"CDN gainers fetched: {len(top)}")
        return top
    except Exception as e:
        log.error(f"CDN gainers FAIL: {e}")
        return []

def _search_news(ticker, cutoff):
    """Fallback source: Yahoo's search endpoint. Ticker.news uses a separate
    feed that can go empty while search still works (seen 2026-10-01 from
    ~16:30: every ticker returned 0 raw items, search returned news fine).
    Search items carry relatedTickers, so only news tagged with this ticker
    is kept; they have no summary text."""
    out = []
    for i in yf.Search(ticker, news_count=10).news or []:
        if ticker not in (i.get("relatedTickers") or []) or not i.get("providerPublishTime"):
            continue
        pub = datetime.fromtimestamp(i["providerPublishTime"], timezone.utc)
        if pub < cutoff:
            continue
        out.append({
            "title": i.get("title", ""),
            "summary": "",
            "pubDate": pub.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "url": i.get("link", ""),
            "provider": i.get("publisher", ""),
        })
    return out


def fetch_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        if not news:
            try:
                fallback = _search_news(ticker, cutoff)
                log.info(f"News fetched: {ticker} | {len(fallback)} articles (last 7 days) [search fallback — Ticker.news empty]")
                return fallback
            except Exception as e:
                log.warning(f"News search fallback FAIL: {ticker} | {e}")
                return []
        filtered = []
        for item in news:
            try:
                content = item.get("content", {})
                pub_date_str = content.get("pubDate") or content.get("displayTime")
                if not pub_date_str:
                    continue
                pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                if pub_date < cutoff:
                    continue
                filtered.append({
                    "title": content.get("title", ""),
                    "summary": content.get("summary", ""),
                    "pubDate": pub_date_str,
                    "url": content.get("canonicalUrl", {}).get("url", ""),
                    "provider": content.get("provider", {}).get("displayName", ""),
                })
            except Exception as e:
                log.warning(f"News item parse FAIL: {ticker} | {e}")
                continue
        log.info(f"News fetched: {ticker} | {len(filtered)} articles (last 7 days)")
        return filtered
    except Exception as e:
        log.error(f"News FAIL: {ticker} | {e}")
        return []
    
def safe_get(df, field):
    try:
        if df is not None and field in df.index and not df.empty:
            val = df.loc[field].iloc[0]
            return None if val != val else int(val)
    except Exception:
        pass
    return None


def extract_periods(df, fields, max_periods):
    result = []
    if df is None or df.empty:
        return result
    cols = df.columns[:max_periods]
    for col in cols:
        period = {"date": str(col.date())}
        for field in fields:
            try:
                val = df.loc[field, col] if field in df.index else None
                period[field] = None if val != val else int(val)
            except Exception:
                period[field] = None
        result.append(period)
    return result


INCOME_FIELDS = [
    "Total Revenue",
    "Gross Profit",
    "Operating Income",
    "EBITDA",
    "Net Income",
    "Basic EPS",
    "Interest Expense",
    "Tax Provision",
    "Operating Expense",
    "Research And Development",
]

BALANCE_FIELDS = [
    "Cash And Cash Equivalents",
    "Total Debt",
    "Total Assets",
    "Total Liabilities Net Minority Interest",
    "Stockholders Equity",
    "Working Capital",
    "Current Assets",
    "Current Liabilities",
    "Long Term Debt",
    "Retained Earnings",
]

CASHFLOW_FIELDS = [
    "Operating Cash Flow",
    "Capital Expenditure",
    "Free Cash Flow",
    "Investing Cash Flow",
    "Financing Cash Flow",
    "Dividends Paid",
    "Repurchase Of Capital Stock",
    "Issuance Of Debt",
    "Repayment Of Debt",
    "Depreciation And Amortization",
]


from datetime import datetime, timezone

def epoch_to_date(ts):
    try:
        return str(datetime.fromtimestamp(ts, tz=timezone.utc).date())
    except Exception:
        return None

def fetch_yfinance(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        calendar = stock.calendar
        dividends = stock.dividends

        annual_income = stock.financials
        quarterly_income = stock.quarterly_financials
        annual_balance = stock.balance_sheet
        quarterly_balance = stock.quarterly_balance_sheet
        annual_cashflow = stock.cashflow
        quarterly_cashflow = stock.quarterly_cashflow

        calendar_data = {}
        if isinstance(calendar, dict):
            earnings_dates = calendar.get("Earnings Date", [])
            calendar_data = {
                "earningsDateStart": str(earnings_dates[0]) if len(earnings_dates) > 0 else None,
                "earningsDateEnd": str(earnings_dates[1]) if len(earnings_dates) > 1 else None,
                "earningsEPSHigh": calendar.get("Earnings High"),
                "earningsEPSLow": calendar.get("Earnings Low"),
                "earningsEPSAvg": calendar.get("Earnings Average"),
                "revenueEstimateHigh": calendar.get("Revenue High"),
                "revenueEstimateLow": calendar.get("Revenue Low"),
                "revenueEstimateAvg": calendar.get("Revenue Average"),
                "exDividendDate": str(calendar.get("Ex-Dividend Date")) if calendar.get("Ex-Dividend Date") else None,
                "dividendPayDate": str(calendar.get("Dividend Date")) if calendar.get("Dividend Date") else None,
            }
            log.info(f"Calendar OK: {ticker} | earnings:{calendar_data.get('earningsDateStart')} exDiv:{calendar_data.get('exDividendDate')}")
        else:
            log.warning(f"Calendar unexpected format: {ticker} | {type(calendar)}")

        events_data = {
            "dividendRate": info.get("dividendRate"),
            "dividendYield": info.get("dividendYield"),
            "trailingAnnualDividendRate": info.get("trailingAnnualDividendRate"),
            "fiveYearAvgDividendYield": info.get("fiveYearAvgDividendYield"),
            "lastDividendValue": info.get("lastDividendValue"),
            "lastDividendDate": epoch_to_date(info.get("lastDividendDate")) if info.get("lastDividendDate") else None,
            "exDividendDate": epoch_to_date(info.get("exDividendDate")) if info.get("exDividendDate") else None,
            "dividendDate": epoch_to_date(info.get("dividendDate")) if info.get("dividendDate") else None,
            "earningsTimestamp": epoch_to_date(info.get("earningsTimestamp")) if info.get("earningsTimestamp") else None,
            "earningsCallStart": epoch_to_date(info.get("earningsCallTimestampStart")) if info.get("earningsCallTimestampStart") else None,
            "earningsCallEnd": epoch_to_date(info.get("earningsCallTimestampEnd")) if info.get("earningsCallTimestampEnd") else None,
            "isEarningsDateEstimate": info.get("isEarningsDateEstimate"),
            "earningsGrowthYoY": info.get("earningsGrowth"),
            "earningsQuarterlyGrowth": info.get("earningsQuarterlyGrowth"),
            "lastSplitFactor": info.get("lastSplitFactor"),
            "lastSplitDate": epoch_to_date(info.get("lastSplitDate")) if info.get("lastSplitDate") else None,
        }
        log.info(f"Events OK: {ticker} | div_date:{events_data.get('dividendDate')} earnings_call:{events_data.get('earningsCallStart')}")

        recent_dividends = []
        if dividends is not None and not dividends.empty:
            recent_dividends = [
                {"date": str(d), "amount": float(v)}
                for d, v in dividends[-3:].items()
            ]

        log.info(f"yfinance OK: {ticker} | rev_annual:{safe_get(annual_income, 'Total Revenue')} rev_q:{safe_get(quarterly_income, 'Total Revenue')}")

        previous_close = None
        try:
            hist2 = stock.history(period="5d")
            if len(hist2) >= 2:
                previous_close = round(float(hist2['Close'].iloc[-2]), 2)
                log.info(f"previousClose OK: {ticker} | {previous_close}")
        except Exception as e:
            log.warning(f"previousClose FAIL: {ticker} | {e}")

        return {
            "price": info.get("currentPrice"),
            "previousClose": previous_close,
            "volume": info.get("volume"),
            "marketCap": info.get("marketCap"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "calendar": calendar_data,
            "events": events_data,
            "dividends": recent_dividends,
            "annualIncome": extract_periods(annual_income, INCOME_FIELDS, 3),
            "quarterlyIncome": extract_periods(quarterly_income, INCOME_FIELDS, 8),
            "annualBalance": extract_periods(annual_balance, BALANCE_FIELDS, 3),
            "quarterlyBalance": extract_periods(quarterly_balance, BALANCE_FIELDS, 8),
            "annualCashflow": extract_periods(annual_cashflow, CASHFLOW_FIELDS, 3),
            "quarterlyCashflow": extract_periods(quarterly_cashflow, CASHFLOW_FIELDS, 8),
        }
    except Exception as e:
        log.error(f"yfinance FAIL: {ticker} | {e}")
        return None


def clean_float(v):
    if v is None:
        return None
    try:
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 2)
    except Exception:
        return None

def fetch_price_only(ticker):
    """Price + technicals fetch: price, previousClose, volume, marketCap, rsi14, bbPercent."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="35d")

        previous_close = None
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")

        if len(hist) >= 2:
            previous_close = round(float(hist['Close'].iloc[-2]), 2)
        if current_price is None and not hist.empty:
            current_price = round(float(hist['Close'].iloc[-1]), 2)

        rsi = None
        bb_percent = None
        try:
            delta = hist['Close'].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            _rsi_val = rsi_series.iloc[-1]
            rsi = round(float(_rsi_val), 1) if not rsi_series.empty and not math.isnan(float(_rsi_val)) else None
        except Exception as e:
            log.warning(f"RSI calc FAIL in fetch_price_only: {ticker} | {e}")

        try:
            close = hist['Close']
            bb_mid = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            last_close = close.iloc[-1]
            band_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
            if band_width and band_width != 0:
                bb_percent = round(float((last_close - bb_lower.iloc[-1]) / band_width), 3)
        except Exception as e:
            log.warning(f"BB%B calc FAIL in fetch_price_only: {ticker} | {e}")

        result = {
            "price": current_price,
            "previousClose": previous_close,
            "volume": info.get("volume"),
            "marketCap": info.get("marketCap"),
            "rsi14": rsi,
            "bbPercent": bb_percent,
        }
        log.info(f"fetch_price_only OK: {ticker} | price:{current_price} prevClose:{previous_close} rsi:{rsi} bb:{bb_percent}")
        return result
    except Exception as e:
        log.error(f"fetch_price_only FAIL: {ticker} | {e}")
        return None


# ── MACRO DATA ────────────────────────────────────────────────────────────────

# FOMC decision dates (rate announcement day only, the second day of each
# 2-day meeting). The Fed publishes roughly a year out and each date is
# "tentative until confirmed at the meeting immediately preceding it," so
# this needs a manual refresh periodically (at least once a year).
FOMC_MEETING_DATES = [
    "2026-09-16", "2026-10-28", "2026-12-09",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09",
    "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
]

def _next_fomc_date():
    today = datetime.now().strftime("%Y-%m-%d")
    future = sorted(d for d in FOMC_MEETING_DATES if d > today)
    return future[0] if future else None

FRED_SERIES = {
    "fed_rate":       {"id": "FEDFUNDS",   "label": "Fed Funds Rate",      "unit": "%", "fomcCalendar": True},
    "cpi":            {"id": "CPIAUCSL",   "label": "CPI (YoY)",           "unit": "%",  "yoy": True, "releaseId": 10},
    "unemployment":   {"id": "UNRATE",     "label": "Unemployment Rate",   "unit": "%",  "releaseId": 50},
    "gdp_growth":     {"id": "A191RL1Q225SBEA", "label": "GDP Growth",     "unit": "%",  "releaseId": 53},
    "ten_yr_yield":   {"id": "DGS10",      "label": "10-Yr Treasury Yield","unit": "%"},
    "two_yr_yield":   {"id": "DGS2",       "label": "2-Yr Treasury Yield", "unit": "%"},
    "vix":            {"id": "VIXCLS",     "label": "VIX (Fear Index)",    "unit": ""},
    "dollar_index":   {"id": "DTWEXBGS",   "label": "USD Dollar Index",    "unit": ""},
    "oil_wti":        {"id": "DCOILWTICO", "label": "WTI Crude Oil",       "unit": "$"},
    "sp500":          {"id": "SP500",      "label": "S&P 500",             "unit": ""},
    "nasdaq100":      {"id": "NASDAQ100",  "label": "Nasdaq 100",          "unit": ""},
    # The monthly jobs report — markets react to the month-over-month
    # change in payroll count, not the raw employment level, so this reads
    # very differently from every other series above (see the "diff" branch
    # in fetch_macro_data below).
    "nonfarm_payrolls": {"id": "PAYEMS",    "label": "Nonfarm Payrolls (m/m)", "unit": "k", "diff": True, "releaseId": 50},
    "jobless_claims":   {"id": "ICSA",      "label": "Initial Jobless Claims", "unit": "k", "scale": 0.001, "releaseId": 180},
    "core_pce":         {"id": "PCEPILFE",  "label": "Core PCE (YoY)",         "unit": "%", "yoy": True, "releaseId": 54},
    "consumer_sentiment": {"id": "UMCSENT", "label": "Consumer Sentiment (U Mich)", "unit": "", "releaseId": 91},
}

def fetch_fred_series(series_id, api_key, limit=2):
    try:
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={api_key}&file_type=json"
            f"&sort_order=desc&limit={limit}"
        )
        r = requests.get(url, timeout=10)
        obs = r.json().get("observations", [])
        valid = [o for o in obs if o.get("value") not in (".", None, "")]
        return valid
    except Exception as e:
        log.warning(f"FRED FAIL: {series_id} | {e}")
        return []

def fetch_next_release_date(release_id, api_key):
    """Next-scheduled date for a FRED "release" (the actual calendar event,
    e.g. BLS's monthly Employment Situation report — shared by both
    unemployment and nonfarm_payrolls, which come from the same release).
    FRED tracks releases separately from series because market-moving
    indicators publish on a schedule (Nonfarm Payrolls: ~first Friday of
    the month) that has nothing to do with which month's data it covers —
    that's the "date" field on the observation itself, handled elsewhere.
    Desc-sorted with a generous limit so the window reliably straddles
    today regardless of how far out FRED has already scheduled releases."""
    try:
        url = (
            f"https://api.stlouisfed.org/fred/release/dates?release_id={release_id}"
            f"&api_key={api_key}&file_type=json&sort_order=desc"
            f"&include_release_dates_with_no_data=true&limit=20"
        )
        r = requests.get(url, timeout=10)
        dates = sorted(x["date"] for x in r.json().get("release_dates", []))
        today = datetime.now().strftime("%Y-%m-%d")
        future = [d for d in dates if d > today]
        return future[0] if future else None
    except Exception as e:
        log.warning(f"FRED release-dates FAIL: {release_id} | {e}")
        return None

def fetch_macro_data(fred_api_key, newsapi_key=None):
    log.info("=== Macro fetch started ===")
    indicators = {}

    for key, cfg in FRED_SERIES.items():
        # "diff" (Nonfarm Payrolls) needs 3 raw levels to show both this
        # month's m/m change and whether that change grew or shrank vs
        # last month's; "yoy" needs 13 to reach back a full year.
        # yoy needs obs[12]/[13] (this month's and last month's year-ago
        # comparison) — request a few extra since FRED occasionally has a
        # blank/pending revision in the window that fetch_fred_series drops.
        limit = 3 if cfg.get("diff") else (16 if cfg.get("yoy") else 2)
        obs = fetch_fred_series(cfg["id"], fred_api_key, limit=limit)
        release_extra = {}
        if cfg.get("releaseId"):
            release_extra = {"nextRelease": fetch_next_release_date(cfg["releaseId"], fred_api_key)}
        elif cfg.get("fomcCalendar"):
            release_extra = {"nextRelease": _next_fomc_date()}
        if not obs:
            indicators[key] = {"label": cfg["label"], "value": None, "date": None, "unit": cfg["unit"], "change": None, **release_extra}
            continue
        try:
            scale = cfg.get("scale", 1)
            latest = float(obs[0]["value"]) * scale
            prev   = float(obs[1]["value"]) * scale if len(obs) > 1 else None
            if cfg.get("diff") and prev is not None:
                # Value shown IS the month-over-month change (what actually
                # moves markets); "change" is that change accelerating (+)
                # or decelerating (-) vs the prior month's change.
                value = round(latest - prev, 2)
                prior_diff = None
                if len(obs) > 2:
                    prior_month = float(obs[2]["value"]) * scale
                    prior_diff = round(prev - prior_month, 2)
                change = round(value - prior_diff, 2) if prior_diff is not None else None
                indicators[key] = {
                    "label": cfg["label"], "value": value, "prev": prior_diff,
                    "change": change, "date": obs[0]["date"], "unit": cfg["unit"],
                    **release_extra,
                }
                log.info(f"FRED OK: {key} | {value} ({obs[0]['date']})")
                continue
            # YoY calculation (CPI: compare to 12 months ago = ~obs[12]). The
            # index level itself (latest) is meaningless on its own — show
            # the YoY% as "value" (matching the "(YoY)" label), and "change"
            # as this month's YoY% vs last month's, i.e. inflation
            # accelerating (+) or cooling (-) — not the raw level delta.
            if cfg.get("yoy") and len(obs) >= 14:
                year_ago = float(obs[12]["value"]) * scale
                year_ago_prev = float(obs[13]["value"]) * scale
                yoy_now = round(((latest - year_ago) / year_ago) * 100, 2)
                yoy_prev = round(((prev - year_ago_prev) / year_ago_prev) * 100, 2)
                indicators[key] = {
                    "label": cfg["label"], "value": yoy_now, "prev": yoy_prev,
                    "change": round(yoy_now - yoy_prev, 2), "date": obs[0]["date"], "unit": cfg["unit"],
                    **release_extra,
                }
                log.info(f"FRED OK: {key} | {yoy_now}% ({obs[0]['date']})")
                continue
            change = round(latest - prev, 2) if prev is not None else None
            indicators[key] = {
                "label":  cfg["label"],
                "value":  round(latest, 2),
                "prev":   round(prev, 2) if prev is not None else None,
                "change": change,
                "date":   obs[0]["date"],
                "unit":   cfg["unit"],
                **release_extra,
            }
            log.info(f"FRED OK: {key} | {latest} ({obs[0]['date']})")
        except Exception as e:
            log.warning(f"FRED parse FAIL: {key} | {e}")
            indicators[key] = {"label": cfg["label"], "value": None, "date": None, "unit": cfg["unit"], "change": None, **release_extra}

    # Yield curve spread
    try:
        t10 = indicators.get("ten_yr_yield", {}).get("value")
        t2  = indicators.get("two_yr_yield",  {}).get("value")
        if t10 is not None and t2 is not None:
            indicators["yield_curve"] = {
                "label": "Yield Curve (10Y-2Y)",
                "value": round(t10 - t2, 2),
                "unit":  "%",
                "date":  indicators["ten_yr_yield"]["date"],
                "change": None,
            }
    except Exception:
        pass

    # Headlines — NewsAPI if key provided, else fallback to yfinance market news
    headlines = []
    if newsapi_key and newsapi_key not in ("your_newsapi_key_here", "", None):
        try:
            r = requests.get(
                "https://newsapi.org/v2/top-headlines",
                params={"category": "business", "language": "en", "pageSize": 10, "apiKey": newsapi_key},
                timeout=10
            )
            articles = r.json().get("articles", [])
            headlines = [
                {"title": a.get("title",""), "source": a.get("source",{}).get("name",""), "url": a.get("url",""), "publishedAt": a.get("publishedAt","")}
                for a in articles if a.get("title")
            ]
            log.info(f"NewsAPI OK: {len(headlines)} headlines")
        except Exception as e:
            log.warning(f"NewsAPI FAIL: {e}")

    # Fallback: pull news from a few major market tickers via yfinance
    if not headlines:
        log.info("Headlines: falling back to yfinance news (SPY, QQQ, GLD)")
        try:
            from datetime import datetime, timezone, timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=3)
            seen = set()
            for sym in ["SPY", "QQQ", "GLD", "TLT"]:
                try:
                    stock = yf.Ticker(sym)
                    for item in (stock.news or []):
                        content_block = item.get("content", {})
                        title = content_block.get("title", "")
                        if not title or title in seen:
                            continue
                        pub = content_block.get("pubDate") or content_block.get("displayTime", "")
                        try:
                            pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                            if pub_dt < cutoff:
                                continue
                        except Exception:
                            pass
                        seen.add(title)
                        headlines.append({
                            "title": title,
                            "source": content_block.get("provider", {}).get("displayName", sym),
                            "url": content_block.get("canonicalUrl", {}).get("url", ""),
                            "publishedAt": pub,
                        })
                        if len(headlines) >= 12:
                            break
                except Exception as e:
                    log.warning(f"yfinance news FAIL: {sym} | {e}")
                if len(headlines) >= 12:
                    break
            log.info(f"yfinance fallback headlines: {len(headlines)}")
        except Exception as e:
            log.warning(f"Headlines fallback FAIL: {e}")

    log.info("=== Macro fetch complete ===")
    return {"indicators": indicators, "headlines": headlines}


# ── DREAM STOCK SCANNER ───────────────────────────────────────────────────────

ARK_ETFS = ["ARKK", "ARKG", "ARKW", "ARKQ", "ARKF"]

def fetch_ark_holdings():
    """Pull current ARK ETF holdings via yfinance."""
    tickers = set()
    for etf in ARK_ETFS:
        try:
            fund = yf.Ticker(etf)
            holdings = fund.funds_data.top_holdings if hasattr(fund, 'funds_data') and fund.funds_data else None
            if holdings is not None and not holdings.empty:
                for sym in holdings.index[:15]:
                    tickers.add(str(sym).upper())
                log.info(f"ARK {etf}: {len(holdings)} holdings fetched")
            else:
                # Fallback: get from info
                info = fund.info
                log.info(f"ARK {etf}: fallback info only | {info.get('symbol','?')}")
        except Exception as e:
            log.warning(f"ARK {etf} FAIL: {e}")
    log.info(f"ARK holdings total unique tickers: {len(tickers)}")
    return list(tickers)


def fetch_yf_growth_screener():
    """Screen for high-growth stocks via all yfinance screeners."""
    tickers = []
    queries = [
        "day_gainers",
        "most_actives",
        "growth_technology_stocks",
        "undervalued_growth_stocks",
        "aggressive_small_caps",
        "small_cap_gainers",
        "undervalued_large_caps",
    ]
    for q in queries:
        try:
            result = yf.screen(q)
            quotes = result.get("quotes", [])
            added = 0
            for quote in quotes[:25]:
                sym = quote.get("symbol", "")
                if sym:
                    tickers.append(sym)
                    added += 1
            log.info(f"Screener {q}: {added} tickers")
        except Exception as e:
            log.warning(f"Screener {q} FAIL: {e}")
    return list(set(tickers))


def fetch_sp500_tickers():
    """Hardcoded S&P 500 components — updated May 2025. No API needed."""
    sp500 = [
        "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM",
        "ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE",
        "AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA",
        "APO","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK",
        "ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC","BAX","BDX","BRK.B","BBY",
        "TECH","BIO","BIIB","BLK","BX","BA","BKNG","BWA","BSX","BMY","AVGO","BR","BRO",
        "BF.B","BLDR","CHRW","CDNS","CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR",
        "CTLT","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNP","CF","CHTR","CVX","CMG",
        "CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH",
        "CL","CMCSA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY","CTVA",
        "CSGP","COST","CTRA","CRWD","CCI","CSX","CMI","CVS","DHR","DRI","DVA","DAY",
        "DE","DAL","XRAY","DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV",
        "DOW","DHI","DTE","DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV",
        "LLY","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS","EL",
        "ETSY","EG","EVRST","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO",
        "FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FMC","F","FTNT","FTV","FOXA",
        "FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM",
        "GPC","GILD","GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE","HLT",
        "HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX",
        "IDXX","ITW","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG",
        "IVZ","INVH","IQV","IRM","JBAL","JKHY","J","JBL","JNPR","JPM","JNPR","K","KVUE",
        "KDP","KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW",
        "LVS","LDOS","LEN","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO",
        "MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK",
        "META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ",
        "MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM","NWSA",
        "NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR",
        "NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PLTR",
        "PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL",
        "PPG","PPL","PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","QRVO","PWR",
        "QCOM","DGX","RL","RJF","RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK",
        "ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW",
        "SPG","SWKS","SJM","SW","SNA","SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE",
        "SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL",
        "TDY","TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB",
        "TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS",
        "VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX","VLTO","VFC","VTRS","VICI","V",
        "VST","VMC","WRB","GWW","WAB","WBA","WMT","DIS","WBD","WM","WAT","WEC","WFC",
        "WELL","WST","WDC","WHR","WMB","WTW","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS"
    ]
    log.info(f"S&P 500: {len(sp500)} tickers (hardcoded)")
    return sp500


def fetch_tsx60_tickers():
    """Fetch TSX 60 components — hardcoded as they rarely change."""
    tsx60 = [
        "AEM.TO","AGI.TO","ATD.TO","BAM.TO","BCE.TO","BHC.TO","BMO.TO","BNS.TO",
        "CAE.TO","CCL-B.TO","CCO.TO","CHP-UN.TO","CM.TO","CNQ.TO","CNR.TO","CP.TO",
        "CSU.TO","CVE.TO","DOL.TO","DSG.TO","EMA.TO","ENB.TO","EQB.TO","FM.TO",
        "FNV.TO","FTS.TO","GFL.TO","GIB-A.TO","GWO.TO","H.TO","IFC.TO","IMO.TO",
        "K.TO","KXS.TO","L.TO","LSPD.TO","LUN.TO","MFC.TO","MG.TO","MRU.TO",
        "NA.TO","NTR.TO","NVEI.TO","OVV.TO","POU.TO","POW.TO","PPL.TO","QBR-B.TO",
        "RCI-B.TO","RY.TO","SAP.TO","SHOP.TO","SLF.TO","SNC.TO","SU.TO","T.TO",
        "TD.TO","TECK-B.TO","TRP.TO","WCN.TO"
    ]
    log.info(f"TSX 60: {len(tsx60)} tickers")
    return tsx60


def fetch_tsx_composite_tickers():
    """S&P/TSX Composite Index components — hardcoded, ~220 names, refreshed
    manually like TSX 60 (composite membership changes a few times a year on
    quarterly rebalances). Sourced 2026-07-13. Class/unit suffixes use a
    hyphen for Yahoo Finance (e.g. RCI.B -> RCI-B.TO, matching TSX 60 above)."""
    tsx_composite_base = [
        "VNP","ABRA","AAV","ARE","AEM","AC","AGI","AQN","ATD","AAUC","AP.UN","ALA",
        "AIF","USA","ARX","ARIS","ATZ","ACO.X","ATH","ATRL","ATS","ASM","AYA","BTO",
        "BDGI","BMO","BNS","ABX","BHC","BTE","BCE","BIR","BDT","BB","BEI.UN","BBD.B",
        "BLX","BYD","BAM","BBUC","BN","BIP.UN","BEP.UN","DOO","CAE","CCO","CAR.UN",
        "CM","CNR","CNQ","CP","CTC.A","CU","CPX","CS","CJT","CCL.B","CLS","CVE","CG",
        "CEU","GIB.A","CSH.UN","CHP.UN","CCA","CIGI","CSU","CRR.UN","CRT.UN","CURA",
        "DFY","DML","DSG","DSV","DOL","DIR.UN","DPM","ELD","EFN","EMA","EMP.A","ENB",
        "EDR","EFR","EQB","EFX","EQX","ERO","EIF","FFH","FTT","FCR.UN","AG","FM",
        "FSV","FTS","FVI","FNV","FRU","GMIN","WN","GFL","GEI","GIL","GSY","GRT.UN",
        "GWO","HR.UN","HWX","HBM","H","IAU","IAG","IMG","IGM","IMO","IFC","IPCO",
        "IIP.UN","IVN","JWEL","KNT","KEL","KEY","KMP.UN","KXS","K","LIF","LB","LSPD",
        "LNR","LAC","L","LUG","LUN","MG","MFC","MFI","MDA","MX","MRU","MAU","MTL",
        "NA","NXE","NFI","NGEX","NWC","NPI","NG","NTR","OGC","ONEX","OTEX","OLA","OR",
        "PAAS","POU","PXT","PPL","PPTA","PET","PEY","POW","PSK","PBH","PMZ.UN",
        "QBR.B","RBA","QSR","RCH","REI.UN","RCI.B","RY","RUS","SAP","SEA","SES",
        "SHOP","SIA","SVM","SKE","SRU.UN","SOBO","SII","SSRM","STN","SJ","SCR","SLF",
        "SU","SPB","TVE","TKO","TRP","TECK.B","T","TVK","TFII","TRI","X","TPZ","TXG",
        "TIH","TD","TOU","TA","TCL.A","TFPM","TSU","VET","VITL.UN","VZLA","WCN","WDO",
        "WFG","WPM","WCP","WPK","WSP",
    ]
    tsx_composite = [t.replace(".", "-") + ".TO" for t in tsx_composite_base]
    log.info(f"TSX Composite: {len(tsx_composite)} tickers")
    return tsx_composite


def fetch_nasdaq100_tickers():
    """Hardcoded NASDAQ-100 components — updated June 2025. No API needed."""
    nasdaq100 = [
        "ADBE","ADP","ABNB","ALGN","GOOGL","GOOG","AMZN","AMD","AEP","AMGN",
        "ADI","ANSS","AAPL","AMAT","ASML","AZN","TEAM","ADSK","BKR","BIIB",
        "BKNG","AVGO","CDNS","CDW","CHTR","CTAS","CSCO","CTSH","CMCSA","CEG",
        "CPRT","CSGP","COST","CRWD","CSX","DDOG","DXCM","FANG","DLTR","DASH",
        "EA","EXC","FAST","FTNT","GEHC","GILD","GFS","HON","ILMN","INTC",
        "INTU","ISRG","KDP","KLAC","KHC","LRCX","LULU","MAR","MRVL","MELI",
        "META","MCHP","MU","MSFT","MRNA","MDLZ","MDB","MNST","NFLX","NVDA",
        "NXPI","ORLY","ON","PCAR","PANW","PAYX","PYPL","PDD","QCOM","REGN",
        "ROST","CRM","SBUX","SNPS","TTWO","TMUS","TSLA","TXN","TTD","VRSK",
        "VRTX","WBA","WBD","WDAY","XEL","ZS","ZM"
    ]
    log.info(f"NASDAQ-100: {len(nasdaq100)} tickers (hardcoded)")
    return nasdaq100


def _hours_since_iso(ts_str):
    if not ts_str:
        return 9999
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except Exception:
        return 9999


def _latest_period_date(periods):
    if periods and isinstance(periods, list) and len(periods) > 0:
        return periods[0].get("date")
    return None


def _fetch_static_enriched(stock_obj, info, ticker):
    result = {}
    institutions = []
    try:
        holders = stock_obj.institutional_holders
        if holders is not None and not holders.empty:
            for _, row in holders.head(3).iterrows():
                name = str(row.get("Holder", row.get("Name", "Unknown")))
                pct = row.get("% Out", row.get("pctHeld", None))
                pct_str = f"{round(float(pct)*100,1)}%" if pct is not None else ""
                institutions.append({"name": name, "pct": pct_str})
    except Exception as e:
        log.warning(f"[Dream Static] institutions FAIL: {ticker} | {e}")
    result["institutions"] = institutions

    insider_net = None
    insider_buys = insider_sells = 0
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        it = stock_obj.insider_transactions
        if it is not None and not it.empty:
            for _, row in it.iterrows():
                try:
                    tx_date = row.get("startDate") or row.get("Start Date")
                    if tx_date is None:
                        continue
                    if hasattr(tx_date, "tzinfo") and tx_date.tzinfo is None:
                        tx_date = tx_date.replace(tzinfo=timezone.utc)
                    if tx_date < cutoff:
                        continue
                    shares = abs(int(row.get("shares", 0) or row.get("Shares", 0) or 0))
                    tx_text = str(row.get("text", "") or row.get("Transaction", "")).lower()
                    if any(w in tx_text for w in ("purchase", "buy", "acquisition")):
                        insider_buys += shares
                    elif any(w in tx_text for w in ("sale", "sell", "disposition")):
                        insider_sells += shares
                except Exception:
                    continue
            if insider_buys > 0 or insider_sells > 0:
                if insider_buys > insider_sells * 1.5:
                    insider_net = "Net Buyer"
                elif insider_sells > insider_buys * 1.5:
                    insider_net = "Net Seller"
                else:
                    insider_net = "Neutral"
    except Exception as e:
        log.warning(f"[Dream Static] insider FAIL: {ticker} | {e}")

    result["insiderNet"]       = insider_net
    result["insiderBuys90d"]   = insider_buys
    result["insiderSells90d"]  = insider_sells
    result["shortFloat"]       = round(info.get("shortPercentOfFloat", 0) * 100, 1) if info.get("shortPercentOfFloat") else None
    result["institutionalPct"] = round(info.get("heldPercentInstitutions", 0) * 100, 1) if info.get("heldPercentInstitutions") else None
    result["insiderPct"]       = round(info.get("heldPercentInsiders", 0) * 100, 1) if info.get("heldPercentInsiders") else None
    result["analystTarget"]    = info.get("targetMeanPrice")
    result["recommendation"]   = info.get("recommendationKey")
    result["description"]      = (info.get("longBusinessSummary") or "")[:400]
    result["sector"]           = info.get("sector", "")
    result["industry"]         = info.get("industry", "")
    log.info(f"[Dream Static] {ticker} | insider:{insider_net} buys:{insider_buys} sells:{insider_sells} short:{result['shortFloat']}% inst:{result['institutionalPct']}%")
    return result


def score_dream_stock(ticker, info, hist, financials):
    """Score a stock 0-100 on dream criteria."""
    score = 0
    breakdown = {}
    flags_good = []
    flags_warn = []

    try:
        # 1. Revenue Growth (20pts)
        rev_growth = info.get("revenueGrowth")
        if rev_growth is not None:
            rev_pct = round(rev_growth * 100, 1)
            if rev_pct >= 40:
                score += 20
                flags_good.append(f"Revenue +{rev_pct}% YoY")
            elif rev_pct >= 20:
                score += 10
                flags_good.append(f"Revenue +{rev_pct}% YoY")
            else:
                flags_warn.append(f"Revenue growth weak: +{rev_pct}%")
            breakdown["revenueGrowth"] = rev_pct
        else:
            flags_warn.append("Revenue growth N/A")
            breakdown["revenueGrowth"] = None

        # 2. Gross Margin (15pts)
        gross_margin = info.get("grossMargins")
        if gross_margin is not None:
            gm_pct = round(gross_margin * 100, 1)
            if gm_pct >= 60:
                score += 15
                flags_good.append(f"Gross margin {gm_pct}%")
            elif gm_pct >= 40:
                score += 8
                flags_good.append(f"Gross margin {gm_pct}%")
            else:
                flags_warn.append(f"Low margin: {gm_pct}%")
            breakdown["grossMargin"] = gm_pct
        else:
            breakdown["grossMargin"] = None

        # 3. Momentum — RSI position (15pts)
        rsi = None
        bb_pct = None
        if hist is not None and not hist.empty and len(hist) >= 15:
            delta = hist['Close'].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            _rsi_val = rsi_series.iloc[-1]
            rsi = round(float(_rsi_val), 1) if not rsi_series.empty and not math.isnan(float(_rsi_val)) else None
        if rsi is not None:
            if 45 <= rsi <= 65:
                score += 15
                flags_good.append(f"RSI {rsi} — momentum, not overextended")
            elif 35 <= rsi < 45:
                score += 10
                flags_good.append(f"RSI {rsi} — building momentum")
            elif 65 < rsi <= 70:
                # Own tier — still strong momentum, not yet overbought. Was
                # previously falling through to "weak momentum" (dead zone).
                score += 12
                flags_good.append(f"RSI {rsi} — strong momentum, approaching overbought")
            elif rsi > 70:
                score += 5
                flags_warn.append(f"RSI {rsi} — overbought, risky entry")
            else:
                flags_warn.append(f"RSI {rsi} — weak momentum")
            breakdown["rsi"] = rsi

            # Volume ratio — today's volume vs trailing 20-day average
            # (surge/breakout signal, same formula as fetch_trade_detail's).
            # Not scored yet — a raw signal for the Identify bucket rules
            # and the tuning-proposal pipeline to evaluate against outcomes.
            volume_ratio = None
            try:
                if 'Volume' in hist.columns and len(hist) >= 2:
                    vol_today = float(hist['Volume'].iloc[-1])
                    vol_window = hist['Volume'].iloc[-21:-1] if len(hist) >= 21 else hist['Volume'].iloc[:-1]
                    vol_avg = float(vol_window.mean()) if len(vol_window) > 0 else None
                    if vol_avg and vol_avg > 0:
                        volume_ratio = round(vol_today / vol_avg, 2)
            except Exception:
                pass
            breakdown["volumeRatio"] = volume_ratio

            # Range expansion — today's High-Low vs trailing 14-day average
            # true range. >1 means today's range is wider than normal, an
            # early breakout/volatility-expansion tell.
            range_expansion = None
            try:
                if {'High', 'Low'}.issubset(hist.columns) and len(hist) >= 15:
                    today_range = float(hist['High'].iloc[-1] - hist['Low'].iloc[-1])
                    range_window = (hist['High'] - hist['Low']).iloc[-15:-1]
                    range_avg = float(range_window.mean()) if len(range_window) > 0 else None
                    if range_avg and range_avg > 0:
                        range_expansion = round(today_range / range_avg, 2)
            except Exception:
                pass
            breakdown["rangeExpansion"] = range_expansion
        else:
            breakdown["rsi"] = None

        # 4. Price trend — above MA50 + near 52w high (15pts)
        trend_score = 0
        week52_high = info.get("fiftyTwoWeekHigh")
        week52_low = info.get("fiftyTwoWeekLow")
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        ma50 = None
        if hist is not None and len(hist) >= 35:
            _ma50_val = hist['Close'].rolling(50).mean().iloc[-1]
            ma50 = round(float(_ma50_val), 2) if not math.isnan(float(_ma50_val)) else None
        if current_price and ma50 and current_price > ma50:
            trend_score += 8
            flags_good.append("Above MA50")
        if current_price and week52_high:
            pct_from_high = (current_price / week52_high) * 100
            if pct_from_high >= 85:
                trend_score += 7
                flags_good.append(f"{round(pct_from_high,0):.0f}% of 52w high — strong trend")
            elif pct_from_high >= 70:
                trend_score += 3
        score += trend_score
        breakdown["ma50"] = ma50
        breakdown["week52High"] = week52_high
        breakdown["priceVs52wHigh"] = round((current_price / week52_high * 100), 1) if current_price and week52_high else None

        # 5. Balance sheet — net cash + FCF (15pts)
        total_cash = info.get("totalCash")
        total_debt = info.get("totalDebt")
        fcf = info.get("freeCashflow")
        bs_score = 0
        if total_cash is not None and total_debt is not None:
            net_cash = total_cash - total_debt
            if net_cash > 0:
                bs_score += 8
                flags_good.append("Net cash positive")
            else:
                flags_warn.append("Net debt position")
            breakdown["netCash"] = net_cash
        if fcf and fcf > 0:
            bs_score += 7
            flags_good.append("FCF positive")
        elif fcf and fcf < 0:
            flags_warn.append("FCF negative — burning cash")
        score += bs_score
        breakdown["fcf"] = fcf

        # 6. Institutional ownership (10pts)
        inst_pct = info.get("institutionPercentHeld")
        if inst_pct is not None:
            inst_val = round(inst_pct * 100, 1)
            if inst_val >= 50:
                score += 10
                flags_good.append(f"Institutional {inst_val}%")
            elif inst_val >= 25:
                score += 5
                flags_good.append(f"Institutional {inst_val}%")
            else:
                flags_warn.append(f"Low institutional: {inst_val}%")
            breakdown["institutionalPct"] = inst_val
        else:
            breakdown["institutionalPct"] = None

        # 7. Market cap — emerging leader (10pts)
        mcap = info.get("marketCap")
        if mcap:
            if 500_000_000 <= mcap <= 10_000_000_000:
                score += 10
                flags_good.append("Emerging leader size ($500M-$10B)")
            elif mcap < 500_000_000:
                score += 5
                flags_warn.append("Micro-cap — higher risk")
            else:
                score += 6
        breakdown["marketCap"] = mcap

        # 8. Insider + short float bonus (10pts) — from enriched if available
        insider_net = (financials or {}).get("insiderNet")
        short_float = (financials or {}).get("shortFloat")
        if insider_net == "Net Buyer":
            score += 5
            flags_good.append("Insider Net Buyer (90d)")
        if short_float is not None and short_float < 5:
            score += 5
            flags_good.append(f"Low short interest {short_float}%")
        elif short_float is not None and short_float > 20:
            flags_warn.append(f"High short interest {short_float}%")
        breakdown["insiderNet"]  = insider_net
        breakdown["shortFloat"]  = short_float

        # aboveMa50 flag for identify buckets
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        ma50 = breakdown.get("ma50")
        breakdown["aboveMa50"] = bool(current_price and ma50 and current_price > ma50)

        score = min(100, score)

        # Data completeness: how many of the 8 scoring components had real
        # data to work with, vs. silently defaulting to "no points". Carried
        # forward (not folded into the score) so downstream stages can
        # require a floor before a half-blank ticker competes for a bucket.
        completeness_checks = [
            rev_growth is not None,
            gross_margin is not None,
            rsi is not None,
            ma50 is not None,
            (total_cash is not None and total_debt is not None),
            inst_pct is not None,
            mcap is not None,
            (insider_net is not None or short_float is not None),
        ]
        breakdown["dataCompleteness"] = round(sum(completeness_checks) / len(completeness_checks), 2)

    except Exception as e:
        log.warning(f"Score calc FAIL: {ticker} | {e}")
        # Partial score is worse than no score — a half-computed sum looks
        # like a real (low) ranking instead of a failure. Surface it as
        # incomplete so it's excluded from bucket selection, not silently
        # ranked against fully-scored tickers.
        score = None
        flags_warn.append("SCORE_INCOMPLETE")

    return score, breakdown, flags_good, flags_warn


DREAM_STATIC_TTL_HOURS = 7 * 24
DREAM_REQUEST_DELAY_SEC = 1.5   # baseline pause between tickers to avoid Yahoo rate limits
DREAM_MAX_RETRIES = 3           # extra attempts specifically for rate-limit errors
DREAM_RETRY_BASE_SEC = 20       # backoff base; attempt N waits DREAM_RETRY_BASE_SEC * N seconds


def _is_rate_limit_error(e):
    msg = str(e)
    return "Too Many Requests" in msg or "Rate limited" in msg or "YFRateLimitError" in type(e).__name__


def _fetch_info_with_retry(stock_obj, ticker):
    """Fetch .info, retrying with backoff if Yahoo rate-limits us. Re-raises non-rate-limit errors immediately."""
    for attempt in range(DREAM_MAX_RETRIES + 1):
        try:
            return stock_obj.info
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < DREAM_MAX_RETRIES:
                wait = DREAM_RETRY_BASE_SEC * (attempt + 1)
                log.warning(f"[Dream] {ticker} rate-limited — retrying in {wait}s (attempt {attempt+1}/{DREAM_MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise


def fetch_dream_candidates(watchlist_tickers, gainer_tickers, existing_candidates=None, progress_callback=None):
    log.info("=== Dream Scan (Incremental Enriched): started ===")

    existing_map = {c["ticker"]: c for c in (existing_candidates or [])}

    # TSX-only build: no ARK holdings or yfinance growth screeners — both
    # return exclusively US-listed tickers with no .TO suffix, which don't
    # belong in this universe. gainer_tickers is expected to be CDN-only
    # (see run_dream_scan, which passes only fetch_daily_gainers()["cdn"]).
    tsx_composite_tickers = fetch_tsx_composite_tickers()

    all_tickers = {}
    for t in watchlist_tickers:                all_tickers[t] = "Watchlist"
    for t in gainer_tickers:
        if t not in all_tickers:               all_tickers[t] = "Daily Gainer"
    for t in tsx_composite_tickers:
        if t not in all_tickers:               all_tickers[t] = "TSX Composite"

    log.info(
        f"[Dream] total:{len(all_tickers)} watchlist:{len(watchlist_tickers)} "
        f"gainers:{len(gainer_tickers)} tsxComposite:{len(tsx_composite_tickers)} existing:{len(existing_map)}"
    )

    candidates = []
    stats = {"quarterly": 0, "annual": 0, "static": 0, "skip": 0, "fail": 0, "recovered": 0}
    now_str = datetime.now(timezone.utc).isoformat()
    total_tickers = len(all_tickers)

    for idx, (ticker, source) in enumerate(all_tickers.items(), start=1):
        log.info(f"--- Dream: {ticker} [{source}] ---")
        if progress_callback:
            try:
                progress_callback(idx, total_tickers, ticker)
            except Exception as cb_err:
                log.warning(f"[Dream] progress_callback FAIL: {cb_err}")
        time.sleep(DREAM_REQUEST_DELAY_SEC)  # pace requests to avoid Yahoo rate limits
        try:
            stock_obj = yf.Ticker(ticker)
            info = _fetch_info_with_retry(stock_obj, ticker)

            if not info.get("regularMarketPrice") and not info.get("currentPrice"):
                stats["skip"] += 1
                prior = existing_map.get(ticker)
                if prior:
                    log.warning(f"[Dream] no price (likely rate-limited) — keeping prior data: {ticker}")
                    candidates.append(prior)
                    stats["recovered"] += 1
                else:
                    log.warning(f"[Dream] skip (no price, no prior data): {ticker}")
                continue

            existing = existing_map.get(ticker, {})
            enriched = dict(existing.get("enriched") or {})

            # ── TECHNICALS: always refresh ────────────────────────────────────
            hist = stock_obj.history(period="60d")
            log.info(f"[Dream Tech] {ticker} | hist rows:{len(hist)}")

            # ── QUARTERLY: only if new period detected ────────────────────────
            stored_q = _latest_period_date(enriched.get("quarterlyIncome", []))
            try:
                probe_q = extract_periods(stock_obj.quarterly_financials, INCOME_FIELDS, 1)
                fresh_q = _latest_period_date(probe_q)
            except Exception:
                fresh_q = None
            if fresh_q and fresh_q != stored_q:
                log.info(f"[Dream] {ticker} new quarter: {stored_q} → {fresh_q}")
                enriched["quarterlyIncome"]   = extract_periods(stock_obj.quarterly_financials,       INCOME_FIELDS,   8)
                enriched["quarterlyBalance"]  = extract_periods(stock_obj.quarterly_balance_sheet,    BALANCE_FIELDS,  8)
                enriched["quarterlyCashflow"] = extract_periods(stock_obj.quarterly_cashflow,         CASHFLOW_FIELDS, 8)
                enriched["annualIncome"]      = extract_periods(stock_obj.financials,                 INCOME_FIELDS,   3)
                enriched["annualBalance"]     = extract_periods(stock_obj.balance_sheet,              BALANCE_FIELDS,  3)
                enriched["annualCashflow"]    = extract_periods(stock_obj.cashflow,                   CASHFLOW_FIELDS, 3)
                enriched["quarterlyUpdatedAt"] = now_str
                stats["quarterly"] += 1
            else:
                log.info(f"[Dream] {ticker} quarter unchanged ({stored_q}) — skip quarterly fetch")

            # ── ANNUAL: only if new period detected ───────────────────────────
            stored_a = _latest_period_date(enriched.get("annualIncome", []))
            try:
                probe_a = extract_periods(stock_obj.financials, INCOME_FIELDS, 1)
                fresh_a = _latest_period_date(probe_a)
            except Exception:
                fresh_a = None
            if fresh_a and fresh_a != stored_a:
                log.info(f"[Dream] {ticker} new annual: {stored_a} → {fresh_a}")
                enriched["annualIncome"]   = extract_periods(stock_obj.financials,    INCOME_FIELDS,   3)
                enriched["annualBalance"]  = extract_periods(stock_obj.balance_sheet, BALANCE_FIELDS,  3)
                enriched["annualCashflow"] = extract_periods(stock_obj.cashflow,      CASHFLOW_FIELDS, 3)
                enriched["annualUpdatedAt"] = now_str
                stats["annual"] += 1
            else:
                log.info(f"[Dream] {ticker} annual unchanged ({stored_a}) — skip annual fetch")

            # ── STATIC: institutions, insider, short float ────────────────────
            static_age = _hours_since_iso(enriched.get("staticUpdatedAt"))
            if static_age > DREAM_STATIC_TTL_HOURS:
                log.info(f"[Dream] {ticker} static refresh (age:{static_age:.1f}h)")
                static = _fetch_static_enriched(stock_obj, info, ticker)
                enriched.update(static)
                enriched["staticUpdatedAt"] = now_str
                stats["static"] += 1
            else:
                log.info(f"[Dream] {ticker} static OK (age:{static_age:.1f}h) — skip")

            enriched["technicalsUpdatedAt"] = now_str

            # ── SCORE ─────────────────────────────────────────────────────────
            score, breakdown, flags_good, flags_warn = score_dream_stock(ticker, info, hist, enriched)

            candidate = {
                **existing,
                "ticker":      ticker,
                "source":      source,
                "score":       score,
                "name":        info.get("shortName", ticker),
                "sector":      enriched.get("sector") or info.get("sector", ""),
                "industry":    enriched.get("industry") or info.get("industry", ""),
                "price":       info.get("currentPrice") or info.get("regularMarketPrice"),
                "marketCap":   info.get("marketCap"),
                "breakdown":   breakdown,
                "flagsGood":   flags_good,
                "flagsWarn":   flags_warn,
                "institutions":enriched.get("institutions", []),
                "description": enriched.get("description") or (info.get("longBusinessSummary") or "")[:400],
                "enriched":    enriched,
                "lastScoredAt":now_str,
            }
            candidates.append(candidate)
            log.info(
                f"[Dream] scored: {ticker} | score:{score} q:{stored_q} a:{stored_a} "
                f"insider:{breakdown.get('insiderNet')} short:{breakdown.get('shortFloat')}% "
                f"static_age:{static_age:.1f}h"
            )

        except Exception as e:
            log.error(f"[Dream] FAIL: {ticker} | {e}")
            stats["fail"] += 1
            prior = existing_map.get(ticker)
            if prior:
                log.warning(f"[Dream] FAIL — keeping prior data: {ticker}")
                candidates.append(prior)
                stats["recovered"] += 1
            continue

    # score can be None (SCORE_INCOMPLETE) — sort those to the bottom instead
    # of crashing on a None-vs-int comparison.
    candidates.sort(key=lambda x: x["score"] if x["score"] is not None else -1, reverse=True)
    log.info(
        f"=== Dream Scan complete: {len(candidates)} scored | "
        f"quarterly:{stats['quarterly']} annual:{stats['annual']} "
        f"static:{stats['static']} skip:{stats['skip']} fail:{stats['fail']} "
        f"recovered:{stats['recovered']} ==="
    )
    return {"candidates": candidates}


# ── TRADE DETAIL FETCH ────────────────────────────────────────────────────────

def price_indicators(hist, ticker=""):
    """Technical indicators from a daily-bar history, as of its LAST bar.
    Shared by fetch_trade_detail (live) and the AI score history rebuild
    (app._rebuild_ai_scores, which passes history truncated to a past
    date) so both compute identical values."""
    ind = dict.fromkeys(["rsi14", "ma50", "ma20", "bbPercent", "atr14", "atr14Pct", "volumeRatio",
                         "change1d", "change2d", "change3d", "change4d", "change5d", "change7d", "change30d"])

    # RSI 14
    try:
        delta = hist['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        _v = rsi_series.iloc[-1]
        ind["rsi14"] = round(float(_v), 1) if not math.isnan(float(_v)) else None
    except Exception as e:
        log.warning(f"price_indicators RSI FAIL: {ticker} | {e}")

    # MA50
    try:
        if len(hist) >= 35:
            _m = hist['Close'].rolling(50).mean().iloc[-1]
            ind["ma50"] = round(float(_m), 2) if not math.isnan(float(_m)) else None
    except Exception as e:
        log.warning(f"price_indicators MA50 FAIL: {ticker} | {e}")

    # MA20
    try:
        if len(hist) >= 20:
            ind["ma20"] = round(float(hist['Close'].rolling(20).mean().iloc[-1]), 2)
    except Exception as e:
        log.warning(f"price_indicators MA20 FAIL: {ticker} | {e}")

    # Bollinger Band %B
    try:
        close = hist['Close']
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        band_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
        if band_width and band_width != 0:
            ind["bbPercent"] = round(float((close.iloc[-1] - bb_lower.iloc[-1]) / band_width), 3)
    except Exception as e:
        log.warning(f"price_indicators BB FAIL: {ticker} | {e}")

    # ATR14 (Average True Range) — feeds the volatility-adjusted stop-loss:
    # a stock's own typical daily range, not a one-size-fits-all flat %.
    try:
        if len(hist) >= 15:
            high, low, close = hist['High'], hist['Low'], hist['Close']
            prev_close = close.shift(1)
            true_range = (high - low).combine((high - prev_close).abs(), max) \
                                      .combine((low - prev_close).abs(), max)
            _atr = true_range.rolling(14).mean().iloc[-1]
            if not math.isnan(float(_atr)):
                ind["atr14"] = round(float(_atr), 4)
                last_price = float(close.iloc[-1])
                if last_price > 0:
                    ind["atr14Pct"] = round(ind["atr14"] / last_price * 100, 2)
    except Exception as e:
        log.warning(f"price_indicators ATR14 FAIL: {ticker} | {e}")

    # Volume ratio — today's volume vs 20-day average (surge/breakout signal)
    try:
        if 'Volume' in hist.columns and len(hist) >= 2:
            vol_today = float(hist['Volume'].iloc[-1])
            avg_window = hist['Volume'].iloc[-21:-1] if len(hist) >= 21 else hist['Volume'].iloc[:-1]
            vol_avg = float(avg_window.mean()) if len(avg_window) > 0 else None
            if vol_avg and vol_avg > 0:
                ind["volumeRatio"] = round(vol_today / vol_avg, 2)
    except Exception as e:
        log.warning(f"price_indicators volumeRatio FAIL: {ticker} | {e}")

    # Price changes
    try:
        c = hist['Close']
        pct = lambda a, b: round((c.iloc[a] - c.iloc[b]) / c.iloc[b] * 100, 2)
        if len(hist) >= 2:
            ind["change1d"] = pct(-1, -2)
        if len(hist) >= 3:
            ind["change2d"] = pct(-2, -3)
        if len(hist) >= 4:
            ind["change3d"] = pct(-3, -4)
        if len(hist) >= 5:
            ind["change4d"] = pct(-4, -5)
        if len(hist) >= 6:
            ind["change5d"] = pct(-5, -6)
        if len(hist) >= 7:
            ind["change7d"] = pct(-1, -7)
        if len(hist) >= 30:
            ind["change30d"] = pct(-1, -30)
    except Exception as e:
        log.warning(f"price_indicators price changes FAIL: {ticker} | {e}")

    return ind


def fetch_trade_detail(ticker):
    """Fetch enriched detail for a trade candidate: price, technicals, key financials."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not current_price:
            log.warning(f"fetch_trade_detail: no price for {ticker}")
            return None

        hist = stock.history(period="60d")
        ind = price_indicators(hist, ticker)
        rsi, ma50, ma20, bb_percent = ind["rsi14"], ind["ma50"], ind["ma20"], ind["bbPercent"]
        atr14, atr14_pct, volume_ratio = ind["atr14"], ind["atr14Pct"], ind["volumeRatio"]
        change1d, change2d, change3d, change4d, change5d, change7d, change30d = (
            ind["change1d"], ind["change2d"], ind["change3d"], ind["change4d"],
            ind["change5d"], ind["change7d"], ind["change30d"])

        # Key financials
        revenue_growth = info.get("revenueGrowth")
        gross_margin = info.get("grossMargins")
        fcf = info.get("freeCashflow")
        pe_ratio = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        eps = info.get("trailingEps")
        debt_to_equity = info.get("debtToEquity")
        return_on_equity = info.get("returnOnEquity")
        week52_high = info.get("fiftyTwoWeekHigh")
        week52_low = info.get("fiftyTwoWeekLow")
        analyst_target = info.get("targetMeanPrice")
        recommendation = info.get("recommendationKey")
        inst_pct = info.get("heldPercentInstitutions") or info.get("institutionPercentHeld")
        insider_pct = info.get("heldPercentInsiders")
        short_float = info.get("shortPercentOfFloat")

        upside = None
        if analyst_target and current_price:
            upside = round((analyst_target - current_price) / current_price * 100, 1)

        price_vs_52w_high = None
        if week52_high and current_price:
            price_vs_52w_high = round(current_price / week52_high * 100, 1)

        # Insider net activity — sum last 90d transactions
        insider_net = None
        insider_buys_90d = 0
        insider_sells_90d = 0
        try:
            from datetime import datetime, timezone, timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=90)
            it = stock.insider_transactions
            if it is not None and not it.empty:
                for _, row in it.iterrows():
                    try:
                        tx_date = row.get("startDate") or row.get("Start Date")
                        if tx_date is None:
                            continue
                        if hasattr(tx_date, "tzinfo") and tx_date.tzinfo is None:
                            tx_date = tx_date.replace(tzinfo=timezone.utc)
                        if tx_date < cutoff:
                            continue
                        shares = abs(int(row.get("shares", 0) or row.get("Shares", 0) or 0))
                        tx_text = str(row.get("text", "") or row.get("Transaction", "")).lower()
                        if "purchase" in tx_text or "buy" in tx_text or "acquisition" in tx_text:
                            insider_buys_90d += shares
                        elif "sale" in tx_text or "sell" in tx_text or "disposition" in tx_text:
                            insider_sells_90d += shares
                    except Exception:
                        continue
                if insider_buys_90d > 0 or insider_sells_90d > 0:
                    if insider_buys_90d > insider_sells_90d * 1.5:
                        insider_net = "Net Buyer"
                    elif insider_sells_90d > insider_buys_90d * 1.5:
                        insider_net = "Net Seller"
                    else:
                        insider_net = "Neutral"
        except Exception as e:
            log.warning(f"fetch_trade_detail insider FAIL: {ticker} | {e}")

        # Next earnings date — buys near earnings are coin flips on the gap
        earnings_date = None
        try:
            cal = stock.calendar
            ed = None
            if isinstance(cal, dict):
                eds = cal.get("Earnings Date") or []
                ed = eds[0] if eds else None
            elif cal is not None and hasattr(cal, "empty") and not cal.empty and "Earnings Date" in getattr(cal, "index", []):
                ed = cal.loc["Earnings Date"].iloc[0]
            if ed is not None:
                earnings_date = str(ed)[:10]
        except Exception as e:
            log.warning(f"fetch_trade_detail earnings date FAIL: {ticker} | {e}")

        result = {
            "ticker": ticker,
            "name": info.get("shortName", ticker),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "price": current_price,
            "marketCap": info.get("marketCap"),
            "volume": info.get("volume"),
            "volumeRatio": volume_ratio,
            "rsi14": rsi,
            "ma20": ma20,
            "ma50": ma50,
            "bbPercent": bb_percent,
            "atr14": atr14,
            "atr14Pct": atr14_pct,
            "change1d": change1d,
            "change2d": change2d,
            "change3d": change3d,
            "change4d": change4d,
            "change5d": change5d,
            "change7d": change7d,
            "change30d": change30d,
            "revenueGrowth": round(revenue_growth * 100, 1) if revenue_growth is not None else None,
            "grossMargin": round(gross_margin * 100, 1) if gross_margin is not None else None,
            "fcf": fcf,
            "peRatio": round(pe_ratio, 1) if pe_ratio and not math.isnan(pe_ratio) else None,
            "forwardPE": round(forward_pe, 1) if forward_pe and not math.isnan(forward_pe) else None,
            "eps": eps,
            "debtToEquity": round(debt_to_equity, 1) if debt_to_equity else None,
            "returnOnEquity": round(return_on_equity * 100, 1) if return_on_equity else None,
            "week52High": week52_high,
            "week52Low": week52_low,
            "priceVs52wHigh": price_vs_52w_high,
            "analystTarget": analyst_target,
            "analystUpside": upside,
            "recommendation": recommendation,
            "institutionalPct": round(inst_pct * 100, 1) if inst_pct else None,
            "insiderPct": round(insider_pct * 100, 1) if insider_pct else None,
            "shortFloat": round(short_float * 100, 1) if short_float else None,
            "insiderNet": insider_net,
            "insiderBuys90d": insider_buys_90d,
            "insiderSells90d": insider_sells_90d,
            "earningsDate": earnings_date,
            "description": (info.get("longBusinessSummary") or "")[:400],
        }

        log.info(
            f"fetch_trade_detail OK: {ticker} | price:{current_price} rsi:{rsi} "
            f"ma50:{ma50} volRatio:{volume_ratio} insider:{insider_net} short:{short_float} inst:{inst_pct}"
        )
        return result

    except Exception as e:
        log.error(f"fetch_trade_detail FAIL: {ticker} | {e}")
        return None


# ── TRADEAI: ON-DEMAND NEWS FETCH ─────────────────────────────────────────────

def fetch_ticker_news(ticker):
    """Fetch latest news for any ticker using same pattern as fetch_news(). Returns list of articles."""
    return fetch_news(ticker)
