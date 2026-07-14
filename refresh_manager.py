import json
import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

TZ = ZoneInfo("America/Toronto")

PRICE_TTL_HOURS = 24
CALENDAR_TTL_DAYS = 7


def now():
    return datetime.now(TZ)


def parse_fetched_at(fetched_at_str):
    if not fetched_at_str:
        return None
    try:
        return datetime.fromisoformat(fetched_at_str.replace(" EDT", "+00:00").replace(" EST", "+00:00"))
    except Exception:
        return None


def hours_since(fetched_at_str):
    dt = parse_fetched_at(fetched_at_str)
    if not dt:
        return 9999
    try:
        return (now() - dt.astimezone(TZ)).total_seconds() / 3600
    except Exception:
        return 9999


def latest_quarter_date(stock):
    qi = stock.get("quarterlyIncome", [])
    return qi[0].get("date") if qi else None


def latest_annual_date(stock):
    ai = stock.get("annualIncome", [])
    return ai[0].get("date") if ai else None


def needs_price_refresh(stock):
    h = hours_since(stock.get("fetchedAt"))
    if h > PRICE_TTL_HOURS:
        log.info(f"{stock['ticker']} needs price refresh ({h:.1f}h old)")
        return True
    return False


def needs_quarterly_refresh(stock, fresh_stock):
    stored = latest_quarter_date(stock)
    fresh = latest_quarter_date(fresh_stock)
    if fresh and stored != fresh:
        log.info(f"{stock['ticker']} new quarter detected: {stored} → {fresh}")
        return True
    return False


def needs_annual_refresh(stock, fresh_stock):
    stored = latest_annual_date(stock)
    fresh = latest_annual_date(fresh_stock)
    if fresh and stored != fresh:
        log.info(f"{stock['ticker']} new annual detected: {stored} → {fresh}")
        return True
    return False


def needs_calendar_refresh(stock):
    h = hours_since(stock.get("calendarFetchedAt"))
    if h > CALENDAR_TTL_DAYS * 24:
        log.info(f"{stock['ticker']} needs calendar refresh ({h:.1f}h old)")
        return True
    return False


def needs_news_refresh(news_entry, today_only=False):
    if not news_entry:
        return True
    key = "todayFetchedAt" if today_only else "sevenDayFetchedAt"
    ttl = 1 if today_only else 6
    h = hours_since(news_entry.get(key))
    return h > ttl


def check_what_needs_refresh(stock, fetcher):
    ticker = stock.get("ticker")
    result = {
        "ticker": ticker,
        "price": False,
        "quarterly": False,
        "annual": False,
        "calendar": False,
        "ai": False,
    }

    if needs_price_refresh(stock):
        result["price"] = True
        result["ai"] = True

    if needs_calendar_refresh(stock):
        result["calendar"] = True
        result["ai"] = True

    try:
        from fetcher import fetch_yfinance
        fresh = fetch_yfinance(ticker)
        if fresh:
            if needs_quarterly_refresh(stock, fresh):
                result["quarterly"] = True
                result["ai"] = True
            if needs_annual_refresh(stock, fresh):
                result["annual"] = True
                result["ai"] = True
            result["_freshData"] = fresh
    except Exception as e:
        log.error(f"refresh check FAIL: {ticker} | {e}")

    log.info(f"Refresh check {ticker}: {result}")
    return result