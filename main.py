import logging
import os
from config import WATCHLIST, FMP_API_KEY
from fetcher import (
    fetch_income_statement,
    fetch_balance_sheet,
    fetch_peers,
    fetch_earnings_calendar,
    fetch_dividends,
    fetch_daily_gainers,
    fetch_yfinance,
)

os.makedirs("logs", exist_ok=True)
os.makedirs("output", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)

log = logging.getLogger(__name__)


def run():
    log.info("=== Phase 2: Data Fetching Started ===")

    for ticker in WATCHLIST:
        log.info(f"--- Fetching: {ticker} ---")
        income = fetch_income_statement(ticker, FMP_API_KEY)
        balance = fetch_balance_sheet(ticker, FMP_API_KEY)
        peers = fetch_peers(ticker, FMP_API_KEY)
        earnings = fetch_earnings_calendar(ticker, FMP_API_KEY)
        dividends = fetch_dividends(ticker, FMP_API_KEY)
        yf_data = fetch_yfinance(ticker)

        log.info(f"{ticker} income: {bool(income)} | balance: {bool(balance)} | peers: {bool(peers)} | yf: {bool(yf_data)}")

    log.info("--- Fetching Daily Gainers ---")
    gainers = fetch_daily_gainers(FMP_API_KEY)
    log.info(f"Gainers fetched: {bool(gainers)}")

    log.info("=== Phase 2: Done ===")


if __name__ == "__main__":
    run()