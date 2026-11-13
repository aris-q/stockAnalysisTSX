import os
from dotenv import load_dotenv

load_dotenv()

def _setup_ca_bundle():
    """Norton Antivirus MITM-proxies HTTPS and re-signs certs with its own CA,
    which Python/curl don't trust — every yfinance/requests call fails with
    CERTIFICATE_VERIFY_FAILED. Build a bundle of certifi + Norton CA and point
    all HTTP clients at it."""
    norton_ca = r"C:\ProgramData\Norton\Antivirus\wscert.pem"
    if not os.path.exists(norton_ca):
        return
    try:
        import certifi, shutil
        os.makedirs("output", exist_ok=True)
        bundle = os.path.abspath("output/ca_bundle.pem")
        shutil.copyfile(certifi.where(), bundle)
        with open(norton_ca, "rb") as f:
            norton = f.read()
        with open(bundle, "ab") as f:
            f.write(b"\n" + norton)
        os.environ.setdefault("SSL_CERT_FILE", bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
        os.environ.setdefault("CURL_CA_BUNDLE", bundle)

        # Python 3.13 verifies certs strictly by default and rejects Norton's
        # proxy CA ("Basic Constraints not marked critical"), which breaks all
        # requests/urllib3 calls (FRED, NewsAPI, TMX). Relax only that flag —
        # chain verification against the bundle above still applies.
        try:
            import ssl
            from urllib3.util import ssl_ as _urllib3_ssl
            _orig_ctx = _urllib3_ssl.create_urllib3_context
            def _lenient_ctx(*args, **kwargs):
                ctx = _orig_ctx(*args, **kwargs)
                try:
                    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
                except Exception:
                    pass
                return ctx
            _urllib3_ssl.create_urllib3_context = _lenient_ctx
            # urllib3.connection binds the function by name at import time —
            # patch that reference too or the fix never takes effect
            import urllib3.connection as _urllib3_conn
            _urllib3_conn.create_urllib3_context = _lenient_ctx
        except Exception:
            pass
    except Exception:
        pass

_setup_ca_bundle()

FMP_API_KEY = os.getenv("FMP_API_KEY")
WATCHLIST = ["SHOP.TO", "RY.TO", "SU.TO", "ABX.TO"]  # TSX starter watchlist
OUTPUT_PATH = "output/analysis.json"
NEWS_PATH = "output/news.json"
WATCHLIST_PATH = "watchlist.json"
LOG_FILE = "logs/app.log"
FOLLOWUP_PATH = "output/followup.json"
FRED_API_KEY = os.getenv("FRED_API_KEY")
NEWSAPI_KEY  = os.getenv("NEWSAPI_KEY")
MACRO_PATH   = "output/macro.json"
TRADES_PATH  = "output/trades.json"
TRADE_CANDIDATES_PATH = "output/trade_candidates.json"
TRADES_AI_PATH = "output/trades_ai.json"
TRADE_AI_CANDIDATES_PATH = "output/trade_ai_candidates.json"
# Reddit stats (2026-11-13) — US Reddit buzz (ApeWisdom), stats only, no
# trading. Canadian forums need Reddit API approval (Responsible Builder
# Policy, Nov 2025), so this tracks the US forums the stock-analysis
# Reddit account trades on.
REDDIT_BUZZ_PATH = "output/reddit_buzz.json"         # latest pull
REDDIT_HISTORY_PATH = "output/reddit_history.json"   # one snapshot per day (with that day's rising list)
REDDIT_STATS_PATH = "output/reddit_stats.json"       # forward returns of each day's rising names vs SPY
OLLAMA_URL   = "http://localhost:11434"
# OLLAMA_MODEL = "gemma2:9b"
OLLAMA_MODEL = "phi3:mini"
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")