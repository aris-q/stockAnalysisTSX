from flask import Flask, render_template, request, redirect, jsonify, make_response
import json, os, logging, threading, time, hashlib, statistics
from config import WATCHLIST, OUTPUT_PATH, NEWS_PATH, WATCHLIST_PATH, TRADES_AI_PATH, TRADE_AI_CANDIDATES_PATH
try:
    from config import FRED_API_KEY, NEWSAPI_KEY, MACRO_PATH
except ImportError:
    FRED_API_KEY = None
    NEWSAPI_KEY  = None
    MACRO_PATH   = "output/macro.json"
DREAM_PATH = "output/dream.json"
TRADE_OUTCOMES_PATH = "output/trade_outcomes.json"
SHADOW_PATH = "output/shadow_candidates.json"
EPOCHS_PATH = "output/account_epochs.json"
RESET_PENDING_PATH = "output/reset_pending.json"

def _load_epochs():
    return load_json(EPOCHS_PATH, {"epochs": []})

def _record_new_epoch(balance, label=None):
    """Called on /tradeai/reset. Trade/scoring history (trade_outcomes.json,
    recommend_versions.json) is deliberately NOT cleared on reset — the user
    wants pre/post-reset performance kept separate for comparison rather than
    lost, so Insights buckets outcomes by which epoch their sell date falls
    in instead of wiping anything."""
    db = _load_epochs()
    epoch_num = len(db["epochs"]) + 1
    entry = {
        "epoch": epoch_num, "startedAt": ts(),
        "label": label or f"Reset on {datetime.now(ET).strftime('%B %-d, %Y')}",
        "startingBalance": balance,
    }
    db["epochs"].append(entry)
    save_json(EPOCHS_PATH, db)
    log.info(f"[Epoch] new epoch {epoch_num} started | balance:${balance:,.2f} | label:{entry['label']}")
    return entry

def _epoch_for_date(date_str, epochs):
    """Which epoch a YYYY-MM-DD... date string falls into (epochs are sorted
    ascending by startedAt; a record before any epoch's start belongs to the
    earliest/implicit epoch 0 — pre-dates this feature)."""
    if not date_str:
        return 0
    matched = 0
    for e in epochs:
        if date_str >= e["startedAt"]:
            matched = e["epoch"]
        else:
            break
    return matched
from fetcher import fetch_yfinance, fetch_daily_gainers, fetch_news, fetch_price_only, fetch_macro_data, fetch_dream_candidates, fetch_trade_detail, fetch_ticker_news, fetch_live_prices, fetch_prev_closes, fetch_index_trend_signal, _next_fomc_date, FOMC_MEETING_DATES
from compute import process_watchlist
from ai_summary import generate_summary, generate_recommendations, generate_followup
from refresh_manager import check_what_needs_refresh, now, hours_since
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

os.makedirs("logs", exist_ok=True)
os.makedirs("output", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8"),
        logging.StreamHandler(stream=open(__import__('sys').stdout.fileno(),
            mode='w', encoding='utf-8', errors='replace', closefd=False))
    ]
)

log = logging.getLogger(__name__)
app = Flask(__name__)


def _sanitize_str(s):
    if not isinstance(s, str):
        return s
    return s.replace('\x00', '').replace('\r', ' ')

def load_watchlist():
    try:
        with open(WATCHLIST_PATH, "r") as f:
            return json.load(f)
    except Exception:
        log.info("No watchlist.json found, using config default")
        return list(WATCHLIST)

def save_watchlist():
    # temp file + rename: a crash mid-write can't leave an empty/corrupt
    # watchlist.json (load_watchlist would then fall back to the 4-ticker
    # config default and the next save would make that permanent)
    tmp = WATCHLIST_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(watchlist, f)
    os.replace(tmp, WATCHLIST_PATH)
    log.info(f"Watchlist saved: {watchlist}")

watchlist = load_watchlist()

def _sync_watchlist_from_holdings(holding_tickers):
    """
    Keep watchlist aligned with TradeAI holdings:
    - Add any holding ticker not already in watchlist, and remember that
      this sync added it (watchlist.json.auto)
    - When a holding is sold, remove it only if this sync added it
    - Tickers you added yourself are never removed, even after the bot
      buys and sells them (before 2026-10-01 they were: every sold holding
      was dropped, manual or not)
    """
    global watchlist
    try:
        current_holdings = set(holding_tickers)
        # No .auto file yet (first run of this logic): nothing is marked
        # auto-added, so nothing already on the list gets removed.
        auto = set(load_json(WATCHLIST_PATH + ".auto", []))

        added, removed = [], []

        for t in current_holdings:
            if t not in watchlist:
                watchlist.append(t)
                added.append(t)
                auto.add(t)

        for t in list(auto):
            if t not in current_holdings:
                if t in watchlist:
                    watchlist.remove(t)
                    removed.append(t)
                auto.discard(t)

        save_watchlist()
        save_json(WATCHLIST_PATH + ".auto", sorted(auto))

        if added or removed:
            log.info(f"[WatchlistSync] added:{added} removed:{removed} | watchlist now:{watchlist}")
        else:
            log.info(f"[WatchlistSync] no changes | holdings:{list(current_holdings)}")

        # New holdings arrive with no watchlist detail data — kick off a full
        # fetch for them so their table isn't empty until manually fetched.
        if added:
            _fetch_new_holdings_async(added)
    except Exception as e:
        log.error(f"[WatchlistSync] FAIL: {e}")


def _fetch_new_holdings_async(tickers):
    """Run a full watchlist fetch for newly held tickers in the background,
    waiting for whatever operation is currently running (e.g. the Recommend
    that triggered this sync) to release fetch_status first."""
    def _worker():
        import time
        for _ in range(60):  # wait up to ~2 minutes
            if not fetch_status.get("running"):
                break
            time.sleep(2)
        if fetch_status.get("running"):
            log.warning(f"[WatchlistSync] auto-fetch skipped, another operation still running: {tickers}")
            return
        log.info(f"[WatchlistSync] auto-fetching watchlist detail for new holdings: {tickers}")
        try:
            run_fetch(list(tickers), "single")
        except Exception as e:
            log.error(f"[WatchlistSync] auto-fetch FAIL: {tickers} | {e}")
    threading.Thread(target=_worker, daemon=True).start()


def _get_holding_tickers():
    """Return current TradeAI holding tickers."""
    try:
        trades = load_json(TRADES_AI_PATH, {})
        return list(trades.get("holdings", {}).keys())
    except Exception:
        return []

fetch_status = {"running": False, "message": "Idle", "operation": None, "current": 0, "total": 0, "current_ticker": ""}
stop_flag = False
op_timestamps = {"prices": None, "smart": None, "new": None, "all": None}
last_fetched_tickers = set()

# Guards every read-modify-write cycle against output/analysis.json. Multiple
# writers (startup price refresh, scheduled sell-checks, manual Fetch, the
# refresh-prices button) can otherwise interleave: each reads the file, works
# for seconds while making network calls, then writes back a full replacement
# built from its now-stale snapshot — silently discarding whatever a
# different writer saved in between (e.g. wiping fetchedAt/sector/financials
# that a concurrent Fetch had just written). This became especially likely
# on 2026-07-13 while the Flask reloader was doubling every background
# thread (see app.run(..., use_reloader=False) below).
_output_lock = threading.Lock()



TZ = ZoneInfo("America/Toronto")

def ts():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M %Z")

def load_json(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None for JSON safety."""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj

def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(_sanitize(data), f, indent=2, allow_nan=False, default=lambda v: None)
    os.replace(tmp, path)

RECOMMEND_VERSIONS_PATH = "output/recommend_versions.json"

def _cfg_hash(cfg):
    import hashlib
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]

def load_recommend_versions():
    """Version log of the Recommend engine. Every buy is stamped with the
    version active at purchase so the Validation tab can compare versions.
    Each version stores its FULL scoring config so any version can be
    restored (rollback = new forward version with an old config — the log
    is append-only and history is never rewritten).
    Seeds v1 (baseline) and v2 (2026-07-07 overhaul) on first use."""
    db = load_json(RECOMMEND_VERSIONS_PATH, None)
    if db and db.get("versions"):
        # Forward-migrate old entries missing config snapshots
        changed = False
        for v in db["versions"]:
            if "configSnapshot" not in v:
                v["configSnapshot"] = None
                changed = True
        if changed:
            save_json(RECOMMEND_VERSIONS_PATH, db)
        return db
    import copy
    v1_cfg = copy.deepcopy(SCORING_DEFAULTS)
    v1_cfg["signalForecast"]["enabled"] = True
    v1_cfg["sellThresholds"]["minHoldDays"] = 0
    v1_cfg["execution"]["slippagePct"] = 0.0
    v2_cfg = copy.deepcopy(SCORING_DEFAULTS)
    db = {"versions": [
        {
            "version": 1,
            "date": "2026-05-16",
            "label": "Baseline",
            "changes": [
                "Original scoring matrix; daily top-5 rotation",
                "KNOWN FLAWS: traded on stale cached prices (Norton TLS bug), no slippage, no hold-time minimum, signal forecast boost ±20 active",
                "Note: rollback restores config only — the data-integrity code fixes (live prices etc.) are permanent",
            ],
            "configSnapshot": v1_cfg,
            "configHash": _cfg_hash(v1_cfg),
        },
        {
            "version": 2,
            "date": "2026-07-07",
            "label": "Data-integrity overhaul",
            "changes": [
                "Live prices required for every buy/sell (no trading on cached data)",
                "0.2% slippage per side on fills",
                "minHoldDays=3 hysteresis on rotation sells",
                "Signal forecast boost disabled (measured: 49% direction accuracy = coin flip)",
                "Rule-based candidate scoring with full 0-100 range",
                "Outcome logging with buy-time factors (this validation system)",
            ],
            "configSnapshot": v2_cfg,
            "configHash": _cfg_hash(v2_cfg),
        },
    ]}
    save_json(RECOMMEND_VERSIONS_PATH, db)
    return db

def current_recommend_version():
    return load_recommend_versions()["versions"][-1]["version"]

def bump_recommend_version(changes, label="", config_snapshot=None, based_on=None, code_change=False):
    db = load_recommend_versions()
    new_v = db["versions"][-1]["version"] + 1
    if config_snapshot is None:
        config_snapshot = load_scoring_config()
    entry = {
        "version": new_v,
        "date": datetime.now(TZ).strftime("%Y-%m-%d"),
        "label": label or "Config change",
        "changes": changes,
        "configSnapshot": config_snapshot,
        "configHash": _cfg_hash(config_snapshot),
    }
    if based_on:
        entry["basedOn"] = based_on
    # Don't-repeat-mistakes guard: flag if this exact config already existed
    # (skipped for code_change: same config, but the scoring code itself changed)
    dup = next((v for v in db["versions"] if v.get("configHash") == entry["configHash"]), None)
    if dup and not based_on and not code_change:
        entry["warning"] = f"Config identical to v{dup['version']} — check its results before expecting different ones"
    # Whole-account equity snapshot at the moment this version starts — lets
    # the Validation tab show a real "before this version vs now" return even
    # if the version never buys or sells anything itself.
    try:
        entry["assetAtStart"] = _snapshot_total_asset()
        entry["assetAtStartCapturedAt"] = ts()
    except Exception as se:
        log.warning(f"[RecVersion] assetAtStart snapshot FAIL for v{new_v}: {se}")
    # Same idea for TSX Composite/TSX 60 — captured once, here, so "is THIS
    # version beating the index" never needs an arbitrary reset-point
    # comparison; it's always scoped to the exact window the version was
    # actually live. (Ported from stock-analysis, adapted to TSX/TSX60.)
    try:
        idx_prices = fetch_live_prices([BENCHMARK_TICKERS["TSX"], BENCHMARK_TICKERS["TSX60"]])
        entry["tsxAtStart"] = idx_prices.get(BENCHMARK_TICKERS["TSX"])
        entry["tsx60AtStart"] = idx_prices.get(BENCHMARK_TICKERS["TSX60"])
    except Exception as se:
        log.warning(f"[RecVersion] TSX/TSX60 snapshot FAIL for v{new_v}: {se}")
    db["versions"].append(entry)
    save_json(RECOMMEND_VERSIONS_PATH, db)
    log.info(f"[RecVersion] bumped to v{new_v} ({label}): {changes}")
    return new_v

def _version_stats_map():
    """Outcome stats grouped by the recommend version that made each trade."""
    outcomes = load_json(TRADE_OUTCOMES_PATH, {"outcomes": []}).get("outcomes", [])
    grouped = {}
    for o in outcomes:
        grouped.setdefault(o.get("recVersion", 1), []).append(o)
    return {v: _outcome_stats(rows) for v, rows in grouped.items()}


def _cap_filtered_candidates(candidates_sorted, holdings, details, identify_rules, exclude_ticker=None):
    """Walk composite-sorted candidates, keeping only those that respect a
    max-per-bucket cap and (optionally) a sector hard-block — both counted
    against current holdings first, then against candidates already kept
    earlier in this same walk. A candidate that's blocked is skipped, not a
    stop condition, so a lower-ranked but compliant candidate can still take
    the slot. `exclude_ticker` lets a rotation sell free its own bucket/
    sector before the replacement is picked. Returns the full filtered list
    in order — callers slice off however many slots they actually need."""
    # maxPerBucketFinal can be a flat int (same cap for every bucket) or a
    # {bucket: cap} dict with an optional "_default" for buckets not listed —
    # lets a proven bucket (e.g. SmartMoney) get more of the limited final
    # slots than a bucket with a thinner track record, without shutting the
    # others out entirely.
    mpb_cfg = identify_rules.get("maxPerBucketFinal", 3)
    def _cap_for_bucket(bucket):
        if isinstance(mpb_cfg, dict):
            return int(mpb_cfg.get(bucket, mpb_cfg.get("_default", 3)))
        return int(mpb_cfg)
    hard_block_sector = bool(identify_rules.get("hardBlockSectorDiversity", False))

    bucket_counts = {}
    held_sectors = set()
    for t, pos in holdings.items():
        if t == exclude_ticker:
            continue
        b = (pos.get("factors") or {}).get("bucket", "Dream")
        bucket_counts[b] = bucket_counts.get(b, 0) + 1
        sector = details.get(t, {}).get("sector") or ""
        if sector:
            held_sectors.add(sector)

    kept = []
    for c in candidates_sorted:
        bucket = c.get("bucket", "Dream")
        if bucket_counts.get(bucket, 0) >= _cap_for_bucket(bucket):
            continue
        sector = details.get(c["ticker"], {}).get("sector") or ""
        if hard_block_sector and sector and sector in held_sectors:
            continue
        kept.append(c)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if sector:
            held_sectors.add(sector)
    return kept


def _keep_recent_if_empty(new_articles, old_articles, days=7):
    """An empty fetch must not wipe stored articles that are still recent —
    on 2026-10-01 a Yahoo news-feed outage replaced every ticker's news
    with an empty list. A real "no news" result loses nothing by this:
    old articles are only kept while they're inside the same 7-day window."""
    if new_articles:
        return new_articles
    cutoff = datetime.now(TZ) - timedelta(days=days)
    kept = []
    for a in old_articles or []:
        try:
            if datetime.fromisoformat(str(a.get("pubDate", "")).replace("Z", "+00:00")) >= cutoff:
                kept.append(a)
        except ValueError:
            pass
    return kept


def _news_tickers():
    """Everything the app shows or scores: holdings, watchlist, today's
    Identify candidates."""
    wl = load_json(WATCHLIST_PATH, [])
    wl = wl if isinstance(wl, list) else wl.get("watchlist", [])
    identified = [i["ticker"] for i in load_json(TRADE_AI_CANDIDATES_PATH, {}).get("identified", [])]
    return sorted(set(load_trades_ai().get("holdings", {})) | set(wl) | set(identified))


def run_news_refresh(trigger="manual", gemini=True):
    """The one news fetch: scheduled at newsTimes (default 09:20, 11:47, 14:47)
    plus the TradeAI REFRESH NEWS button. Refreshes every ticker from
    _news_tickers(), then re-runs the severe-news scan on holdings. Other
    code paths read the stored news instead of fetching their own."""
    global fetch_status
    tickers = _news_tickers()
    total = len(tickers)
    fetch_status = {"running": True, "message": f"News: refreshing {total} tickers...", "operation": "news_refresh",
                    "current": 0, "total": total, "current_ticker": ""}
    log.info(f"=== News Refresh Started [{trigger}]: {total} tickers ===")
    try:
        news_db = load_json(NEWS_PATH, {})
        with_news = 0
        for i, t in enumerate(tickers, 1):
            fetch_status.update({"current": i, "current_ticker": t, "message": f"News: {i} of {total} — {t}"})
            _refresh_ticker_news_if_stale(t, news_db, max_age_hours=-1)  # -1 forces a refresh
            with_news += bool(news_db.get(t, {}).get("articles"))
        save_json(NEWS_PATH, news_db)
        try:
            _scan_news_warnings(list(load_trades_ai().get("holdings", {})), news_db)
        except Exception as e:
            log.warning(f"[NewsWarning] scan FAIL: {e}")
        msg = f"News refreshed: {total} tickers, {with_news} with recent news"
        ok, why = _gemini_allow("news-check", trigger == "scheduler") if gemini else (False, "")
        if ok:
            fetch_status.update({"message": "News: Gemini checking today's headlines..."})
            try:
                nc = run_news_check(trigger)
                msg += f" · Gemini: {nc['counts']['negative']} negative today" if nc["status"] == "ok" else " · Gemini check FAILED"
            except Exception as e:
                log.error(f"[NewsCheck] FAIL: {e}")
                msg += " · Gemini check FAILED"
        elif why:
            msg += " · " + why
        fetch_status = {"running": False, "message": msg, "operation": None, "current": total, "total": total, "current_ticker": ""}
        log.info(f"=== News Refresh Complete [{trigger}]: {msg} ===")
    except Exception as e:
        log.error(f"News refresh FAIL: {e}")
        fetch_status = {"running": False, "message": f"News refresh FAIL: {e}", "operation": None}


# ── Gemini budget: free tier = 20 requests/day per Google project (429s in
# logs/app.log, 2026-06/07). This app keeps to geminiDailyCap (default 4) so
# other apps sharing the project keep theirs. Manual presses may only use
# calls not needed by today's remaining scheduled Gemini jobs.
GEMINI_USAGE_PATH = "output/gemini_usage.json"
_gemini_job = ["?"]


def _gemini_usage():
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    db = load_json(GEMINI_USAGE_PATH, {})
    return db if db.get("date") == today else {"date": today, "calls": []}


def _gemini_record():
    db = _gemini_usage()
    db["calls"].append({"at": ts(), "job": _gemini_job[0]})
    save_json(GEMINI_USAGE_PATH, db)


import news_check as _news_check_mod
_news_check_mod.USAGE_HOOK = _gemini_record


def _gemini_reserved(now=None):
    """Scheduled Gemini jobs still to come today (market days only)."""
    now = now or datetime.now(TZ)
    cfg = load_scheduler_config()
    if now.weekday() >= 5 or now.strftime("%Y-%m-%d") in cfg.get("marketHolidays", []):
        return 0
    hhmm = now.strftime("%H:%M")
    news = [t for t in cfg.get("geminiNewsTimes", []) if t in cfg.get("newsTimes", []) and t > hhmm]
    return len(news) + sum(1 for k in ("buyReviewTime", "lossReviewTime") if (cfg.get(k) or "") > hhmm)


def _gemini_status():
    cfg = load_scheduler_config()
    used = len(_gemini_usage()["calls"])
    return {"used": used, "cap": int(cfg.get("geminiDailyCap", 4)), "reserved": _gemini_reserved()}


def _gemini_allow(job, scheduled):
    """(ok, reason). Scheduled jobs may use any call left under the cap;
    manual ones only calls not reserved for later scheduled jobs."""
    st = _gemini_status()
    left = st["cap"] - st["used"]
    spare = left if scheduled else left - st["reserved"]
    if spare >= 1:
        _gemini_job[0] = job
        return True, ""
    why = (f"Gemini budget used: {st['used']}/{st['cap']} today" if left < 1 else
           f"Gemini skipped: {left} call(s) left today, kept for {st['reserved']} scheduled job(s)")
    log.info(f"[GeminiBudget] {job} {'(scheduled)' if scheduled else '(manual)'} blocked — {why}")
    return False, why


# ═══════════════════════════ GEMINI NEWS CHECK ═══════════════════════════
# After every news refresh, one batched Gemini call judges TODAY's headlines
# for every TradeAI holding and candidate (news_check.py). The latest run is
# shown in a popup that stays up until dismissed; every run is kept in the
# history file so the flags can later be checked against what prices did.
NEWS_CHECK_PATH = "output/news_check.json"
NEWS_CHECK_HISTORY_PATH = "output/news_check_history.json"
NEWS_CHECK_ALERT_SEVERITY = 4  # phone/in-app alert at this severity and above
MARKET_NEWS_QUERIES = ["TSX", "Bank of Canada", "Canadian economy", "stock market", "Federal Reserve", "tariffs"]  # Yahoo searches for the morning market-risk read


def _news_check_roles():
    roles = {i["ticker"]: "candidate" for i in load_json(TRADE_AI_CANDIDATES_PATH, {}).get("identified", []) if i.get("ticker")}
    roles.update({t: "held" for t in load_trades_ai().get("holdings", {})})
    return roles


def run_news_check(trigger="manual"):
    from news_check import run_check, fetch_market_headlines
    from config import GEMINI_API_KEY
    prev = load_json(NEWS_CHECK_PATH, {})
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    # Market read: once a day, inside the first news-check call (normally 09:20); later runs carry it forward.
    pm = prev.get("market") or {}
    have_market = pm.get("date") == today and pm.get("status") == "ok"
    mkt_heads = None if have_market else fetch_market_headlines(MARKET_NEWS_QUERIES, TZ)
    rec = run_check(_news_check_roles(), load_json(NEWS_PATH, {}), GEMINI_API_KEY, TZ, trigger, market_headlines=mkt_heads)
    if have_market:
        rec["market"] = pm
    else:
        m = rec.get("market") or {}
        log.info(f"[MarketRisk] {m.get('status')} | {m.get('riskLevel')} | {m.get('summary') or m.get('error')}")
        send = globals().get("_send_alert")
        if send and m.get("riskLevel") == "high":
            send("news", "🌐 Market risk HIGH today", m.get("summary") or "", priority=4, tags=["warning"])
    rec["usage"] = _gemini_status()
    rec["acked"] = False
    save_json(NEWS_CHECK_PATH, rec)
    hist = load_json(NEWS_CHECK_HISTORY_PATH, {"runs": []})
    hist["runs"] = (hist.get("runs", []) + [{k: v for k, v in rec.items() if k != "acked"}])[-300:]
    save_json(NEWS_CHECK_HISTORY_PATH, hist)
    c = rec["counts"]
    log.info(f"[NewsCheck] [{trigger}] {rec['status']} | window {rec['window']} | negative:{c['negative']} deal:{c['deal']} "
             f"clear:{c['clear']} no-news:{c['no_news']} error:{c['error']} | gemini called:{rec['geminiCalled']}"
             + (f" | {rec['error']}" if rec.get("error") else ""))
    send = globals().get("_send_alert")  # alerts exist in the US app only
    if send:
        for r in rec["tickers"]:
            if r["state"] == "negative" and r["severity"] >= NEWS_CHECK_ALERT_SEVERITY:
                send("news", f"⚠ {r['ticker']} negative news (severity {r['severity']}/5)",
                     f"{'HELD' if r['role'] == 'held' else 'Candidate'} · {r['category']} — {r['reason']}",
                     priority=4, tags=["warning"])
    return rec


def _news_check_map():
    """Today's flags from the latest run: {ticker: {negative, severity, category, deal, dealNote}}."""
    rec = load_json(NEWS_CHECK_PATH, {})
    if rec.get("date") != datetime.now(TZ).strftime("%Y-%m-%d"):
        return {}
    return {r["ticker"]: {"negative": r.get("state") == "negative", "severity": r["severity"], "category": r["category"],
                          "deal": r.get("deal", False), "dealNote": r.get("dealNote", "")}
            for r in rec.get("tickers", []) if r.get("state") == "negative" or r.get("deal")}


def _market_risk_today():
    m = load_json(NEWS_CHECK_PATH, {}).get("market") or {}
    return m if m.get("date") == datetime.now(TZ).strftime("%Y-%m-%d") else None


# ── Buy review: after the last Recommend, Gemini second-opinions today's buys ──
BUY_REVIEWS_PATH = "output/buy_reviews.json"


def run_buy_review(trigger="manual"):
    """One Gemini call reviewing every BUY made today that has no verdict yet.
    Flags only — agree / caution / avoid; nothing is sold on it."""
    from news_check import review_buys
    from config import GEMINI_API_KEY
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    db = load_json(BUY_REVIEWS_PATH, {"reviews": []})
    done = {r["key"] for r in db.get("reviews", []) if r.get("verdict")}
    trades = load_trades_ai()
    holdings = trades.get("holdings", {})
    buys = []
    for tx in trades.get("transactions", []):
        if str(tx.get("action", "")).upper() != "BUY" or (tx.get("date") or "")[:10] != today:
            continue
        key = f"{tx.get('ticker')}|{tx.get('date')}"
        if key in done:
            continue
        pos = holdings.get(tx.get("ticker")) or {}
        buys.append({**tx, "key": key, "factors": pos.get("factors") if pos.get("purchasedAt") == tx.get("date") else None})
    if not buys:
        db["lastRun"] = {"at": ts(), "trigger": trigger, "status": "ok", "reviewed": 0, "note": "no unreviewed buys today"}
        save_json(BUY_REVIEWS_PATH, db)
        log.info(f"[BuyReview] [{trigger}] nothing to review")
        return db["lastRun"]
    ok, why = _gemini_allow("buy-review", trigger == "scheduler")
    if not ok:
        db["lastRun"] = {"at": ts(), "trigger": trigger, "status": "skipped", "reviewed": 0, "note": why}
        save_json(BUY_REVIEWS_PATH, db)
        return db["lastRun"]
    nc = load_json(NEWS_CHECK_PATH, {})
    fresh = nc.get("date") == today
    out = review_buys(buys, load_json(TRADE_AI_CANDIDATES_PATH, {}).get("details", {}),
                      {r["ticker"]: r for r in nc.get("tickers", [])} if fresh else {},
                      nc.get("market") if fresh else None, GEMINI_API_KEY, TZ)
    keep = [r for r in db.get("reviews", []) if r["key"] not in out]
    db["reviews"] = (keep + list(out.values()))[-500:]
    ok = sum(1 for r in out.values() if r.get("verdict"))
    db["lastRun"] = {"at": ts(), "trigger": trigger, "status": "ok" if ok == len(out) else "partial",
                     "reviewed": ok, "failed": len(out) - ok}
    save_json(BUY_REVIEWS_PATH, db)
    log.info(f"[BuyReview] [{trigger}] reviewed {ok}/{len(out)} buys | "
             + ", ".join(f"{r['ticker']}:{r.get('verdict')}" for r in out.values()))
    send = globals().get("_send_alert")  # alerts exist in the US app only
    if send:
        for r in out.values():
            if r.get("verdict") == "avoid":
                send("news", f"✋ Gemini disagrees with today's {r['ticker']} buy",
                     f"{r['thesis']} — risks: {'; '.join(r.get('risks') or [])}", priority=4, tags=["warning"])
    return db["lastRun"]


def _buy_reviews_payload():
    """Reviews (newest first), per-verdict results on closed trades so the
    verdicts can be judged against outcomes, and the verdict of each
    current holding's purchase for the badge."""
    db = load_json(BUY_REVIEWS_PATH, {"reviews": []})
    rows = sorted(db.get("reviews", []), key=lambda r: r.get("date") or "", reverse=True)
    closed = {(o.get("ticker"), o.get("buyDate")): o for o in load_json(TRADE_OUTCOMES_PATH, {"outcomes": []}).get("outcomes", [])}
    stats = {v: {"reviewed": 0, "closed": 0, "wins": 0, "sumPct": 0.0} for v in ("agree", "caution", "avoid")}
    for r in rows:
        v = r.get("verdict")
        if v not in stats:
            continue
        stats[v]["reviewed"] += 1
        o = closed.get((r["ticker"], r.get("date")))
        if o and o.get("gainLossPct") is not None:
            r["closedPct"] = o["gainLossPct"]
            stats[v]["closed"] += 1
            stats[v]["wins"] += o["gainLossPct"] > 0
            stats[v]["sumPct"] = round(stats[v]["sumPct"] + o["gainLossPct"], 2)
    held = {}
    for t, pos in load_trades_ai().get("holdings", {}).items():
        r = next((r for r in rows if r["ticker"] == t and r.get("date") == pos.get("purchasedAt") and r.get("verdict")), None)
        if r:
            held[t] = {"verdict": r["verdict"], "thesis": r.get("thesis", ""), "risks": r.get("risks", [])}
    return {"reviews": rows[:60], "stats": stats, "held": held, "lastRun": db.get("lastRun")}


@app.route("/tradeai/buy_review/run", methods=["POST"])
def buy_review_run():
    try:
        return jsonify(run_buy_review("manual"))
    except Exception as e:
        log.error(f"[BuyReview] FAIL: {e}")
        return jsonify({"error": str(e)}), 500


# ── Loss review: after the close, Gemini tags why each losing trade lost ──
LOSS_REVIEWS_PATH = "output/loss_reviews.json"
LOSS_REVIEW_LOOKBACK_DAYS = 7  # headlines are only stored for ~7 days


def run_loss_review(trigger="manual"):
    """Review losing trades closed in the last LOSS_REVIEW_LOOKBACK_DAYS that
    don't have a verdict yet — one Gemini call for all of them."""
    from news_check import review_losses
    from config import GEMINI_API_KEY
    db = load_json(LOSS_REVIEWS_PATH, {"reviews": []})
    done = {r["key"] for r in db.get("reviews", []) if r.get("cause")}
    cutoff = (datetime.now(TZ) - timedelta(days=LOSS_REVIEW_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    todo = []
    for o in load_json(TRADE_OUTCOMES_PATH, {"outcomes": []}).get("outcomes", []):
        key = f"{o.get('ticker')}|{o.get('sellDate')}"
        if (o.get("gainLossPct") or 0) < 0 and (o.get("sellDate") or "")[:10] >= cutoff and key not in done:
            todo.append({**o, "key": key})
    todo = todo[-25:]
    if not todo:
        db["lastRun"] = {"at": ts(), "trigger": trigger, "status": "ok", "reviewed": 0, "note": "no unreviewed losing trades"}
        save_json(LOSS_REVIEWS_PATH, db)
        log.info(f"[LossReview] [{trigger}] nothing to review")
        return db["lastRun"]
    ok, why = _gemini_allow("loss-review", trigger == "scheduler")
    if not ok:
        db["lastRun"] = {"at": ts(), "trigger": trigger, "status": "skipped", "reviewed": 0, "note": why}
        save_json(LOSS_REVIEWS_PATH, db)
        return db["lastRun"]
    out = review_losses(todo, load_json(NEWS_PATH, {}), load_json(NEWS_CHECK_HISTORY_PATH, {"runs": []}).get("runs", []),
                        GEMINI_API_KEY, TZ)
    keep = [r for r in db.get("reviews", []) if r["key"] not in out]
    db["reviews"] = (keep + list(out.values()))[-500:]
    ok = sum(1 for r in out.values() if r.get("cause"))
    db["lastRun"] = {"at": ts(), "trigger": trigger, "status": "ok" if ok == len(out) else "partial",
                     "reviewed": ok, "failed": len(out) - ok}
    save_json(LOSS_REVIEWS_PATH, db)
    log.info(f"[LossReview] [{trigger}] reviewed {ok}/{len(out)} losing trades")
    return db["lastRun"]


def _loss_reviews_payload():
    db = load_json(LOSS_REVIEWS_PATH, {"reviews": []})
    rows = sorted(db.get("reviews", []), key=lambda r: r.get("sellDate") or "", reverse=True)
    causes = {}
    for r in rows:
        if r.get("cause"):
            c = causes.setdefault(r["cause"], {"count": 0, "lossPct": 0.0})
            c["count"] += 1
            c["lossPct"] = round(c["lossPct"] + (r.get("gainLossPct") or 0), 2)
    return {"reviews": rows[:100], "causes": causes, "lastRun": db.get("lastRun")}


@app.route("/news/loss_review/run", methods=["POST"])
def loss_review_run():
    try:
        return jsonify(run_loss_review("manual"))
    except Exception as e:
        log.error(f"[LossReview] FAIL: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/news/check")
def news_check_latest():
    return jsonify(load_json(NEWS_CHECK_PATH, {}))


@app.route("/news/check/ack", methods=["POST"])
def news_check_ack():
    rec = load_json(NEWS_CHECK_PATH, {})
    if rec and rec.get("runId") == (request.get_json(silent=True) or {}).get("runId"):
        rec["acked"] = True
        save_json(NEWS_CHECK_PATH, rec)
    return jsonify({"status": "ok"})


# ═══════════════════════════ OPTIONS HISTORY ═══════════════════════════
# Yahoo keeps no options history, so it has to be recorded daily to ever
# test whether unusual options activity predicts moves. One file per
# trading day: output/options/YYYY-MM-DD.json (gitignored).
# TSX options aren't on Yahoo, so each ticker is recorded through its
# US-listed twin when it has one (RY.TO -> RY); see resolve_us_twin.
OPTIONS_DIR = "output/options"
OPTIONS_SYMBOL_MAP_PATH = "output/options_symbol_map.json"
OPTIONS_TWIN_RECHECK_DAYS = 30  # retry tickers with no twin monthly (new US listings)


def _options_tickers():
    """Every ticker ever bought (transactions, holdings and closed-trade
    outcomes — outcomes survive account resets) plus every ticker the
    shadow simulation has scored."""
    t = load_json(TRADES_AI_PATH, {})
    tickers = {x["ticker"] for x in t.get("transactions", []) if x.get("ticker")} | set(t.get("holdings", {}))
    tickers |= {o["ticker"] for o in load_json(TRADE_OUTCOMES_PATH, {"outcomes": []}).get("outcomes", []) if o.get("ticker")}
    for snap in load_json(SHADOW_PATH, {"snapshots": []}).get("snapshots", []):
        tickers |= {r["ticker"] for r in snap.get("rows", []) if r.get("composite") is not None}
    return sorted(tickers)


def _options_twin_map(tickers):
    """{tsx_ticker: us_symbol or None}, cached in OPTIONS_SYMBOL_MAP_PATH.
    Only new tickers and stale no-twin entries are looked up."""
    from fetcher import resolve_us_twin
    cache = load_json(OPTIONS_SYMBOL_MAP_PATH, {})
    today = datetime.now(TZ).date()
    changed = False
    for t in tickers:
        e = cache.get(t)
        if e and (e.get("us") or (today - datetime.strptime(e["checkedAt"], "%Y-%m-%d").date()).days < OPTIONS_TWIN_RECHECK_DAYS):
            continue
        fetch_status["message"] = f"Options: matching US listing — {t}"
        try:
            cache[t] = {"us": resolve_us_twin(t), "checkedAt": today.isoformat()}
            changed = True
        except Exception as ex:
            log.warning(f"[Options] twin lookup FAIL: {t} | {ex}")  # retried next run
    if changed:
        save_json(OPTIONS_SYMBOL_MAP_PATH, cache)
    return {t: (cache.get(t) or {}).get("us") for t in tickers}


def run_options_snapshot(trigger="manual"):
    """Record today's end-of-day options summary for _options_tickers().
    Re-running the same day only fills tickers not yet recorded (or that
    failed), so a manual re-run after a partial failure is cheap."""
    global fetch_status
    from fetcher import fetch_options_summary
    os.makedirs(OPTIONS_DIR, exist_ok=True)
    day = datetime.now(TZ).strftime("%Y-%m-%d")
    path = os.path.join(OPTIONS_DIR, f"{day}.json")
    db = load_json(path, {"date": day, "tickers": {}})
    todo = [t for t in _options_tickers() if t not in db["tickers"] or "error" in db["tickers"][t]]
    total = len(todo)
    fetch_status = {"running": True, "message": f"Options: {total} tickers...", "operation": "options_snapshot",
                    "current": 0, "total": total, "current_ticker": ""}
    twins = _options_twin_map(todo)
    log.info(f"=== Options Snapshot Started [{trigger}]: {total} tickers ===")
    ok = no_opts = failed = 0
    try:
        for i, t in enumerate(todo, 1):
            if stop_flag:
                break
            fetch_status.update({"current": i, "current_ticker": t, "message": f"Options: {i} of {total} — {t}"})
            for attempt in (1, 2):
                try:
                    us = twins.get(t)
                    s = {**fetch_options_summary(us), "usSymbol": us} if us else {"noOptions": True, "usSymbol": None}
                    db["tickers"][t] = s
                    if s.get("noOptions"):
                        no_opts += 1
                    else:
                        ok += 1
                    break
                except Exception as e:
                    if attempt == 2:
                        db["tickers"][t] = {"error": str(e)[:200]}
                        failed += 1
                        log.warning(f"[Options] FAIL: {t} | {e}")
                    else:
                        time.sleep(2)
            if i % 25 == 0:
                save_json(path, db)  # partial progress survives a crash
        db["capturedAt"] = ts()
        save_json(path, db)
        # only name failures when there are some: any "FAIL" in the message pops the global error popup
        msg = f"Options saved {day}: {ok} with options, {no_opts} without (no US listing)" + (f", {failed} FAILED" if failed else "")
        fetch_status = {"running": False, "message": msg, "operation": None, "current": total, "total": total, "current_ticker": ""}
        log.info(f"=== Options Snapshot Complete [{trigger}]: {msg} ===")
    except Exception as e:
        save_json(path, db)
        log.error(f"Options snapshot FAIL: {e}")
        fetch_status = {"running": False, "message": f"Options snapshot FAIL: {e}", "operation": None}


@app.route("/options/snapshot/run", methods=["POST"])
def options_snapshot_run():
    if fetch_status.get("running"):
        return jsonify({"error": "Another operation is running"}), 409
    threading.Thread(target=run_options_snapshot, args=("manual",), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/options/<ticker>")
def options_history(ticker):
    """Recorded daily options summaries for one ticker, newest first."""
    days = int(request.args.get("days", 30))
    out = []
    if os.path.isdir(OPTIONS_DIR):
        for f in sorted(os.listdir(OPTIONS_DIR), reverse=True)[:days]:
            rec = load_json(os.path.join(OPTIONS_DIR, f), {}).get("tickers", {}).get(ticker)
            if rec is not None:
                out.append({"date": f[:-5], **rec})
    return jsonify({"ticker": ticker, "history": out})


@app.route("/news/refresh_all", methods=["POST"])
def news_refresh_all():
    if fetch_status.get("running"):
        return jsonify({"error": "Another operation is running"}), 409
    threading.Thread(target=run_news_refresh, args=("manual",), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/news/headlines/<ticker>")
def news_headlines(ticker):
    """Stored headlines only — no fetching, no AI analysis."""
    entry = load_json(NEWS_PATH, {}).get(ticker, {})
    items = sorted(entry.get("articles", []), key=lambda a: a.get("pubDate", ""), reverse=True)
    return jsonify({"ticker": ticker, "fetchedAt": entry.get("fetchedAt"),
                    "items": [{k: a.get(k, "") for k in ("title", "url", "provider", "pubDate")} for a in items]})


def _refresh_ticker_news_if_stale(ticker, news_db, max_age_hours=4):
    """Fetch and merge fresh news articles for `ticker` into news_db if the
    cached copy is older than max_age_hours (or missing). Merges into
    whatever's already there (fetchedAt/articles keys only) rather than
    replacing the entry outright, so it never clobbers the separate
    todayItems/sevenDayItems/impact fields the on-demand /news/<ticker>
    detail-panel route writes into the same per-ticker record. This is what
    feeds _rule_based_assessment's newsSentiment and the severe-keyword
    warning scan — without it, both run on an empty article list most days."""
    entry = news_db.get(ticker, {})
    if hours_since(entry.get("fetchedAt")) <= max_age_hours:
        return False
    try:
        articles = fetch_ticker_news(ticker)
        entry["articles"] = _keep_recent_if_empty(articles, entry.get("articles"))
        entry["fetchedAt"] = ts()
        news_db[ticker] = entry
        return True
    except Exception as e:
        log.warning(f"[News] refresh FAIL: {ticker} | {e}")
        return False


NEWS_WARNINGS_PATH = "output/news_warnings.json"

# Deliberately narrow and severe — distinct from _rule_based_assessment's
# mild bullish/bearish keyword scoring. This list is for "a human should
# look at this now" events, never for routine sentiment.
SEVERE_NEWS_KEYWORDS = [
    "bankruptcy", "chapter 11", "insolvency", "going concern",
    "securities fraud", "sec investigation", "doj investigation", "fraud charges",
    "class action", "restatement", "accounting irregularit",
    "delisting", "delisted", "trading halt", "halted trading",
    "product recall", "debt default", "default on",
    "ceo resigns", "cfo resigns", "resignation of", "executive resigns",
    "short seller report", "going private investigation",
]

def _scan_news_warnings(tickers, news_db):
    """Scan cached news for `tickers` (expected: currently-held positions)
    for severe red-flag keywords. Flags only — never sells anything. A
    ticker is re-flagged only when a NEW matching headline appears; if the
    same headline that triggered a dismissed warning is still the most
    recent match, it stays dismissed. Returns the updated warnings dict."""
    warnings_db = load_json(NEWS_WARNINGS_PATH, {"warnings": {}})
    warnings = warnings_db.get("warnings", {})
    changed = False
    for ticker in tickers:
        articles = (news_db.get(ticker, {}) or {}).get("articles", [])[:5]
        hit = None
        for a in articles:
            title = str(a.get("title", "")).lower()
            for kw in SEVERE_NEWS_KEYWORDS:
                if kw in title:
                    hit = (kw, a)
                    break
            if hit:
                break
        if not hit:
            continue
        kw, article = hit
        headline = article.get("title", "")
        headline_hash = hashlib.md5(f"{ticker}|{headline}".encode()).hexdigest()[:12]
        existing = warnings.get(ticker)
        if existing and existing.get("headlineHash") == headline_hash:
            continue  # same headline already on record (active or dismissed) — don't re-flag
        warnings[ticker] = {
            "status": "active",
            "keyword": kw,
            "headline": headline,
            "url": article.get("url"),
            "provider": article.get("provider"),
            "pubDate": article.get("pubDate"),
            "headlineHash": headline_hash,
            "detectedAt": ts(),
        }
        changed = True
        log.warning(f"[NewsWarning] {ticker}: '{kw}' matched — {headline}")
    if changed:
        warnings_db["warnings"] = warnings
        save_json(NEWS_WARNINGS_PATH, warnings_db)
    return warnings


MACRO_NEGATIVE_WORDS = ("recession", "crash", "selloff", "sell-off", "correction", "bear market",
                         "layoffs", "default", "crisis", "downturn", "stagflation", "plunge")
MACRO_POSITIVE_WORDS = ("rally", "record high", "soft landing", "rate cut", "growth beat", "strong jobs")

def _macro_risk_signal():
    """Rule-based macro risk-off read: yield curve inversion, negative
    keyword density in fetched market headlines, and four hard economic-
    data triggers (each individually rare and unambiguous, not a fuzzy
    blend of every indicator on the Macro tab — most of those, e.g. VIX
    level or the market-price series, are already covered by the separate
    VIX/trend gates or don't have a defensible risk-off threshold at all).
    Same additive pattern as the existing VIX regime gate — an extra input
    to it, not a replacement. (Ported from stock-analysis — these are US
    macro triggers; TSX-listed names still trade largely on US Fed/economic
    data, same as the main account, so no Canada-specific equivalent yet.)"""
    macro = load_json(MACRO_PATH, {})
    indicators = macro.get("indicators", {})
    headlines = macro.get("headlines", [])

    yc = (indicators.get("yield_curve") or {}).get("value")
    inverted = yc is not None and yc < 0

    text = " ".join(str(h.get("title", "")).lower() for h in headlines)
    neg = sum(text.count(w) for w in MACRO_NEGATIVE_WORDS)
    pos = sum(text.count(w) for w in MACRO_POSITIVE_WORDS)
    headline_bearish = neg > pos and neg >= 2  # require 2+ hits, not one stray word
    headline_bullish = pos > neg and pos >= 2  # mirror of the bearish check, display-only (no buy boost yet)

    # Payrolls turning negative (outright job losses, not just slower
    # growth) is a hard, rare, unambiguous recession signal.
    payrolls = (indicators.get("nonfarm_payrolls") or {}).get("value")
    payrolls_negative = payrolls is not None and payrolls < 0

    # Unemployment rising fast — a simplified proxy for the Sahm Rule
    # (a 0.3pt+ single-month jump is itself unusual; the real rule uses a
    # 3-month average vs. its 12-month low, which would need retained
    # history this fetch doesn't keep).
    unemp_chg = (indicators.get("unemployment") or {}).get("change")
    unemployment_spiking = unemp_chg is not None and unemp_chg >= 0.3

    # Inflation clearly reaccelerating (this month's YoY vs last month's)
    # feeds the same fear either headline handles — the Fed staying
    # restrictive/hiking into a slowing economy. CPI and Core PCE share one
    # trigger since they mostly move together; report separately if not.
    cpi_chg = (indicators.get("cpi") or {}).get("change")
    pce_chg = (indicators.get("core_pce") or {}).get("change")
    inflation_reaccelerating = (cpi_chg is not None and cpi_chg >= 0.3) or (pce_chg is not None and pce_chg >= 0.3)

    # Fed Funds Rate actually moving up — unlike a market series, FEDFUNDS
    # (the monthly EFFR average) doesn't drift on its own; a rate hike hits
    # the target within days, so any real m/m rise reflects an actual FOMC
    # decision, not noise. 0.05pt (5bp) filters out the rare tiny-basis-point
    # drift within an unchanged target so this doesn't false-trigger every
    # month the rate merely holds.
    fed_chg = (indicators.get("fed_rate") or {}).get("change")
    fed_hiking = fed_chg is not None and fed_chg >= 0.05

    risk_off = inverted or headline_bearish or payrolls_negative or unemployment_spiking or inflation_reaccelerating or fed_hiking

    reasons = []
    if inverted:
        reasons.append(f"yield curve inverted ({yc}%)")
    if headline_bearish:
        reasons.append(f"{neg} bearish vs {pos} bullish macro headline keyword hits")
    if payrolls_negative:
        reasons.append(f"Nonfarm Payrolls fell {payrolls}k m/m — outright job losses")
    if unemployment_spiking:
        reasons.append(f"Unemployment rate jumped +{unemp_chg}pt m/m")
    if inflation_reaccelerating:
        which = "CPI" if (cpi_chg is not None and cpi_chg >= 0.3) else "Core PCE"
        chg = cpi_chg if which == "CPI" else pce_chg
        reasons.append(f"{which} YoY reaccelerating (+{chg}pt vs last month)")
    if fed_hiking:
        reasons.append(f"Fed Funds Rate rose +{fed_chg}pt m/m — tightening")
    if not risk_off and headline_bullish:
        reasons.append(f"{pos} bullish vs {neg} bearish macro headline keyword hits")

    signal = "riskOff" if risk_off else ("riskOn" if headline_bullish else "neutral")

    return {
        "riskOff": risk_off, "signal": signal,
        "yieldCurve": yc, "negHits": neg, "posHits": pos,
        "reasons": reasons, "macroFetchedAt": macro.get("fetchedAt"),
    }


BENCHMARK_TICKERS = {"TSX": "^GSPTSE", "TSX60": "XIU.TO"}  # TSX60 uses the XIU.TO ETF proxy — ^TX60 isn't available on Yahoo Finance

def _refresh_benchmarks():
    """Fetch live price + yesterday's close for the S&P/TSX Composite index
    directly and the TSX 60 via its XIU.TO ETF proxy (Yahoo Finance has no
    raw ^TX60 index symbol), and store into analysis.json's top-level
    "benchmarks" key, alongside the watchlist. Called on the same cadence as
    holding price refreshes (the sell-check cycle and the manual refresh
    button) so they're always as fresh as your own positions."""
    tickers = list(BENCHMARK_TICKERS.values())
    live = fetch_live_prices(tickers)
    prevs = fetch_prev_closes(tickers)
    now_ts = ts()
    with _output_lock:
        out = load_json(OUTPUT_PATH, {"watchlist": []})
        benchmarks = out.get("benchmarks", {})
        for name, tk in BENCHMARK_TICKERS.items():
            price = live.get(tk)
            prev = prevs.get(tk)
            if not price:
                continue
            day_change_pct = round((price - prev) / prev * 100, 2) if prev else None
            benchmarks[name] = {
                "ticker": tk, "price": price, "previousClose": prev,
                "dayChangePct": day_change_pct, "fetchedAt": now_ts,
            }
        out["benchmarks"] = benchmarks
        save_json(OUTPUT_PATH, out)
    return benchmarks


# ═══════════════════════════ ACCOUNT TRACKER ═══════════════════════════
# Whole-account value vs. the TSX Composite / TSX 60 since the current
# account period started (ported from stock-analysis, adapted to this
# repo's TSX/TSX 60 benchmarks in place of SPY/Nasdaq).
ACCOUNT_TRACKER_PATH = "output/account_tracker.json"

def _replay_balance_and_holdings(transactions, as_of_date_str, starting_balance):
    """Reconstruct {balance, holdings: {ticker: shares}} as of the end of
    as_of_date_str (YYYY-MM-DD) by replaying every transaction up to and
    including that date, starting from starting_balance and no holdings.
    Used to mark the account to market on a PAST date — the only way to
    get an actual account-value time series when no daily snapshot was
    being recorded at the time."""
    balance = starting_balance
    holdings = {}
    for t in sorted(transactions, key=lambda x: x.get("date", "")):
        if (t.get("date") or "")[:10] > as_of_date_str:
            break
        ticker, shares = t.get("ticker"), t.get("shares") or 0
        if t.get("action") == "BUY":
            holdings[ticker] = holdings.get(ticker, 0) + shares
            balance -= t.get("amount") or 0
        elif t.get("action") == "SELL":
            holdings[ticker] = holdings.get(ticker, 0) - shares
            if holdings.get(ticker, 0) <= 0:
                holdings.pop(ticker, None)
            balance += t.get("amount") or 0
    return balance, holdings


def _close_on_or_before(daily_closes, date_obj):
    """Latest (date, close) from a _daily_closes() list at or before
    date_obj — a held ticker/index may not have traded exactly on the
    reconstruction date (holiday, gap in data)."""
    candidates = [(d, c) for d, c in daily_closes if d <= date_obj]
    return candidates[-1][1] if candidates else None


def _update_account_tracker():
    """Portfolio value vs. the TSX Composite / TSX 60, tracked daily since
    the start of the CURRENT account period (the earliest transaction still
    on record — a reset wipes trades_ai.json's transaction list, so this
    naturally starts a fresh comparison rather than conflating pre/post-
    reset performance). Backfills any trading day between that start and
    yesterday using historical closes (nothing was being snapshotted at the
    time), then records today live. Safe to call repeatedly — today's entry
    is always replaced, not appended again."""
    import yfinance as yf
    trades = load_trades_ai()
    transactions = trades.get("transactions", [])
    if not transactions:
        return None
    period_start = min(t["date"][:10] for t in transactions if t.get("date"))
    starting_balance = None
    first_tx = min((t for t in transactions if t.get("date", "")[:10] == period_start), key=lambda t: t["date"])
    if first_tx.get("action") == "BUY":
        starting_balance = round((first_tx.get("balance") or 0) + (first_tx.get("amount") or 0), 2)
    else:
        starting_balance = round((first_tx.get("balance") or 0) - (first_tx.get("amount") or 0), 2)

    db = load_json(ACCOUNT_TRACKER_PATH, {"periodStart": None, "snapshots": []})
    if db.get("periodStart") != period_start:
        # New account period (reset since last run) — start over.
        db = {"periodStart": period_start, "snapshots": []}

    today_str = datetime.now(TZ).strftime("%Y-%m-%d")
    existing_dates = {s["date"] for s in db["snapshots"] if s["date"] != today_str}
    start_date = datetime.strptime(period_start, "%Y-%m-%d").date()
    end_date = datetime.now(TZ).date() - timedelta(days=1)  # backfill through yesterday; today done live below
    missing_days = []
    d = start_date
    while d <= end_date:
        if d.weekday() < 5 and str(d) not in existing_dates:  # trading days only
            missing_days.append(d)
        d += timedelta(days=1)

    if missing_days:
        tickers = {t["ticker"] for t in transactions if t.get("ticker")}
        history_period = f"{(datetime.now(TZ).date() - start_date).days + 5}d"
        closes_by_ticker = {}
        for tk in tickers | {BENCHMARK_TICKERS["TSX"], BENCHMARK_TICKERS["TSX60"]}:
            try:
                closes_by_ticker[tk] = _daily_closes(yf.Ticker(tk).history(period=history_period))
            except Exception as e:
                log.warning(f"[AccountTracker] history FAIL: {tk} | {e}")
                closes_by_ticker[tk] = []
        tsx_closes = closes_by_ticker.get(BENCHMARK_TICKERS["TSX"], [])
        tsx60_closes = closes_by_ticker.get(BENCHMARK_TICKERS["TSX60"], [])

        for day in missing_days:
            day_str = str(day)
            balance, holdings = _replay_balance_and_holdings(transactions, day_str, starting_balance)
            value = balance
            ok = True
            for ticker, shares in holdings.items():
                close = _close_on_or_before(closes_by_ticker.get(ticker, []), day)
                if close is None:
                    ok = False
                    break
                value += shares * close
            tsx_close = _close_on_or_before(tsx_closes, day)
            tsx60_close = _close_on_or_before(tsx60_closes, day)
            if ok and tsx_close and tsx60_close:
                db["snapshots"].append({
                    "date": day_str, "accountValue": round(value, 2),
                    "tsxClose": tsx_close, "tsx60Close": tsx60_close,
                })

    # Today, live.
    live_tickers = list({t["ticker"] for t in transactions if t.get("ticker")}
                         | {BENCHMARK_TICKERS["TSX"], BENCHMARK_TICKERS["TSX60"]})
    live_prices = fetch_live_prices(live_tickers) if live_tickers else {}
    balance, holdings = _replay_balance_and_holdings(transactions, today_str, starting_balance)
    value = balance
    ok = True
    for ticker, shares in holdings.items():
        price = live_prices.get(ticker)
        if price is None:
            ok = False
            break
        value += shares * price
    tsx_now = live_prices.get(BENCHMARK_TICKERS["TSX"])
    tsx60_now = live_prices.get(BENCHMARK_TICKERS["TSX60"])
    if ok and tsx_now and tsx60_now:
        db["snapshots"] = [s for s in db["snapshots"] if s["date"] != today_str]
        db["snapshots"].append({"date": today_str, "accountValue": round(value, 2),
                                 "tsxClose": tsx_now, "tsx60Close": tsx60_now})

    db["snapshots"].sort(key=lambda s: s["date"])
    save_json(ACCOUNT_TRACKER_PATH, db)
    return db


ACCOUNT_TRACKER_WINDOWS = [
    ("today", 1), ("week", 7), ("month", 30), ("year", 365), ("allTime", None),
]

def _account_tracker_summary():
    """Cumulative % return for the account vs. TSX Composite vs. TSX 60 over
    several lookback windows (day/week/month/year/all-time since the
    account period started), from whatever's already recorded — read-only,
    no fetches (see _update_account_tracker for the write side). A window
    with no snapshot far enough back yet (e.g. asking for "month" with only
    4 days of history) returns None for that window rather than silently
    duplicating all-time's number."""
    db = load_json(ACCOUNT_TRACKER_PATH, {"periodStart": None, "snapshots": []})
    snaps = db.get("snapshots", [])
    if len(snaps) < 1:
        return None
    by_date = {s["date"]: s for s in snaps}
    dates = sorted(by_date.keys())
    last = by_date[dates[-1]]
    last_date = datetime.strptime(dates[-1], "%Y-%m-%d").date()

    def pct(a, b):
        return round((b - a) / a * 100, 2) if a else None

    def window(days_back):
        if days_back is None:
            base_date = dates[0]
        else:
            target = last_date - timedelta(days=days_back)
            candidates = [d for d in dates if d <= str(target)]
            if not candidates:
                return None
            base_date = candidates[-1]
        base = by_date[base_date]
        n_trading_days = dates.index(dates[-1]) - dates.index(base_date)
        return {
            "fromDate": base_date, "tradingDays": n_trading_days,
            "accountReturnPct": pct(base["accountValue"], last["accountValue"]),
            "tsxReturnPct": pct(base["tsxClose"], last["tsxClose"]),
            "tsx60ReturnPct": pct(base["tsx60Close"], last["tsx60Close"]),
        }

    windows = {name: window(days_back) for name, days_back in ACCOUNT_TRACKER_WINDOWS}
    return {
        "periodStart": db.get("periodStart"),
        "asOf": dates[-1],
        "days": len(dates),
        "windows": windows,
        "snapshots": snaps,
    }


def _resolve_holding_price(ticker, pos, wl_map, details, dream_map):
    """Best-available current price for a held ticker: prefer the watchlist
    price when fresher than the Fetch Info detail price, else fall back
    through detail -> dream -> purchase price."""
    detail = details.get(ticker, {})
    d_data = dream_map.get(ticker, {})
    wl_entry = wl_map.get(ticker, {})
    wl_price = wl_entry.get("price")
    wl_ts = wl_entry.get("priceFetchedAt") or ""
    det_price = detail.get("price")
    det_ts = detail.get("fetchCompletedAt") or ""
    if wl_price and (not det_price or wl_ts >= det_ts):
        return wl_price
    return det_price or d_data.get("price") or pos.get("purchasePrice")


def _snapshot_total_asset():
    """Whole-account equity right now: cash balance + market value of every
    currently open position, regardless of which version bought it. Uses
    cached prices only (analysis.json / Fetch Info detail / dream) — no live
    network call, safe to run on every version bump and every stats request."""
    trades = load_trades_ai()
    holdings = trades.get("holdings", {})
    balance = trades.get("balance", STARTING_BALANCE)
    candidates = load_json(TRADE_AI_CANDIDATES_PATH, {})
    details = candidates.get("details", {})
    dream = load_json(DREAM_PATH, {})
    dream_map = {c["ticker"]: c for c in dream.get("candidates", [])}
    analysis = load_json(OUTPUT_PATH, {})
    wl_map = {s.get("ticker"): s for s in analysis.get("watchlist", [])}
    value = balance
    for ticker, pos in holdings.items():
        price = _resolve_holding_price(ticker, pos, wl_map, details, dream_map) or pos.get("purchasePrice") or 0
        value += price * (pos.get("shares") or 0)
    return round(value, 2)


def _version_asset_stats(holdings, price_map, current_version):
    """Dollar-weighted, per-version account impact under a "management
    period" model: a version owns whatever happens to the book while it's
    the live version, not just the trades it personally opened.

    - Closed trades count toward whichever version was ACTIVE WHEN THE SELL
      FIRED (sellRecVersion) — a position bought under v1 and stopped out
      today under v4's thresholds is v4's exit, not v1's.
    - Every position still open right now counts toward the CURRENT version
      — it inherited the whole book at version-start and is responsible for
      continuing to hold (or not hold) every position in it, regardless of
      who originally bought it.
    - A retired version's contribution is therefore frozen the moment the
      next version takes over: whatever hadn't been sold yet transferred
      to the new version's book, so a retired version's own totalReturnPct
      never changes again once superseded.

    totalReturnPct = totalPnl / totalCost, matching "compare total account
    asset after N transactions" — not a per-trade average."""
    outcomes = load_json(TRADE_OUTCOMES_PATH, {"outcomes": []}).get("outcomes", [])
    closed_by_v = {}
    for o in outcomes:
        v = o.get("sellRecVersion", o.get("recVersion", 1))
        e = closed_by_v.setdefault(v, {"pnl": 0.0, "cost": 0.0, "n": 0})
        e["pnl"]  += o.get("gainLoss") or 0
        e["cost"] += (o.get("buyPrice") or 0) * (o.get("shares") or 0)
        e["n"]    += 1

    open_total = {"pnl": 0.0, "cost": 0.0, "n": 0}
    for ticker, pos in holdings.items():
        price = price_map.get(ticker) or pos.get("purchasePrice")
        buy = pos.get("purchasePrice") or 0
        shares = pos.get("shares") or 0
        if not price or not buy:
            continue
        open_total["pnl"]  += (price - buy) * shares
        open_total["cost"] += buy * shares
        open_total["n"]    += 1

    result = {}
    for v in set(closed_by_v) | {current_version}:
        c = closed_by_v.get(v, {"pnl": 0.0, "cost": 0.0, "n": 0})
        o = open_total if v == current_version else {"pnl": 0.0, "cost": 0.0, "n": 0}
        total_pnl  = c["pnl"] + o["pnl"]
        total_cost = c["cost"] + o["cost"]
        result[v] = {
            "closedPnl": round(c["pnl"], 2), "closedCost": round(c["cost"], 2), "closedN": c["n"],
            "openPnl": round(o["pnl"], 2), "openCost": round(o["cost"], 2), "openN": o["n"],
            "totalPnl": round(total_pnl, 2), "totalCost": round(total_cost, 2),
            "totalN": c["n"] + o["n"],
            "totalReturnPct": round(total_pnl / total_cost * 100, 2) if total_cost > 0 else None,
        }
    return result


def _version_vs_index_stats(versions_db, live_prices=None):
    """TSX Composite/TSX 60 return over the EXACT date range each version
    was live — the per-version complement to accountReturnPct/
    _version_asset_stats. Answers "is THIS version beating the index," not
    "is the whole account beating the index since some arbitrary reset
    point" (which a reset invalidates every time it happens). Uses
    tsxAtStart/tsx60AtStart captured once at version-creation time (see
    bump_recommend_version) — a retired version's window is therefore
    frozen forever once the next version starts, same as assetAtEnd; only
    the current (last) version's end boundary is live. (Ported from
    stock-analysis, adapted to TSX/TSX60 in place of SPY/Nasdaq.)"""
    versions = versions_db.get("versions", [])
    if not versions:
        return {}
    live_prices = live_prices or {}
    tsx_now = live_prices.get(BENCHMARK_TICKERS["TSX"])
    tsx60_now = live_prices.get(BENCHMARK_TICKERS["TSX60"])
    result = {}
    for i, v in enumerate(versions):
        tsx_start, tsx60_start = v.get("tsxAtStart"), v.get("tsx60AtStart")
        if not tsx_start or not tsx60_start:
            continue  # predates this tracking — nothing to compare
        is_current = (i == len(versions) - 1)
        if is_current:
            tsx_end, tsx60_end = tsx_now, tsx60_now
        else:
            nxt = versions[i + 1]
            tsx_end, tsx60_end = nxt.get("tsxAtStart"), nxt.get("tsx60AtStart")
        if not tsx_end or not tsx60_end:
            continue
        result[v["version"]] = {
            "tsxReturnPct": round((tsx_end - tsx_start) / tsx_start * 100, 2),
            "tsx60ReturnPct": round((tsx60_end - tsx60_start) / tsx60_start * 100, 2),
        }
    return result


ROLLBACK_PROPOSAL_PATH = "output/rollback_proposal.json"

# Registry of protective changes that apply immediately to whatever version
# is live (entry guards / regime gates added directly to SCORING_DEFAULTS,
# not a scored version bump) — used by _diagnose_version_gap to check
# whether a version's underperformance predates a fix that would already
# stop the same trades from happening today. Append here whenever a new
# safeguard like this ships; nothing else needs to change for the
# diagnostic to pick it up. (Ported from stock-analysis; dates below are
# when each safeguard shipped there — this repo just got them.)
SAFEGUARD_LOG = [
    {"addedAt": "2026-09-17", "description": "FOMC blackout gate — blocks new buys within fomcBlackoutDays of a scheduled Fed decision"},
    {"addedAt": "2026-09-17", "description": "Already-down-today entry guard — refuses to buy a ticker already deep into a bad day"},
    {"addedAt": "2026-09-17", "description": "Overextension penalty — dampens Accelerating + RSI≥75 + 30d change≥25% picks"},
    {"addedAt": "2026-09-17", "description": "Fed Funds Rate hike trigger added to the macro risk-off gate"},
]


def _diagnose_version_gap(version, outcomes):
    """Best-effort automated diagnosis of WHY a version's return looks bad —
    the same kind of investigation a human should do before trusting a
    rollback signal: is the underperformance a broad, structural pattern
    across the whole sample, or a correlated cluster of losses from a
    narrow window (a known event, a bug since fixed) that isn't actually
    evidence against the version's scoring logic itself? Attached to the
    rollback proposal so every proposal carries its own "why", not just a
    bottom-line number.

    Trades are attributed to `version` the same way _version_asset_stats
    does — by sellRecVersion (who was live when the position closed), not
    who originally bought it."""
    v_outs = [o for o in outcomes if o.get("sellRecVersion", o.get("recVersion", 1)) == version]
    losers = [o for o in v_outs if (o.get("gainLossPct") or 0) <= 0 and o.get("sellDate")]
    if len(losers) < 2:
        return None  # not enough losing trades to look for a pattern
    losers.sort(key=lambda o: o["sellDate"])
    total_loss_pct = sum(o.get("gainLossPct") or 0 for o in losers)  # negative
    if total_loss_pct >= 0:
        return None

    # ── Loss-cluster detection: shortest contiguous date window whose
    # losing trades account for >=60% of this version's total loss. ──
    dates = sorted({o["sellDate"][:10] for o in losers})
    best = None
    for i in range(len(dates)):
        for j in range(i, len(dates)):
            window = set(dates[i:j + 1])
            w_losers = [o for o in losers if o["sellDate"][:10] in window]
            w_loss = sum(o.get("gainLossPct") or 0 for o in w_losers)
            if w_loss <= total_loss_pct * 0.6:  # both negative — captures 60%+ of the loss
                span = (datetime.strptime(dates[j], "%Y-%m-%d") - datetime.strptime(dates[i], "%Y-%m-%d")).days + 1
                if best is None or span < best["spanDays"] or (span == best["spanDays"] and len(w_losers) > len(best["trades"])):
                    best = {"start": dates[i], "end": dates[j], "spanDays": span,
                            "trades": w_losers, "lossPct": round(w_loss, 2)}
                break  # smallest j for this i that clears the bar; larger j only widens the window
    if not best:
        return None

    cluster = best["trades"]
    cluster_tickers = [o["ticker"] for o in cluster]
    share_of_total_loss = round(best["lossPct"] / total_loss_pct * 100, 1)

    # ── Idiosyncratic vs. broad-market: did SPY actually fall too? ──
    spy_vals = [o.get("spyReturnPct") for o in cluster if o.get("spyReturnPct") is not None]
    own_vals = [o.get("gainLossPct") for o in cluster if o.get("gainLossPct") is not None]
    cluster_spy_avg = round(sum(spy_vals) / len(spy_vals), 2) if spy_vals else None
    cluster_own_avg = round(sum(own_vals) / len(own_vals), 2) if own_vals else None
    idiosyncratic = (cluster_spy_avg is not None and cluster_own_avg is not None
                      and abs(cluster_spy_avg) < abs(cluster_own_avg) * 0.3)

    # ── FOMC proximity — only meaningful for dates on/after the earliest
    # entry in FOMC_MEETING_DATES; older trades simply can't be checked. ──
    fomc_near = 0
    for o in cluster:
        buy_date_str = (o.get("buyDate") or "")[:10]
        if not buy_date_str:
            continue
        try:
            bd = datetime.strptime(buy_date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        for fd_str in FOMC_MEETING_DATES:
            fd = datetime.strptime(fd_str, "%Y-%m-%d").date()
            if abs((fd - bd).days) <= 2:
                fomc_near += 1
                break

    # ── Shared factor overrepresentation: Accelerating trajectory, high
    # composite — cheap signal for "these were all the same kind of pick." ──
    rest = [o for o in v_outs if o not in cluster]
    def _traj_share(rows):
        tagged = [o for o in rows if (o.get("factors") or {}).get("trajectory")]
        if not tagged:
            return None
        return round(sum(1 for o in tagged if o["factors"]["trajectory"] == "Accelerating") / len(tagged) * 100, 0)
    cluster_accel_pct = _traj_share(cluster)
    rest_accel_pct = _traj_share(rest)

    # ── Safeguards shipped since — does a fix already exist for this? ──
    relevant_safeguards = []
    for sg in SAFEGUARD_LOG:
        added = sg["addedAt"]
        predates = sum(1 for o in cluster if (o.get("buyDate") or "")[:10] < added)
        if predates:
            relevant_safeguards.append({**sg, "clusterTradesBeforeIt": predates})

    lines = [
        f"{share_of_total_loss}% of v{version}'s total loss came from {len(cluster)} trade(s) "
        f"between {best['start']} and {best['end']} ({', '.join(cluster_tickers)})."
    ]
    if idiosyncratic:
        lines.append(f"SPY averaged {cluster_spy_avg:+}% on those trades' sell dates while these lost {cluster_own_avg:+}% on average — "
                      f"idiosyncratic to these picks, not a broad market selloff.")
    elif cluster_spy_avg is not None:
        lines.append(f"SPY also averaged {cluster_spy_avg:+}% on those dates — part of a broader market move, not isolated to these picks.")
    if fomc_near:
        lines.append(f"{fomc_near}/{len(cluster)} of those trades were bought within 2 days of a scheduled FOMC decision.")
    if cluster_accel_pct is not None and rest_accel_pct is not None and cluster_accel_pct > rest_accel_pct:
        lines.append(f"{cluster_accel_pct:.0f}% of the cluster's trades were 'Accelerating' trajectory at buy time, vs {rest_accel_pct:.0f}% for this version's other trades.")
    if relevant_safeguards:
        lines.append("Since then, safeguards have shipped that would apply to new trades like these: " +
                      "; ".join(f"{sg['description']} (added {sg['addedAt']}, predates {sg['clusterTradesBeforeIt']} of these trades)" for sg in relevant_safeguards) + ".")

    return {
        "clusterStart": best["start"], "clusterEnd": best["end"], "clusterSpanDays": best["spanDays"],
        "clusterTickers": cluster_tickers, "clusterN": len(cluster), "clusterLossPct": best["lossPct"],
        "shareOfTotalLossPct": share_of_total_loss,
        "clusterSpyAvgPct": cluster_spy_avg, "clusterOwnAvgPct": cluster_own_avg, "idiosyncratic": idiosyncratic,
        "fomcNearCount": fomc_near,
        "clusterAcceleratingPct": cluster_accel_pct, "restAcceleratingPct": rest_accel_pct,
        "safeguardsSince": relevant_safeguards,
        "summary": " ".join(lines),
    }


def _check_rollback_candidate(sl_cfg, holdings=None, live_prices=None):
    """Champion/challenger check: once the current version has enough total
    buy transactions (closed + still-open) to trust the comparison, does its
    dollar-weighted asset return lag the best-proven prior version (the
    champion) by more than the threshold? If so, record a PROPOSAL for the
    user to review and approve in the TradeAI account area — this never
    changes the live config on its own. Only /tradeai/rollback/approve does
    that. The champion itself is only ever replaced by a version that beats
    it on sufficient data — the ratchet moves up, never back and forth."""
    min_n  = int(sl_cfg.get("minTrades", 10))
    thresh = float(sl_cfg.get("rollbackThresholdPct", 1.0))
    db = load_recommend_versions()
    cur = db["versions"][-1]
    stats_map = _version_asset_stats(holdings or {}, live_prices or {}, cur["version"])
    cur_s = stats_map.get(cur["version"])
    if not cur_s or cur_s["totalN"] < min_n or cur_s["totalReturnPct"] is None:
        return None  # still testing — not enough evidence to judge

    prior = [v for v in db["versions"][:-1]
             if v.get("configSnapshot")
             and stats_map.get(v["version"])
             and stats_map[v["version"]]["totalN"] >= min_n
             and stats_map[v["version"]]["totalReturnPct"] is not None]
    if not prior:
        return None
    champ = max(prior, key=lambda v: stats_map[v["version"]]["totalReturnPct"])
    champ_s = stats_map[champ["version"]]
    gap = champ_s["totalReturnPct"] - cur_s["totalReturnPct"]

    existing_proposal = load_json(ROLLBACK_PROPOSAL_PATH, None)
    if gap <= thresh:
        # No longer underperforming (e.g. an open position swung back) —
        # clear a stale pending proposal so it doesn't linger after the
        # numbers improved.
        if (existing_proposal and existing_proposal.get("status") == "pending"
                and existing_proposal.get("currentVersion") == cur["version"]):
            save_json(ROLLBACK_PROPOSAL_PATH, {"status": "none"})
        return None

    already_dismissed = (existing_proposal and existing_proposal.get("status") == "dismissed"
                          and existing_proposal.get("currentVersion") == cur["version"]
                          and existing_proposal.get("championVersion") == champ["version"])
    if already_dismissed:
        return None  # user already said no to this exact comparison

    already_pending = (existing_proposal and existing_proposal.get("status") == "pending"
                        and existing_proposal.get("currentVersion") == cur["version"]
                        and existing_proposal.get("championVersion") == champ["version"])
    if already_pending:
        return existing_proposal

    outcomes = load_json(TRADE_OUTCOMES_PATH, {"outcomes": []}).get("outcomes", [])
    diagnostics = None
    try:
        diagnostics = _diagnose_version_gap(cur["version"], outcomes)
    except Exception as de:
        log.warning(f"[SelfLearning] diagnostics FAIL for v{cur['version']}: {de}")

    proposal = {
        "status": "pending",
        "currentVersion": cur["version"], "currentLabel": cur.get("label"),
        "championVersion": champ["version"], "championLabel": champ.get("label"),
        "currentReturnPct": cur_s["totalReturnPct"], "currentN": cur_s["totalN"],
        "championReturnPct": champ_s["totalReturnPct"], "championN": champ_s["totalN"],
        "gapPct": round(gap, 2), "thresholdPct": thresh,
        "detectedAt": ts(),
        "diagnostics": diagnostics,
    }
    save_json(ROLLBACK_PROPOSAL_PATH, proposal)
    log.warning(f"[SelfLearning] ROLLBACK PROPOSED: v{cur['version']} ({cur_s['totalReturnPct']}%) vs "
                f"champion v{champ['version']} ({champ_s['totalReturnPct']}%) — gap {round(gap,2)}% "
                f"exceeds {thresh}% — awaiting approval")
    return proposal

TUNING_PROPOSAL_PATH = "output/tuning_proposal.json"

def _check_tuning_candidate(min_gainers=10, gain_threshold=5.0, window_days=30):
    """Compare tickers the shadow log shows had a real 5%+ day against what
    actually got bought, on the factors behind each configurable Identify
    gate (Momentum RSI band, data-completeness floor). If relaxing one gate
    would provably have let a meaningful number of those missed gainers
    through, propose that ONE relaxation — never applied automatically, see
    /tradeai/tuning/approve. Forecast is honest counts from real history
    (how many missed gainers would now qualify), not a fabricated P&L
    number — there isn't enough data for that kind of claim yet."""
    db = load_json(SHADOW_PATH, {"snapshots": []})
    cutoff = datetime.now(TZ).date() - timedelta(days=window_days)
    missed, bought_n = [], 0
    for snap in db.get("snapshots", []):
        try:
            snap_date = datetime.strptime(snap["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        if snap_date < cutoff:
            continue
        for r in snap.get("rows", []):
            if r.get("maxDailyGainPct") is None:
                continue  # not graded yet
            if r.get("outcome") == "bought":
                bought_n += 1
            elif r["maxDailyGainPct"] >= gain_threshold:
                missed.append(r)

    if len(missed) < min_gainers:
        return None  # not enough evidence yet

    ir          = load_scoring_config().get("identifyRules", {})
    cur_rsi_min = float(ir.get("momentumRsiMin", 45))
    cur_rsi_max = float(ir.get("momentumRsiMax", 70))
    cur_floor   = float(ir.get("dataCompletenessFloor", 0.6))
    catch_bar   = max(3, len(missed) * 0.2)

    rule_candidates = []
    for new_max in [72, 75, 80, 85]:
        catch = [r for r in missed
                 if r.get("rsi") is not None and r.get("aboveMa50") is True
                 and cur_rsi_min <= r["rsi"] <= new_max
                 and not (cur_rsi_min <= r["rsi"] <= cur_rsi_max)]
        if len(catch) >= catch_bar:
            rule_candidates.append({
                "path": "identifyRules.momentumRsiMax", "from": cur_rsi_max, "to": new_max,
                "wouldCatch": len(catch),
                "reason": f"{len(catch)} of {len(missed)} missed 5%+ gainers had RSI between "
                          f"{cur_rsi_max} and {new_max} (above MA50) — currently excluded by the "
                          f"Momentum RSI cap.",
            })
            break  # least-aggressive value that clears the bar
    for new_min in [40, 35, 30]:
        catch = [r for r in missed
                 if r.get("rsi") is not None and r.get("aboveMa50") is True
                 and new_min <= r["rsi"] <= cur_rsi_max
                 and not (cur_rsi_min <= r["rsi"] <= cur_rsi_max)]
        if len(catch) >= catch_bar:
            rule_candidates.append({
                "path": "identifyRules.momentumRsiMin", "from": cur_rsi_min, "to": new_min,
                "wouldCatch": len(catch),
                "reason": f"{len(catch)} of {len(missed)} missed 5%+ gainers had RSI between "
                          f"{new_min} and {cur_rsi_min} (above MA50) — currently excluded by the "
                          f"Momentum RSI floor.",
            })
            break
    for new_floor in [0.5, 0.4, 0.3]:
        catch = [r for r in missed
                 if r.get("outcome") == "scanned_not_identified"
                 and r.get("dataCompleteness") is not None
                 and new_floor <= r["dataCompleteness"] < cur_floor]
        if len(catch) >= catch_bar:
            rule_candidates.append({
                "path": "identifyRules.dataCompletenessFloor", "from": cur_floor, "to": new_floor,
                "wouldCatch": len(catch),
                "reason": f"{len(catch)} of {len(missed)} missed 5%+ gainers were excluded entirely "
                          f"by the data-completeness floor ({cur_floor}) before any bucket even saw "
                          f"them. They'd still need to pass a bucket's own criteria — this only "
                          f"removes the completeness block.",
            })
            break

    if not rule_candidates:
        return None
    best = max(rule_candidates, key=lambda c: c["wouldCatch"])

    proposal_key = f"{best['path']}={best['to']}"
    existing = load_json(TUNING_PROPOSAL_PATH, None)
    if existing and existing.get("status") == "dismissed" and existing.get("proposalKey") == proposal_key:
        return None  # user already said no to this exact change
    if existing and existing.get("status") == "pending" and existing.get("proposalKey") == proposal_key:
        return existing  # already surfaced, don't spam a duplicate

    proposal = {
        "status": "pending",
        "proposalKey": proposal_key,
        "detectedAt": ts(),
        "windowDays": window_days,
        "gainThresholdPct": gain_threshold,
        "missedGainersAnalyzed": len(missed),
        "boughtRowsAnalyzed": bought_n,
        "proposedChange": {"path": best["path"], "from": best["from"], "to": best["to"]},
        "reason": best["reason"],
        "forecast": {
            "historicalMissedGainersWouldHaveCaught": best["wouldCatch"],
            "historicalMissedGainersTotal": len(missed),
            # Every rule in this catalog only ever ADMITS candidates it
            # previously excluded (a raised/lowered RSI bound, a lowered
            # floor) — none can exclude something that already qualified,
            # so there is no "cost" side to this forecast by construction.
            "currentPicksWouldHaveExcluded": 0,
        },
    }
    save_json(TUNING_PROPOSAL_PATH, proposal)
    log.warning(f"[Tuning] PROPOSED: {best['path']} {best['from']} -> {best['to']} "
                f"— would have caught {best['wouldCatch']}/{len(missed)} missed gainers — awaiting approval")
    return proposal


def _missed_gainers_summary(window_days=30, gain_threshold=5.0, limit=25):
    """Top non-bought tickers from the last N days that had a real 5%+ day —
    the same pool _check_tuning_candidate reasons over, surfaced directly in
    the Validation tab so you can see WHICH tickers and WHY they were missed,
    not just the aggregate proposal."""
    db = load_json(SHADOW_PATH, {"snapshots": []})
    cutoff = datetime.now(TZ).date() - timedelta(days=window_days)
    rows = []
    for snap in db.get("snapshots", []):
        try:
            snap_date = datetime.strptime(snap["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        if snap_date < cutoff:
            continue
        for r in snap.get("rows", []):
            if (r.get("outcome") != "bought" and r.get("maxDailyGainPct") is not None
                    and r["maxDailyGainPct"] >= gain_threshold):
                rows.append({**r, "snapshotDate": snap["date"]})
    rows.sort(key=lambda r: r["maxDailyGainPct"], reverse=True)
    return rows[:limit]


WEIGHT_BACKTEST_PATH = "output/weight_backtest.json"

# Full 0/100 -> 100/0 sweep in 10pt steps. Adjacent splits within the
# dream-dominant range pick nearly identical tickers (~5.9/6 overlap,
# checked 2026-09-25), so finer steps would only add noise.
WEIGHT_BACKTEST_CANDIDATES = [(d / 10, (10 - d) / 10) for d in range(11)]

# Date the current AI score formula went live. Older shadow rows carry a
# different AI score (Gemini until 2026-08-24, then a rule formula that
# also scored fundamentals), so judging today's weighting on them would
# be judging a different score. Bump this whenever _rule_based_assessment
# changes.
AI_SCORE_FORMULA_SINCE = "2026-09-26"


def _rebuild_ai_scores(bars=42):
    """Backfill `aiScoreRebuilt` on identified+ shadow rows dated before
    AI_SCORE_FORMULA_SINCE: the CURRENT _rule_based_assessment, fed price
    indicators rebuilt from daily bars strictly BEFORE each snapshot date
    (no lookahead; same fetcher.price_indicators math the live fetch uses,
    over ~the same 60-calendar-day window). Analyst upside and news can't
    be rebuilt — Yahoo only has today's analyst target and old headlines
    were never stored — so both count as neutral. Kept separate from
    `aiScore` so live-recorded scores stay pure, and tagged with the
    formula date so a future formula change triggers a fresh rebuild."""
    import yfinance as yf
    import pandas as pd
    from fetcher import price_indicators

    db = load_json(SHADOW_PATH, {"snapshots": []})
    todo = [(snap["date"], r) for snap in db.get("snapshots", [])
            if snap.get("date", "") < AI_SCORE_FORMULA_SINCE
            for r in snap.get("rows", [])
            if r.get("composite") is not None and r.get("aiScoreRebuiltFor") != AI_SCORE_FORMULA_SINCE]
    if not todo:
        return {"rebuilt": 0, "noHistory": 0}

    tickers = sorted({r["ticker"] for _, r in todo})
    start = (datetime.strptime(min(d for d, _ in todo), "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")
    end = max(d for d, _ in todo)  # yfinance end is exclusive — only bars before the last snapshot are needed
    hist_by = {}
    for i in range(0, len(tickers), 40):
        chunk = tickers[i:i + 40]
        try:
            df = yf.download(chunk, start=start, end=end, group_by="ticker",
                             auto_adjust=True, progress=False, threads=True)
        except Exception as e:
            log.warning(f"[AIScoreRebuild] download FAIL for {chunk[0]}..: {e}")
            continue
        for t in chunk:
            try:
                h = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                h = h.dropna(subset=["Close"])
                if h.index.tz is not None:
                    h.index = h.index.tz_localize(None)
                if not h.empty:
                    hist_by[t] = h
            except KeyError:
                pass

    rebuilt = no_hist = 0
    for date, r in todo:
        h = hist_by.get(r["ticker"])
        h = h[h.index < pd.Timestamp(date)].tail(bars) if h is not None else None
        if h is None or len(h) < 30:
            r["aiScoreRebuilt"] = None
            no_hist += 1
        else:
            ind = price_indicators(h, r["ticker"])
            detail = {k: ind[k] for k in ("rsi14", "bbPercent", "volumeRatio", "change1d", "change7d", "change30d")}
            r["aiScoreRebuilt"] = _rule_based_assessment(r["ticker"], detail, [], r.get("bucket") or "Dream")["aiScore"]
            rebuilt += 1
        r["aiScoreRebuiltFor"] = AI_SCORE_FORMULA_SINCE
    save_json(SHADOW_PATH, db)
    log.info(f"[AIScoreRebuild] rebuilt {rebuilt} rows, {no_hist} without enough price history ({len(tickers)} tickers)")
    return {"rebuilt": rebuilt, "noHistory": no_hist, "tickers": len(tickers)}

FIXED_HORIZONS = [5, 10, 20]  # holding periods in trading days


def _grade_fixed_horizons():
    """Fill r["fwdH"] = {"5": pct, "10": pct, "20": pct} on identified+
    shadow rows: return from the snapshot price to the close N trading days
    later. fwdReturnPct is graded once, whenever the grading job first
    reaches a snapshot (9 to 50 days later in practice), so days measured
    different holding periods and couldn't be compared fairly — every row
    here is measured over the same holds. Today's bar is never used (it
    may be incomplete), and a horizon is only filled once it has fully
    elapsed, so values never change after being written."""
    import yfinance as yf
    import pandas as pd

    db = load_json(SHADOW_PATH, {"snapshots": []})
    todo = [(snap["date"], r) for snap in db.get("snapshots", []) for r in snap.get("rows", [])
            if r.get("composite") is not None and r.get("price") and not r.get("fwdHNoData")
            and not all(str(h) in (r.get("fwdH") or {}) for h in FIXED_HORIZONS)]
    if not todo:
        return {"graded": 0}

    tickers = sorted({r["ticker"] for _, r in todo})
    start = min(d for d, _ in todo)
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    closes = {}
    for i in range(0, len(tickers), 40):
        chunk = tickers[i:i + 40]
        try:
            df = yf.download(chunk, start=start, end=today, group_by="ticker",
                             auto_adjust=True, progress=False, threads=True)
        except Exception as e:
            log.warning(f"[FixedHorizon] download FAIL for {chunk[0]}..: {e}")
            continue
        for t in chunk:
            try:
                c = (df[t] if isinstance(df.columns, pd.MultiIndex) else df)["Close"].dropna()
                if c.index.tz is not None:
                    c.index = c.index.tz_localize(None)
                closes[t] = c
            except KeyError:
                closes[t] = None

    graded = 0
    for date, r in todo:
        if r["ticker"] not in closes:
            continue  # download failed this run — retry next time
        c = closes[r["ticker"]]
        if c is None or c.empty:
            r["fwdHNoData"] = True
            continue
        after = c[c.index > pd.Timestamp(date)]
        fwd_h = r.setdefault("fwdH", {})
        for h in FIXED_HORIZONS:
            if str(h) not in fwd_h and len(after) >= h:
                fwd_h[str(h)] = round((float(after.iloc[h - 1]) - r["price"]) / r["price"] * 100, 2)
                graded += 1
    save_json(SHADOW_PATH, db)
    log.info(f"[FixedHorizon] filled {graded} horizon values across {len(todo)} rows")
    return {"graded": graded}


def _backtest_bucket_weights(top_n=6):
    """Day-by-day A/B test: for each shadow snapshot, re-rank that day's
    identified-or-further candidate pool by each dream/ai split and compare
    the top-N picks' return against what the CURRENT formula would have
    picked that same day, at each fixed holding period (FIXED_HORIZONS,
    buy-and-hold — stops and rotations are not simulated). "Current" uses
    each row's own bucket weights; the alternative applies the same split
    to every bucket.

    Only judges the current AI score formula: days on/after
    AI_SCORE_FORMULA_SINCE use the live aiScore, earlier days use
    aiScoreRebuilt (no analyst/news parts). Older recorded aiScores
    (Gemini, prior rule formula) are never used. Refreshes the rebuilt
    scores and fixed-horizon returns first (network), then computes."""
    for step in (_rebuild_ai_scores, _grade_fixed_horizons):
        try:
            step()
        except Exception as e:
            log.warning(f"[WeightBacktest] {step.__name__} FAIL, using existing data: {e}")

    bw = load_scoring_config().get("bucketWeights", {})
    db = load_json(SHADOW_PATH, {"snapshots": []})

    def current_score(r):
        w = bw.get(r["bucket"], {"dream": 0.2, "ai": 0.6})
        return r["dream"] * w.get("dream", 0.2) + r["ai"] * w.get("ai", 0.6)

    days = []
    for snap in db.get("snapshots", []):
        live = snap.get("date", "") >= AI_SCORE_FORMULA_SINCE
        ai_key = "aiScore" if live else "aiScoreRebuilt"
        rows = [{"bucket": r["bucket"], "dream": r["dreamScore"], "ai": r[ai_key], "fwdH": r.get("fwdH") or {}}
                for r in snap.get("rows", [])
                if r.get("composite") is not None and r.get(ai_key) is not None
                and r.get("dreamScore") is not None and r.get("bucket") in bw]
        days.append((snap["date"], rows, live))

    median_top = lambda rows, key, h: statistics.median([r["fwdH"][h] for r in sorted(rows, key=key, reverse=True)[:top_n]])
    horizons = []
    for hz in FIXED_HORIZONS:
        h = str(hz)
        usable = [(d, [r for r in rows if h in r["fwdH"]], live) for d, rows, live in days]
        usable = [u for u in usable if len(u[1]) >= top_n]
        if not usable:
            horizons.append({"h": hz, "days": 0})
            continue
        cur = [median_top(rows, current_score, h) for _, rows, _ in usable]
        pool = [statistics.median([r["fwdH"][h] for r in rows]) for _, rows, _ in usable]
        variants = []
        for ad, aa in WEIGHT_BACKTEST_CANDIDATES:
            alt = [median_top(rows, lambda r: r["dream"] * ad + r["ai"] * aa, h) for _, rows, _ in usable]
            # ties counted separately — both weightings often pick the same top-N
            variants.append({
                "label": f"{round(ad*100)}/{round(aa*100)}", "dream": ad, "ai": aa,
                "w": sum(1 for a, c in zip(alt, cur) if a > c),
                "t": sum(1 for a, c in zip(alt, cur) if a == c),
                "l": sum(1 for a, c in zip(alt, cur) if a < c),
                "avgPct": round(sum(alt) / len(alt), 2),
            })
        horizons.append({
            "h": hz, "days": len(usable), "from": usable[0][0], "to": usable[-1][0],
            "daysLive": sum(1 for u in usable if u[2]),
            "currentAvgPct": round(sum(cur) / len(cur), 2),
            "poolAvgPct": round(sum(pool) / len(pool), 2),
            "variants": variants,
        })

    if horizons[0]["days"] < 5:
        out = {"computedAt": ts(), "insufficient": True, "daysAvailable": horizons[0]["days"],
               "daysNeeded": 5, "since": AI_SCORE_FORMULA_SINCE, "topN": top_n}
    else:
        out = {"computedAt": ts(), "since": AI_SCORE_FORMULA_SINCE, "topN": top_n, "horizons": horizons}
    save_json(WEIGHT_BACKTEST_PATH, out)
    log.info(f"[WeightBacktest] computed over {[(x['h'], x['days']) for x in horizons]} (horizon, days)")
    return out


def _flatten_cfg(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_cfg(v, key))
        else:
            out[key] = v
    return out

def sync_watchlist_to_analysis(existing_watchlist, watchlist):
    """Ensure all tickers in watchlist exist in analysis, never remove existing cards."""
    existing_map = {s["ticker"]: s for s in existing_watchlist}
    for ticker in watchlist:
        if ticker not in existing_map:
            log.info(f"Adding missing ticker to analysis: {ticker}")
            existing_map[ticker] = {"ticker": ticker, "fetchedAt": None}
    return existing_map


def run_price_refresh(triggered_by="startup"):
    global fetch_status, op_timestamps
    fetch_status = {"running": True, "message": f"⚡ Updating prices for all tickers...", "operation": "prices"}
    log.info(f"=== Price Refresh Started [{triggered_by}]: {watchlist} ===")

    # Collect price updates in memory only — don't carry an early snapshot of
    # the rest of the record across this loop's network calls (see _output_lock).
    price_updates = {}
    for ticker in watchlist:
        log.info(f"--- Price refresh: {ticker} ---")
        try:
            price_data = fetch_price_only(ticker)
            if not price_data:
                log.error(f"Price refresh no data: {ticker}")
                continue
            price_updates[ticker] = {
                "ticker": ticker,
                "price": price_data.get("price"),
                "previousClose": price_data.get("previousClose"),
                "volume": price_data.get("volume"),
                "marketCap": price_data.get("marketCap"),
                "rsi14": price_data.get("rsi14"),
                "bbPercent": price_data.get("bbPercent"),
                "priceFetchedAt": ts(),
            }
            log.info(f"Price refresh OK: {ticker} | price:{price_data.get('price')} prevClose:{price_data.get('previousClose')} rsi:{price_data.get('rsi14')} bb:{price_data.get('bbPercent')}")
        except Exception as e:
            log.error(f"Price refresh FAIL: {ticker} | {e}")

    now_ts = ts()
    op_timestamps["prices"] = now_ts

    with _output_lock:
        out = load_json(OUTPUT_PATH, {"watchlist": [], "dailyGainers": []})
        out_map = {s["ticker"]: s for s in out.get("watchlist", [])}
        for ticker in watchlist:
            stock = out_map.get(ticker, {"ticker": ticker, "fetchedAt": None})
            if ticker in price_updates:
                stock.update(price_updates[ticker])
            out_map[ticker] = stock
        out["watchlist"] = [out_map[t] for t in watchlist if t in out_map]
        out["lastPriceRefresh"] = now_ts
        save_json(OUTPUT_PATH, out)

    fetch_status = {"running": False, "message": f"Prices updated: {now_ts}", "operation": None}
    log.info(f"=== Price Refresh Complete [{triggered_by}] ===")


def run_ai_summary(ticker):
    global fetch_status
    fetch_status = {"running": True, "message": f"⚡ Generating AI summary: {ticker}...", "operation": "ai"}
    log.info(f"=== AI Summary Started: {ticker} ===")
    try:
        existing = load_json(OUTPUT_PATH, {"watchlist": []})
        existing_map = {s["ticker"]: s for s in existing.get("watchlist", [])}
        stock = existing_map.get(ticker)
        if not stock:
            log.error(f"AI summary FAIL: {ticker} not found in data")
            fetch_status = {"running": False, "message": f"AI summary FAIL: {ticker} not found.", "operation": None}
            return
        now_ts = ts()
        stock["aiSummary"] = generate_summary(stock)
        stock["aiSummaryUpdatedAt"] = now_ts
        existing_map[ticker] = stock
        out = load_json(OUTPUT_PATH, {})
        out["watchlist"] = [existing_map.get(t, s) for t, s in [(s["ticker"], s) for s in existing.get("watchlist", [])]]
        save_json(OUTPUT_PATH, out)
        fetch_status = {"running": False, "message": f"AI summary done: {ticker} at {now_ts}", "operation": None}
        log.info(f"=== AI Summary Complete: {ticker} ===")
    except Exception as e:
        log.error(f"AI summary FAIL: {ticker} | {e}")
        fetch_status = {"running": False, "message": f"AI summary FAIL: {ticker}", "operation": None}


def run_fetch(tickers_to_fetch, mode="all"):
    global fetch_status, last_fetched_tickers, op_timestamps
    fetch_status = {"running": True, "message": f"[{mode.upper()}] Fetching {len(tickers_to_fetch)} ticker(s)...", "operation": mode}
    log.info(f"=== Fetch Started [{mode}]: {tickers_to_fetch} ===")

    # Collect fetched detail in memory only — merged into a FRESH read of the
    # file at save time (see _output_lock) so a concurrent writer (scheduled
    # sell-check, startup price refresh) can't have its update clobbered by
    # this loop's early snapshot.
    fetched = {}
    for ticker in tickers_to_fetch:
        log.info(f"--- Fetching [{mode}]: {ticker} ---")
        try:
            yf_data = fetch_yfinance(ticker)
            if not yf_data:
                log.error(f"No data returned for {ticker}")
                continue
            fetched[ticker] = {
                "ticker": ticker,
                "fetchedAt": ts(),
                "calendarFetchedAt": ts(),
                "price": yf_data.get("price"),
                "volume": yf_data.get("volume"),
                "marketCap": yf_data.get("marketCap"),
                "sector": yf_data.get("sector"),
                "industry": yf_data.get("industry"),
                "calendar": yf_data.get("calendar", {}),
                "events": yf_data.get("events", {}),
                "dividends": yf_data.get("dividends", []),
                "annualIncome": yf_data.get("annualIncome", []),
                "quarterlyIncome": yf_data.get("quarterlyIncome", []),
                "annualBalance": yf_data.get("annualBalance", []),
                "quarterlyBalance": yf_data.get("quarterlyBalance", []),
                "annualCashflow": yf_data.get("annualCashflow", []),
                "quarterlyCashflow": yf_data.get("quarterlyCashflow", []),
            }
            last_fetched_tickers.add(ticker)
        except Exception as e:
            log.error(f"Fetch FAIL: {ticker} | {e}")

    gainers = fetch_daily_gainers()
    now_ts = ts()
    if mode == "all":
        op_timestamps["all"] = now_ts
    elif mode == "new":
        op_timestamps["new"] = now_ts

    with _output_lock:
        out = load_json(OUTPUT_PATH, {})
        existing_map = reconcile_with_watchlist({s["ticker"]: s for s in out.get("watchlist", [])})
        for ticker, fields in fetched.items():
            stock = existing_map.get(ticker, {"ticker": ticker})
            stock.update(fields)
            existing_map[ticker] = stock

        results = [existing_map[t] for t in watchlist if t in existing_map]
        results = process_watchlist(results)

        log.info(f"Generating recommendations...")
        recommendations = generate_recommendations(results)
        log.info("AI summaries skipped — run per ticker manually.")
        log.info(f"Total before save: {len(results)} | {[r['ticker'] for r in results]}")

        out.update({
            "watchlist": results,
            "dailyGainers": gainers.get("us", []),
            "dailyGainersCDN": gainers.get("cdn", []),
            "recommendations": recommendations,
            "lastUpdated": now_ts,
            "opTimestamps": {**out.get("opTimestamps", {}), **{k: v for k, v in op_timestamps.items() if v}},
        })
        save_json(OUTPUT_PATH, out)

    fetch_status = {"running": False, "message": f"Done! [{mode}] {len(tickers_to_fetch)} ticker(s) updated.", "operation": None}
    log.info("=== Fetch Complete ===")


def run_smart_refresh():
    global fetch_status, op_timestamps
    fetch_status = {"running": True, "message": "Smart refresh: checking what needs updating..."}
    log.info("=== Smart Refresh Started ===")

    existing = load_json(OUTPUT_PATH, {"watchlist": [], "dailyGainers": []})
    existing_map = reconcile_with_watchlist(
    {s["ticker"]: s for s in existing.get("watchlist", [])}
)

    tickers_needing_refresh = []
    refresh_plans = {}

    for ticker in watchlist:
        stock = existing_map.get(ticker, {"ticker": ticker})
        plan = check_what_needs_refresh(stock, None)
        if any([plan["price"], plan["quarterly"], plan["annual"], plan["calendar"]]):
            tickers_needing_refresh.append(ticker)
            refresh_plans[ticker] = plan

    if not tickers_needing_refresh:
        fetch_status = {"running": False, "message": "Smart refresh: everything up to date."}
        log.info("Smart refresh: nothing to update")
        return

    fetch_status = {"running": True, "message": f"Smart refresh: updating {len(tickers_needing_refresh)} ticker(s)..."}
    log.info(f"Smart refresh: {tickers_needing_refresh}")

    # Collect only the CHANGED fields in memory — merged into a FRESH read of
    # the file at save time (see _output_lock) rather than carried forward
    # on the early snapshot across this loop's network calls.
    updates = {}
    for ticker in tickers_needing_refresh:
        plan = refresh_plans[ticker]
        fields = {}
        try:
            yf_data = plan.get("_freshData") or fetch_yfinance(ticker)
            if not yf_data:
                continue
            if plan["price"]:
                fields.update({
                    "fetchedAt": ts(),
                    "price": yf_data.get("price"),
                    "volume": yf_data.get("volume"),
                    "marketCap": yf_data.get("marketCap"),
                })
            if plan["quarterly"]:
                fields["quarterlyIncome"] = yf_data.get("quarterlyIncome", [])
                fields["quarterlyBalance"] = yf_data.get("quarterlyBalance", [])
                fields["quarterlyCashflow"] = yf_data.get("quarterlyCashflow", [])
            if plan["annual"]:
                fields["annualIncome"] = yf_data.get("annualIncome", [])
                fields["annualBalance"] = yf_data.get("annualBalance", [])
                fields["annualCashflow"] = yf_data.get("annualCashflow", [])
            if plan["calendar"]:
                fields["calendar"] = yf_data.get("calendar", {})
                fields["events"] = yf_data.get("events", {})
                fields["dividends"] = yf_data.get("dividends", [])
                fields["calendarFetchedAt"] = ts()
            updates[ticker] = fields
        except Exception as e:
            log.error(f"Smart refresh FAIL: {ticker} | {e}")

    gainers = fetch_daily_gainers()
    now_ts = ts()
    op_timestamps["smart"] = now_ts

    with _output_lock:
        out = load_json(OUTPUT_PATH, {})
        existing_map = reconcile_with_watchlist({s["ticker"]: s for s in out.get("watchlist", [])})
        for ticker, fields in updates.items():
            stock = existing_map.get(ticker, {"ticker": ticker})
            stock.update(fields)
            existing_map[ticker] = stock

        results = list(existing_map.values())
        results = process_watchlist(results)
        recommendations = generate_recommendations(results)
        log.info("Smart refresh: AI summaries skipped — run per ticker manually.")

        out.update({
            "watchlist": results,
            "dailyGainers": gainers or out.get("dailyGainers", []),
            "recommendations": recommendations,
            "lastUpdated": now_ts,
            "opTimestamps": {**out.get("opTimestamps", {}), **{k: v for k, v in op_timestamps.items() if v}},
        })
        save_json(OUTPUT_PATH, out)

    fetch_status = {"running": False, "message": f"Smart refresh done: {len(tickers_needing_refresh)} ticker(s) updated.", "operation": None}
    log.info("=== Smart Refresh Complete ===")

def reconcile_with_watchlist(existing_map):
    """Remove tickers from existing_map that are no longer in watchlist."""
    removed = [t for t in list(existing_map.keys()) if t not in watchlist]
    for t in removed:
        del existing_map[t]
        log.info(f"Reconcile: removed {t} (no longer in watchlist)")
    added = [t for t in watchlist if t not in existing_map]
    for t in added:
        existing_map[t] = {"ticker": t, "fetchedAt": None}
        log.info(f"Reconcile: added placeholder for {t}")
    return existing_map

@app.route("/fetch")
def trigger_fetch():
    if not fetch_status["running"]:
        thread = threading.Thread(target=run_fetch, args=(list(watchlist), "all"))
        thread.start()
    return redirect("/")


@app.route("/fetch/new")
def trigger_fetch_new():
    if not fetch_status["running"]:
        existing = load_json(OUTPUT_PATH, {"watchlist": []})
        existing_tickers = {s["ticker"] for s in existing.get("watchlist", [])}
        new_tickers = [t for t in watchlist if t not in existing_tickers]
        log.info(f"Fetch New: new={new_tickers}")
        thread = threading.Thread(target=run_fetch, args=(new_tickers or [], "new"))
        thread.start()
    return redirect("/")


@app.route("/fetch/prices")
def trigger_fetch_prices():
    if not fetch_status["running"]:
        thread = threading.Thread(target=run_price_refresh, args=("manual",))
        thread.start()
    return redirect("/")


@app.route("/fetch/smart")
def trigger_smart_refresh():
    if not fetch_status["running"]:
        thread = threading.Thread(target=run_smart_refresh)
        thread.start()
    return redirect("/")


@app.route("/fetch/ticker/<ticker>")
def trigger_fetch_ticker(ticker):
    if not fetch_status["running"] and ticker in watchlist:
        thread = threading.Thread(target=run_fetch, args=([ticker], "single"))
        thread.start()
    return "", 204

@app.route("/fetch/ticker/<ticker>/ai")
def trigger_fetch_ticker_ai(ticker):
    if not fetch_status["running"] and ticker in watchlist:
        thread = threading.Thread(target=run_ai_summary, args=(ticker,))
        thread.start()
    return redirect("/")

@app.route("/status")
def status():
    out = load_json(OUTPUT_PATH, {})
    return jsonify({
        **fetch_status,
        "opTimestamps": out.get("opTimestamps", {}),
        "lastPriceRefresh": out.get("lastPriceRefresh"),
    })

@app.route("/stop", methods=["POST"])
def stop_operation():
    global stop_flag
    stop_flag = True
    log.info("[STOP] Stop flag set by user")
    return jsonify({"status": "stopping"})


@app.route("/api/holding-tickers")
def api_holding_tickers():
    """Returns current TradeAI holding tickers so UI can lock them in watchlist bar."""
    tickers = _get_holding_tickers()
    log.info(f"[HoldingTickers] {tickers}")
    return jsonify({"holdingTickers": tickers})

@app.route("/")
def index():
    resp = make_response(render_template("index.html", watchlist=watchlist, status=fetch_status))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/add", methods=["POST", "GET"])
def add_ticker():
    ticker = (request.form.get("ticker") or request.args.get("ticker") or "").upper().strip()
    if not ticker or ticker in watchlist:
        return redirect("/")
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        if not info.get("regularMarketPrice") and not info.get("currentPrice"):
            log.warning(f"Ticker validation FAIL: {ticker} — not found on yfinance")
            return redirect("/?error=invalid_ticker")
        watchlist.append(ticker)
        save_watchlist()
        log.info(f"Added ticker: {ticker}")
    except Exception as e:
        log.error(f"Ticker validation error: {ticker} | {e}")
    return redirect("/")


@app.route("/remove/<ticker>")
def remove_ticker(ticker):
    if ticker in watchlist:
        watchlist.remove(ticker)
        save_watchlist()
        log.info(f"Removed ticker: {ticker}")
    try:
        existing = load_json(OUTPUT_PATH, {"watchlist": [], "dailyGainers": []})
        existing["watchlist"] = [s for s in existing["watchlist"] if s["ticker"] != ticker]
        save_json(OUTPUT_PATH, existing)
        log.info(f"Removed {ticker} from analysis.json")
    except Exception as e:
        log.error(f"Remove from analysis FAIL: {ticker} | {e}")
    return redirect("/")


@app.route("/logs")
def view_logs():
    try:
        with open("logs/app.log", "r") as f:
            lines = f.readlines()
        count = len(lines)
        last = lines[-500:]
        return jsonify({"lines": [l.rstrip() for l in last], "total": count})
    except Exception as e:
        return jsonify({"lines": [], "total": 0, "error": str(e)})

@app.route("/logs/clear", methods=["POST"])
def clear_logs():
    try:
        with open("logs/app.log", "w") as f:
            f.write("")
        log.info("Log file cleared by user")
        return jsonify({"status": "cleared"})
    except Exception as e:
        log.error(f"Log clear FAIL: {e}")
        return jsonify({"error": str(e)})

@app.route("/data")
def get_data():
    try:
        data = load_json(OUTPUT_PATH, {"watchlist": [], "dailyGainers": []})
        return jsonify(data)
    except Exception as e:
        log.error(f"Data load FAIL: {e}")
        return jsonify({"watchlist": [], "dailyGainers": []})
    
from config import WATCHLIST, OUTPUT_PATH, NEWS_PATH, WATCHLIST_PATH, FOLLOWUP_PATH

@app.route("/followup/<ticker>/<event_name>")
def followup_plan(ticker, event_name):
    try:
        force = request.args.get("refresh") == "1"
        followup_db = load_json(FOLLOWUP_PATH, {})
        cache_key = f"{ticker}_{event_name}".replace(" ", "_")

        if not force and cache_key in followup_db:
            log.info(f"Followup from cache: {ticker} | {event_name}")
            return jsonify({**followup_db[cache_key], "cached": True})

        with open(OUTPUT_PATH, "r") as f:
            data = json.load(f)
        stock = next((s for s in data["watchlist"] if s["ticker"] == ticker), None)
        if not stock:
            return jsonify({"error": "Ticker not found"})

        plan = generate_followup(stock, event_name)
        plan["savedAt"] = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y-%m-%d %H:%M %Z")
        plan["cached"] = False

        followup_db[cache_key] = plan
        save_json(FOLLOWUP_PATH, followup_db)
        log.info(f"Followup saved: {ticker} | {event_name}")
        return jsonify(plan)
    except Exception as e:
        log.error(f"Followup FAIL: {ticker} | {e}")
        return jsonify({"error": str(e)})

@app.route("/followup/check/<ticker>/<event_name>")
def followup_check(ticker, event_name):
    followup_db = load_json(FOLLOWUP_PATH, {})
    cache_key = f"{ticker}_{event_name}".replace(" ", "_")
    entry = followup_db.get(cache_key)
    return jsonify({"exists": entry is not None, "savedAt": entry.get("savedAt") if entry else None})

@app.route("/technical/<ticker>")
def technical_analysis(ticker):
    try:
        existing = load_json(OUTPUT_PATH, {"watchlist": []})
        stock = next((s for s in existing.get("watchlist", []) if s["ticker"] == ticker), {})
        rsi = stock.get("rsi14")
        bb = stock.get("bbPercent")
        log.info(f"Technical analysis: {ticker} | rsi:{rsi} bb:{bb}")

        if rsi is None and bb is None:
            return jsonify({"error": "No technical data. Run a price refresh first.", "ticker": ticker})

        def rsi_label(v):
            if v is None: return "Unknown", "neutral"
            if v < 30: return "Oversold", "bullish"
            if v > 70: return "Overbought", "bearish"
            if v < 45: return "Mildly Oversold", "mild-bullish"
            if v > 55: return "Mildly Overbought", "mild-bearish"
            return "Neutral", "neutral"

        def bb_label(v):
            if v is None: return "Unknown", "neutral"
            if v < 0: return "Below Lower Band", "bullish"
            if v > 1: return "Above Upper Band", "bearish"
            if v < 0.2: return "Near Lower Band", "mild-bullish"
            if v > 0.8: return "Near Upper Band", "mild-bearish"
            return "Mid-Range", "neutral"

        rsi_text, rsi_signal = rsi_label(rsi)
        bb_text, bb_signal = bb_label(bb)

        signals = [rsi_signal, bb_signal]
        bullish = signals.count("bullish") + signals.count("mild-bullish") * 0.5
        bearish = signals.count("bearish") + signals.count("mild-bearish") * 0.5

        if bullish > bearish:
            if bullish >= 2:
                verdict = "STRONG BUY WATCH"
                action = "Both indicators confirm oversold conditions. This is a potential bounce candidate. Watch for a reversal confirmation candle (e.g. green day on above-average volume) before entering. Set a stop below recent low."
            else:
                verdict = "MILD BUY WATCH"
                action = "One indicator suggests oversold but signals are mixed. Proceed cautiously. Wait for price to stabilize or show momentum before considering entry."
        elif bearish > bullish:
            if bearish >= 2:
                verdict = "CAUTION — OVERBOUGHT"
                action = "Both indicators confirm overbought conditions. Elevated pullback risk. Avoid chasing the price here. If you hold, consider tightening your stop-loss or taking partial profits."
            else:
                verdict = "MILD CAUTION"
                action = "One indicator suggests overbought. Not an immediate sell signal, but avoid adding to position at current levels. Monitor for signs of weakening momentum."
        else:
            verdict = "NEUTRAL"
            action = "Indicators show no strong directional bias. The stock is trading within normal range. No immediate action required — wait for a clearer signal before committing."

        rsi_str = f"RSI {rsi}" if rsi is not None else "RSI N/A"
        bb_str = f"BB%B {bb}" if bb is not None else "BB%B N/A"

        return jsonify({
            "ticker": ticker,
            "rsi14": rsi,
            "bbPercent": bb,
            "rsiLabel": rsi_text,
            "bbLabel": bb_text,
            "verdict": verdict,
            "action": action,
            "summary": f"{rsi_str} ({rsi_text}) · {bb_str} ({bb_text})",
        })
    except Exception as e:
        log.error(f"Technical analysis FAIL: {ticker} | {e}")
        return jsonify({"error": str(e), "ticker": ticker})



def run_macro_refresh():
    global fetch_status
    fetch_status = {"running": True, "message": "Fetching macro data...", "operation": "macro"}
    log.info("=== Macro Refresh Started ===")
    try:
        if not FRED_API_KEY or FRED_API_KEY == "your_fred_api_key_here":
            log.error("Macro refresh FAIL: FRED_API_KEY not set in config.py")
            fetch_status = {"running": False, "message": "Macro FAIL: FRED_API_KEY not set", "operation": None}
            return
        data = fetch_macro_data(FRED_API_KEY, NEWSAPI_KEY)
        now_ts = ts()
        data["fetchedAt"] = now_ts
        save_json(MACRO_PATH, data)
        fetch_status = {"running": False, "message": f"Macro updated: {now_ts}", "operation": None}
        log.info(f"=== Macro Refresh Complete: {now_ts} ===")
    except Exception as e:
        log.error(f"Macro refresh FAIL: {e}")
        fetch_status = {"running": False, "message": f"Macro FAIL: {e}", "operation": None}


@app.route("/macro")
def get_macro():
    data = load_json(MACRO_PATH, {})
    return jsonify(data)


@app.route("/macro/refresh", methods=["GET","POST"])
def macro_refresh():
    if fetch_status.get("running"):
        return jsonify({"error": "Another operation is running"}), 409
    t = threading.Thread(target=run_macro_refresh, daemon=True)
    t.start()
    log.info("Macro refresh triggered via UI")
    return jsonify({"status": "started"})


@app.route("/tradeai/account_tracker/run", methods=["POST"])
def tradeai_account_tracker_run():
    """Manual trigger — pure computation (transaction replay + a handful of
    live/historical price fetches), fast enough to run synchronously."""
    try:
        _update_account_tracker()
        return jsonify({"status": "ok", "summary": _account_tracker_summary()})
    except Exception as e:
        log.error(f"[AccountTracker] manual run FAIL: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/tradeai/weight_backtest")
def tradeai_weight_backtest_get():
    """Read-only — whatever the last computed run (scheduled or manual)
    left in output/weight_backtest.json. Returns {result: null} if it's
    never been run yet rather than computing on a page load."""
    return jsonify({"result": load_json(WEIGHT_BACKTEST_PATH, None)})


@app.route("/tradeai/weight_backtest/run", methods=["POST"])
def tradeai_weight_backtest_run():
    """Manual trigger — refreshes rebuilt AI scores and fixed-horizon
    returns (one batch price download), then computes. Synchronous."""
    try:
        return jsonify({"status": "ok", "result": _backtest_bucket_weights()})
    except Exception as e:
        log.error(f"[WeightBacktest] manual run FAIL: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── MARKET ALERT ──────────────────────────────────────────────────────────────
MARKET_ALERT_CACHE = {"data": None, "cachedAt": None}
MARKET_ALERT_TTL   = 300   # seconds (5 min)
ALERT_THRESHOLD    = -0.02  # -2% S&P 500
VIX_FEAR_THRESHOLD = 20.0

def _fetch_market_alert_data():
    """Fetch ^GSPC and ^VIX via yfinance. Returns dict or None on failure."""
    try:
        import yfinance as yf
        spx = yf.Ticker("^GSPC")
        vix = yf.Ticker("^VIX")

        spx_info = spx.fast_info
        vix_info = vix.fast_info

        spx_price = getattr(spx_info, "last_price", None)
        spx_prev  = getattr(spx_info, "previous_close", None)
        vix_price = getattr(vix_info, "last_price", None)

        if spx_price is None or spx_prev is None:
            log.warning("[MarketAlert] Could not fetch ^GSPC fast_info — trying history fallback")
            hist = spx.history(period="2d", interval="1d")
            if len(hist) >= 2:
                spx_price = float(hist["Close"].iloc[-1])
                spx_prev  = float(hist["Close"].iloc[-2])
            else:
                log.error("[MarketAlert] ^GSPC history fallback failed")
                return None

        if vix_price is None:
            try:
                vhist = vix.history(period="1d", interval="1m")
                vix_price = float(vhist["Close"].iloc[-1]) if not vhist.empty else None
            except Exception as ve:
                log.warning(f"[MarketAlert] ^VIX fallback failed: {ve}")

        spx_change = (spx_price - spx_prev) / spx_prev if spx_prev else 0.0
        result = {
            "spxPrice":  round(float(spx_price), 2),
            "spxPrev":   round(float(spx_prev), 2),
            "spxChange": round(spx_change, 5),
            "vix":       round(float(vix_price), 2) if vix_price else None,
        }
        log.info(f"[MarketAlert] fetched | SPX:{result['spxPrice']} chg:{result['spxChange']:.2%} VIX:{result['vix']}")
        return result
    except Exception as e:
        log.error(f"[MarketAlert] fetch FAIL: {e}")
        return None


def _generate_alert_cause(spx_change, vix):
    """Rule-based cause for the market drop. No AI call (quota preserved for ticker analysis)."""
    try:
        chg_pct = f"{spx_change * 100:.1f}%"
        if vix is None:
            vix_str = "VIX unavailable"
        elif vix >= 30:
            vix_str = f"VIX spiking to {vix:.1f}, signaling high fear"
        elif vix >= 20:
            vix_str = f"VIX elevated at {vix:.1f}"
        else:
            vix_str = f"VIX calm at {vix:.1f}"
        cause = f"S&P 500 down {chg_pct}; {vix_str}."
        log.info(f"[MarketAlert] rule-based cause: {cause}")
        return cause
    except Exception as e:
        log.warning(f"[MarketAlert] cause FAIL (non-critical): {e}")
        return None


@app.route("/api/market-alert")
def api_market_alert():
    """
    Returns market alert status. Caches for MARKET_ALERT_TTL seconds.
    Only calls Ollama for cause when alert is active.
    Response: { alert, spxChange, spxPrice, vix, cause, level, checkedAt, cached }
    """
    try:
        import time
        now_epoch = time.time()
        cached = MARKET_ALERT_CACHE.get("data")
        cached_at = MARKET_ALERT_CACHE.get("cachedAt") or 0

        if cached and (now_epoch - cached_at) < MARKET_ALERT_TTL:
            log.info(f"[MarketAlert] served from cache | age:{int(now_epoch - cached_at)}s")
            return jsonify({**cached, "cached": True})

        raw = _fetch_market_alert_data()
        if not raw:
            log.error("[MarketAlert] route: fetch returned None")
            if cached:
                return jsonify({**cached, "cached": True, "stale": True})
            return jsonify({"alert": False, "error": "fetch_failed", "cached": False}), 200

        spx_change = raw["spxChange"]
        vix        = raw["vix"]
        alert      = spx_change <= ALERT_THRESHOLD

        level  = "normal"
        if spx_change <= -0.03:
            level = "severe"
        elif spx_change <= -0.02:
            level = "warning"
        elif spx_change <= -0.01:
            level = "watch"

        cause = None
        if alert:
            # Reuse cached cause if alert was already active and cause exists
            prev_cause = (cached or {}).get("cause")
            cause = prev_cause if prev_cause else _generate_alert_cause(spx_change, vix)

        vix_fear = vix is not None and vix >= VIX_FEAR_THRESHOLD

        data = {
            "alert":      alert,
            "level":      level,
            "spxChange":  raw["spxChange"],
            "spxPrice":   raw["spxPrice"],
            "spxPrev":    raw["spxPrev"],
            "vix":        vix,
            "vixFear":    vix_fear,
            "cause":      cause,
            "checkedAt":  ts(),
            "cached":     False,
        }

        MARKET_ALERT_CACHE["data"]     = data
        MARKET_ALERT_CACHE["cachedAt"] = now_epoch
        log.info(f"[MarketAlert] alert:{alert} level:{level} spxChange:{spx_change:.2%} vix:{vix} vixFear:{vix_fear}")
        return jsonify(data)

    except Exception as e:
        log.error(f"[MarketAlert] route FAIL: {e}")
        return jsonify({"alert": False, "error": str(e), "cached": False}), 200
# ── END MARKET ALERT ───────────────────────────────────────────────────────────


@app.route("/dream")
def get_dream():
    data = load_json(DREAM_PATH, {})
    return jsonify(data)


def run_dream_scan(trigger="manual"):
    """Run dream candidate scan and save results to DREAM_PATH."""
    global fetch_status, stop_flag
    stop_flag = False
    fetch_status = {"running": True, "message": "Dream scan running...", "operation": "dream_scan", "current": 0, "total": 0, "current_ticker": ""}
    log.info(f"=== Dream Scan Started (trigger={trigger}) ===")
    try:
        watchlist = load_json(WATCHLIST_PATH, [])
        watchlist_tickers = [w["ticker"] for w in watchlist if isinstance(w, dict)] if watchlist and isinstance(watchlist[0], dict) else watchlist
        gainers = fetch_daily_gainers()
        # TSX-only build: use CDN gainers only — US gainers have no .TO
        # suffix and don't belong in a TSX Composite universe.
        cdn = gainers.get("cdn", []) if isinstance(gainers, dict) else []
        gainer_tickers = [g.get("ticker", "") for g in cdn if isinstance(g, dict)]
        existing_raw = load_json(DREAM_PATH, {}).get("candidates", [])
        # Guard against stale nested structure from old bug
        if isinstance(existing_raw, dict):
            existing = existing_raw.get("candidates", [])
        else:
            existing = existing_raw

        def _on_dream_progress(current, total, ticker):
            global fetch_status
            fetch_status = {
                "running": True,
                "message": f"Dream scan: {current}/{total} — {ticker}",
                "operation": "dream_scan",
                "current": current,
                "total": total,
                "current_ticker": ticker,
            }

        result = fetch_dream_candidates(watchlist_tickers, gainer_tickers, existing, progress_callback=_on_dream_progress)
        # fetch_dream_candidates returns {"candidates": [...]} — unwrap it
        candidates = result.get("candidates", []) if isinstance(result, dict) else result
        from datetime import datetime
        save_json(DREAM_PATH, {"candidates": candidates, "scannedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "trigger": trigger})
        log.info(f"=== Dream Scan Complete: {len(candidates)} candidates ===")
        fetch_status = {"running": False, "message": f"Dream scan complete: {len(candidates)} candidates", "operation": None, "current": len(candidates), "total": len(candidates), "current_ticker": ""}
    except Exception as e:
        log.error(f"Dream scan FAIL: {e}", exc_info=True)
        fetch_status = {"running": False, "message": f"Dream scan FAIL: {e}", "operation": None, "current": 0, "total": 0, "current_ticker": ""}


@app.route("/dream/scan", methods=["GET", "POST"])
def dream_scan():
    if fetch_status.get("running"):
        return jsonify({"error": "Another operation is running"}), 409
    t = threading.Thread(target=run_dream_scan, args=("manual",), daemon=True)
    t.start()
    log.info("Dream scan triggered via UI")
    return jsonify({"status": "started"})


@app.route("/dream/refresh-tickers", methods=["POST"])
def dream_refresh_tickers():
    if fetch_status.get("running"):
        return jsonify({"error": "Another operation is running"}), 409
    def _do_refresh():
        global fetch_status, stop_flag
        stop_flag = False
        fetch_status = {"running": True, "message": "Refreshing ticker lists (TSX 60, TSX Composite)...", "operation": "refresh_tickers", "current": 0, "total": 2, "current_ticker": ""}
        log.info("=== Ticker List Refresh Started ===")
        try:
            from fetcher import fetch_tsx60_tickers, fetch_tsx_composite_tickers
            fetch_status["current"] = 1
            fetch_status["current_ticker"] = "TSX60"
            fetch_status["message"] = "Refreshing ticker lists: 1 of 2 — TSX60"
            tsx60 = fetch_tsx60_tickers()
            fetch_status["current"] = 2
            fetch_status["current_ticker"] = "TSX Composite"
            fetch_status["message"] = "Refreshing ticker lists: 2 of 2 — TSX Composite"
            tsx_composite = fetch_tsx_composite_tickers()
            log.info(f"=== Ticker Refresh Complete: TSX60:{len(tsx60)} TSXComposite:{len(tsx_composite)} ===")
            fetch_status = {"running": False, "message": f"Ticker lists refreshed: TSX60={len(tsx60)} TSXComposite={len(tsx_composite)}", "operation": None, "current": 2, "total": 2, "current_ticker": ""}
        except Exception as e:
            log.error(f"Ticker refresh FAIL: {e}", exc_info=True)
            fetch_status = {"running": False, "message": f"Ticker refresh FAIL: {e}", "operation": None, "current": 0, "total": 0, "current_ticker": ""}
    t = threading.Thread(target=_do_refresh, daemon=True)
    t.start()
    log.info("Ticker refresh triggered via UI")
    return jsonify({"status": "started"})


def _maybe_run_dream_scan():
    import time
    time.sleep(60)
    # Wait up to 5 extra minutes for any in-progress operation (e.g. startup price refresh) to finish,
    # rather than skipping the daily scan entirely just because it overlapped.
    wait_attempts = 0
    while fetch_status.get("running") and wait_attempts < 30:
        log.info("Dream auto-scan waiting — another operation running")
        time.sleep(10)
        wait_attempts += 1
    if fetch_status.get("running"):
        log.info("Dream auto-scan skipped — another operation still running after wait")
        return
    dream = load_json(DREAM_PATH, {})
    scanned_at = dream.get("scannedAt")
    if scanned_at:
        try:
            from datetime import datetime
            last = datetime.strptime(scanned_at[:16], "%Y-%m-%d %H:%M")
            diff = (datetime.now() - last).total_seconds() / 3600
            if diff < 20:
                log.info(f"Dream auto-scan skipped — last ran {diff:.1f}h ago")
                return
        except Exception:
            pass
    log.info("Dream auto-scan starting (stale or never run)")
    run_dream_scan("startup")


# ── Scoring Matrix Config ─────────────────────────────────────────────────────
SCORING_CONFIG_PATH = "output/scoring_config.json"

SCORING_DEFAULTS = {
    "buySignalMap": {
        "Strong Buy": 100,
        "Buy": 75,
        "Hold": 40,
        "Avoid": 0,
        "Unknown": 10
    },
    "bucketWeights": {
        "Momentum":      {"dream": 0.15, "ai": 0.65},
        "Reversal":      {"dream": 0.15, "ai": 0.65},
        "SmartMoney":    {"dream": 0.30, "ai": 0.55},
        "SmartMoney+": {"dream": 0.30, "ai": 0.55},
        "Dream":         {"dream": 0.25, "ai": 0.65}
    },
    "trajectoryBonus": {
        "Momentum":      {"Accelerating": 20, "Stable": 8,  "Decelerating": 0},
        "Reversal":      {"Accelerating": 0,  "Stable": 0,  "Decelerating": 0},
        "SmartMoney":    {"Accelerating": 10, "Stable": 0,  "Decelerating": 0},
        "SmartMoney+": {"Accelerating": 10, "Stable": 0,  "Decelerating": 0},
        "Dream":         {"Accelerating": 12, "Stable": 0,  "Decelerating": 0}
    },
    "stageBonus": {
        "Momentum":      {"Early-Stage": 0,  "Growth": 8,  "Mature": 5,  "Declining": 0},
        "Reversal":      {"Early-Stage": 0,  "Growth": 0,  "Mature": 0,  "Declining": 0},
        "SmartMoney":    {"Early-Stage": 15, "Growth": 10, "Mature": 0,  "Declining": 0},
        "SmartMoney+": {"Early-Stage": 15, "Growth": 10, "Mature": 0,  "Declining": 0},
        "Dream":         {"Early-Stage": 8,  "Growth": 4,  "Mature": 0,  "Declining": 0}
    },
    "sharedBonuses": {
        "volumeSurgeHigh":    10,
        "volumeSurgeMed":     5,
        "volumeSurgeHigh_threshold": 2.0,
        "volumeSurgeMed_threshold":  1.5,
        "analystUpsideHigh":  8,
        "analystUpsideMed":   4,
        "analystUpsideHigh_threshold": 20.0,
        "analystUpsideMed_threshold":  10.0,
        "newsBullish":        6,
        "newsNeutral":        3,
        "insiderNetBuy":      5,
        "insiderNetSell":     -3
    },
    "reversalBonuses": {
        "rsiDeep":    20,
        "rsiMed":     12,
        "rsiMild":    6,
        "rsiDeep_threshold":  30,
        "rsiMed_threshold":   35,
        "rsiMild_threshold":  40,
        "bbLowerBand": 10,
        "bbLowerBand_threshold": 0.10
    },
    "signalForecast": {
        # Disabled 2026-07-07: 386 graded forecasts showed 49% directional accuracy
        # (coin flip) and slightly inverted bias — pure noise worth ±20 composite pts.
        "enabled": False,
        "returnMultiplier": 3.0,
        "maxBoost":  20,
        "maxPenalty": -20,
        "neutralZone": 1.0
    },
    "sellThresholds": {
        "hardSellFloor":     25,
        "softSellFloor":     50,
        "stopLossHard":      10.0,  # ceiling for the ATR-based stop below — worst case allowed regardless of volatility
        "atrStopMultiplier": 2.0,   # effective stop = min(atrStopMultiplier * ticker's own ATR14%, stopLossHard)
        "stopLossSoft":      6.0,
        "trailingStopPeak":  10.0,
        "dailyDropStop":     8.0,
        "minHoldDays":       3
    },
    "execution": {
        "slippagePct":         0.2,
        "slippageMidCapPct":   0.4,
        "slippageSmallCapPct": 0.6,
        "midCapBelow":         2000000000,
        "smallCapBelow":       300000000
    },
    "regime": {
        "vixNoBuyAbove": 30,
        "macroRiskGateEnabled": True,
        "trendFilterEnabled": True,
        "trendFilterTicker": "SPY",
        "trendFilterDays": 200,
        # A scheduled FOMC decision moves high-beta/extended names before it
        # ever shows up as a VIX spike or a realized rate change — the other
        # gates are all backward-looking, this one is the only
        # forward-looking one. Blocks new buys for N days up to and
        # including the decision day itself (see fetcher.FOMC_MEETING_DATES).
        "fomcBlackoutEnabled": True,
        "fomcBlackoutDays": 1
    },
    # Cross-bucket dampener: a stock that's already run up hard and is
    # scored "Accelerating" is exactly the extended, high-beta profile that
    # gets hit hardest in a risk-off wobble — nothing previously penalized
    # "already extended," only the Reversal bucket rewarded "oversold."
    "overextension": {
        "rsiThreshold": 75,
        "change30dThreshold": 25.0,
        "penalty": -15
    },
    # Qualify-stage guard: refuse to buy a ticker that's already deep into
    # a bad day — buying at, say, -7% intraday just to get stopped out by
    # the daily-drop stop an hour later wastes the buy+sell slippage for a
    # position that never had a chance. Expressed as a fraction of
    # sellThresholds.dailyDropStop so it scales with that setting.
    "entryGuards": {
        "maxAlreadyDownPctOfDailyDropStop": 0.5
    },
    "selfLearning": {
        "autoRollback": True,
        "minTrades": 10,
        "rollbackThresholdPct": 1.0
    },
    "breakoutScore": {
        "rsiTurning":      8,
        "bbSqueeze":       8,
        "shortSqueeze":    5,
        "volumeBuilding":  4,
        "rsiTurning_low":  25,
        "rsiTurning_high": 42,
        "bbSqueeze_threshold": 0.20,
        "shortSqueeze_threshold": 15.0,
        "volumeBuilding_low":  1.2,
        "volumeBuilding_high": 1.9
    },
    "sectorDiversity": {
        "penalty": -15
    },
    "purchaseMatrix": {
        "slots":         6,
        "maxDeployPct":  100,
        "minBuyComposite": 60,
        "earningsBlackoutDays": 3,
        "rotationEnabled": True,
        "rotationMinScoreGap": 15,
        "rotationCooldownTradingDays": 5
    },
    "identifyRules": {
        # A half-scored ticker (missing data, or SCORE_INCOMPLETE) shouldn't
        # compete for bucket slots against fully-scored ones. See
        # score_dream_stock's dataCompleteness field.
        "dataCompletenessFloor": 0.6,
        # Max tickers any single bucket can contribute to the final buy
        # slots. Per-bucket dict (not a flat cap): SmartMoney and SmartMoney+
        # get more room (SmartMoney on an earned 11-trade track record;
        # SmartMoney+ by inheriting SmartMoney's own validated fundamental
        # screen, even though it has no closed trades of its own yet — a
        # deliberate judgment call, not evidence-backed like SmartMoney's).
        # The rest stay capped tighter to limit exposure to thinner-evidence
        # buckets and keep some diversification. "_default" covers any
        # bucket not listed here.
        "maxPerBucketFinal": {"SmartMoney": 5, "SmartMoney+": 5, "Momentum": 2, "Reversal": 2, "Dream": 2, "_default": 2},
        # Hard-block a second buy in an already-held sector at final
        # selection, instead of only soft-penalizing it in the composite.
        "hardBlockSectorDiversity": False,
        # Momentum bucket RSI band — moved out of hardcoded Python so it's
        # tunable (and proposable by the tuning-analysis pipeline, see
        # _check_tuning_candidate). Defaults match prior hardcoded behavior.
        "momentumRsiMin": 45,
        "momentumRsiMax": 70,
        # Volume-surge gate for Momentum — None means not enforced (day-one
        # default: the signal is new, no evidence yet on where to set it).
        # A candidate with no volumeRatio data always passes through
        # regardless of this setting. Set by approving a tuning proposal,
        # or by hand once you have a view.
        "momentumVolumeRatioMin": None
    }
}

def _record_shadow_snapshot(scored, qualified, actual_buys, dream_map, holdings, live_prices, spy_now, rec_version):
    """Shadow validation: snapshot the FULL daily funnel, not just what got
    bought — every dream-scanned ticker gets a row, tagged with how far it
    got: scanned_not_identified (never made the 30-shortlist) ->
    identified_not_qualified (shortlisted, filtered at Recommend — cooldown,
    earnings blackout, composite too low, etc.) -> qualified_not_bought
    (passed every check, lost the slot race or blocked by bucket/sector
    caps) -> bought. This is what lets the tuning-analysis pipeline compare
    "what we bought" against "what we missed" on the same factors, instead
    of only ever seeing the ~30 tickers that made the shortlist."""
    db = load_json(SHADOW_PATH, {"snapshots": []})
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    if any(s.get("date") == today for s in db["snapshots"]):
        return

    qualified_tickers = {q["ticker"] for q in qualified}
    bought_tickers = {b["ticker"] for b in actual_buys} | set(holdings.keys())
    scored_by_ticker = {s["ticker"]: s for s in scored}

    rows = []
    for ticker, c in dream_map.items():
        bd = c.get("breakdown") or {}
        s = scored_by_ticker.get(ticker)
        if s:
            outcome = ("bought" if ticker in bought_tickers
                       else "qualified_not_bought" if ticker in qualified_tickers
                       else "identified_not_qualified")
            price = live_prices.get(ticker) or s.get("price") or c.get("price")
            composite = s.get("composite")
            bucket = s.get("bucket")
            # None (not 0) when no assessment ran, so the weight backtest can
            # exclude it instead of treating "never scored" as a real 0.
            ai_score = s.get("aiScore") if s.get("hasAI") else None
        else:
            outcome = "scanned_not_identified"
            price = c.get("price")
            composite = None
            bucket = None
            ai_score = None
        if not price:
            continue
        rows.append({
            "ticker": ticker, "bucket": bucket, "outcome": outcome,
            "composite": composite, "dreamScore": c.get("score"), "aiScore": ai_score,
            "rsi": bd.get("rsi"), "volumeRatio": bd.get("volumeRatio"),
            "rangeExpansion": bd.get("rangeExpansion"), "aboveMa50": bd.get("aboveMa50"),
            "dataCompleteness": bd.get("dataCompleteness"),
            "price": price, "spyPrice": spy_now, "recVersion": rec_version,
            "fwdReturnPct": None, "fwdExcessPct": None,
            "maxDailyGainPct": None, "maxDailyGainDate": None,
        })
    if rows:
        db["snapshots"].append({"date": today, "rows": rows})
        # Full-universe rows are much smaller per-row than the old
        # scored-only shape, but there are ~20x more of them — shorter
        # retention keeps total size in the same ballpark.
        db["snapshots"] = db["snapshots"][-45:]
        save_json(SHADOW_PATH, db)
        outcomes = {}
        for r in rows:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        log.info(f"[Shadow] snapshot {today}: {len(rows)} tickers | {outcomes}")


def _daily_closes(hist):
    """(date, close) pairs from a yfinance history DataFrame, oldest first."""
    if hist is None or hist.empty:
        return []
    return [(idx.date() if hasattr(idx, "date") else idx, float(c)) for idx, c in zip(hist.index, hist["Close"])]


def _grade_shadow_snapshots(min_age_days=5, max_fetches=40):
    """Fill in forward returns (vs SPY) AND the max single-day gain seen in
    the window for shadow snapshots old enough to judge. Max daily gain is
    what actually answers "did we miss a real mover" — a stock that spiked
    8% on day 2 then gave it back by day 5 would show ~0% cumulative return
    here but was a real 5%+ day someone could have caught."""
    import yfinance as yf
    db = load_json(SHADOW_PATH, {"snapshots": []})
    today = datetime.now(TZ).date()
    pending_rows = []
    for snap in db["snapshots"]:
        try:
            snap_date = datetime.strptime(snap["date"], "%Y-%m-%d").date()
            age = (today - snap_date).days
        except Exception:
            continue
        if age < min_age_days:
            continue
        for r in snap.get("rows", []):
            if r.get("fwdReturnPct") is None:
                pending_rows.append((r, snap_date, age))
    if not pending_rows:
        return 0

    # Cap DISTINCT tickers fetched this cycle (rate-limit protection); grade
    # every pending row for those tickers while their history is in hand.
    tickers_to_grade = sorted({r["ticker"] for r, _, _ in pending_rows})[:max_fetches]

    spy_closes = []
    try:
        spy_closes = _daily_closes(yf.Ticker("SPY").history(period="2mo"))
    except Exception as e:
        log.warning(f"[Shadow] SPY history FAIL: {e}")

    def _after(closes, d):
        return [(cd, c) for cd, c in closes if cd > d]

    graded = 0
    for ticker in tickers_to_grade:
        try:
            closes = _daily_closes(yf.Ticker(ticker).history(period="2mo"))
        except Exception as e:
            log.warning(f"[Shadow] {ticker} history FAIL: {e}")
            continue
        if not closes:
            continue
        for r, snap_date, age in pending_rows:
            if r["ticker"] != ticker:
                continue
            after = _after(closes, snap_date)
            base_price = r.get("price")
            if not after or not base_price:
                continue
            latest_close = after[-1][1]
            r["fwdReturnPct"] = round((latest_close - base_price) / base_price * 100, 2)

            # Day-over-day % change through the window, first day anchored
            # off the snapshot's own recorded price.
            prev_close = base_price
            max_gain, max_gain_date = None, None
            for cd, c in after:
                if prev_close:
                    pct = (c - prev_close) / prev_close * 100
                    if max_gain is None or pct > max_gain:
                        max_gain, max_gain_date = pct, str(cd)
                prev_close = c
            if max_gain is not None:
                r["maxDailyGainPct"] = round(max_gain, 2)
                r["maxDailyGainDate"] = max_gain_date

            spy_after = _after(spy_closes, snap_date)
            spy_base = r.get("spyPrice")
            if spy_after and spy_base:
                spy_ret = (spy_after[-1][1] - spy_base) / spy_base * 100
                r["fwdExcessPct"] = round(r["fwdReturnPct"] - spy_ret, 2)
            r["gradedAfterDays"] = age
            graded += 1
    if graded:
        save_json(SHADOW_PATH, db)
        log.info(f"[Shadow] graded {graded} candidate rows (max-daily-gain tracking)")
    return graded


def _slippage_for(detail, exec_cfg):
    """Liquidity-aware slippage: small caps pay wider spreads than a flat 0.2%.
    Returns a fraction (e.g. 0.006 for 0.6%). Unknown market cap = assume mid-cap."""
    cap = (detail or {}).get("marketCap")
    small = float(exec_cfg.get("slippageSmallCapPct", 0.6)) / 100.0
    mid   = float(exec_cfg.get("slippageMidCapPct", 0.4)) / 100.0
    base  = float(exec_cfg.get("slippagePct", 0.2)) / 100.0
    if cap is None:
        return mid
    if cap < float(exec_cfg.get("smallCapBelow", 3e8)):
        return small
    if cap < float(exec_cfg.get("midCapBelow", 2e9)):
        return mid
    return base


def _log_trade_outcome(ticker, pos, sell_price, sell_reason, trigger, composite, spy_now, days_held, sell_rec_version=None):
    """Log a closed-trade outcome with buy-time factors for scoring feedback,
    benchmarked against SPY over the same holding period.

    Two different versions can be credited: `recVersion` is whichever
    version's scoring picked this trade at BUY time (entry attribution —
    "was this version's stock-picking good?"); `sellRecVersion` is whichever
    version was live when the SELL decision fired (exit attribution — "was
    this version's stop-loss/rotation logic good?"). A position bought under
    v1 and sold today under v4's thresholds shows recVersion=1,
    sellRecVersion=4 — both are real, answering different questions."""
    purchase_price = pos.get("purchasePrice") or 0
    gl_pct = round((sell_price - purchase_price) / purchase_price * 100, 2) if purchase_price else None
    spy_buy = pos.get("spyAtBuy")
    spy_return_pct = round((spy_now - spy_buy) / spy_buy * 100, 2) if (spy_now and spy_buy) else None
    excess_pct = round(gl_pct - spy_return_pct, 2) if (gl_pct is not None and spy_return_pct is not None) else None
    outcomes_db = load_json(TRADE_OUTCOMES_PATH, {"outcomes": []})
    outcomes_db["outcomes"].append({
        "ticker": ticker,
        "buyDate": pos.get("purchasedAt"),
        "sellDate": ts(),
        "daysHeld": round(days_held, 1) if days_held is not None else None,
        "shares": pos["shares"],
        "buyPrice": purchase_price,
        "sellPrice": sell_price,
        "gainLoss": round((sell_price - purchase_price) * pos["shares"], 2) if purchase_price else None,
        "gainLossPct": gl_pct,
        "spyReturnPct": spy_return_pct,
        "excessReturnPct": excess_pct,
        "vixAtBuy": pos.get("vixAtBuy"),
        "trigger": trigger,
        "sellReason": sell_reason,
        "sellComposite": composite,
        "factors": pos.get("factors"),
        "recVersion": pos.get("recVersion", 1),
        "sellRecVersion": sell_rec_version if sell_rec_version is not None else current_recommend_version(),
    })
    save_json(TRADE_OUTCOMES_PATH, outcomes_db)


def _days_held(pos):
    """Days since purchase, or None if the timestamp is unparseable."""
    try:
        purchased_at = pos.get("purchasedAt", "")
        dt = datetime.strptime(purchased_at[:16], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        return (datetime.now(TZ) - dt).total_seconds() / 86400.0
    except Exception:
        return None


def _effective_stop_loss_pct(ticker, details, sell_cfg):
    """Volatility-adjusted hard stop: min(atrMultiplier x the ticker's own
    ATR14, hard cap). Tighter for calm stocks, looser for volatile ones, but
    never worse than the hard cap regardless of what ATR says. Falls back to
    the hard cap itself when ATR data isn't available (conservative — treat
    unknown volatility as the worst case, not an arbitrary guess)."""
    hard_cap = float(sell_cfg.get("stopLossHard", 10.0))
    multiplier = float(sell_cfg.get("atrStopMultiplier", 2.0))
    atr_pct = (details.get(ticker) or {}).get("atr14Pct")
    if atr_pct is None:
        return hard_cap
    return min(multiplier * atr_pct, hard_cap)


def load_scoring_config():
    saved = load_json(SCORING_CONFIG_PATH, {})
    if not saved:
        return SCORING_DEFAULTS.copy()
    import copy
    cfg = copy.deepcopy(SCORING_DEFAULTS)
    def deep_merge(base, override):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                deep_merge(base[k], v)
            else:
                base[k] = v
    deep_merge(cfg, saved)
    # Fill any missing keys in saved sub-dicts with defaults (forward migration)
    def fill_missing(base, saved_ref):
        for k, v in base.items():
            if k not in saved_ref:
                saved_ref[k] = v
                log.info(f"[ScoringConfig] migrated missing key: {k} = {v}")
            elif isinstance(v, dict) and isinstance(saved_ref.get(k), dict):
                fill_missing(v, saved_ref[k])
    fill_missing(SCORING_DEFAULTS, cfg)
    return cfg

@app.route("/scoring/config", methods=["GET"])
def get_scoring_config():
    return jsonify(load_scoring_config())

@app.route("/scoring/config", methods=["POST"])
def save_scoring_config():
    try:
        data = request.get_json()
        # Auto-bump recommend version when scoring parameters actually change,
        # so the Validation tab can attribute performance to each config.
        old_flat = _flatten_cfg(load_scoring_config())
        new_flat = _flatten_cfg(data)
        diffs = [f"{k}: {old_flat.get(k)} → {new_flat[k]}"
                 for k in new_flat if k in old_flat and old_flat.get(k) != new_flat[k]]
        save_json(SCORING_CONFIG_PATH, data)
        warning = None
        if diffs:
            new_v = bump_recommend_version(diffs, label="Scoring config change", config_snapshot=data)
            db = load_recommend_versions()
            warning = db["versions"][-1].get("warning")
        log.info(f"[ScoringConfig] saved | {len(diffs)} changed keys")
        return jsonify({"status": "ok", "changed": diffs, "warning": warning})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/scoring/reset", methods=["POST"])
def reset_scoring_config():
    try:
        save_json(SCORING_CONFIG_PATH, SCORING_DEFAULTS)
        log.info("[ScoringConfig] reset to defaults")
        return jsonify({"status": "ok", "config": SCORING_DEFAULTS})
    except Exception as e:
        return jsonify({"error": str(e)}), 500




# ── TRADEAI SIMULATION ────────────────────────────────────────────────────────

STARTING_BALANCE = 100_000.0

def load_trades_ai():
    default = {"balance": STARTING_BALANCE, "holdings": {}, "transactions": []}
    return load_json(TRADES_AI_PATH, default)

def save_trades_ai(data):
    save_json(TRADES_AI_PATH, data)


def run_tradeai_identify():
    global fetch_status, stop_flag
    stop_flag = False
    fetch_status = {"running": True, "message": "TradeAI: identifying candidates (3-bucket)...", "operation": "tradeai_identify", "current": 0, "total": 0, "current_ticker": ""}
    log.info("=== TradeAI Identify Started (3-bucket) ===")
    try:
        dream = load_json(DREAM_PATH, {})
        all_candidates = dream.get("candidates", [])

        # Helper to safely get breakdown field
        def bd(c, key):
            return (c.get("breakdown") or {}).get(key)

        # Exclude anything that failed to score (SCORE_INCOMPLETE -> score is
        # None) or is too data-sparse to trust — a half-blank ticker
        # shouldn't compete for a bucket slot against a fully-scored one.
        # dataCompleteness is a new field: candidates scored before this fix
        # won't have it at all yet, so only apply the floor when it's
        # actually present — otherwise every pre-existing candidate would
        # get excluded until the next full Dream Scan refreshes them.
        identify_rules = load_scoring_config().get("identifyRules", {})
        completeness_floor = float(identify_rules.get("dataCompletenessFloor", 0.6))
        candidates = [
            c for c in all_candidates
            if c.get("score") is not None
            and (bd(c, "dataCompleteness") is None or bd(c, "dataCompleteness") >= completeness_floor)
        ]
        excluded = len(all_candidates) - len(candidates)
        if excluded:
            log.info(f"[TradeAI Identify] Excluded {excluded} ticker(s) below dataCompleteness floor {completeness_floor} or with no score")

        seen = set()  # held tickers now compete for slots on merit
        result = []

        CANDIDATE_SLOTS = 30  # 8 Momentum + 8 Reversal + 8 SmartMoney + 6 Dream fallback

        # ── BUCKET 1: MOMENTUM (8 slots) ──────────────────────────────────────
        # RSI band (configurable) + price above MA50, optionally requiring a
        # volume surge. Fast signal: money already moving, confirm with trend
        momentum_rsi_min = float(identify_rules.get("momentumRsiMin", 45))
        momentum_rsi_max = float(identify_rules.get("momentumRsiMax", 70))
        momentum_vol_min = identify_rules.get("momentumVolumeRatioMin")
        momentum_pool = [
            c for c in candidates
            if c["ticker"] not in seen
            and bd(c, "rsi") is not None
            and momentum_rsi_min <= bd(c, "rsi") <= momentum_rsi_max
            and bd(c, "aboveMa50") is True
            # Only enforced once a value is set (via tuning proposal or by
            # hand) AND the ticker actually has volume data — missing data
            # never excludes, matching the dataCompleteness lesson.
            and (momentum_vol_min is None or bd(c, "volumeRatio") is None
                 or bd(c, "volumeRatio") >= momentum_vol_min)
        ]
        # Rank the RSI/aboveMa50-eligible pool by dream score, not raw RSI.
        # RSI/aboveMa50 still DEFINE "momentum" (the eligibility filter
        # above is unchanged) — but taking the top 8 by RSI alone let a
        # fundamentally weak name bump a fundamentally strong one off the
        # shortlist purely for being a couple RSI points more overbought.
        # Ported from stock-analysis, where it was backtested on 14 days of
        # shadow data (11/14 days better); TSX has no graded shadow history
        # yet to re-test it on.
        momentum_pool.sort(key=lambda c: c.get("score", 0) or 0, reverse=True)
        for c in momentum_pool[:8]:
            if len(result) >= CANDIDATE_SLOTS: break
            ticker = c["ticker"]
            seen.add(ticker)
            rsi_val = bd(c, "rsi")
            reason = f"Momentum: RSI {rsi_val} (trending, not overextended), dream score {c.get('score', '?')} — ranked by score among RSI-eligible names"
            result.append({"ticker": ticker, "reason": reason, "source": c.get("source", "Dream"), "bucket": "Momentum"})
        log.info(f"[TradeAI Identify] Bucket 1 Momentum: {[r['ticker'] for r in result if r.get('bucket')=='Momentum']}")

        # ── BUCKET 2: REVERSAL (8 slots) ──────────────────────────────────────
        # RSI < 40 (oversold) + dream score 20-60 (not trash, not already priced in)
        # Goal: catch accumulation before it shows — early-stage signal
        reversal_pool = [
            c for c in candidates
            if c["ticker"] not in seen
            and bd(c, "rsi") is not None
            and bd(c, "rsi") < 40
            and 20 <= c.get("score", 0) <= 60
        ]
        # Ascending by RSI (most oversold first) — explicit None-check
        # instead of `or 99`, which would wrongly sort a literal RSI of 0
        # to the bottom instead of the top.
        reversal_pool.sort(key=lambda c: bd(c, "rsi") if bd(c, "rsi") is not None else 99)
        for c in reversal_pool[:8]:
            if len(result) >= CANDIDATE_SLOTS: break
            ticker = c["ticker"]
            seen.add(ticker)
            rsi_val = bd(c, "rsi")
            score = c.get("score", "?")
            reason = f"Reversal: RSI {rsi_val} oversold, dream score {score} — early accumulation signal"
            result.append({"ticker": ticker, "reason": reason, "source": c.get("source", "Dream"), "bucket": "Reversal"})
        log.info(f"[TradeAI Identify] Bucket 2 Reversal: {[r['ticker'] for r in result if r.get('bucket')=='Reversal']}")

        # ── BUCKET 3: SMART MONEY (8 slots) ───────────────────────────────────
        # High gross margin + revenue growth + score below top tier
        # Dream score 35-70: fundamentally strong but not yet consensus buy
        smart_pool = [
            c for c in candidates
            if c["ticker"] not in seen
            and bd(c, "grossMargin") is not None
            and bd(c, "grossMargin") >= 40
            and bd(c, "revenueGrowth") is not None
            and bd(c, "revenueGrowth") >= 15
            and 35 <= c.get("score", 0) <= 75
        ]
        # Sort by revenue growth — accelerating fundamentals
        smart_pool.sort(key=lambda c: bd(c, "revenueGrowth") or 0, reverse=True)
        for c in smart_pool[:8]:
            if len(result) >= CANDIDATE_SLOTS: break
            ticker = c["ticker"]
            seen.add(ticker)
            rev = bd(c, "revenueGrowth")
            gm = bd(c, "grossMargin")
            reason = f"Smart Money: rev growth {rev}%, margin {gm}% — strong fundamentals, not yet consensus"
            result.append({"ticker": ticker, "reason": reason, "source": c.get("source", "Dream"), "bucket": "SmartMoney"})
        log.info(f"[TradeAI Identify] Bucket 3 SmartMoney: {[r['ticker'] for r in result if r.get('bucket')=='SmartMoney']}")

        # ── BUCKET 3b: SMART MONEY+ (8 slots) ─────────────────────────────────
        # Same fundamental screen as SmartMoney (margin + revenue growth), but
        # for candidates whose dream score is ABOVE SmartMoney's 75 ceiling —
        # i.e. the fundamentals earned them a spot on their own merits, they're
        # just no longer "under the radar." Exists so these don't fall through
        # to the Dream/Filler catch-all, which has no real thesis and (so far)
        # the weakest track record of any bucket.
        smart_plus_pool = [
            c for c in candidates
            if c["ticker"] not in seen
            and bd(c, "grossMargin") is not None
            and bd(c, "grossMargin") >= 40
            and bd(c, "revenueGrowth") is not None
            and bd(c, "revenueGrowth") >= 15
            and c.get("score", 0) > 75
        ]
        smart_plus_pool.sort(key=lambda c: bd(c, "revenueGrowth") or 0, reverse=True)
        for c in smart_plus_pool[:8]:
            if len(result) >= CANDIDATE_SLOTS: break
            ticker = c["ticker"]
            seen.add(ticker)
            rev = bd(c, "revenueGrowth")
            gm = bd(c, "grossMargin")
            score = c.get("score", 0)
            reason = f"Smart Money+: rev growth {rev}%, margin {gm}%, dream {score} — strong fundamentals AND already a top scorer"
            result.append({"ticker": ticker, "reason": reason, "source": c.get("source", "Dream"), "bucket": "SmartMoney+"})
        log.info(f"[TradeAI Identify] Bucket 3b SmartMoney+: {[r['ticker'] for r in result if r.get('bucket')=='SmartMoney+']}")

        # ── FILLER: top dream scorers if any bucket came up short ─────────────
        candidate_count = len(result)
        if candidate_count < CANDIDATE_SLOTS:
            filler_pool = [c for c in candidates if c["ticker"] not in seen]
            filler_pool.sort(key=lambda c: c.get("score", 0), reverse=True)
            for c in filler_pool[:6]:
                if len(result) >= CANDIDATE_SLOTS:
                    break
                ticker = c["ticker"]
                seen.add(ticker)
                reason = f"Dream score {c['score']}/100 (filler)"
                result.append({"ticker": ticker, "reason": reason, "source": c.get("source", "Dream"), "bucket": "Dream"})
            log.info(f"[TradeAI Identify] Filler added: {len(result)} total")

        # Clear first so frontend sees empty list, then add one by one
        existing = load_json(TRADE_AI_CANDIDATES_PATH, {})
        existing["identified"] = []
        existing["identifiedAt"] = ts()
        existing["aiAssessments"] = {}
        save_json(TRADE_AI_CANDIDATES_PATH, existing)
        log.info("[TradeAI Identify] Cleared previous candidates, adding incrementally...")
        for item in result:
            existing = load_json(TRADE_AI_CANDIDATES_PATH, {})
            existing["identified"].append(item)
            existing["identifiedAt"] = ts()
            save_json(TRADE_AI_CANDIDATES_PATH, existing)
            log.info(f"[TradeAI Identify] Added: {item['ticker']} ({item['bucket']})")
        buckets = {}
        for r in result:
            b = r.get("bucket", "Dream")
            buckets[b] = buckets.get(b, 0) + 1
        fetch_status = {"running": False, "message": f"TradeAI identify done: {len(result)} candidates | {buckets}", "operation": None, "current": len(result), "total": len(result), "current_ticker": ""}
        log.info(f"=== TradeAI Identify Complete: {len(result)} tickers | buckets:{buckets} ===")
    except Exception as e:
        log.error(f"TradeAI identify FAIL: {e}")
        fetch_status = {"running": False, "message": f"TradeAI identify FAIL: {e}", "operation": None, "current": 0, "total": 0, "current_ticker": ""}


def run_tradeai_fetch():
    global fetch_status, stop_flag
    stop_flag = False
    total = 0
    fetch_status = {"running": True, "message": "TradeAI: fetching detailed info...", "operation": "tradeai_fetch", "current": 0, "total": 0, "current_ticker": ""}
    log.info("=== TradeAI Fetch Started ===")
    try:
        candidates_data = load_json(TRADE_AI_CANDIDATES_PATH, {})
        identified = candidates_data.get("identified", [])
        if not identified:
            fetch_status = {"running": False, "message": "TradeAI fetch: no identified tickers. Run Identify first.", "operation": None, "current": 0, "total": 0, "current_ticker": ""}
            return
        trades = load_trades_ai()
        held_tickers = list(trades.get("holdings", {}).keys())
        identified_tickers = {item["ticker"] for item in identified}
        fetch_list = list(identified) + [{"ticker": t, "bucket": "Holding"} for t in held_tickers if t not in identified_tickers]
        total = len(fetch_list)
        fetch_status["total"] = total
        details = candidates_data.get("details", {})
        for idx, item in enumerate(fetch_list, 1):
            if stop_flag:
                log.info(f"[TradeAI Fetch] Stopped by user at {idx}/{total}")
                candidates_data["details"] = details
                candidates_data["detailsFetchedAt"] = ts()
                save_json(TRADE_AI_CANDIDATES_PATH, candidates_data)
                fetch_status = {"running": False, "message": f"TradeAI fetch stopped: {len(details)}/{total} fetched", "operation": None, "current": idx, "total": total, "current_ticker": ""}
                return
            ticker = item["ticker"]
            fetch_status["current"] = idx
            fetch_status["current_ticker"] = ticker
            fetch_status["message"] = f"TradeAI fetch: {idx} of {total} — {ticker}"
            log.info(f"--- TradeAI fetch detail: {ticker} [{idx}/{total}] ---")
            try:
                detail = fetch_trade_detail(ticker)
                if detail:
                    detail["fetchCompletedAt"] = ts()
                    details[ticker] = detail
                    candidates_data["details"] = details
                    # Store RSI snapshot history for Reversal confirmation (rising RSI check)
                    rsi_history = candidates_data.get("rsiHistory", {})
                    if ticker not in rsi_history:
                        rsi_history[ticker] = []
                    rsi_val = detail.get("rsi14")
                    if rsi_val is not None:
                        rsi_history[ticker].append({"rsi": rsi_val, "ts": ts()})
                        rsi_history[ticker] = rsi_history[ticker][-5:]  # keep last 5 readings
                    candidates_data["rsiHistory"] = rsi_history
                    save_json(TRADE_AI_CANDIDATES_PATH, candidates_data)
                    log.info(f"TradeAI fetch OK: {ticker} | price:{detail.get('price')} rsi:{rsi_val}")
            except Exception as e:
                log.error(f"TradeAI fetch FAIL: {ticker} | {e}")
            # News is no longer fetched here — run_news_refresh (09:20,
            # 11:47 and 14:47, plus the REFRESH NEWS button) keeps it fresh for every
            # ticker before the 10:00+ Recommend runs read it.
        candidates_data["details"] = details
        candidates_data["detailsFetchedAt"] = ts()
        save_json(TRADE_AI_CANDIDATES_PATH, candidates_data)
        fetch_status = {"running": False, "message": f"TradeAI fetch done: {len(details)} tickers enriched", "operation": None, "current": total, "total": total, "current_ticker": ""}
        log.info(f"=== TradeAI Fetch Complete: {len(details)} tickers ===")
    except Exception as e:
        log.error(f"TradeAI fetch runner FAIL: {e}")
        fetch_status = {"running": False, "message": f"TradeAI fetch FAIL: {e}", "operation": None, "current": 0, "total": total, "current_ticker": ""}


def run_tradeai_price_refresh():
    """Refresh only fast-changing fields for the fixed daily shortlist."""
    global fetch_status, stop_flag
    stop_flag = False
    candidates_data = load_json(TRADE_AI_CANDIDATES_PATH, {})
    identified = candidates_data.get("identified", [])
    details = candidates_data.get("details", {})
    holdings = load_trades_ai().get("holdings", {})
    tickers = [item["ticker"] for item in identified]
    tickers += [ticker for ticker in holdings if ticker not in tickers]
    total = len(tickers)
    fetch_status = {"running": True, "message": "TradeAI: refreshing prices...", "operation": "tradeai_prices", "current": 0, "total": total, "current_ticker": ""}
    try:
        rsi_history = candidates_data.get("rsiHistory", {})
        for idx, ticker in enumerate(tickers, 1):
            if stop_flag:
                break
            fetch_status.update({"current": idx, "current_ticker": ticker, "message": f"TradeAI prices: {idx} of {total} — {ticker}"})
            price_data = fetch_price_only(ticker)
            if not price_data or ticker not in details:
                continue
            details[ticker].update(price_data)
            details[ticker]["priceFetchedAt"] = ts()
            rsi = price_data.get("rsi14")
            if rsi is not None:
                history = rsi_history.setdefault(ticker, [])
                history.append({"rsi": rsi, "ts": ts()})
                rsi_history[ticker] = history[-5:]
        candidates_data["details"] = details
        candidates_data["rsiHistory"] = rsi_history
        candidates_data["pricesRefreshedAt"] = ts()
        save_json(TRADE_AI_CANDIDATES_PATH, candidates_data)
        fetch_status = {"running": False, "message": f"TradeAI prices refreshed: {total} tickers", "operation": None, "current": total, "total": total, "current_ticker": ""}
    except Exception as e:
        log.error(f"TradeAI price refresh FAIL: {e}")
        fetch_status = {"running": False, "message": f"TradeAI price refresh FAIL: {e}", "operation": None, "current": 0, "total": total, "current_ticker": ""}


def _rule_based_assessment(ticker, detail, news_items, bucket):
    """Create the assessment contract used by Recommend without an AI call."""
    detail = detail or {}
    value = lambda key, default=0: detail.get(key) if detail.get(key) is not None else default
    # Missing RSI stays missing — defaulting it to 50 landed inside the
    # 45-65 reward band and gave a no-data ticker +12.
    rsi = float(detail["rsi14"]) if detail.get("rsi14") is not None else None
    change1d = float(value("change1d") or 0)
    change7d = float(value("change7d") or 0)
    change30d = float(value("change30d") or 0)
    revenue_growth = float(value("revenueGrowth") or 0)
    gross_margin = float(value("grossMargin") or 0)
    analyst_upside = float(value("analystUpside") or 0)
    volume_ratio = float(value("volumeRatio") or 0)
    bb_percent = detail.get("bbPercent")

    momentum = (change1d > 0) + (change7d > 0) + (change30d > 0)
    if momentum == 3 or (change7d > 2 and change30d > 5):
        trajectory = "Accelerating"
    elif momentum == 0 or (change7d < -2 and change30d < 0):
        trajectory = "Decelerating"
    else:
        trajectory = "Stable"

    if revenue_growth >= 20 and change30d > 0:
        stage = "Growth"
    elif revenue_growth >= 10 and gross_margin >= 40:
        stage = "Mature"
    elif revenue_growth < 0 and change30d < 0:
        stage = "Declining"
    else:
        stage = "Early-Stage"

    positive_words = ("beat", "growth", "surge", "upgrade", "profit", "strong", "record", "approve")
    negative_words = ("miss", "cut", "loss", "downgrade", "risk", "lawsuit", "fall", "weak")
    news_text = " ".join(str(n.get("title", "")).lower() for n in news_items or [])
    positive_news = sum(news_text.count(word) for word in positive_words)
    negative_news = sum(news_text.count(word) for word in negative_words)
    news_sentiment = "Bullish" if positive_news > negative_news else "Bearish" if negative_news > positive_news else "Neutral"

    # Recent clues only — revenue growth and gross margin are long-term
    # fundamentals already scored by the Dream score, and scoring them here
    # too made the two scores ~0.67 rank-correlated. Base raised 35 -> 44 to
    # offset their removal so the Strong Buy/Buy/Hold/Avoid mix (which
    # drives forced "Avoid" sells) stays about the same. Same formula as
    # stock-analysis: the removed points average +10.6 on TSX's identified
    # pool vs +9.5 on the US one, and +6.4 vs +6.9 across each Dream universe.
    score = 44
    score += 15 if change30d > 10 else 8 if change30d > 0 else -8 if change30d < -10 else 0
    if rsi is not None:
        score += 12 if 45 <= rsi <= 65 else 5 if 35 <= rsi < 45 else -8 if rsi > 75 else 0
    score += 8 if analyst_upside >= 20 else 4 if analyst_upside >= 10 else 0
    score += 5 if volume_ratio >= 1.5 else 0
    score += 5 if trajectory == "Accelerating" else -5 if trajectory == "Decelerating" else 0
    score += 5 if news_sentiment == "Bullish" else -5 if news_sentiment == "Bearish" else 0
    if bucket == "Reversal" and rsi is not None and rsi < 40:
        score += 8
    if bb_percent is not None and bb_percent < 0.1:
        score += 5
    score = max(0, min(100, round(score)))
    signal = "Strong Buy" if score >= 80 else "Buy" if score >= 60 else "Hold" if score >= 40 else "Avoid"
    sentiment = "Bullish" if score >= 60 else "Bearish" if score < 40 else "Neutral"
    return {
        "ticker": ticker, "aiScore": score, "buySignal": signal,
        "sentiment": sentiment, "newsSentiment": news_sentiment,
        "trajectory": trajectory, "stage": stage, "confidence": "Rule-based",
        "reasoning": f"Rule assessment: {trajectory.lower()} price trend, {stage.lower()} profile, and {news_sentiment.lower()} recent-news signal.",
        "forecastReason": f"Rule forecast uses RSI {f'{rsi:.1f}' if rsi is not None else 'n/a'}, momentum {change7d:.1f}% over 7 days, and the current technical range.",
        "promptSent": "Rule-based assessment; no AI provider used.",
    }


def run_tradeai_analyze():
    global fetch_status, stop_flag
    stop_flag = False
    total = 0
    fetch_status = {"running": True, "message": "TradeAI: running rule assessment...", "operation": "tradeai_analyze", "current": 0, "total": 0, "current_ticker": ""}
    log.info("=== TradeAI Rule Assessment Started ===")
    try:
        candidates_data = load_json(TRADE_AI_CANDIDATES_PATH, {})
        identified = candidates_data.get("identified", [])
        details = candidates_data.get("details", {})
        news_db = load_json(NEWS_PATH, {})
        held_tickers = list(load_trades_ai().get("holdings", {}).keys())
        identified_map = {i["ticker"]: i for i in identified}
        eligible = [identified_map.get(t, {"ticker": t, "bucket": "Holding"}) for t in held_tickers if details.get(t)]
        eligible += [i for i in identified if details.get(i["ticker"]) and i["ticker"] not in held_tickers]
        total = len(eligible)
        if not total:
            fetch_status = {"running": False, "message": "Rule assessment: no fetched candidates", "operation": None, "current": 0, "total": 0, "current_ticker": ""}
            return
        assessments = candidates_data.get("aiAssessments", {})
        for idx, item in enumerate(eligible, 1):
            if stop_flag:
                break
            ticker = item["ticker"]
            fetch_status.update({"current": idx, "current_ticker": ticker, "message": f"TradeAI rule assessment: {idx} of {total} — {ticker}"})
            assessments[ticker] = _rule_based_assessment(ticker, details.get(ticker), news_db.get(ticker, {}).get("articles", [])[:5], item.get("bucket", "Dream"))
        candidates_data["aiAssessments"] = assessments
        candidates_data["aiAnalyzedAt"] = ts()
        save_json(TRADE_AI_CANDIDATES_PATH, candidates_data)
        fetch_status = {"running": False, "message": f"Rule assessment done: {len(eligible)} assessed", "operation": None, "current": total, "total": total, "current_ticker": ""}
        log.info(f"=== TradeAI Rule Assessment Complete: {len(eligible)} tickers ===")
    except Exception as e:
        log.error(f"TradeAI rule assessment FAIL: {e}")
        fetch_status = {"running": False, "message": f"Rule assessment FAIL: {e}", "operation": None, "current": 0, "total": total, "current_ticker": ""}


def run_tradeai_recommend():
    global fetch_status
    fetch_status = {"running": True, "message": "TradeAI: running AI recommendation...", "operation": "tradeai_recommend"}
    log.info("=== TradeAI Recommend Started ===")
    try:
        trades = load_trades_ai()
        balance = trades.get("balance", STARTING_BALANCE)
        holdings = trades.get("holdings", {})
        transactions = trades.get("transactions", [])

        candidates_data = load_json(TRADE_AI_CANDIDATES_PATH, {})
        identified = candidates_data.get("identified", [])
        details = candidates_data.get("details", {})
        ai_assessments = candidates_data.get("aiAssessments", {})
        dream = load_json(DREAM_PATH, {})
        dream_map = {c["ticker"]: c for c in dream.get("candidates", [])}

        # ── Load scoring config (user-tweakable weights) ─────────────────────
        cfg = load_scoring_config()
        BKT_WEIGHTS     = cfg["bucketWeights"]
        TRAJ_BONUS      = cfg["trajectoryBonus"]
        STAGE_BONUS_CFG = cfg["stageBonus"]
        SH              = cfg["sharedBonuses"]
        RB              = cfg["reversalBonuses"]
        SF              = cfg["signalForecast"]
        SELL_CFG        = cfg["sellThresholds"]
        EXEC_CFG        = cfg.get("execution", {})
        REGIME_CFG      = cfg.get("regime", {})
        BKT_SC          = cfg.get("breakoutScore", {})
        SD_PENALTY      = cfg.get("sectorDiversity", {}).get("penalty", -15)
        OVEREXT_CFG     = cfg.get("overextension", {})
        ENTRY_GUARDS    = cfg.get("entryGuards", {})
        pm              = cfg.get("purchaseMatrix", {})
        max_slots       = int(pm.get("slots", 5))
        identify_rules  = cfg.get("identifyRules", {})

        # Load today's signal forecasts for boost calculation
        today_str = datetime.now(TZ).strftime("%Y-%m-%d")
        fc_db = load_json(SIGNAL_FORECAST_PATH, {"snapshots": []})
        today_snap = next((s for s in fc_db.get("snapshots", []) if s.get("date") == today_str), None)
        fc_map = {}
        if today_snap:
            for row in today_snap.get("rows", []):
                fc_map[row["ticker"]] = row

        # Build previous composite score lookup for sell reason delta
        prev_scored = {s["ticker"]: s["composite"] for s in (candidates_data.get("lastRecommend") or {}).get("scored", [])}

        # Build candidate lookup for bucket-aware scoring
        cand_map = {item["ticker"]: item for item in identified}

        # Build held sector set for diversity penalty
        held_sectors = set()
        for ht in holdings:
            hs = details.get(ht, {}).get("sector", "")
            if hs:
                held_sectors.add(hs)

        scoring_pool = list(identified) + [{"ticker": t, "bucket": "Dream"} for t in holdings if t not in cand_map]

        rec_version = current_recommend_version()

        # ── Fetch fresh market prices — never trade on cached detail prices ──
        # SPY rides along as the benchmark; ^VIX as the regime gauge.
        fetch_status["message"] = "TradeAI: fetching live prices..."
        live_prices = fetch_live_prices([item["ticker"] for item in scoring_pool] + ["SPY", "^VIX"])
        spy_now = live_prices.get("SPY")
        vix_now = live_prices.get("^VIX")
        missing_live = [item["ticker"] for item in scoring_pool if item["ticker"] not in live_prices]
        if missing_live:
            log.warning(f"TradeAI recommend: no live price for {missing_live} — cached prices will NOT be used for trades")

        # Yesterday's closes — powers the daily-drop stop for holdings, and
        # the already-down-today entry guard for buy candidates below.
        prev_close_tickers = list({*holdings.keys(), *[item["ticker"] for item in scoring_pool]})
        prev_closes = fetch_prev_closes(prev_close_tickers) if prev_close_tickers else {}

        scored = []
        for item in scoring_pool:
            ticker = item["ticker"]
            bucket = item.get("bucket", "Dream")
            detail = details.get(ticker, {})
            ai = ai_assessments.get(ticker, {})
            d_data = dream_map.get(ticker, {})
            dream_score = d_data.get("score") or 0  # score can be None (SCORE_INCOMPLETE)
            price = live_prices.get(ticker) or detail.get("price") or d_data.get("price")

            ai_score    = ai.get("aiScore") or 0
            trajectory  = ai.get("trajectory", "")
            stage       = ai.get("stage", "")
            bkt_key     = bucket if bucket in BKT_WEIGHTS else "Dream"
            w           = BKT_WEIGHTS[bkt_key]

            # ── Breakout Score (all buckets) ──────────────────────────────────
            rsi        = detail.get("rsi14") or 50
            bb_percent = detail.get("bbPercent")
            vol_ratio  = detail.get("volumeRatio") or 0
            short_fl   = detail.get("shortFloat") or 0

            breakout_score = 0
            # RSI turning up from oversold zone
            if BKT_SC.get("rsiTurning_low", 25) <= rsi <= BKT_SC.get("rsiTurning_high", 42):
                breakout_score += BKT_SC.get("rsiTurning", 8)
            # BB% squeeze — low volatility about to break out
            if bb_percent is not None and bb_percent < BKT_SC.get("bbSqueeze_threshold", 0.20):
                breakout_score += BKT_SC.get("bbSqueeze", 8)
            # Short squeeze potential
            if short_fl * 100 >= BKT_SC.get("shortSqueeze_threshold", 15.0):
                breakout_score += BKT_SC.get("shortSqueeze", 5)
            # Volume building (not yet surging — anticipatory)
            if BKT_SC.get("volumeBuilding_low", 1.2) <= vol_ratio <= BKT_SC.get("volumeBuilding_high", 1.9):
                breakout_score += BKT_SC.get("volumeBuilding", 4)

            # ── Sector diversity penalty ───────────────────────────────────────
            ticker_sector = detail.get("sector", "")
            sector_penalty = 0
            if ticker_sector and ticker_sector in held_sectors and ticker not in holdings:
                sector_penalty = SD_PENALTY
                log.info(f"[SectorPenalty] {ticker} sector:{ticker_sector} matches held position → {sector_penalty} pts")

            # ── Shared bonuses ────────────────────────────────────────────────
            volume_bonus = (SH["volumeSurgeHigh"] if vol_ratio >= SH["volumeSurgeHigh_threshold"]
                            else SH["volumeSurgeMed"] if vol_ratio >= SH["volumeSurgeMed_threshold"] else 0)

            upside        = detail.get("analystUpside") or 0
            analyst_bonus = (SH["analystUpsideHigh"] if upside >= SH["analystUpsideHigh_threshold"]
                             else SH["analystUpsideMed"] if upside >= SH["analystUpsideMed_threshold"] else 0)

            news_sentiment = ai.get("newsSentiment", "")
            news_bonus     = SH["newsBullish"] if news_sentiment == "Bullish" else (SH["newsNeutral"] if news_sentiment == "Neutral" else 0)

            # ── Trajectory & stage bonuses (bucket-specific) ──────────────────
            traj_cfg  = TRAJ_BONUS.get(bkt_key, {})
            trajectory_bonus = traj_cfg.get(trajectory, 0) if trajectory else 0

            stage_cfg  = STAGE_BONUS_CFG.get(bkt_key, {})
            stage_bonus = stage_cfg.get(stage, 0) if stage else 0

            # ── Bucket-specific base composite (scaled to 70% for score spread) ──
            if bucket == "Reversal":
                rsi_bonus  = (RB["rsiDeep"] if rsi < RB["rsiDeep_threshold"]
                              else RB["rsiMed"] if rsi < RB["rsiMed_threshold"]
                              else RB["rsiMild"] if rsi < RB["rsiMild_threshold"] else 0)
                bb_bonus   = RB["bbLowerBand"] if (bb_percent is not None and bb_percent < RB["bbLowerBand_threshold"]) else 0
                base       = ((dream_score * w["dream"]) + (ai_score * w["ai"])) * 0.70
                composite  = base + rsi_bonus + bb_bonus + analyst_bonus + news_bonus

            else:
                base      = ((dream_score * w["dream"]) + (ai_score * w["ai"])) * 0.70
                composite = base + trajectory_bonus + stage_bonus + volume_bonus + analyst_bonus + news_bonus

            # ── Insider activity bonus (all buckets) ──────────────────────────
            insider_net_status = detail.get("insiderNet")
            insider_bonus = (SH.get("insiderNetBuy", 5) if insider_net_status == "Net Buyer"
                             else (SH.get("insiderNetSell", -3) if insider_net_status == "Net Seller" else 0))

            # ── Overextension penalty (all buckets) — Accelerating + deeply
            # overbought + already run up hard is the exact profile that gets
            # hit hardest in a risk-off wobble. Only fires alongside
            # Accelerating so it doesn't punish a stock merely having a good
            # month — specifically "extended AND still scored like it has
            # more room to run." (Ported from stock-analysis.)
            change_30d = detail.get("change30d") or 0
            overextension_penalty = 0
            if (trajectory == "Accelerating"
                    and rsi >= OVEREXT_CFG.get("rsiThreshold", 75)
                    and change_30d >= OVEREXT_CFG.get("change30dThreshold", 25.0)):
                overextension_penalty = OVEREXT_CFG.get("penalty", -15)
                log.info(f"[Overextension] {ticker} RSI:{rsi} change30d:{change_30d}% Accelerating → {overextension_penalty} pts")

            # ── Breakout score + sector penalty (all buckets) ─────────────────
            composite = composite + breakout_score + sector_penalty + insider_bonus + overextension_penalty

            # ── Signal Forecast Boost ─────────────────────────────────────────
            signal_boost = 0
            signal_fc_direction = ""
            signal_fc_target = None
            signal_fc_return = None
            if SF.get("enabled") and ticker in fc_map and price:
                fc = fc_map[ticker]
                direction = fc.get("direction", "")
                signal_fc_direction = direction
                if direction == "Bullish":
                    signal_fc_target = fc.get("forecastHigh")
                elif direction == "Bearish":
                    signal_fc_target = fc.get("forecastLow")
                else:
                    signal_fc_target = fc.get("forecastMid")
                if signal_fc_target:
                    exp_return_pct = (signal_fc_target - price) / price * 100
                    signal_fc_return = round(exp_return_pct, 2)
                    neutral_zone = SF.get("neutralZone", 1.0)
                    if abs(exp_return_pct) > neutral_zone:
                        raw_boost = exp_return_pct * SF.get("returnMultiplier", 3.0)
                        signal_boost = round(max(SF["maxPenalty"], min(SF["maxBoost"], raw_boost)), 1)

            composite = round(min(100, max(0, composite + signal_boost)), 1)

            log.info(
                f"[Score] {ticker} bucket:{bucket} dream:{dream_score} ai:{ai_score} "
                f"traj:{trajectory_bonus} stage:{stage_bonus} "
                f"vol:{volume_bonus} analyst:{analyst_bonus} news:{news_bonus} "
                f"breakout:{breakout_score} sector_pen:{sector_penalty} insider:{insider_bonus} "
                f"overext:{overextension_penalty} "
                f"fc_dir:{signal_fc_direction} fc_return:{signal_fc_return}% fc_boost:{signal_boost} "
                f"composite:{composite}"
            )

            scored.append({
                "ticker": ticker,
                "bucket": bucket,
                "price": price,
                "dreamScore": dream_score,
                "aiScore": ai_score,
                "buySignal": ai.get("buySignal", "Unknown"),
                "trajectory": trajectory,
                "stage": stage,
                "sentiment": ai.get("sentiment", "—"),
                "reasoning": ai.get("reasoning", ""),
                "breakoutScore": breakout_score,
                "sectorPenalty": sector_penalty,
                "overextensionPenalty": overextension_penalty,
                "composite": composite,
                "hasAI": ai.get("aiScore") is not None,
                "signalFcDirection": signal_fc_direction,
                "signalFcTarget": signal_fc_target,
                "signalFcReturn": signal_fc_return,
                "signalFcBoost": signal_boost,
                "isHeld": ticker in holdings,
            })

        scored.sort(key=lambda x: x["composite"], reverse=True)
        top_slots = [s["ticker"] for s in scored[:max_slots]]

        # Grade old snapshots now; recording today's happens at the end of
        # this function once qualify/buy outcomes are known (see below).
        try:
            _grade_shadow_snapshots()
        except Exception as she:
            log.error(f"[Shadow] grade FAIL: {she}")

        sells = []
        for ticker, pos in list(holdings.items()):
            scored_entry   = next((s for s in scored if s["ticker"] == ticker), None)
            composite      = scored_entry["composite"] if scored_entry else 0
            buy_signal     = scored_entry["buySignal"] if scored_entry else "Unknown"
            current_price  = fetch_live_prices([ticker]).get(ticker)
            current_price_fetched_at = ts() if current_price else None
            purchase_price = pos.get("purchasePrice") or current_price or 0

            if not current_price:
                log.warning(f"TradeAI SELL-CHECK skip: {ticker} — no live price, holding position (will not trade on stale data)")
                continue

            # Days held — soft (rotation) sells are blocked before minHoldDays
            days_held = _days_held(pos)

            # ── Peak price tracking ───────────────────────────────────────────
            peak_price = pos.get("peakPrice") or purchase_price
            if current_price and current_price > peak_price:
                peak_price = current_price
                pos["peakPrice"] = round(peak_price, 4)
                log.info(f"[Peak] {ticker} new peak: ${peak_price:.2f}")

            pct_from_purchase = ((current_price - purchase_price) / purchase_price * 100) if current_price and purchase_price else 0
            pct_from_peak     = ((current_price - peak_price) / peak_price * 100) if current_price and peak_price else 0
            prev_close        = prev_closes.get(ticker)
            day_change_pct    = ((current_price - prev_close) / prev_close * 100) if current_price and prev_close else None

            # ── Stop-loss checks (price-based, highest priority) ──────────────
            stop_loss_hard  = _effective_stop_loss_pct(ticker, details, SELL_CFG)
            stop_loss_soft  = SELL_CFG.get("stopLossSoft", 6.0)
            trailing_pct    = SELL_CFG.get("trailingStopPeak", 10.0)
            daily_drop_stop = SELL_CFG.get("dailyDropStop", 8.0)

            stop_loss_triggered = False
            stop_loss_reason    = ""
            if pct_from_purchase <= -stop_loss_hard:
                stop_loss_triggered = True
                atr_pct = (details.get(ticker) or {}).get("atr14Pct")
                atr_note = f", ATR-based ({SELL_CFG.get('atrStopMultiplier', 2.0)}x{atr_pct}%)" if atr_pct is not None else ", ATR unavailable — used hard cap"
                stop_loss_reason = f"Stop-loss triggered: down {abs(pct_from_purchase):.1f}% from purchase ${purchase_price:.2f} (stop at -{stop_loss_hard:.1f}%{atr_note})"
            elif day_change_pct is not None and day_change_pct <= -daily_drop_stop:
                stop_loss_triggered = True
                stop_loss_reason = f"Daily-drop stop triggered: down {abs(day_change_pct):.1f}% vs yesterday's close ${prev_close:.2f} (stop at -{daily_drop_stop}%)"
            elif pct_from_purchase <= -stop_loss_soft and composite < 50:
                stop_loss_triggered = True
                stop_loss_reason = f"Stop-loss triggered: down {abs(pct_from_purchase):.1f}% from purchase and composite {composite}/100 below 50"
            elif pct_from_peak <= -trailing_pct and pct_from_purchase > 0:
                stop_loss_triggered = True
                stop_loss_reason = f"Trailing stop triggered: down {abs(pct_from_peak):.1f}% from peak ${peak_price:.2f} — protecting gains"

            # ── Score-based sell checks ───────────────────────────────────────
            hard_sell = buy_signal == "Avoid" or composite < SELL_CFG["hardSellFloor"]
            soft_sell = ticker not in top_slots and composite < SELL_CFG["softSellFloor"]

            # Missing-data guard: a held ticker with NO fresh AI assessment
            # computes composite from aiScore=0 — that's absent data, not a
            # score collapse. Suppress score-based sells; price-based stops
            # above remain fully active so real losers still get cut.
            has_ai = bool(scored_entry and scored_entry.get("hasAI"))
            if not has_ai and (hard_sell or soft_sell):
                log.warning(f"TradeAI SELL-GUARD: {ticker} has no fresh AI assessment — "
                            f"score-based sell suppressed (composite {composite} is data-starved); stops still active")
                hard_sell = soft_sell = False

            # Hold-time hysteresis: rotation (soft) sells only after minHoldDays.
            # Stop-loss and hard sells are exempt — risk exits stay immediate.
            min_hold_days = float(SELL_CFG.get("minHoldDays", 3))
            if soft_sell and not (stop_loss_triggered or hard_sell):
                if days_held is not None and days_held < min_hold_days:
                    log.info(f"TradeAI HOLD (min-hold): {ticker} | held {days_held:.1f}d < {min_hold_days}d — rotation sell deferred")
                    soft_sell = False

            if stop_loss_triggered or hard_sell or soft_sell:
                sell_slip = _slippage_for(details.get(ticker), EXEC_CFG)
                price = round(current_price * (1 - sell_slip), 4)  # sell fill below market: spread + slippage
                if price:
                    proceeds = round(price * pos["shares"], 2)
                    balance += proceeds
                    if stop_loss_triggered:
                        sell_reason = stop_loss_reason
                    elif hard_sell and buy_signal == "Avoid":
                        sell_reason = f"AI signal: Avoid — {ai_assessments.get(ticker, {}).get('reasoning', 'deteriorating outlook')}"
                    elif composite < SELL_CFG["hardSellFloor"]:
                        sell_reason = f"Composite score collapsed to {composite}/100 — exiting position"
                    else:
                        prev_score = prev_scored.get(ticker)
                        delta_str  = ""
                        if prev_score is not None:
                            delta = composite - prev_score
                            delta_str = f" (was {prev_score}/100, delta {'+' if delta >= 0 else ''}{delta:.1f})"
                        sell_reason = f"Outside top 5 composite (score {composite}/100{delta_str}) — {ai_assessments.get(ticker, {}).get('reasoning', 'better opportunities available')}"
                    sells.append({
                        "ticker": ticker, "action": "SELL", "shares": pos["shares"],
                        "price": price, "amount": proceeds, "date": ts(),
                        "priceFetchedAt": current_price_fetched_at,
                        "balance": round(balance, 2), "reason": sell_reason,
                        "composite": composite, "buySignal": buy_signal,
                        "pctFromPurchase": round(pct_from_purchase, 2),
                        "pctFromPeak": round(pct_from_peak, 2),
                        "dayChangePct": round(day_change_pct, 2) if day_change_pct is not None else None,
                        "stopLoss": stop_loss_triggered,
                        "gainLoss": round((price - purchase_price) * pos["shares"], 2) if purchase_price else None,
                        "daysHeld": round(days_held, 1) if days_held is not None else None,
                        "purchasePrice": purchase_price, "purchasedAt": pos.get("purchasedAt"),
                    })
                    transactions.append(sells[-1])
                    trigger = "stop-loss" if stop_loss_triggered else ("hard" if hard_sell else "soft")
                    # Log closed-trade outcome with buy-time factors for scoring feedback
                    try:
                        _log_trade_outcome(ticker, pos, price, sell_reason, trigger, composite, spy_now, days_held, sell_rec_version=rec_version)
                    except Exception as oe:
                        log.error(f"[TradeOutcome] log FAIL: {ticker} | {oe}")
                    del holdings[ticker]
                    log.info(f"TradeAI SELL: {ticker} | shares:{pos['shares']} price:{price} proceeds:{proceeds} | trigger:{trigger} pct_purchase:{pct_from_purchase:.1f}% composite:{composite}")
            else:
                log.info(f"TradeAI HOLD: {ticker} | composite:{composite} signal:{buy_signal} pct_purchase:{pct_from_purchase:.1f}% — keeping position")

        # Load RSI history for Reversal entry confirmation
        rsi_history = candidates_data.get("rsiHistory", {})

        # Purchase Matrix config (pm, max_slots loaded earlier alongside cfg)
        max_deploy       = float(pm.get("maxDeployPct", 80)) / 100.0
        min_composite    = float(pm.get("minBuyComposite", 60))
        rotation_enabled = bool(pm.get("rotationEnabled", True))
        rotation_gap     = float(pm.get("rotationMinScoreGap", 15))
        rotation_cooldown = int(pm.get("rotationCooldownTradingDays", 5))
        deployable       = round(balance * max_deploy, 2)
        budget_per_slot  = round(deployable / max(1, max_slots), 2)
        log.info(f"TradeAI BUY config | balance:{balance} deployable:{deployable} slots:{max_slots} per_slot:{budget_per_slot} min_composite:{min_composite}")

        buys = []
        held_count = len(holdings)

        # ── Step 1: qualify ALL scored candidates ─────────────────────────────
        qualified = []
        skip_reasons = []
        for s in scored:
            ticker = s["ticker"]
            if ticker in holdings:
                log.info(f"TradeAI QUALIFY skip (already held): {ticker}")
                skip_reasons.append({"ticker": ticker, "bucket": s.get("bucket"), "composite": s.get("composite"), "reason": "Already held in portfolio"})
                continue
            if rotation_cooldown > 0:
                recent_sells = [t for t in transactions if t.get("ticker") == ticker and t.get("action") == "SELL" and t.get("date")]
                if recent_sells:
                    try:
                        last_sell_date = datetime.strptime(max(t["date"][:10] for t in recent_sells), "%Y-%m-%d").date()
                        trading_days_since = sum(
                            (last_sell_date + timedelta(days=offset)).weekday() < 5
                            for offset in range(1, (datetime.now(TZ).date() - last_sell_date).days + 1)
                        )
                        if trading_days_since < rotation_cooldown:
                            skip_reasons.append({
                                "ticker": ticker, "bucket": s.get("bucket"), "composite": s.get("composite"),
                                "reason": f"Re-entry cooldown: {rotation_cooldown - trading_days_since} trading day(s) remaining"
                            })
                            continue
                    except Exception:
                        pass
            if s.get("buySignal") == "Avoid":
                log.info(f"TradeAI QUALIFY skip (AI says Avoid): {ticker}")
                skip_reasons.append({"ticker": ticker, "bucket": s.get("bucket"), "composite": s.get("composite"), "reason": "AI buy signal is Avoid"})
                continue
            if s.get("bucket") == "Reversal":
                ticker_rsi_hist = rsi_history.get(ticker, [])
                if len(ticker_rsi_hist) >= 2:
                    rsi_today = ticker_rsi_hist[-1]["rsi"]
                    rsi_prev  = ticker_rsi_hist[-2]["rsi"]
                    if rsi_today <= rsi_prev:
                        log.info(f"TradeAI QUALIFY skip Reversal (RSI not rising): {ticker} | rsi:{rsi_prev}\u2192{rsi_today}")
                        skip_reasons.append({"ticker": ticker, "bucket": s.get("bucket"), "composite": s.get("composite"), "reason": f"Reversal RSI not yet rising ({rsi_prev}\u2192{rsi_today})"})
                        continue
                    log.info(f"TradeAI QUALIFY confirm Reversal (RSI rising): {ticker} | rsi:{rsi_prev}\u2192{rsi_today}")
                else:
                    log.info(f"TradeAI QUALIFY skip Reversal (insufficient RSI history): {ticker} | history:{len(ticker_rsi_hist)}")
                    skip_reasons.append({"ticker": ticker, "bucket": s.get("bucket"), "composite": s.get("composite"), "reason": "Insufficient RSI history for Reversal confirmation"})
                    continue
            price = s.get("price")
            if not price or price <= 0:
                log.warning(f"TradeAI QUALIFY skip (no price): {ticker}")
                skip_reasons.append({"ticker": ticker, "bucket": s.get("bucket"), "composite": s.get("composite"), "reason": "No valid price data"})
                continue
            if ticker not in live_prices:
                log.warning(f"TradeAI QUALIFY skip (no live price): {ticker} — refusing to buy on cached price")
                skip_reasons.append({"ticker": ticker, "bucket": s.get("bucket"), "composite": s.get("composite"), "reason": "No live price — cached data only"})
                continue
            # Already-down-today guard: refuse to buy a ticker that's deep
            # into a bad day already — buying at e.g. -7% intraday just to
            # get stopped out by the daily-drop stop an hour later wastes
            # the buy+sell slippage on a position that never had a chance.
            # (Ported from stock-analysis.)
            prev_close = prev_closes.get(ticker)
            if prev_close and price:
                already_down_pct = (prev_close - price) / prev_close * 100
                guard_pct = SELL_CFG.get("dailyDropStop", 8.0) * ENTRY_GUARDS.get("maxAlreadyDownPctOfDailyDropStop", 0.5)
                if already_down_pct >= guard_pct:
                    log.info(f"TradeAI QUALIFY skip (already down {already_down_pct:.1f}% today): {ticker}")
                    skip_reasons.append({"ticker": ticker, "bucket": s.get("bucket"), "composite": s.get("composite"),
                                          "reason": f"Already down {already_down_pct:.1f}% vs yesterday's close ${prev_close:.2f} — too close to the daily-drop stop to buy into"})
                    continue
            # Earnings blackout: buying right before earnings is a coin flip on the gap
            blackout_days = int(pm.get("earningsBlackoutDays", 3))
            e_date_str = details.get(ticker, {}).get("earningsDate")
            if blackout_days > 0 and e_date_str:
                try:
                    e_date = datetime.strptime(e_date_str, "%Y-%m-%d").date()
                    days_to_earnings = (e_date - datetime.now(TZ).date()).days
                    if 0 <= days_to_earnings <= blackout_days:
                        log.info(f"TradeAI QUALIFY skip (earnings blackout): {ticker} reports in {days_to_earnings}d ({e_date_str})")
                        skip_reasons.append({"ticker": ticker, "bucket": s.get("bucket"), "composite": s.get("composite"), "reason": f"Earnings in {days_to_earnings} day(s) ({e_date_str}) — inside {blackout_days}-day blackout"})
                        continue
                except Exception:
                    pass
            if s.get("composite", 0) < min_composite:
                log.info(f"TradeAI QUALIFY skip (composite {s.get('composite')} < min {min_composite}): {ticker}")
                skip_reasons.append({"ticker": ticker, "bucket": s.get("bucket"), "composite": s.get("composite"), "reason": f"Composite score {s.get('composite')} below minimum {min_composite}"})
                continue
            full_book_rotation = rotation_enabled and held_count >= max_slots
            if budget_per_slot < price and not full_book_rotation:
                log.warning(f"TradeAI QUALIFY skip (price ${price} > per-slot budget ${budget_per_slot}): {ticker}")
                skip_reasons.append({"ticker": ticker, "bucket": s.get("bucket"), "composite": s.get("composite"), "reason": f"Price ${price} exceeds per-slot budget ${budget_per_slot}"})
                continue
            qualified.append(s)
            log.info(f"TradeAI QUALIFIED: {ticker} | bucket:{s.get('bucket')} composite:{s.get('composite')} signal:{s.get('buySignal')}")

        # ── VIX regime gate: no new buys into a volatility spike ─────────────
        vix_ceiling = float(REGIME_CFG.get("vixNoBuyAbove", 30))
        if vix_now and vix_now > vix_ceiling:
            log.warning(f"TradeAI REGIME GATE: VIX {vix_now} > {vix_ceiling} — no new buys this run (sells unaffected)")
            skip_reasons.append({"ticker": "*", "reason": f"VIX {vix_now} above {vix_ceiling} — regime gate blocked all new buys"})
            qualified = []
        elif bool(REGIME_CFG.get("macroRiskGateEnabled", True)):
            # Macro regime gate: same additive pattern as VIX above, sourced
            # from the daily macro refresh (yield curve + headline keywords)
            # instead of live VIX. Only evaluated when VIX itself is fine —
            # the two gates don't stack, either is sufficient to pause buys.
            macro_signal = _macro_risk_signal()
            if macro_signal["riskOff"]:
                reason_str = "; ".join(macro_signal["reasons"])
                log.warning(f"TradeAI REGIME GATE: macro risk-off ({reason_str}) — no new buys this run (sells unaffected)")
                skip_reasons.append({"ticker": "*", "reason": f"Macro risk-off: {reason_str} — regime gate blocked all new buys"})
                qualified = []
            elif bool(REGIME_CFG.get("trendFilterEnabled", True)):
                # Trend regime gate: same additive pattern as VIX/macro above —
                # only evaluated when neither of those already blocked buys.
                # Pauses new buys during a sustained downtrend (index below its
                # own long-run moving average), separate from short-term fear
                # (VIX) or structural risk (macro headlines/yield curve).
                trend = fetch_index_trend_signal(
                    REGIME_CFG.get("trendFilterTicker", "SPY"),
                    int(REGIME_CFG.get("trendFilterDays", 200)))
                if trend.get("available") and not trend["aboveSma"]:
                    reason_str = (f"{trend['ticker']} ${trend['lastClose']} is {abs(trend['pctFromSma'])}% "
                                  f"below its {REGIME_CFG.get('trendFilterDays', 200)}-day average ${trend['sma']}")
                    log.warning(f"TradeAI REGIME GATE: trend filter ({reason_str}) — no new buys this run (sells unaffected)")
                    skip_reasons.append({"ticker": "*", "reason": f"Trend filter: {reason_str} — regime gate blocked all new buys"})
                    qualified = []
                elif bool(REGIME_CFG.get("fomcBlackoutEnabled", True)):
                    # FOMC blackout gate: same additive pattern as the three
                    # above, only evaluated when none of them already
                    # blocked buys. The other gates are all backward-looking
                    # (a rate hike that already happened, inflation that
                    # already reaccelerated, a VIX spike already underway) —
                    # this is the only forward-looking one. A scheduled Fed
                    # decision moves high-beta/extended names before it ever
                    # shows up in those other signals. (Ported from
                    # stock-analysis, where 5 of 6 closed trades lost 0.5-7%
                    # on days SPY barely moved, 1-2 days before an FOMC
                    # decision.)
                    next_fomc = _next_fomc_date()
                    if next_fomc:
                        days_to_fomc = (datetime.strptime(next_fomc, "%Y-%m-%d").date() - datetime.now(TZ).date()).days
                        blackout_days = int(REGIME_CFG.get("fomcBlackoutDays", 1))
                        if 0 <= days_to_fomc <= blackout_days:
                            reason_str = f"FOMC decision in {days_to_fomc} day(s) ({next_fomc})"
                            log.warning(f"TradeAI REGIME GATE: FOMC blackout ({reason_str}) — no new buys this run (sells unaffected)")
                            skip_reasons.append({"ticker": "*", "reason": f"FOMC blackout: {reason_str} — regime gate blocked all new buys"})
                            qualified = []

        slots_to_fill = max(0, max_slots - held_count)
        qualified_sorted = sorted(qualified, key=lambda x: x.get("composite", 0), reverse=True)

        rotation_seller = None
        rotation_candidate = None
        if rotation_enabled and slots_to_fill == 0 and qualified_sorted:
            min_hold_days = float(SELL_CFG.get("minHoldDays", 3))
            eligible_held = []
            for held_ticker, held_pos in holdings.items():
                held_entry = next((s for s in scored if s["ticker"] == held_ticker), None)
                held_days = _days_held(held_pos)
                if (held_entry and held_entry.get("hasAI") and held_days is not None
                        and held_days >= min_hold_days):
                    eligible_held.append((held_entry, held_pos, held_days))

            if eligible_held:
                weakest_entry, weakest_pos, weakest_days = min(
                    eligible_held, key=lambda row: row[0].get("composite", 0))
                weakest_ticker = weakest_entry["ticker"]
                weakest_quote = fetch_live_prices([weakest_ticker]).get(weakest_ticker)
                if weakest_quote:
                    weakest_slip = _slippage_for(details.get(weakest_ticker), EXEC_CFG)
                    estimated_proceeds = round(
                        weakest_quote * (1 - weakest_slip) * weakest_pos["shares"], 2)
                    post_sale_budget = round((balance + estimated_proceeds) * max_deploy, 2)
                    rotation_pool = _cap_filtered_candidates(
                        qualified_sorted, holdings, details, identify_rules, exclude_ticker=weakest_ticker)
                    for candidate in rotation_pool:
                        if (candidate.get("composite", 0) >= weakest_entry.get("composite", 0) + rotation_gap
                                and candidate.get("price", 0) <= post_sale_budget):
                            rotation_seller = (weakest_entry, weakest_pos, weakest_days, weakest_quote)
                            rotation_candidate = candidate
                            break

                if rotation_candidate:
                    log.info(
                        f"TradeAI ROTATION: replacing {weakest_ticker} ({weakest_entry.get('composite')}) "
                        f"with {rotation_candidate['ticker']} ({rotation_candidate.get('composite')}) "
                        f"gap:{rotation_candidate.get('composite', 0) - weakest_entry.get('composite', 0):.1f}")
                else:
                    skip_reasons.append({
                        "ticker": "*", "reason": f"No rotation: best affordable candidate did not exceed "
                        f"weakest eligible holding by {rotation_gap} points"})

        if rotation_seller:
            weakest_entry, weakest_pos, weakest_days, current_price = rotation_seller
            weakest_ticker = weakest_entry["ticker"]
            sell_slip = _slippage_for(details.get(weakest_ticker), EXEC_CFG)
            sell_price = round(current_price * (1 - sell_slip), 4)
            proceeds = round(sell_price * weakest_pos["shares"], 2)
            balance += proceeds
            sell_reason = (
                f"Portfolio rotation: replacing composite {weakest_entry.get('composite')}/100 "
                f"with {rotation_candidate['ticker']} at {rotation_candidate.get('composite')}/100 "
                f"(minimum advantage {rotation_gap} points)"
            )
            weakest_purchase_price = weakest_pos.get("purchasePrice") or 0
            rotation_sell = {
                "ticker": weakest_ticker, "action": "SELL", "shares": weakest_pos["shares"],
                "price": sell_price, "amount": proceeds, "date": ts(),
                "priceFetchedAt": ts(), "balance": round(balance, 2), "reason": sell_reason,
                "composite": weakest_entry.get("composite"), "buySignal": weakest_entry.get("buySignal", "Unknown"),
                "stopLoss": False, "gainLoss": round((sell_price - weakest_purchase_price) * weakest_pos["shares"], 2),
                "pctFromPurchase": round((sell_price - weakest_purchase_price) / weakest_purchase_price * 100, 2) if weakest_purchase_price else None,
                "daysHeld": round(weakest_days, 1), "rotation": True,
                "purchasePrice": weakest_purchase_price, "purchasedAt": weakest_pos.get("purchasedAt"),
            }
            sells.append(rotation_sell)
            transactions.append(rotation_sell)
            try:
                _log_trade_outcome(weakest_ticker, weakest_pos, sell_price, sell_reason, "rotation",
                                   weakest_entry.get("composite", 0), spy_now, weakest_days, sell_rec_version=rec_version)
            except Exception as oe:
                log.error(f"[TradeOutcome] rotation log FAIL: {weakest_ticker} | {oe}")
            del holdings[weakest_ticker]
            held_count -= 1
            slots_to_fill = 1
            deployable = round(balance * max_deploy, 2)
            qualified_sorted = [rotation_candidate]

        actual_buys = _cap_filtered_candidates(qualified_sorted, holdings, details, identify_rules)[:slots_to_fill]

        # Equal budget split across actual buys — not wasted across empty slots
        actual_buy_count = len(actual_buys)
        if actual_buy_count > 0:
            equal_budget = round(deployable / actual_buy_count, 2)
        else:
            equal_budget = budget_per_slot
        log.info(f"TradeAI QUALIFY results | {len(qualified)} qualified from {len(scored)} scored | buying top {slots_to_fill} | equal_budget:{equal_budget} each")

        # ── Step 2: buy top N qualifiers by composite score ───────────────────
        for s in actual_buys:
            ticker = s["ticker"]
            fresh_price = fetch_live_prices([ticker]).get(ticker)
            if not fresh_price:
                log.warning(f"TradeAI BUY skip (no fresh price): {ticker} — refusing to trade without an order-time quote")
                skip_reasons.append({"ticker": ticker, "bucket": s.get("bucket"), "composite": s.get("composite"), "reason": "No order-time live price"})
                continue
            price_fetched_at = ts()
            buy_slip = _slippage_for(details.get(ticker), EXEC_CFG)
            price  = round(fresh_price * (1 + buy_slip), 4)  # buy fill above market: spread + slippage
            shares = int(equal_budget // price)
            if shares < 1:
                continue
            cost = round(shares * price, 2)
            balance -= cost
            # Snapshot the score breakdown at buy time so closed trades can be
            # attributed back to the factors that recommended them.
            factors = {
                "bucket":        s.get("bucket"),
                "composite":     s.get("composite"),
                "aiScore":       s.get("aiScore"),
                "dreamScore":    s.get("dreamScore"),
                "breakoutScore": s.get("breakoutScore"),
                "buySignal":     s.get("buySignal"),
                "trajectory":    s.get("trajectory"),
                "stage":         s.get("stage"),
                "signalFcBoost": s.get("signalFcBoost"),
                "sectorPenalty": s.get("sectorPenalty"),
            }
            holdings[ticker] = {"shares": shares, "purchasePrice": price, "purchasedAt": ts(), "factors": factors,
                                "priceFetchedAt": price_fetched_at,
                                "recVersion": rec_version, "spyAtBuy": spy_now, "vixAtBuy": vix_now}
            buys.append({
                "ticker": ticker, "action": "BUY", "shares": shares,
                "price": price, "amount": cost, "date": ts(), "priceFetchedAt": price_fetched_at,
                "balance": round(balance, 2), "reason": s.get("reasoning", ""),
                "composite": s.get("composite"), "bucket": s.get("bucket"),
            })
            transactions.append(buys[-1])
            held_count += 1
            log.info(f"TradeAI BUY: {ticker} | shares:{shares} price:{price} cost:{cost} composite:{s.get('composite')} signal:{s.get('buySignal')}")

        trades["balance"] = round(balance, 2)
        trades["holdings"] = holdings
        trades["transactions"] = transactions
        trades["lastRecommendAt"] = ts()
        save_trades_ai(trades)

        # Shadow validation: full-funnel snapshot now that qualify/buy
        # outcomes are known (see _record_shadow_snapshot).
        try:
            _record_shadow_snapshot(scored, qualified, actual_buys, dream_map, holdings, live_prices, spy_now, rec_version)
        except Exception as she:
            log.error(f"[Shadow] record FAIL: {she}")

        # ── Self-learning check: does the current version still deserve to run? ──
        # Detection only — never changes the live config. A rollback needs the
        # user's explicit approval via /tradeai/rollback/approve.
        self_learning_action = None
        try:
            sl_cfg = cfg.get("selfLearning", {})
            if sl_cfg.get("autoRollback", True):
                self_learning_action = _check_rollback_candidate(sl_cfg, holdings, live_prices)
        except Exception as sle:
            log.error(f"[SelfLearning] check FAIL: {sle}")

        candidates_data["lastRecommend"] = {"buys": buys, "sells": sells, "scoredAt": ts(), "scored": scored, "skipReasons": skip_reasons,
                                            "selfLearning": self_learning_action}
        save_json(TRADE_AI_CANDIDATES_PATH, candidates_data)

        # ── Sync watchlist from TradeAI holdings ──────────────────────────────
        _sync_watchlist_from_holdings(list(holdings.keys()))

        fetch_status = {"running": False, "message": f"TradeAI recommend done: {len(buys)} buys, {len(sells)} sells", "operation": None}
        log.info(f"=== TradeAI Recommend Complete: {len(buys)} buys {len(sells)} sells | balance:{balance} ===")
    except Exception as e:
        log.error(f"TradeAI recommend FAIL: {e}")
        fetch_status = {"running": False, "message": f"TradeAI recommend FAIL: {e}", "operation": None}


@app.route("/tradeai")
def get_tradeai():
    trades = load_trades_ai()
    candidates = load_json(TRADE_AI_CANDIDATES_PATH, {})
    dream = load_json(DREAM_PATH, {})
    dream_map = {c["ticker"]: c for c in dream.get("candidates", [])}
    details = candidates.get("details", {})
    news_db = load_json(NEWS_PATH, {})

    # Watchlist price refresh writes to analysis.json — holdings are synced into
    # the watchlist, so prefer that price when it's fresher than the Fetch Info one
    analysis_data = load_json(OUTPUT_PATH, {})
    wl_map = {s.get("ticker"): s for s in analysis_data.get("watchlist", [])}

    # Keep displayed candidate prices aligned with the scheduled watchlist
    # refresh without replacing the deeper Fetch Info fields.
    display_details = {ticker: dict(detail) for ticker, detail in details.items()}
    for ticker, detail in display_details.items():
        wl_entry = wl_map.get(ticker, {})
        wl_ts = wl_entry.get("priceFetchedAt") or ""
        detail_ts = detail.get("fetchCompletedAt") or ""
        if wl_entry.get("price") is not None and wl_ts >= detail_ts:
            for field in ("price", "previousClose", "volume", "marketCap", "rsi14", "bbPercent"):
                if field in wl_entry:
                    detail[field] = wl_entry[field]
            detail["priceFetchedAt"] = wl_ts

    holdings_out = []
    for ticker, pos in trades.get("holdings", {}).items():
        detail = details.get(ticker, {})
        wl_entry = wl_map.get(ticker, {})
        current_price = _resolve_holding_price(ticker, pos, wl_map, details, dream_map)
        purchase_price = pos.get("purchasePrice", 0)
        shares = pos.get("shares", 0)
        gain_loss = round((current_price - purchase_price) * shares, 2) if current_price else None
        gain_loss_pct = round(((current_price - purchase_price) / purchase_price) * 100, 2) if current_price and purchase_price else None
        prev_close = wl_entry.get("previousClose") or detail.get("previousClose")
        day_change_pct = round((current_price - prev_close) / prev_close * 100, 2) if (current_price and prev_close) else None
        day_change_amount = round((current_price - prev_close) * shares, 2) if (current_price and prev_close) else None
        holdings_out.append({
            "ticker": ticker, "shares": shares,
            "purchasePrice": purchase_price, "currentPrice": current_price,
            "previousClose": prev_close, "dayChangePct": day_change_pct, "dayChangeAmount": day_change_amount,
            "gainLoss": gain_loss, "gainLossPct": gain_loss_pct,
            "purchasedAt": pos.get("purchasedAt"),
            "factors": pos.get("factors"),
            "change1d": detail.get("change1d"),
            "change2d": detail.get("change2d"),
            "change3d": detail.get("change3d"),
            "change4d": detail.get("change4d"),
            "change5d": detail.get("change5d"),
        })

    # Build news timestamp map for all identified tickers
    identified = candidates.get("identified", [])
    news_ts_map = {}
    for item in identified:
        ticker = item["ticker"]
        entry = news_db.get(ticker, {})
        news_ts_map[ticker] = entry.get("fetchedAt") or entry.get("updatedAt") or None
    news_count_map = {t: len(news_db.get(t, {}).get("articles", []))
                      for t in {*trades.get("holdings", {}), *(i["ticker"] for i in identified)}}

    # Load today's signal forecasts for per-ticker display
    today_str = datetime.now(TZ).strftime("%Y-%m-%d")
    fc_db = load_json(SIGNAL_FORECAST_PATH, {"snapshots": []})
    today_snap = next((s for s in fc_db.get("snapshots", []) if s.get("date") == today_str), None)
    signal_fc_map = {}
    if today_snap:
        for row in today_snap.get("rows", []):
            signal_fc_map[row["ticker"]] = {
                "direction":    row.get("direction"),
                "forecastLow":  row.get("forecastLow"),
                "forecastHigh": row.get("forecastHigh"),
                "forecastMid":  row.get("forecastMid"),
                "biasScore":    row.get("biasScore"),
                "generatedAt":  row.get("generatedAt"),
            }

    log.info(f"[TradeAI API] aiAssessments:{len(candidates.get('aiAssessments', {}))} keys | aiAnalyzedAt:{candidates.get('aiAnalyzedAt')}")
    return jsonify({
        "balance": trades.get("balance", STARTING_BALANCE),
        "holdings": holdings_out,
        "transactions": trades.get("transactions", []),
        "identified": identified,
        "details": display_details,
        "aiAssessments": candidates.get("aiAssessments", {}),
        "lastRecommend": candidates.get("lastRecommend"),
        "identifiedAt":       candidates.get("identifiedAt"),
        "detailsFetchedAt":   candidates.get("detailsFetchedAt"),
        "macroFetchedAt":     load_json(MACRO_PATH, {}).get("fetchedAt"),
        "signalForecastAt":   candidates.get("signalForecastAt"),
        "aiAnalyzedAt":       candidates.get("aiAnalyzedAt"),
        "newsTsMap":          news_ts_map,
        "newsCountMap":       news_count_map,
        "newsCheckMap":       _news_check_map(),
        "marketRisk":         _market_risk_today(),
        "lossReviews":        _loss_reviews_payload(),
        "buyReviews":         _buy_reviews_payload(),
        "signalFcMap":        signal_fc_map,
        "rollbackProposal":   load_json(ROLLBACK_PROPOSAL_PATH, {"status": "none"}),
        "tuningProposal":     load_json(TUNING_PROPOSAL_PATH, {"status": "none"}),
        "benchmarks":         analysis_data.get("benchmarks", {}),
        "newsWarnings":       load_json(NEWS_WARNINGS_PATH, {"warnings": {}}).get("warnings", {}),
        "macroSignal":        _macro_risk_signal(),
        "weightBacktest":     load_json(WEIGHT_BACKTEST_PATH, None),
        "resetPending":       load_json(RESET_PENDING_PATH, {"status": "none"}),
    })


def run_tradeai_sell_check(trigger="manual"):
    """Sell-only cycle: refresh holdings prices and run the price-based stops
    (hard stop, daily-drop vs yesterday's close, soft stop, trailing stop).
    No scoring, no buys, no AI calls — light enough to run every few
    minutes during market hours. Composite context comes from the last full
    Recommend; rotation (top-5) sells are deliberately excluded here."""
    global fetch_status
    fetch_status = {"running": True, "message": "TradeAI: sell-only stop check...", "operation": "tradeai_sellcheck"}
    log.info(f"=== TradeAI Sell-Check Started [{trigger}] ===")
    try:
        _refresh_benchmarks()
    except Exception as be:
        log.warning(f"[Benchmarks] refresh FAIL: {be}")
    try:
        trades = load_trades_ai()
        holdings = trades.get("holdings", {})
        if not holdings:
            fetch_status = {"running": False, "message": "Sell-check: no holdings", "operation": None}
            log.info("=== TradeAI Sell-Check Complete: no holdings ===")
            return

        balance      = trades.get("balance", STARTING_BALANCE)
        transactions = trades.get("transactions", [])
        cfg      = load_scoring_config()
        SELL_CFG = cfg["sellThresholds"]
        EXEC_CFG = cfg.get("execution", {})

        # Severe-news watch on held positions: scan the stored news (kept
        # fresh by run_news_refresh at 09:20/11:47/14:47 and the REFRESH NEWS
        # button) for red-flag keywords. Flags only — never sells on its own.
        try:
            news_db = load_json(NEWS_PATH, {})
            _scan_news_warnings(list(holdings.keys()), news_db)
        except Exception as nwe:
            log.warning(f"[NewsWarning] scan FAIL: {nwe}")

        candidates_data = load_json(TRADE_AI_CANDIDATES_PATH, {})
        details = candidates_data.get("details", {})
        last_scored = {s["ticker"]: s for s in (candidates_data.get("lastRecommend") or {}).get("scored", [])}

        tickers = list(holdings.keys())
        live_prices = fetch_live_prices(tickers + ["SPY"])
        prev_closes = fetch_prev_closes(tickers)
        spy_now = live_prices.get("SPY")

        sells = []
        for ticker, pos in list(holdings.items()):
            current_price = fetch_live_prices([ticker]).get(ticker)
            current_price_fetched_at = ts() if current_price else None
            if not current_price:
                log.warning(f"SellCheck skip: {ticker} — no live price, holding position")
                continue
            purchase_price = pos.get("purchasePrice") or current_price

            peak_price = pos.get("peakPrice") or purchase_price
            if current_price > peak_price:
                peak_price = current_price
                pos["peakPrice"] = round(peak_price, 4)
                log.info(f"[Peak] {ticker} new peak: ${peak_price:.2f}")

            entry     = last_scored.get(ticker)
            composite = entry.get("composite", 0) if entry else 0
            has_ai    = bool(entry and entry.get("hasAI"))

            pct_from_purchase = ((current_price - purchase_price) / purchase_price * 100) if purchase_price else 0
            pct_from_peak     = ((current_price - peak_price) / peak_price * 100) if peak_price else 0
            prev_close        = prev_closes.get(ticker)
            day_change_pct    = ((current_price - prev_close) / prev_close * 100) if prev_close else None

            stop_loss_hard  = _effective_stop_loss_pct(ticker, details, SELL_CFG)
            stop_loss_soft  = SELL_CFG.get("stopLossSoft", 6.0)
            trailing_pct    = SELL_CFG.get("trailingStopPeak", 10.0)
            daily_drop_stop = SELL_CFG.get("dailyDropStop", 8.0)

            reason = ""
            if pct_from_purchase <= -stop_loss_hard:
                atr_pct = (details.get(ticker) or {}).get("atr14Pct")
                atr_note = f", ATR-based ({SELL_CFG.get('atrStopMultiplier', 2.0)}x{atr_pct}%)" if atr_pct is not None else ", ATR unavailable — used hard cap"
                reason = f"Stop-loss triggered: down {abs(pct_from_purchase):.1f}% from purchase ${purchase_price:.2f} (stop at -{stop_loss_hard:.1f}%{atr_note})"
            elif day_change_pct is not None and day_change_pct <= -daily_drop_stop:
                reason = f"Daily-drop stop triggered: down {abs(day_change_pct):.1f}% vs yesterday's close ${prev_close:.2f} (stop at -{daily_drop_stop}%)"
            elif pct_from_purchase <= -stop_loss_soft and has_ai and composite < 50:
                reason = f"Stop-loss triggered: down {abs(pct_from_purchase):.1f}% from purchase and composite {composite}/100 below 50"
            elif pct_from_peak <= -trailing_pct and pct_from_purchase > 0:
                reason = f"Trailing stop triggered: down {abs(pct_from_peak):.1f}% from peak ${peak_price:.2f} — protecting gains"

            if not reason:
                day_str = f"{day_change_pct:.1f}" if day_change_pct is not None else "n/a"
                log.info(f"SellCheck HOLD: {ticker} | day:{day_str}% purchase:{pct_from_purchase:.1f}% peak:{pct_from_peak:.1f}%")
                continue

            days_held = _days_held(pos)
            sell_slip = _slippage_for(details.get(ticker), EXEC_CFG)
            price     = round(current_price * (1 - sell_slip), 4)  # sell fill below market: spread + slippage
            proceeds  = round(price * pos["shares"], 2)
            balance  += proceeds
            sells.append({
                "ticker": ticker, "action": "SELL", "shares": pos["shares"],
                "price": price, "amount": proceeds, "date": ts(), "priceFetchedAt": current_price_fetched_at,
                "balance": round(balance, 2), "reason": reason,
                "composite": composite, "buySignal": (entry or {}).get("buySignal", "Unknown"),
                "pctFromPurchase": round(pct_from_purchase, 2),
                "pctFromPeak": round(pct_from_peak, 2),
                "dayChangePct": round(day_change_pct, 2) if day_change_pct is not None else None,
                "stopLoss": True,
                "gainLoss": round((price - purchase_price) * pos["shares"], 2) if purchase_price else None,
                "daysHeld": round(days_held, 1) if days_held is not None else None,
                "purchasePrice": purchase_price, "purchasedAt": pos.get("purchasedAt"),
            })
            transactions.append(sells[-1])
            try:
                _log_trade_outcome(ticker, pos, price, reason, "stop-loss", composite, spy_now, days_held)
            except Exception as oe:
                log.error(f"[TradeOutcome] log FAIL: {ticker} | {oe}")
            del holdings[ticker]
            log.info(f"TradeAI SELL (sell-check): {ticker} | shares:{pos['shares']} price:{price} proceeds:{proceeds} | {reason}")

        trades["balance"]      = round(balance, 2)
        trades["holdings"]     = holdings
        trades["transactions"] = transactions
        save_trades_ai(trades)

        # Push fresh prices into analysis.json so the account tab shows them
        try:
            with _output_lock:
                out = load_json(OUTPUT_PATH, {"watchlist": []})
                now_ts = ts()
                for s in out.get("watchlist", []):
                    t = s.get("ticker")
                    if t in tickers and t in live_prices:
                        s["price"] = live_prices[t]
                        if t in prev_closes:
                            s["previousClose"] = prev_closes[t]
                        s["priceFetchedAt"] = now_ts
                save_json(OUTPUT_PATH, out)
        except Exception as pe:
            log.warning(f"SellCheck price sync FAIL: {pe}")

        if sells:
            _sync_watchlist_from_holdings(list(holdings.keys()))

        fetch_status = {"running": False, "message": f"Sell-check done: {len(sells)} sells", "operation": None}
        log.info(f"=== TradeAI Sell-Check Complete [{trigger}]: {len(sells)} sells | balance:{trades['balance']} ===")
    except Exception as e:
        log.error(f"TradeAI Sell-Check FAIL: {e}")
        fetch_status = {"running": False, "message": f"Sell-check FAIL: {e}", "operation": None}


@app.route("/tradeai/force_sell", methods=["POST"])
def tradeai_force_sell():
    """Sell one holding immediately at a fresh quote and record the exit."""
    try:
        if fetch_status.get("running"):
            return jsonify({"error": "Another TradeAI operation is running; try again when it finishes"}), 409
        body = request.get_json(silent=True) or {}
        ticker = str(body.get("ticker") or "").upper().strip()
        if not ticker:
            return jsonify({"error": "ticker is required"}), 400

        trades = load_trades_ai()
        holdings = trades.get("holdings", {})
        pos = holdings.get(ticker)
        if not pos:
            return jsonify({"error": f"{ticker} is not currently held"}), 404

        quotes = fetch_live_prices([ticker, "SPY"])
        current_price = quotes.get(ticker)
        if not current_price:
            return jsonify({"error": f"No live price available for {ticker}; nothing was sold"}), 503

        candidates_data = load_json(TRADE_AI_CANDIDATES_PATH, {})
        details = candidates_data.get("details", {})
        exec_cfg = load_scoring_config().get("execution", {})
        sell_slip = _slippage_for(details.get(ticker), exec_cfg)
        sell_price = round(current_price * (1 - sell_slip), 4)
        shares = int(pos.get("shares") or 0)
        if shares < 1:
            return jsonify({"error": f"{ticker} has no sellable shares"}), 400

        proceeds = round(sell_price * shares, 2)
        balance = round(trades.get("balance", STARTING_BALANCE) + proceeds, 2)
        now_ts = ts()
        last_scored = {s["ticker"]: s for s in (candidates_data.get("lastRecommend") or {}).get("scored", [])}
        scored = last_scored.get(ticker, {})
        sell_reason = "Manual force sell requested by user"
        force_sell_purchase_price = pos.get("purchasePrice") or 0
        transaction = {
            "ticker": ticker, "action": "SELL", "shares": shares,
            "price": sell_price, "amount": proceeds, "date": now_ts,
            "priceFetchedAt": now_ts, "balance": balance, "reason": sell_reason,
            "composite": scored.get("composite", 0), "buySignal": scored.get("buySignal", "Unknown"),
            "stopLoss": False, "manual": True,
            "gainLoss": round((sell_price - force_sell_purchase_price) * shares, 2),
            "pctFromPurchase": round((sell_price - force_sell_purchase_price) / force_sell_purchase_price * 100, 2) if force_sell_purchase_price else None,
            "daysHeld": round(_days_held(pos), 1) if _days_held(pos) is not None else None,
            "purchasePrice": force_sell_purchase_price, "purchasedAt": pos.get("purchasedAt"),
        }
        trades["balance"] = balance
        trades["holdings"] = holdings
        trades["transactions"] = trades.get("transactions", []) + [transaction]
        del holdings[ticker]
        save_trades_ai(trades)

        try:
            _log_trade_outcome(ticker, pos, sell_price, sell_reason, "manual", scored.get("composite", 0), quotes.get("SPY"), _days_held(pos))
        except Exception as outcome_error:
            log.error(f"[TradeOutcome] force sell log FAIL: {ticker} | {outcome_error}")

        _sync_watchlist_from_holdings(list(holdings.keys()))
        log.info(f"TradeAI FORCE SELL: {ticker} | shares:{shares} price:{sell_price} proceeds:{proceeds}")
        return jsonify({"status": "ok", "transaction": transaction, "balance": balance})
    except Exception as e:
        log.error(f"TradeAI FORCE SELL FAIL: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/tradeai/refresh_prices", methods=["POST"])
def tradeai_refresh_prices():
    """Refresh live price + yesterday's close for held tickers only (~6 Yahoo
    calls) and sync into analysis.json so the account view updates. No trading."""
    try:
        try:
            _refresh_benchmarks()
        except Exception as be:
            log.warning(f"[Benchmarks] refresh FAIL: {be}")
        holdings = load_trades_ai().get("holdings", {})
        tickers = list(holdings.keys())
        if not tickers:
            return jsonify({"status": "ok", "updated": 0})
        live = fetch_live_prices(tickers)
        prevs = fetch_prev_closes(tickers)
        with _output_lock:
            out = load_json(OUTPUT_PATH, {"watchlist": []})
            wl_map = {s.get("ticker"): s for s in out.get("watchlist", [])}
            now_ts = ts()
            updated = 0
            for t in tickers:
                if t not in live:
                    continue
                s = wl_map.get(t)
                if not s:
                    s = {"ticker": t}
                    out["watchlist"].append(s)
                s["price"] = live[t]
                if t in prevs:
                    s["previousClose"] = prevs[t]
                s["priceFetchedAt"] = now_ts
                updated += 1
            save_json(OUTPUT_PATH, out)
        log.info(f"[AccountRefresh] updated {updated}/{len(tickers)} holding prices")
        return jsonify({"status": "ok", "updated": updated})
    except Exception as e:
        log.error(f"[AccountRefresh] FAIL: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/tradeai/sellcheck", methods=["POST"])
def tradeai_sellcheck():
    if fetch_status.get("running"):
        return jsonify({"error": "Another operation is running"}), 409
    t = threading.Thread(target=run_tradeai_sell_check, args=("manual",), daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/tradeai/rollback/approve", methods=["POST"])
def tradeai_rollback_approve():
    """Execute a pending self-learning rollback proposal — the only path that
    actually changes the live scoring config in response to underperformance."""
    proposal = load_json(ROLLBACK_PROPOSAL_PATH, None)
    if not proposal or proposal.get("status") != "pending":
        return jsonify({"error": "No pending rollback proposal"}), 400
    db = load_recommend_versions()
    cur = db["versions"][-1]
    if cur["version"] != proposal["currentVersion"]:
        save_json(ROLLBACK_PROPOSAL_PATH, {"status": "none"})
        return jsonify({"error": "Proposal is stale — the current version changed since it was raised"}), 409
    champ = next((v for v in db["versions"] if v["version"] == proposal["championVersion"]), None)
    if not champ or not champ.get("configSnapshot"):
        return jsonify({"error": "Champion config is no longer available"}), 400
    save_json(SCORING_CONFIG_PATH, champ["configSnapshot"])
    new_v = bump_recommend_version(
        [f"APPROVED ROLLBACK: v{cur['version']} returned {proposal['currentReturnPct']}% on capital deployed over "
         f"{proposal['currentN']} buys (realized+unrealized) vs champion v{champ['version']} at "
         f"{proposal['championReturnPct']}% — gap {proposal['gapPct']}% exceeded threshold {proposal['thresholdPct']}%",
         f"Restored v{champ['version']} config — approved by user"],
        label=f"Rollback to v{champ['version']} config (approved)",
        config_snapshot=champ["configSnapshot"],
        based_on=champ["version"],
    )
    save_json(ROLLBACK_PROPOSAL_PATH, {"status": "none"})
    log.warning(f"[SelfLearning] ROLLBACK APPROVED by user: v{cur['version']} → v{new_v} (config of v{champ['version']})")
    return jsonify({"status": "ok", "newVersion": new_v, "restoredFrom": champ["version"]})


@app.route("/tradeai/rollback/dismiss", methods=["POST"])
def tradeai_rollback_dismiss():
    """Decline a pending rollback proposal. It won't resurface for this exact
    current-vs-champion version pair, but a fresh comparison (new champion,
    or the current version changing) can raise a new one."""
    proposal = load_json(ROLLBACK_PROPOSAL_PATH, None)
    if not proposal or proposal.get("status") != "pending":
        return jsonify({"error": "No pending rollback proposal"}), 400
    proposal["status"] = "dismissed"
    proposal["dismissedAt"] = ts()
    save_json(ROLLBACK_PROPOSAL_PATH, proposal)
    log.info(f"[SelfLearning] rollback proposal dismissed by user: v{proposal['currentVersion']} vs champion v{proposal['championVersion']}")
    return jsonify({"status": "ok"})


def _set_cfg_path(cfg, path, value):
    """Set a dotted path (e.g. 'identifyRules.momentumRsiMax') in a nested
    config dict, creating intermediate dicts as needed."""
    parts = path.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


@app.route("/tradeai/tuning/approve", methods=["POST"])
def tradeai_tuning_approve():
    """Apply a pending tuning proposal — the only path that actually changes
    the live scoring config based on missed-gainer evidence. Always creates
    a new tracked version, so it's rollback-able like any other change."""
    proposal = load_json(TUNING_PROPOSAL_PATH, None)
    if not proposal or proposal.get("status") != "pending":
        return jsonify({"error": "No pending tuning proposal"}), 400
    change = proposal["proposedChange"]
    new_cfg = load_scoring_config()
    _set_cfg_path(new_cfg, change["path"], change["to"])
    save_json(SCORING_CONFIG_PATH, new_cfg)
    f = proposal["forecast"]
    new_v = bump_recommend_version(
        [f"APPROVED TUNING: {change['path']} {change['from']} -> {change['to']} — based on "
         f"{proposal['missedGainersAnalyzed']} missed {proposal['gainThresholdPct']}%+ gainers over the "
         f"last {proposal['windowDays']} days ({proposal['boughtRowsAnalyzed']} bought rows in the same window)",
         proposal["reason"],
         f"Historical backtest: would have caught {f['historicalMissedGainersWouldHaveCaught']}/"
         f"{f['historicalMissedGainersTotal']} of those missed gainers, would have excluded "
         f"{f['currentPicksWouldHaveExcluded']} of what was actually bought"],
        label=f"Tuning: {change['path'].split('.')[-1]} -> {change['to']} (approved)",
        config_snapshot=new_cfg,
    )
    save_json(TUNING_PROPOSAL_PATH, {"status": "none"})
    log.warning(f"[Tuning] APPROVED by user: {change['path']} {change['from']} -> {change['to']} (v{new_v})")
    return jsonify({"status": "ok", "newVersion": new_v})


@app.route("/tradeai/tuning/dismiss", methods=["POST"])
def tradeai_tuning_dismiss():
    """Decline a pending tuning proposal. Won't resurface for this exact
    proposed change, but a fresh analysis cycle can propose a different one."""
    proposal = load_json(TUNING_PROPOSAL_PATH, None)
    if not proposal or proposal.get("status") != "pending":
        return jsonify({"error": "No pending tuning proposal"}), 400
    proposal["status"] = "dismissed"
    proposal["dismissedAt"] = ts()
    save_json(TUNING_PROPOSAL_PATH, proposal)
    log.info(f"[Tuning] proposal dismissed by user: {proposal['proposalKey']}")
    return jsonify({"status": "ok"})


@app.route("/tradeai/news_warning/dismiss", methods=["POST"])
def tradeai_news_warning_dismiss():
    """Acknowledge a severe-news warning on a held ticker. It won't resurface
    for this exact headline — a genuinely new severe headline on the same
    ticker still raises a fresh warning."""
    body = request.get_json(silent=True) or {}
    ticker = str(body.get("ticker") or "").upper().strip()
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    warnings_db = load_json(NEWS_WARNINGS_PATH, {"warnings": {}})
    warning = warnings_db.get("warnings", {}).get(ticker)
    if not warning or warning.get("status") != "active":
        return jsonify({"error": f"No active news warning for {ticker}"}), 400
    warning["status"] = "dismissed"
    warning["dismissedAt"] = ts()
    save_json(NEWS_WARNINGS_PATH, warnings_db)
    log.info(f"[NewsWarning] dismissed by user: {ticker} ({warning.get('keyword')})")
    return jsonify({"status": "ok"})


@app.route("/tradeai/identify", methods=["POST"])
def tradeai_identify():
    if fetch_status.get("running"):
        return jsonify({"error": "Another operation is running"}), 409
    t = threading.Thread(target=run_tradeai_identify, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/tradeai/fetch", methods=["POST"])
def tradeai_fetch():
    if fetch_status.get("running"):
        return jsonify({"error": "Another operation is running"}), 409
    t = threading.Thread(target=run_tradeai_fetch, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/tradeai/analyze", methods=["POST"])
def tradeai_analyze():
    if fetch_status.get("running"):
        return jsonify({"error": "Another operation is running"}), 409
    t = threading.Thread(target=run_tradeai_analyze, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/tradeai/recommend", methods=["POST"])
def tradeai_recommend():
    if fetch_status.get("running"):
        return jsonify({"error": "Another operation is running"}), 409
    t = threading.Thread(target=run_tradeai_recommend, daemon=True)
    t.start()
    return jsonify({"status": "started"})


def _outcome_stats(rows):
    """Aggregate win/loss stats for a list of outcome records."""
    n = len(rows)
    if n == 0:
        return None
    pnl_pcts = [r["gainLossPct"] for r in rows if r.get("gainLossPct") is not None]
    wins   = [p for p in pnl_pcts if p > 0]
    losses = [p for p in pnl_pcts if p <= 0]
    total_gl = sum(r.get("gainLoss") or 0 for r in rows)
    gross_win  = sum(max(0, r.get("gainLoss") or 0) for r in rows)
    gross_loss = abs(sum(min(0, r.get("gainLoss") or 0) for r in rows))
    excess = [r["excessReturnPct"] for r in rows if r.get("excessReturnPct") is not None]
    return {
        "n": n,
        "winRate": round(len(wins) / len(pnl_pcts) * 100, 1) if pnl_pcts else None,
        "avgWinPct": round(sum(wins) / len(wins), 2) if wins else None,
        "avgLossPct": round(sum(losses) / len(losses), 2) if losses else None,
        "expectancyPct": round(sum(pnl_pcts) / len(pnl_pcts), 2) if pnl_pcts else None,
        "excessExpectancyPct": round(sum(excess) / len(excess), 2) if excess else None,
        "excessN": len(excess),
        "totalGainLoss": round(total_gl, 2),
        "profitFactor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "avgDaysHeld": round(sum(r["daysHeld"] for r in rows if r.get("daysHeld") is not None) /
                             max(1, len([r for r in rows if r.get("daysHeld") is not None])), 1),
    }


def _composite_band(c):
    if c is None: return "unknown"
    if c >= 90: return "90+"
    if c >= 80: return "80-89"
    if c >= 70: return "70-79"
    if c >= 60: return "60-69"
    return "<60"


@app.route("/tradeai/stats")
def tradeai_stats():
    """Scoring feedback: closed-trade outcomes attributed to buy-time factors,
    plus rule-based tuning suggestions for the scoring matrix.

    ?epoch=<n> filters to trades whose sellDate falls within that account
    epoch (epoch 0 = everything before the first reset). Default/omitted =
    "all" = every trade ever, unfiltered — a reset never deletes history,
    it only lets the UI look at one slice of it at a time."""
    epochs_db = _load_epochs()
    epochs = epochs_db.get("epochs", [])
    epoch_param = request.args.get("epoch")

    db = load_json(TRADE_OUTCOMES_PATH, {"outcomes": []})
    outcomes = db.get("outcomes", [])
    for o in outcomes:
        o["epoch"] = _epoch_for_date(o.get("sellDate") or o.get("date"), epochs)
    if epoch_param not in (None, "", "all"):
        try:
            target_epoch = int(epoch_param)
            outcomes = [o for o in outcomes if o["epoch"] == target_epoch]
        except ValueError:
            pass
    with_factors = [o for o in outcomes if o.get("factors")]

    def group_by(rows, key_fn):
        groups = {}
        for r in rows:
            groups.setdefault(key_fn(r) or "unknown", []).append(r)
        return {k: _outcome_stats(v) for k, v in sorted(groups.items())}

    # ── Per-version performance: is each Recommend version an improvement? ──
    # Primary metric is dollar-weighted asset return under a "management
    # period" model: a version owns every sell that fires while it's live
    # (regardless of who bought the position) PLUS every position still open
    # right now (it inherited the whole book at version-start), against
    # capital deployed. A version with a handful of small closed sells but a
    # big still-open winner is NOT penalized the way a naive
    # average-%-per-closed-trade metric would penalize it.
    versions_db = load_recommend_versions()

    # Backfill sellRecVersion for outcomes logged before that field existed —
    # infer which version was live on the sell date from each version's date
    # (the same object `outcomes` points into `db`, so this mutates in place).
    # Runs BEFORE _version_asset_stats below, which keys closed trades off
    # sellRecVersion — otherwise the very first call after deploy would still
    # read the un-backfilled file.
    sorted_versions_by_date = sorted(versions_db["versions"], key=lambda v: v.get("date") or "")
    backfilled_exits = 0
    for o in outcomes:
        if "sellRecVersion" not in o:
            sell_date = (o.get("sellDate") or "")[:10]
            active = None
            for v in sorted_versions_by_date:
                if (v.get("date") or "") <= sell_date:
                    active = v["version"]
            o["sellRecVersion"] = active if active is not None else o.get("recVersion", 1)
            backfilled_exits += 1
    if backfilled_exits:
        save_json(TRADE_OUTCOMES_PATH, db)
        log.info(f"[Outcomes] backfilled sellRecVersion for {backfilled_exits} legacy outcome(s)")

    trades_now = load_trades_ai()
    holdings_now = trades_now.get("holdings", {})
    candidates_now = load_json(TRADE_AI_CANDIDATES_PATH, {})
    details_now = candidates_now.get("details", {})
    dream_now = load_json(DREAM_PATH, {})
    dream_map_now = {c["ticker"]: c for c in dream_now.get("candidates", [])}
    analysis_now = load_json(OUTPUT_PATH, {})
    wl_map_now = {s.get("ticker"): s for s in analysis_now.get("watchlist", [])}
    price_map_now = {t: _resolve_holding_price(t, p, wl_map_now, details_now, dream_map_now)
                      for t, p in holdings_now.items()}
    asset_stats_map = _version_asset_stats(holdings_now, price_map_now, versions_db["versions"][-1]["version"])

    # Whole-account equity snapshots let every version show a real return even
    # with zero trades of its own (a version that just holds inherited
    # positions still has an answerable "before vs now"). Backfill the
    # snapshot for the current version if it predates this tracking, then
    # chain each version's end-equity to the next version's start-equity
    # (the current version's "end" is right now).
    latest_v = versions_db["versions"][-1]
    if latest_v.get("assetAtStart") is None:
        latest_v["assetAtStart"] = _snapshot_total_asset()
        latest_v["assetAtStartCapturedAt"] = ts()
        latest_v["assetAtStartBackfilled"] = True
        save_json(RECOMMEND_VERSIONS_PATH, versions_db)
        log.info(f"[Versions] backfilled assetAtStart for v{latest_v['version']}: ${latest_v['assetAtStart']}")
    current_total_asset = _snapshot_total_asset()

    # Same backfill idea for TSX/TSX60 (see bump_recommend_version) — only
    # meaningful for the CURRENT version, backfilled with today's price as
    # an approximation of "start." A retired version missing this predates
    # the tracking entirely and is intentionally left without an index
    # comparison rather than guessed at. (Ported from stock-analysis.)
    index_live_prices = fetch_live_prices([BENCHMARK_TICKERS["TSX"], BENCHMARK_TICKERS["TSX60"]])
    if not latest_v.get("tsxAtStart"):
        latest_v["tsxAtStart"] = index_live_prices.get(BENCHMARK_TICKERS["TSX"])
        latest_v["tsx60AtStart"] = index_live_prices.get(BENCHMARK_TICKERS["TSX60"])
        save_json(RECOMMEND_VERSIONS_PATH, versions_db)
    version_vs_index = _version_vs_index_stats(versions_db, index_live_prices)

    by_version_rows = {}
    for o in outcomes:
        by_version_rows.setdefault(o.get("recVersion", 1), []).append(o)
    # Exit attribution: which version's sell logic (stop-loss/rotation
    # thresholds) actually closed this trade — distinct from recVersion,
    # which credits whichever version's scoring PICKED it at buy time. A
    # position bought under v1 and stopped out today under v4's thresholds
    # counts as v1's pick but v4's exit — both are real and answer different
    # questions. Legacy outcomes logged before this field existed fall back
    # to their buy-time version so they don't just vanish from the exit view.
    by_sell_version_rows = {}
    for o in outcomes:
        by_sell_version_rows.setdefault(o.get("sellRecVersion", o.get("recVersion", 1)), []).append(o)
    version_report = []
    prev_asset = None
    for i, v in enumerate(versions_db["versions"]):
        vstats = _outcome_stats(by_version_rows.get(v["version"], []))
        vexitstats = _outcome_stats(by_sell_version_rows.get(v["version"], []))
        vasset = asset_stats_map.get(v["version"])
        entry = dict(v)
        entry["hasConfig"] = bool(entry.pop("configSnapshot", None))
        entry["stats"] = vstats
        entry["exitStats"] = vexitstats
        entry["assetStats"] = vasset
        entry["current"] = (v["version"] == versions_db["versions"][-1]["version"])

        # Account-level return: whole portfolio, independent of trade count.
        asset_at_start = v.get("assetAtStart")
        if i + 1 < len(versions_db["versions"]):
            nxt = versions_db["versions"][i + 1]
            # If the NEXT version's assetAtStart was re-anchored by a reset
            # (see _reanchor_current_version_on_reset), it reflects the
            # post-reset balance, not where THIS retiring version's book
            # actually stood the moment it was superseded — that's
            # assetAtStartOriginal, captured before any reset touched it.
            # Falling back to the (possibly re-anchored) assetAtStart would
            # silently reintroduce the same pre/post-reset mismatch bug one
            # version earlier. (Ported from stock-analysis.)
            asset_at_end = nxt.get("assetAtStartOriginal", nxt.get("assetAtStart"))
        else:
            asset_at_end = current_total_asset
        entry["assetAtEnd"] = asset_at_end
        entry["accountReturnPct"] = (round((asset_at_end - asset_at_start) / asset_at_start * 100, 2)
                                      if asset_at_start else None)
        vs_index = version_vs_index.get(v["version"])
        entry["vsIndex"] = vs_index
        if vs_index and entry["accountReturnPct"] is not None:
            entry["vsIndex"] = {
                **vs_index,
                "vsTsxPct": round(entry["accountReturnPct"] - vs_index["tsxReturnPct"], 2),
                "vsTsx60Pct": round(entry["accountReturnPct"] - vs_index["tsx60ReturnPct"], 2),
            }

        if vasset and prev_asset and vasset["totalReturnPct"] is not None and prev_asset["totalReturnPct"] is not None:
            entry["improvement"] = {
                "returnDelta": round(vasset["totalReturnPct"] - prev_asset["totalReturnPct"], 2),
                "basis": "asset-return (realized+unrealized vs capital deployed)",
                "vsVersion": prev_asset_version,
                "reliable": vasset["totalN"] >= 10 and prev_asset["totalN"] >= 10,
            }
        if vasset:
            prev_asset = vasset
            prev_asset_version = v["version"]
        version_report.append(entry)

    # Champion = best dollar-weighted asset return among versions with enough
    # total buy transactions (closed + still-open); the ratchet moves up only.
    sl_cfg = load_scoring_config().get("selfLearning", {})
    sl_min_n = int(sl_cfg.get("minTrades", 10))
    proven = [e for e in version_report
              if e["assetStats"] and e["assetStats"]["totalN"] >= sl_min_n and e["assetStats"]["totalReturnPct"] is not None]
    champion_v = max(proven, key=lambda e: e["assetStats"]["totalReturnPct"])["version"] if proven else None
    for e in version_report:
        if champion_v is not None and e["version"] == champion_v:
            e["status"] = "champion"
        elif e["current"]:
            e["status"] = "testing"
        elif e["assetStats"] and e["assetStats"]["totalN"] >= sl_min_n:
            e["status"] = "retired"
        else:
            e["status"] = "insufficient-data"

    by_bucket    = group_by(with_factors, lambda r: r["factors"].get("bucket"))
    by_band      = group_by(with_factors, lambda r: _composite_band(r["factors"].get("composite")))
    by_signal    = group_by(with_factors, lambda r: r["factors"].get("buySignal"))
    by_trigger   = group_by(outcomes, lambda r: r.get("trigger"))
    by_trajectory = group_by(with_factors, lambda r: r["factors"].get("trajectory"))

    # Factor discrimination: mean of each numeric factor in winners vs losers.
    # A factor whose mean is the same in both groups isn't predicting anything.
    numeric_factors = ["composite", "aiScore", "dreamScore", "breakoutScore", "signalFcBoost"]
    winners = [o for o in with_factors if (o.get("gainLossPct") or 0) > 0]
    losers  = [o for o in with_factors if (o.get("gainLossPct") or 0) <= 0]
    factor_comparison = {}
    for f in numeric_factors:
        w_vals = [o["factors"][f] for o in winners if o["factors"].get(f) is not None]
        l_vals = [o["factors"][f] for o in losers if o["factors"].get(f) is not None]
        if w_vals or l_vals:
            factor_comparison[f] = {
                "winnersAvg": round(sum(w_vals) / len(w_vals), 1) if w_vals else None,
                "losersAvg": round(sum(l_vals) / len(l_vals), 1) if l_vals else None,
                "winnersN": len(w_vals), "losersN": len(l_vals),
            }

    # ── Rule-based tuning suggestions (min sample sizes to avoid noise) ──────
    suggestions = []
    MIN_N = 5
    overall = _outcome_stats(outcomes)
    if overall and overall["n"] < MIN_N:
        suggestions.append(f"Only {overall['n']} closed trades with outcome data so far — suggestions unlock at {MIN_N}+. Keep trading; every sell now logs its buy-time factors.")
    else:
        if overall and overall["expectancyPct"] is not None and overall["expectancyPct"] < 0:
            suggestions.append(f"Overall expectancy is {overall['expectancyPct']}% per trade — the system is losing on average. Prioritize cutting the worst factor groups below before tuning anything else.")
        if overall and overall.get("excessExpectancyPct") is not None and overall["excessN"] >= MIN_N \
                and overall["excessExpectancyPct"] < 0 and (overall["expectancyPct"] or 0) >= 0:
            suggestions.append(f"Trades average {overall['expectancyPct']}% raw but {overall['excessExpectancyPct']}% vs SPY — the gains are just the market rising. Buying SPY would have done better; the stock selection is not adding value yet.")
        for bucket, s in by_bucket.items():
            if s and s["n"] >= MIN_N and s["expectancyPct"] is not None and s["expectancyPct"] < 0:
                suggestions.append(f"Bucket '{bucket}': {s['winRate']}% win rate, {s['expectancyPct']}% avg per trade over {s['n']} trades — consider lowering its bucketWeights or excluding it from Identify.")
        band_low, band_high = by_band.get("60-69"), by_band.get("80-89") or by_band.get("90+")
        if band_low and band_low["n"] >= MIN_N and band_low["expectancyPct"] is not None and band_low["expectancyPct"] < 0:
            suggestions.append(f"Composite 60-69 trades average {band_low['expectancyPct']}% — consider raising purchaseMatrix.minBuyComposite to 70.")
        if band_high and band_low and band_high["expectancyPct"] is not None and band_low["expectancyPct"] is not None and band_high["expectancyPct"] <= band_low["expectancyPct"] and band_high["n"] >= MIN_N:
            suggestions.append("Higher composite scores are NOT outperforming lower ones — the scoring matrix isn't ranking predictively. Check factorComparison to see which components aren't discriminating.")
        for f, cmpv in factor_comparison.items():
            if cmpv["winnersN"] >= MIN_N and cmpv["losersN"] >= MIN_N and cmpv["winnersAvg"] is not None and cmpv["losersAvg"] is not None:
                if cmpv["winnersAvg"] <= cmpv["losersAvg"]:
                    suggestions.append(f"Factor '{f}' averages {cmpv['winnersAvg']} in winners vs {cmpv['losersAvg']} in losers — it is not separating good trades from bad. Consider reducing its weight/bonus in the scoring config.")
        sl = by_trigger.get("stop-loss")
        if sl and sl["n"] >= MIN_N and overall and overall["n"] > 0 and sl["n"] / overall["n"] > 0.5:
            suggestions.append(f"{sl['n']}/{overall['n']} exits were stop-losses — entries are frequently wrong. Tighten entry criteria (raise minBuyComposite) rather than widening stops.")
        soft = by_trigger.get("soft")
        if soft and soft["n"] >= MIN_N and soft["expectancyPct"] is not None and soft["expectancyPct"] > 0 and soft["avgDaysHeld"] is not None and soft["avgDaysHeld"] <= float(load_scoring_config()["sellThresholds"].get("minHoldDays", 3)) + 1:
            suggestions.append(f"Rotation (soft) sells exit at +{soft['expectancyPct']}% avg after only {soft['avgDaysHeld']} days — winners may be cut early. Consider raising sellThresholds.minHoldDays.")

    # ── Shadow validation: does the composite ranking predict forward returns? ──
    shadow_db = load_json(SHADOW_PATH, {"snapshots": []})
    shadow = None
    graded_snaps = [s for s in shadow_db.get("snapshots", [])
                    if any(r.get("fwdReturnPct") is not None for r in s.get("rows", []))]
    if graded_snaps:
        all_graded = [r for s in graded_snaps for r in s["rows"] if r.get("fwdReturnPct") is not None]
        def _avg(vals):
            return round(sum(vals) / len(vals), 2) if vals else None
        band_groups = {}
        for r in all_graded:
            band_groups.setdefault(_composite_band(r.get("composite")), []).append(r)
        by_band_shadow = {
            band: {"n": len(rs),
                   "avgFwdPct": _avg([r["fwdReturnPct"] for r in rs]),
                   "avgExcessPct": _avg([r["fwdExcessPct"] for r in rs if r.get("fwdExcessPct") is not None])}
            for band, rs in sorted(band_groups.items())
        }
        # Top-5 by composite in each snapshot vs the rest — the actual selection test
        top_excess, rest_excess = [], []
        for s in graded_snaps:
            rows_g = sorted([r for r in s["rows"] if r.get("fwdReturnPct") is not None],
                            key=lambda r: r.get("composite") or 0, reverse=True)
            for i, r in enumerate(rows_g):
                val = r.get("fwdExcessPct", r.get("fwdReturnPct"))
                if val is None:
                    continue
                (top_excess if i < 5 else rest_excess).append(val)
        shadow = {
            "graded": len(all_graded),
            "snapshots": len(graded_snaps),
            "byCompositeBand": by_band_shadow,
            "top5AvgExcessPct": _avg(top_excess),
            "restAvgExcessPct": _avg(rest_excess),
        }
        if len(top_excess) >= 15 and len(rest_excess) >= 15:
            edge = (shadow["top5AvgExcessPct"] or 0) - (shadow["restAvgExcessPct"] or 0)
            if edge <= 0:
                suggestions.append(f"Shadow test over {shadow['graded']} candidate-days: the composite's top-5 picks averaged {shadow['top5AvgExcessPct']}% vs SPY while the REST averaged {shadow['restAvgExcessPct']}% — the ranking is not selecting winners. Revisit the factor weights before trusting any buy.")
            else:
                suggestions.append(f"Shadow test over {shadow['graded']} candidate-days: top-5 picks beat the rest by {round(edge,2)}% (5-day, vs SPY) — the composite ranking is adding real selection value.")

    return jsonify({
        "epochs": epochs,
        "selectedEpoch": epoch_param if epoch_param not in (None, "") else "all",
        "overall": overall,
        "shadow": shadow,
        "byBucket": by_bucket,
        "byCompositeBand": by_band,
        "byBuySignal": by_signal,
        "byTrigger": by_trigger,
        "byTrajectory": by_trajectory,
        "factorComparison": factor_comparison,
        "suggestions": suggestions,
        "outcomesLogged": len(outcomes),
        "outcomesWithFactors": len(with_factors),
        "recent": list(reversed(outcomes[-100:])),
        "versions": version_report,
        "currentVersion": versions_db["versions"][-1]["version"],
        "championVersion": champion_v,
        "rollbackProposal": load_json(ROLLBACK_PROPOSAL_PATH, {"status": "none"}),
        "tuningProposal": load_json(TUNING_PROPOSAL_PATH, {"status": "none"}),
        "missedGainers": _missed_gainers_summary(),
        "selfLearning": {
            "autoRollback": bool(sl_cfg.get("autoRollback", True)),
            "minTrades": sl_min_n,
            "rollbackThresholdPct": float(sl_cfg.get("rollbackThresholdPct", 1.0)),
        },
    })


@app.route("/admin/purge", methods=["POST"])
def admin_purge():
    """Remove trading data dated before a given YYYY-MM-DD. `targets` selects
    which stores to purge (default: all). Holdings, balance, watchlist,
    configs and version history are untouched."""

    def _purge_transactions(before):
        trades = load_trades_ai()
        txns = trades.get("transactions", [])
        keep = [t for t in txns if (t.get("date") or "9999")[:10] >= before]
        trades["transactions"] = keep
        save_trades_ai(trades)
        return len(txns) - len(keep)

    def _purge_outcomes(before):
        odb = load_json(TRADE_OUTCOMES_PATH, {"outcomes": []})
        outs = odb.get("outcomes", [])
        keep = [o for o in outs if (o.get("sellDate") or "9999")[:10] >= before]
        odb["outcomes"] = keep
        save_json(TRADE_OUTCOMES_PATH, odb)
        return len(outs) - len(keep)

    def _purge_shadow(before):
        sdb = load_json(SHADOW_PATH, {"snapshots": []})
        snaps = sdb.get("snapshots", [])
        keep = [s for s in snaps if (s.get("date") or "9999") >= before]
        sdb["snapshots"] = keep
        save_json(SHADOW_PATH, sdb)
        return len(snaps) - len(keep)

    def _purge_forecasts(before):
        fdb = load_json(SIGNAL_FORECAST_PATH, {"snapshots": []})
        fsnaps = fdb.get("snapshots", [])
        keep = [s for s in fsnaps if (s.get("date") or "9999") >= before]
        fdb["snapshots"] = keep
        save_json(SIGNAL_FORECAST_PATH, fdb)
        return len(fsnaps) - len(keep)

    purge_fns = {
        "transactions":      _purge_transactions,
        "outcomes":          _purge_outcomes,
        "shadowSnapshots":   _purge_shadow,
        "forecastSnapshots": _purge_forecasts,
    }

    try:
        body = request.get_json(silent=True) or {}
        before = (body.get("before") or "").strip()
        datetime.strptime(before, "%Y-%m-%d")  # validate format

        targets = body.get("targets") or list(purge_fns.keys())
        unknown = [t for t in targets if t not in purge_fns]
        if unknown:
            return jsonify({"error": f"unknown purge targets: {unknown}"}), 400

        removed = {t: purge_fns[t](before) for t in targets}
        log.info(f"[AdminPurge] removed data before {before}: {removed}")
        return jsonify({"status": "ok", "before": before, "removed": removed})
    except ValueError:
        return jsonify({"error": "before must be YYYY-MM-DD"}), 400
    except Exception as e:
        log.error(f"[AdminPurge] FAIL: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/tradeai/version/rollback", methods=["POST"])
def tradeai_version_rollback():
    """Restore an earlier version's config as a new forward version."""
    try:
        body = request.get_json(silent=True) or {}
        target = int(body.get("toVersion", 0))
        db = load_recommend_versions()
        v = next((x for x in db["versions"] if x["version"] == target), None)
        if not v:
            return jsonify({"error": f"version v{target} not found"}), 400
        if not v.get("configSnapshot"):
            return jsonify({"error": f"v{target} has no saved config snapshot — cannot restore"}), 400
        save_json(SCORING_CONFIG_PATH, v["configSnapshot"])
        new_v = bump_recommend_version(
            [f"Manual rollback: restored v{target} config"],
            label=f"Rollback to v{target} config",
            config_snapshot=v["configSnapshot"], based_on=target)
        log.info(f"[RecVersion] manual rollback to v{target} config as v{new_v}")
        return jsonify({"status": "ok", "newVersion": new_v, "restoredFrom": target})
    except Exception as e:
        log.error(f"[RecVersion] rollback FAIL: {e}")
        return jsonify({"error": str(e)}), 500


def _reanchor_current_version_on_reset(new_balance):
    """A reset breaks the version tracker's assetAtStart/tsxAtStart chain —
    accountReturnPct for the CURRENT version would otherwise compare its
    PRE-reset starting asset against the POST-reset current asset (two
    different starting points, not a real performance number). Re-anchors
    the current version's start snapshot to right after the reset — its
    scoring config didn't change, just the account balance — so
    accountReturnPct going forward measures "since the last reset OR since
    this version started, whichever is more recent," matching how the
    whole-account tracker (account_tracker.json) already behaves. The
    original creation-time snapshot is preserved (never overwritten) so
    nothing is silently lost, and every re-anchor is logged. (Ported from
    stock-analysis, where this fixed a version showing a bogus -16% return
    purely from an untracked reset, not real trading.)"""
    try:
        db = load_recommend_versions()
        v = db["versions"][-1]
        if v.get("assetAtStartOriginal") is None:
            v["assetAtStartOriginal"] = v.get("assetAtStart")
            v["assetAtStartOriginalCapturedAt"] = v.get("assetAtStartCapturedAt")
        idx_prices = fetch_live_prices([BENCHMARK_TICKERS["TSX"], BENCHMARK_TICKERS["TSX60"]])
        reset_anchors = v.setdefault("resetAnchors", [])
        reset_anchors.append({
            "at": ts(), "previousAssetAtStart": v.get("assetAtStart"),
            "newAssetAtStart": new_balance,
        })
        v["assetAtStart"] = new_balance
        v["assetAtStartCapturedAt"] = ts()
        v["tsxAtStart"] = idx_prices.get(BENCHMARK_TICKERS["TSX"])
        v["tsx60AtStart"] = idx_prices.get(BENCHMARK_TICKERS["TSX60"])
        save_json(RECOMMEND_VERSIONS_PATH, db)
        log.info(f"[RecVersion] re-anchored v{v['version']} on reset — assetAtStart now ${new_balance:,.2f}")
    except Exception as e:
        log.warning(f"[RecVersion] re-anchor on reset FAIL: {e}")


@app.route("/tradeai/reset", methods=["POST"])
def tradeai_reset():
    try:
        body = request.get_json(silent=True) or {}
        balance = float(body.get("balance", STARTING_BALANCE))
        auto_buy = bool(body.get("autoBuyIn", False))
        save_json(TRADES_AI_PATH, {"balance": balance, "holdings": {}, "transactions": []})
        save_json(TRADE_AI_CANDIDATES_PATH, {})
        _reanchor_current_version_on_reset(balance)
        epoch = _record_new_epoch(balance, label=body.get("label"))
        if auto_buy:
            save_json(RESET_PENDING_PATH, {
                "status": "pending", "requestedAt": ts(),
                "targetBalance": balance, "epoch": epoch["epoch"],
            })
            log.info(f"[Reset] auto buy-in queued — will run at the next Recommend slot (10am/11am ET)")
        else:
            save_json(RESET_PENDING_PATH, {"status": "none"})
        log.info(f"TradeAI simulation reset | balance: ${balance:,.2f} | epoch:{epoch['epoch']} | autoBuyIn:{auto_buy}")
        return jsonify({"status": "reset", "balance": balance, "epoch": epoch["epoch"], "autoBuyIn": auto_buy})
    except Exception as e:
        log.error(f"TradeAI reset FAIL: {e}")
        return jsonify({"error": str(e)})


@app.route("/tradeai/delete_transactions", methods=["POST"])
def tradeai_delete_transactions():
    try:
        body = request.get_json(silent=True) or {}
        month = body.get("month")  # e.g. "2026-06" or "all"
        trades = load_json(TRADES_AI_PATH, {})
        txns = trades.get("transactions", [])
        before = len(txns)
        if month == "all":
            trades["transactions"] = []
        else:
            trades["transactions"] = [t for t in txns if not (t.get("date") or "").startswith(month)]
        after = len(trades["transactions"])
        save_json(TRADES_AI_PATH, trades)
        log.info(f"TradeAI transactions deleted: month={month} | {before - after} removed, {after} remaining")
        return jsonify({"status": "ok", "removed": before - after, "remaining": after})
    except Exception as e:
        log.error(f"TradeAI delete_transactions FAIL: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/tradeai/signal_ts", methods=["POST"])
def tradeai_signal_ts():
    """Save signalForecastAt timestamp into candidates file after signal forecast runs."""
    try:
        data = load_json(TRADE_AI_CANDIDATES_PATH, {})
        data["signalForecastAt"] = ts()
        save_json(TRADE_AI_CANDIDATES_PATH, data)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/tradeai/edit_holding", methods=["POST"])
def tradeai_edit_holding():
    try:
        body = request.get_json()
        ticker = body.get("ticker")
        shares = body.get("shares")
        purchase_price = body.get("purchasePrice")
        trades = load_json(TRADES_AI_PATH, {})
        holdings = trades.get("holdings", {})
        if ticker not in holdings:
            return jsonify({"error": f"{ticker} not in holdings"}), 404
        if shares is not None:
            holdings[ticker]["shares"] = int(shares)
        if purchase_price is not None:
            holdings[ticker]["purchasePrice"] = round(float(purchase_price), 4)
        trades["holdings"] = holdings
        save_json(TRADES_AI_PATH, trades)
        log.info(f"TradeAI holding edited: {ticker} | shares:{shares} purchasePrice:{purchase_price}")
        return jsonify({"status": "ok", "ticker": ticker})
    except Exception as e:
        log.error(f"TradeAI edit_holding FAIL: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/tradeai/edit_balance", methods=["POST"])
def tradeai_edit_balance():
    try:
        body = request.get_json()
        new_balance = float(body.get("balance", 0))
        trades = load_json(TRADES_AI_PATH, {})
        old = trades.get("balance", 0)
        trades["balance"] = round(new_balance, 2)
        save_json(TRADES_AI_PATH, trades)
        log.info(f"TradeAI balance edited: {old} → {new_balance}")
        return jsonify({"status": "ok", "balance": trades["balance"]})
    except Exception as e:
        log.error(f"TradeAI edit_balance FAIL: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/tradeai/edit_transaction", methods=["POST"])
def tradeai_edit_transaction():
    try:
        body = request.get_json()
        idx = int(body.get("index"))
        new_price = float(body.get("price"))
        trades = load_json(TRADES_AI_PATH, {})
        txns = trades.get("transactions", [])
        if idx < 0 or idx >= len(txns):
            return jsonify({"error": "invalid index"}), 400
        old_price = txns[idx].get("price")
        shares = txns[idx].get("shares", 0)
        action = txns[idx].get("action")
        txns[idx]["price"] = round(new_price, 4)
        txns[idx]["amount"] = round(new_price * shares, 2)

        # Propagate: cash balance moves by the fill-price delta
        delta = (new_price - (old_price or 0)) * shares
        if action == "SELL":
            trades["balance"] = round(trades.get("balance", 0) + delta, 2)
        elif action == "BUY":
            trades["balance"] = round(trades.get("balance", 0) - delta, 2)
            # If still held, the position's cost basis changes too
            if txns[idx].get("ticker") in trades.get("holdings", {}):
                trades["holdings"][txns[idx]["ticker"]]["purchasePrice"] = round(new_price, 4)
        trades["transactions"] = txns
        save_json(TRADES_AI_PATH, trades)

        # Propagate: the validation outcome record must match the corrected fill
        outcome_updated = False
        if action == "SELL":
            try:
                odb = load_json(TRADE_OUTCOMES_PATH, {"outcomes": []})
                for o in odb["outcomes"]:
                    if o.get("ticker") == txns[idx].get("ticker") and o.get("sellDate") == txns[idx].get("date"):
                        o["sellPrice"] = round(new_price, 4)
                        bp = o.get("buyPrice")
                        if bp:
                            o["gainLoss"] = round((new_price - bp) * o.get("shares", shares), 2)
                            o["gainLossPct"] = round((new_price - bp) / bp * 100, 2)
                            if o.get("spyReturnPct") is not None:
                                o["excessReturnPct"] = round(o["gainLossPct"] - o["spyReturnPct"], 2)
                        outcome_updated = True
                if outcome_updated:
                    save_json(TRADE_OUTCOMES_PATH, odb)
            except Exception as oe:
                log.error(f"TradeAI edit_transaction outcome sync FAIL: {oe}")

        log.info(f"TradeAI transaction edited: idx:{idx} {action} price:{old_price}→{new_price} | balance delta:{round(delta,2) if action=='SELL' else round(-delta,2)} | outcome synced:{outcome_updated}")
        return jsonify({"status": "ok", "index": idx, "newBalance": trades.get("balance"), "outcomeUpdated": outcome_updated})
    except Exception as e:
        log.error(f"TradeAI edit_transaction FAIL: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/news/refresh/<ticker>", methods=["POST"])
def refresh_ticker_news(ticker):
    """On-demand news refresh for any ticker. Saves into shared news.json."""
    try:
        news_db = load_json(NEWS_PATH, {})
        articles = _keep_recent_if_empty(fetch_ticker_news(ticker), (news_db.get(ticker) or {}).get("articles"))
        news_db[ticker] = {
            "articles": articles,
            "fetchedAt": ts(),
        }
        save_json(NEWS_PATH, news_db)
        log.info(f"News refresh OK: {ticker} | {len(articles)} articles")
        return jsonify({"status": "ok", "ticker": ticker, "count": len(articles), "fetchedAt": news_db[ticker]["fetchedAt"]})
    except Exception as e:
        log.error(f"News refresh FAIL: {ticker} | {e}")
        return jsonify({"error": str(e)}), 500


# ── Market-day Scheduler ──────────────────────────────────────────────────────
# All times US-Eastern. On market days:
#   sell-only stop checks  :20/:40 during the 9/10/11am hours (post-9:30 only)
#   hourly refresh→Fetch→Rule assess→Recommend during market hours
#   price refresh + Dream→Identify→Fetch→Rule assess after the close
ET = ZoneInfo("America/New_York")
SCHEDULER_CONFIG_PATH = "output/scheduler_config.json"
SCHEDULER_STATE_PATH  = "output/scheduler_state.json"

SCHEDULER_DEFAULTS = {
    "enabled": True,
    "sellCheckHours":   [9, 10, 11],
    "sellCheckMinutes": [20, 40],
    "recommendTimes":   ["10:00", "11:00"],
    "macroTime":        "13:00",  # daily — before dream/chain so the macro risk gate has today's data
    "dreamTime":        "13:30",
    "accountTrackerTime": "17:04",  # daily, after close — portfolio value vs TSX/TSX 60
    "weightBacktestTime": "17:06",  # daily — dream/AI weighting backtest over shadow history
    "geminiDailyCap":   4,        # Gemini calls/day this app may use (free tier: 20/day per Google project, shared)
    "geminiNewsTimes":  ["09:20", "14:47"],  # which news refreshes also run the Gemini check (09:20 also reads market risk)
    "buyReviewTime":    "15:10",  # daily, after the last (15:00) Recommend — Gemini second opinion on today's buys
    "lossReviewTime":   "16:20",  # daily, after the close — Gemini tags why each losing trade lost
    "optionsTime":      "17:15",  # daily, after the close — end-of-day options volume/OI (via US-listed twin) for every ticker ever bought or simulated
    "newsTimes":        ["09:20", "11:47", "14:47"],  # the only scheduled news fetch — before the open (catches the ~53% published after close/overnight), then just before the 12:00 and 15:00 Recommends; :47 avoids the 5-min sell checks
    "tuningTime":       "16:15",  # Friday only — weekly missed-gainer tuning check
    "marketHolidays": [
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
        "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
        "2027-01-01",
    ],
    "earlyCloseDates": ["2026-11-27", "2026-12-24"],  # 1pm ET close — afternoon jobs skipped
}

def load_scheduler_config():
    cfg = dict(SCHEDULER_DEFAULTS)
    cfg.update(load_json(SCHEDULER_CONFIG_PATH, {}))
    return cfg

_sched_job_lock = threading.Lock()  # serializes scheduler-initiated operations

def _sched_record_run(job_name, note=""):
    try:
        st = load_json(SCHEDULER_STATE_PATH, {"lastRuns": {}})
        st.setdefault("lastRuns", {})[job_name] = {"at": ts(), "note": note}
        save_json(SCHEDULER_STATE_PATH, st)
    except Exception as e:
        log.warning(f"[Scheduler] state save FAIL: {e}")

def _sched_wait_idle(max_wait_sec=1800):
    """Wait for any in-flight (e.g. manually triggered) operation to finish."""
    waited = 0
    while fetch_status.get("running") and waited < max_wait_sec:
        time.sleep(10)
        waited += 10
    return not fetch_status.get("running")

def _sched_tuning_check():
    """Weekly: does the shadow-snapshot history make a strong enough case
    for a specific Identify config relaxation? Only ever records a pending
    proposal — see /tradeai/tuning/approve."""
    try:
        _check_tuning_candidate()
    except Exception as e:
        log.error(f"[Tuning] scheduled check FAIL: {e}")

def _sched_dream_scan():
    """Scheduled dream scan — skip if one already ran in the last few hours
    (e.g. manually this morning) to avoid a redundant 75-minute Yahoo sweep."""
    dream = load_json(DREAM_PATH, {})
    scanned_at = dream.get("scannedAt")
    if scanned_at:
        try:
            last = datetime.strptime(scanned_at[:16], "%Y-%m-%d %H:%M")
            hours = (datetime.now() - last).total_seconds() / 3600
            if hours < 3:
                log.info(f"[Scheduler] dream scan skipped — last ran {hours:.1f}h ago")
                return
        except Exception:
            pass
    run_dream_scan("scheduler")

def _clear_reset_pending():
    save_json(RESET_PENDING_PATH, {"status": "none"})

def _sched_run_steps(job_name, steps):
    """Run pipeline steps sequentially; one scheduled job at a time."""
    with _sched_job_lock:
        for label, fn in steps:
            if not _sched_wait_idle():
                log.warning(f"[Scheduler] {job_name}: '{label}' skipped — another operation never finished")
                _sched_record_run(job_name, f"skipped at {label} (busy)")
                return
            log.info(f"[Scheduler] {job_name}: step '{label}' starting")
            try:
                fn()
            except Exception as e:
                log.error(f"[Scheduler] {job_name}: step '{label}' FAIL: {e}")
                _sched_record_run(job_name, f"failed at {label}: {e}")
                return
    _sched_record_run(job_name, "ok")
    log.info(f"[Scheduler] {job_name}: complete")

def _scheduler_loop():
    log.info("[Scheduler] thread started")
    fired = set()
    while True:
        try:
            cfg = load_scheduler_config()
            if not cfg.get("enabled"):
                time.sleep(30)
                continue
            now_et = datetime.now(ET)
            today  = now_et.strftime("%Y-%m-%d")
            if now_et.weekday() >= 5 or today in cfg.get("marketHolidays", []):
                time.sleep(60)
                continue
            early_close = today in cfg.get("earlyCloseDates", [])
            hhmm = now_et.strftime("%H:%M")

            slots = []
            if (now_et.hour in cfg.get("sellCheckHours", []) and now_et.minute in cfg.get("sellCheckMinutes", [])
                    and (now_et.hour, now_et.minute) >= (9, 30) and (early_close is False or now_et.hour < 13)):
                slots.append((f"sellCheck-{hhmm}", "sellCheck",
                              [("sell-check", lambda: run_tradeai_sell_check("scheduler"))]))
            if hhmm in cfg.get("recommendTimes", []):
                reset_pending = load_json(RESET_PENDING_PATH, {"status": "none"}).get("status") == "pending"
                if reset_pending:
                    # Account was reset with auto buy-in requested — this is the
                    # first Recommend slot since then (always 10am/11am, so the
                    # "only buy between 10am-3pm" condition holds automatically).
                    # Candidates were wiped by the reset, so run the full chain
                    # from dream-scan, not just the normal 3 steps.
                    slots.append((f"recommend-{hhmm}", "resetBuyIn",
                                  [("dream-scan", _sched_dream_scan),
                                   ("identify", run_tradeai_identify),
                                   ("fetch", run_tradeai_fetch),
                                   ("rule-assess", run_tradeai_analyze),
                                   ("recommend", run_tradeai_recommend),
                                   ("clear-reset-pending", _clear_reset_pending)]))
                else:
                    slots.append((f"recommend-{hhmm}", "hourlyRecommend",
                                  [("price-refresh", run_tradeai_price_refresh),
                                   ("rule-assess", run_tradeai_analyze),
                                   ("recommend", run_tradeai_recommend)]))
            if hhmm == cfg.get("macroTime"):
                slots.append((f"macro-{hhmm}", "macroRefresh",
                              [("macro-refresh", run_macro_refresh)]))
            if hhmm == cfg.get("accountTrackerTime"):
                slots.append((f"accountTracker-{hhmm}", "accountTracker",
                              [("account-tracker", _update_account_tracker)]))
            if hhmm == cfg.get("buyReviewTime"):
                slots.append((f"buyReview-{hhmm}", "buyReview",
                              [("buy-review", lambda: run_buy_review("scheduler"))]))
            if hhmm == cfg.get("lossReviewTime"):
                slots.append((f"lossReview-{hhmm}", "lossReview",
                              [("loss-review", lambda: run_loss_review("scheduler"))]))
            if hhmm == cfg.get("optionsTime") and not early_close:
                slots.append((f"options-{hhmm}", "optionsSnapshot",
                              [("options-snapshot", lambda: run_options_snapshot("scheduler"))]))
            if hhmm in cfg.get("newsTimes", []):
                g = hhmm in cfg.get("geminiNewsTimes", [])
                slots.append((f"news-{hhmm}", "newsRefresh",
                              [("news-refresh", lambda g=g: run_news_refresh("scheduler", gemini=g))]))
            if hhmm == cfg.get("weightBacktestTime"):
                slots.append((f"weightBacktest-{hhmm}", "weightBacktest",
                              [("weight-backtest", _backtest_bucket_weights)]))
            if hhmm == cfg.get("dreamTime") and not early_close:
                slots.append((f"dream-{hhmm}", "dreamPreparation",
                              [("price-refresh", lambda: run_price_refresh("scheduler")),
                               ("dream-scan", _sched_dream_scan),
                               ("identify", run_tradeai_identify),
                               ("fetch", run_tradeai_fetch),
                               ("rule-assess", run_tradeai_analyze)]))
            if hhmm == cfg.get("tuningTime") and now_et.weekday() == 4:  # Friday
                slots.append((f"tuning-{today}", "tuningCheck",
                              [("tuning-analysis", _sched_tuning_check)]))

            for slot_key, job_name, steps in slots:
                key = f"{today}-{slot_key}"
                if key in fired:
                    continue
                fired.add(key)
                log.info(f"[Scheduler] firing {job_name} ({key})")
                threading.Thread(target=_sched_run_steps, args=(job_name, steps), daemon=True).start()

            if len(fired) > 500:
                fired = set(sorted(fired)[-200:])
        except Exception as e:
            log.error(f"[Scheduler] loop error: {e}")
        time.sleep(20)


@app.route("/admin/scheduler", methods=["GET", "POST"])
def admin_scheduler():
    if request.method == "POST":
        body = request.get_json(force=True, silent=True) or {}
        saved = load_json(SCHEDULER_CONFIG_PATH, {})
        if "enabled" in body:
            saved["enabled"] = bool(body["enabled"])
            log.info(f"[Scheduler] enabled set to {saved['enabled']}")
        # Times are matched against strftime("%H:%M") — normalize to zero-padded HH:MM
        def norm_time(t):
            return datetime.strptime(str(t).strip(), "%H:%M").strftime("%H:%M")
        try:
            if "recommendTimes" in body:
                saved["recommendTimes"] = [norm_time(t) for t in body["recommendTimes"]]
            if "dreamTime" in body:
                saved["dreamTime"] = norm_time(body["dreamTime"])
            if "accountTrackerTime" in body:
                saved["accountTrackerTime"] = norm_time(body["accountTrackerTime"])
            if "geminiDailyCap" in body:
                cap = int(body["geminiDailyCap"])
                if not 0 <= cap <= 20:
                    raise ValueError("Gemini daily cap must be 0-20")
                saved["geminiDailyCap"] = cap
            if "geminiNewsTimes" in body:
                saved["geminiNewsTimes"] = [norm_time(t) for t in body["geminiNewsTimes"]]
            if "buyReviewTime" in body:
                saved["buyReviewTime"] = norm_time(body["buyReviewTime"])
            if "lossReviewTime" in body:
                saved["lossReviewTime"] = norm_time(body["lossReviewTime"])
            if "optionsTime" in body:
                saved["optionsTime"] = norm_time(body["optionsTime"])
            if "newsTimes" in body:
                saved["newsTimes"] = [norm_time(t) for t in body["newsTimes"]]
            if "weightBacktestTime" in body:
                saved["weightBacktestTime"] = norm_time(body["weightBacktestTime"])
            if "sellCheckHours" in body:
                hours = sorted({int(h) for h in body["sellCheckHours"]})
                if any(h < 0 or h > 23 for h in hours):
                    raise ValueError("hours must be 0-23")
                saved["sellCheckHours"] = hours
            if "sellCheckMinutes" in body:
                mins = sorted({int(m) for m in body["sellCheckMinutes"]})
                if any(m < 0 or m > 59 for m in mins):
                    raise ValueError("minutes must be 0-59")
                saved["sellCheckMinutes"] = mins
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid schedule value: {e}"}), 400
        save_json(SCHEDULER_CONFIG_PATH, saved)
        if len(body) > 1 or "enabled" not in body:
            log.info(f"[Scheduler] config updated: {saved}")
    return jsonify({
        "config": load_scheduler_config(),
        "state": load_json(SCHEDULER_STATE_PATH, {"lastRuns": {}}),
        "geminiUsage": {**_gemini_status(), "calls": _gemini_usage()["calls"]},
    })


_scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True)
_scheduler_thread.start()

# Auto price refresh on startup
_startup_holdings = _get_holding_tickers()
if _startup_holdings:
    _sync_watchlist_from_holdings(_startup_holdings)
    log.info(f"=== Startup watchlist sync from holdings: {_startup_holdings} ===")
_startup_thread = threading.Thread(target=run_price_refresh, args=("startup",), daemon=True)
_startup_thread.start()
log.info("=== Startup price refresh triggered ===")

log.info("=== Stock.AI started ===")

# ── Signal Forecast ───────────────────────────────────────────────────────────
SIGNAL_FORECAST_PATH = "output/signal_forecasts.json"

def _rule_forecast(detail):
    """Generate a price forecast range using technical rules."""
    price    = detail.get("price") or 0
    rsi      = detail.get("rsi14") or 50
    bb       = detail.get("bbPercent")
    chg1d    = detail.get("change1d") or 0
    chg7d    = detail.get("change7d") or 0
    chg30d   = detail.get("change30d") or 0
    ma50     = detail.get("ma50")
    vol_rat  = detail.get("volumeRatio") or 1.0
    short_fl = detail.get("shortFloat") or 0
    upside   = detail.get("analystUpside") or 0

    if not price:
        return None

    # ── Volatility estimate ───────────────────────────────────────────────────
    # Tier 1: actual yesterday→today move (best single-day reference)
    prev_close   = detail.get("previousClose")
    actual_1d    = abs(price - prev_close) if prev_close else None

    # Tier 2: 7-day average daily move (captures recent swing size)
    daily_from_7d = abs(chg7d / 100 * price) / 5 if chg7d else None  # 5 trading days

    # Tier 3: 30-day average (smoothed baseline, last resort)
    daily_from_30d = abs(chg30d / 100 * price) / 21 if chg30d else price * 0.015

    # Use best available, scaled up to reflect realistic intraday range
    # Stocks rarely move their exact average — use 1.8x as a realistic daily range factor
    if actual_1d and actual_1d > 0:
        daily_vol = max(actual_1d, daily_from_7d or 0, daily_from_30d) * 1.8
    elif daily_from_7d:
        daily_vol = daily_from_7d * 2.0
    else:
        daily_vol = daily_from_30d * 2.0

    # Floor: at minimum 0.5% of price (very low vol stocks still move something)
    # Ceiling: cap at 8% of price (avoid absurd ranges on meme stocks)
    daily_vol = max(daily_vol, price * 0.005)
    daily_vol = min(daily_vol, price * 0.08)

    # ── Direction bias score (-1 to +1) ──────────────────────────────────────
    bias = 0.0

    # RSI contribution
    if rsi < 30:   bias += 0.4   # deeply oversold → bounce likely
    elif rsi < 40: bias += 0.2
    elif rsi < 50: bias += 0.05
    elif rsi < 65: bias += 0.1   # momentum zone
    elif rsi < 75: bias -= 0.1   # overbought caution
    else:          bias -= 0.3   # very overbought

    # BB%B contribution
    if bb is not None:
        if bb < 0.1:   bias += 0.3   # touching lower band
        elif bb < 0.2: bias += 0.15
        elif bb > 0.9: bias -= 0.3   # touching upper band
        elif bb > 0.8: bias -= 0.15

    # Short-term momentum
    if chg1d > 1:    bias += 0.1
    elif chg1d < -1: bias -= 0.1
    if chg7d > 3:    bias += 0.1
    elif chg7d < -3: bias -= 0.1

    # MA50 vs price
    if ma50 and price > ma50 * 1.01:  bias += 0.1
    elif ma50 and price < ma50 * 0.99: bias -= 0.1

    # Volume surge adds conviction to the current direction
    if vol_rat >= 2.0:
        bias = bias * 1.3
    elif vol_rat >= 1.5:
        bias = bias * 1.15

    # Short squeeze potential adds upside buffer
    squeeze_buffer = price * 0.005 * min(short_fl / 10, 2.0) if short_fl > 15 else 0

    # ── Build range ──────────────────────────────────────────────────────────
    bias = max(-1.0, min(1.0, bias))
    midpoint = round(price + (bias * daily_vol * 1.2), 2)
    half_range = round(daily_vol * 1.0, 2)

    low  = round(midpoint - half_range, 2)
    high = round(midpoint + half_range + squeeze_buffer, 2)

    if bias > 0.2:   direction = "Bullish"
    elif bias < -0.2: direction = "Bearish"
    else:             direction = "Neutral"

    return {
        "currentPrice": round(price, 2),
        "forecastLow":  low,
        "forecastMid":  midpoint,
        "forecastHigh": high,
        "direction":    direction,
        "biasScore":    round(bias, 3),
        "dailyVol":     round(daily_vol, 3),
    }


# Module-level progress state for signal forecast
_signal_progress = {"running": False, "current": 0, "total": 0, "ticker": "", "done": 0}

@app.route("/signal/progress")
def signal_progress():
    return jsonify(_signal_progress)

@app.route("/signal/forecast", methods=["POST"])
def signal_forecast():
    """Generate and save forecasts for all identified candidates."""
    global _signal_progress
    try:
        candidates_data = load_json(TRADE_AI_CANDIDATES_PATH, {})
        identified      = candidates_data.get("identified", [])
        details         = candidates_data.get("details", {})
        cand_map        = {item["ticker"]: item for item in identified}

        forecasts_db = load_json(SIGNAL_FORECAST_PATH, {"snapshots": []})
        today = datetime.now(TZ).strftime("%Y-%m-%d")

        forecasts_db["snapshots"] = [s for s in forecasts_db["snapshots"] if s.get("date") != today]

        eligible = {t: d for t, d in details.items() if d}
        total = len(eligible)
        _signal_progress = {"running": True, "current": 0, "total": total, "ticker": "", "done": 0}

        rows = []
        ai_assessments = load_json(TRADES_AI_PATH, {})
        for idx, (ticker, detail) in enumerate(eligible.items(), 1):
            _signal_progress.update({"current": idx, "ticker": ticker})
            fc = _rule_forecast(detail)
            if not fc:
                log.info(f"[Forecast] {idx}/{total} {ticker} — skipped (no price)")
                continue
            bucket = cand_map.get(ticker, {}).get("bucket", "Dream")
            # Use forecastReason from AI Analyze (no extra Gemini call needed)
            reason = ai_assessments.get(ticker, {}).get("forecastReason", "")
            rows.append({
                "ticker":      ticker,
                "bucket":      bucket,
                "date":        today,
                **fc,
                "aiReason":    reason,
                "actualPrice": None,
                "accuracy":    None,
                "generatedAt": ts(),
            })
            _signal_progress["done"] = len(rows)
            log.info(f"[Forecast] {idx}/{total} {ticker} OK | {fc['direction']} | ${fc['forecastLow']}–${fc['forecastHigh']}")

        forecasts_db["snapshots"].append({"date": today, "rows": rows, "generatedAt": ts()})
        save_json(SIGNAL_FORECAST_PATH, forecasts_db)
        _signal_progress = {"running": False, "current": total, "total": total, "ticker": "", "done": len(rows)}
        return jsonify({"status": "ok", "date": today, "count": len(rows), "total": total})
    except Exception as e:
        _signal_progress = {"running": False, "current": 0, "total": 0, "ticker": "", "done": 0}
        log.error(f"Signal forecast FAIL: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/signal/actual", methods=["POST"])
def signal_actual():
    """Update all past snapshots missing actual prices using today's current price."""
    try:
        from fetcher import fetch_price_only
        forecasts_db = load_json(SIGNAL_FORECAST_PATH, {"snapshots": []})
        today = datetime.now(TZ).strftime("%Y-%m-%d")

        past_snaps = [s for s in forecasts_db["snapshots"] if s.get("date") != today]
        if not past_snaps:
            return jsonify({"status": "ok", "message": "No past snapshots to update", "updated": 0, "snapshots": 0})

        price_cache = {}
        total_updated = 0
        snaps_updated = 0

        for snap in past_snaps:
            snap_updated = 0
            for row in snap.get("rows", []):
                if row.get("actualPrice") is not None:
                    continue
                ticker = row["ticker"]
                try:
                    if ticker not in price_cache:
                        result = fetch_price_only(ticker)
                        price_cache[ticker] = result.get("price") if isinstance(result, dict) else result
                    actual = price_cache[ticker]
                    if not actual:
                        continue
                    actual = round(float(actual), 2)
                    row["actualPrice"] = actual
                    snap_date = snap["date"]
                    flo = row["forecastLow"]
                    fhi = row["forecastHigh"]
                    if flo <= actual <= fhi:
                        row["accuracy"] = "hit"
                    elif actual > fhi:
                        row["accuracy"] = "miss-low"
                    else:
                        row["accuracy"] = "miss-high"
                    acc = row["accuracy"]
                    log.info(f"[Actual] {ticker} snap:{snap_date} forecast ${flo}-${fhi} actual ${actual} | {acc}")
                    snap_updated += 1
                    total_updated += 1
                except Exception as e:
                    log.warning(f"[Actual] price fetch FAIL {ticker}: {e}")
            if snap_updated > 0:
                snap["updatedAt"] = ts()
                snaps_updated += 1

        save_json(SIGNAL_FORECAST_PATH, forecasts_db)
        log.info(f"Signal actuals updated: {total_updated} rows across {snaps_updated} snapshots")
        return jsonify({"status": "ok", "updated": total_updated, "snapshots": snaps_updated})
    except Exception as e:
        log.error(f"Signal actual FAIL: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/signal/history")
def signal_history():
    """Return all forecast snapshots."""
    data = load_json(SIGNAL_FORECAST_PATH, {"snapshots": []})
    return jsonify(data)

if __name__ == "__main__":
    # use_reloader=False is required: the reloader forks a second process that
    # re-executes this module, doubling every background thread (scheduler,
    # startup refresh) and causing duplicate scheduled runs — e.g. two
    # concurrent Analyze passes blowing the 20-request/day Gemini quota.
    # 5051, not 5050: stock-analysis (US) uses 5050 and its stock.bat
    # refuses to start while anything holds that port, so sharing it meant
    # only one of the two apps could ever run at a time.
    app.run(debug=True, port=5051, use_reloader=False)
