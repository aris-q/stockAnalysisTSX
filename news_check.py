"""Gemini analyses for the TradeAI account (REST, no google-genai dependency;
the rest of the app's Gemini use stays disabled in ai_summary._call_gemini).

1. run_check — after every news refresh, ONE batched call judges each
   holding's and candidate's headlines: negative news (severity 0-5) and
   takeover/merger "deal" status (a pending buyout caps a candidate's
   upside). Window: today's headlines, except before the 09:30 open, when
   it reaches back to the previous trading day's 16:00 so after-close and
   overnight news (about half of all headlines) gets checked too.
2. Market risk — once a day, folded into the first news check's call
   (normally 09:20): reads market-wide headlines and names today's event
   risk (Fed, CPI, tariffs). No call of its own.
3. review_buys — after the last Recommend (15:10): a second opinion on each
   of today's purchases (agree / caution / avoid). Flags only.
5. Corporate events — splits from Yahoo's split calendar (exact ratios and
   dates, no Gemini) and takeovers from market-wide M&A headlines, which
   the news-check call itself extracts (no extra Gemini call).
4. review_losses — after the close: tags each losing trade's cause
   (company news / sector / market / no news = technical) from the
   headlines seen while it was held plus its return vs the index.

Every ticker gets a result so the UI can always list everything.

The free tier allows 20 gemini-2.5-flash requests/day per Google project
(proven by 429s in logs/app.log), so the app caps itself: USAGE_HOOK is
called once per request actually sent, and the app's budget check decides
whether a job may call at all.
"""
import json
import logging
import time
from datetime import datetime, timedelta

import requests

log = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
MAX_HEADLINES_PER_TICKER = 15
MARKET_HEADLINE_MAX_AGE_H = 18
MAX_MARKET_HEADLINES = 40
USAGE_HOOK = None  # set by the app: called with no args before every request sent

NEWS_PROMPT = """You screen stock news for a trading account. You get, per ticker, the headlines from the stated window (newest first, numbered), with publisher and time, and whether the account HOLDS it or it is a buy CANDIDATE. You only see headlines, not article bodies: judge only what a headline clearly says.

For EVERY ticker given, decide:
1. negative: is there company-specific negative news? Count: fraud/investigation, lawsuit, guidance cut or earnings miss, analyst downgrade or price-target cut, share offering/dilution, CEO/CFO exit, product failure/recall, regulatory block, lost major customer, credit downgrade, bankruptcy risk.
2. severity 0-5: 0 = none / only positive or neutral; 1-2 = minor or routine (one downgrade, small miss); 3 = material (guidance cut, offering, exec exit); 4-5 = serious (fraud, investigation, bankruptcy, delisting).
3. category: one of fraud/investigation, lawsuit, guidance/earnings, downgrade, dilution, management, product/regulatory, customer, credit, bankruptcy, macro/sector, none. Use macro/sector when the news is about the whole market or industry, not the company (those are severity <= 1).
4. headlines: the numbers of the headline(s) the verdict is based on ([] if none).
5. reason: <= 20 words.
6. confidence: low / medium / high (low when a headline is ambiguous).
8. split: true if a headline says THIS company announced a stock split (forward or reverse); splitNote: <= 12 words (ratio, forward/reverse, date if stated), "" if none.
7. deal: true if a headline says THIS company agreed to be acquired, received a takeover bid/offer, is in merger talks as the target, or has a pending merger. Acquiring another company is NOT a deal for this purpose. dealNote: <= 15 words (acquirer, price per share if stated, status), "" if no deal.

Rules: a price move by itself ("stock falls", "slips", "down 12% this quarter", "underperforms") is NOT negative news — severity 0 unless the headline names a company-specific cause from the list above; an investor's short position or bearish opinion piece is at most severity 1; ignore listicles and "stocks to watch" pieces; ignore headlines that mention the ticker but are really about another company; count a repeated story once; do not predict prices.

Reply with JSON only: {"results":[{"ticker":"...","negative":true|false,"severity":0,"category":"...","headlines":[1],"reason":"...","confidence":"...","deal":false,"dealNote":"","split":false,"splitNote":""}]} with one entry per ticker given."""

EVENT_ADDON = """

ALSO, separately: you get MARKET-WIDE merger/acquisition and stock-split headlines (numbered E1, E2, ..., each with the tickers Yahoo tagged). List every company that is the TARGET of a takeover: agreed acquisition, takeover bid/offer, merger where it is acquired, or reported talks/exploring a sale. Skip acquirers, skip completed deals more than a few days old, skip asset/division sales.
Per target: {"ticker": "<target's ticker from the tags if clear, else ''>", "company": "...", "acquirer": "...", "pricePerShare": "<as stated, else ''>", "premium": "<as stated, else ''>", "stage": "rumor"|"talks"|"bid"|"agreed"|"completed", "headlines": [E-numbers as integers, without the E], "note": "<= 15 words"}
Add them to the same JSON object as "takeovers": [...] ([] if none).
ALSO list every company in those headlines that ANNOUNCED a stock split: {"ticker": "...", "company": "...", "kind": "forward"|"reverse", "ratio": "e.g. 10-for-1 or 1-for-20", "date": "<effective/record date if stated, else ''>", "headlines": [E-numbers]} as "splits": [...] ([] if none)."""

MARKET_ADDON = """

ALSO, separately: you get MARKET-WIDE headlines (numbered M1, M2, ...). Brief a trader on today's session:
- riskLevel: "low" (nothing scheduled or unusual), "elevated" (a scheduled event or developing story that can move the whole market today, e.g. Fed decision or Fed chair speech, CPI/jobs/GDP release, tariff deadline, major geopolitical escalation, big-index earnings), or "high" (a shock already under way: crash, war escalation, emergency policy action, systemic bank stress).
- events: up to 5 {"event": "...", "when": "time/date if stated, else ''", "impact": "<= 15 words on why it matters to stocks"}, market-wide only.
- summary: <= 30 words.
- headlines: the M-numbers (as integers, without the M) your verdict relies on.
Add it to the same JSON object as "market": {"riskLevel":"...","summary":"...","events":[...],"headlines":[1]}"""

LOSS_PROMPT = """You review losing trades from a stock trading account to find the CAUSE of each loss. Per trade you get: ticker, bucket (the strategy that picked it), days held, the trade's return %, the index return over the same period, the excess return, the exit trigger, and the headlines published while it was held (numbered; may be empty or incomplete — only the last ~7 days of news are stored).

For EVERY trade decide:
1. cause, one of:
   - "company news": a company-specific headline explains the drop (downgrade, miss, offering, lawsuit...)
   - "sector": headlines show its industry/peers falling for a shared reason
   - "market": the index fell about as much or more (excess return near 0 or better) and no company news explains it
   - "no news (technical)": no explaining headline and the index did not fall comparably — the entry signal simply failed
2. reason: <= 25 words.
3. confidence: low / medium / high (low when headlines are missing or ambiguous).
4. headlines: numbers of the headlines that explain it ([] if none).

Reply with JSON only: {"results":[{"key":"...","cause":"...","reason":"...","confidence":"...","headlines":[1]}]} with one entry per trade, echoing its key."""


BUY_PROMPT = """You give a second opinion on stock purchases a rule-based trading system made today. Per purchase you get: the strategy bucket and composite score that triggered it, the system's stated reason, price and size, key metrics (trend, momentum, valuation, analyst view, short interest, upcoming earnings date), today's headlines for the stock with a prior negative-news verdict, and today's market-risk read.

The system's style: short-to-medium holds (days to a few weeks), ATR-based stop-loss, buys strength (momentum/smart-money/reversal buckets).

For EVERY purchase decide:
1. verdict: "agree" (the setup is sound for this style), "caution" (a real but manageable risk: e.g. earnings within ~7 days, extended far above averages, negative news, crowded short, market risk elevated), or "avoid" (a serious red flag the rules missed: e.g. pending buyout capping upside, severe company news, earnings tomorrow on a volatile name, thesis contradicted by the data).
2. thesis: <= 20 words — what the buy is betting on.
3. risks: up to 3 short items, most important first.
4. watch: <= 15 words — what would show the trade is failing.
5. earningsSoon: true if earnings are within 7 days of today.
6. confidence: low / medium / high.

Judge only from the data given; do not invent facts or predict exact prices.

Reply with JSON only: {"results":[{"key":"...","verdict":"agree|caution|avoid","thesis":"...","risks":["..."],"watch":"...","earningsSoon":false,"confidence":"..."}]} with one entry per purchase, echoing its key."""

BUY_METRICS = ["sector", "industry", "marketCap", "price", "rsi14", "ma20", "ma50", "priceVs52wHigh", "change1d", "change5d",
               "change30d", "volumeRatio", "atr14Pct", "bbPercent", "peRatio", "forwardPE", "revenueGrowth", "grossMargin",
               "analystTarget", "analystUpside", "recommendation", "shortFloat", "institutionalPct", "insiderNet", "earningsDate"]


def review_buys(buys, details, news_rows, market, api_key, tz):
    """buys: BUY transactions (each with a unique "key", plus optional
    "factors" from the holding). details: candidate detail dicts by ticker.
    news_rows: {ticker: latest news-check row}. Returns {key: review}."""
    now = datetime.now(tz)
    reviews, payload = {}, []
    mkt = f"{(market or {}).get('riskLevel') or 'unknown'} — {(market or {}).get('summary') or 'no read today'}"
    for b in buys:
        t = b["ticker"]
        d = details.get(t) or {}
        nr = news_rows.get(t) or {}
        heads = (nr.get("headlines") or [])[:8]
        metrics = {k: d.get(k) for k in BUY_METRICS if d.get(k) is not None}
        reviews[b["key"]] = {"key": b["key"], "ticker": t, "date": b.get("date"), "price": b.get("price"),
                             "amount": b.get("amount"), "bucket": b.get("bucket"), "composite": b.get("composite"),
                             "buyReason": b.get("reason"), "earningsDate": d.get("earningsDate"),
                             "verdict": None, "thesis": "", "risks": [], "watch": "", "earningsSoon": False,
                             "confidence": "", "reviewedAt": now.strftime("%Y-%m-%d %H:%M %Z")}
        payload.append(f"\nkey={b['key']} | {t} | bought {b.get('date')} @ {b.get('price')} (${b.get('amount')}) | bucket {b.get('bucket')} "
                       f"| composite {b.get('composite')} | factors {json.dumps(b.get('factors') or {})}"
                       f"\n  system reason: {b.get('reason')}"
                       f"\n  metrics: {json.dumps(metrics, default=str)}"
                       f"\n  news verdict today: {nr.get('state', 'not checked')}"
                       + (f" (severity {nr.get('severity')}, {nr.get('category')}: {nr.get('reason')})" if nr.get("state") == "negative" else "")
                       + (f" | DEAL: {nr.get('dealNote')}" if nr.get("deal") else "")
                       + "".join(f"\n  - [{h.get('time')}, {h.get('provider')}] {h.get('title')}" for h in heads))
    if not payload:
        return {}
    if not api_key:
        for r in reviews.values():
            r["thesis"] = "Not reviewed — GEMINI_API_KEY is not set"
        return reviews
    try:
        out = call_gemini_json(api_key, BUY_PROMPT,
                               f"Today: {now.strftime('%Y-%m-%d %H:%M %Z')}. Market risk today: {mkt}.\nPurchases:" + "".join(payload))
        for x in out.get("results", []):
            r = reviews.get(str(x.get("key")))
            if not r:
                continue
            v = str(x.get("verdict") or "").lower()
            r.update(verdict=v if v in ("agree", "caution", "avoid") else None, thesis=x.get("thesis") or "",
                     risks=[str(s) for s in (x.get("risks") or [])][:3], watch=x.get("watch") or "",
                     earningsSoon=bool(x.get("earningsSoon")), confidence=x.get("confidence") or "")
    except Exception as e:
        log.error(f"[BuyReview] Gemini FAIL: {e}")
        for r in reviews.values():
            r["thesis"] = "Not reviewed — " + str(e)[:200]
    return reviews


def call_gemini_json(api_key, system_prompt, user_prompt):
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
    }
    last = None
    for attempt in range(3):
        if USAGE_HOOK:
            USAGE_HOOK()
        r = requests.post(GEMINI_URL, headers={"x-goog-api-key": api_key}, json=body, timeout=180)
        if r.status_code == 429 and "PerDay" in r.text:
            raise RuntimeError("Gemini daily free-tier limit reached (20/day per project); try again tomorrow")
        if r.status_code in (429, 500, 503) and attempt < 2:
            last = f"HTTP {r.status_code}"
            time.sleep(20 * (attempt + 1))  # free tier: per-minute limit / overload
            continue
        if r.status_code != 200:
            raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:200]}")
        parts = r.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        return json.loads(text)
    raise RuntimeError(f"Gemini unavailable after retries ({last})")


def _parse_pub(p, tz):
    try:
        return datetime.fromisoformat(p.replace("Z", "+00:00")).astimezone(tz)
    except Exception:
        return None


def _parse_local(s, tz):
    """'2026-09-29 10:00 EDT' (the app's ts() format) -> aware datetime."""
    try:
        return datetime.strptime(s[:16], "%Y-%m-%d %H:%M").replace(tzinfo=tz)
    except Exception:
        return None


def news_window_start(now):
    """Before the 09:30 open: previous weekday 16:00 (after-close + overnight
    news). Otherwise: today 00:00."""
    if (now.hour, now.minute) >= (9, 30):
        return now.replace(hour=0, minute=0, second=0, microsecond=0), "today"
    d = now - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    start = d.replace(hour=16, minute=0, second=0, microsecond=0)
    return start, f"since {start.strftime('%a %H:%M')}"


def headlines_since(articles, start, tz, end=None):
    out = []
    today = datetime.now(tz).date()
    for a in articles or []:
        d = _parse_pub(a.get("pubDate") or "", tz)
        if not d or d < start or (end and d > end):
            continue
        label = d.strftime("%m-%d %H:%M") if end else d.strftime("%H:%M") if d.date() == today else d.strftime("%a %H:%M")
        out.append({"title": a.get("title", ""), "url": a.get("url", ""), "provider": a.get("provider", ""),
                    "time": label, "_ts": d.isoformat()})
    out.sort(key=lambda x: x["_ts"], reverse=True)
    for h in out:
        h.pop("_ts")
    return out[:MAX_HEADLINES_PER_TICKER]


def run_check(roles, news_db, api_key, tz, trigger, market_headlines=None, event_headlines=None):
    """roles: {ticker: "held"|"candidate"}. market_headlines (optional, from
    fetch_market_headlines) adds the day's market-risk read to the SAME call.
    Returns the full result record (rec["market"] set when asked)."""
    now = datetime.now(tz)
    start, window_label = news_window_start(now)
    rec = {"runId": str(int(time.time() * 1000)), "runAt": now.strftime("%Y-%m-%d %H:%M %Z"),
           "date": now.strftime("%Y-%m-%d"), "trigger": trigger, "model": GEMINI_MODEL,
           "window": window_label, "status": "ok", "error": None, "geminiCalled": False, "tickers": []}
    news = {t: headlines_since((news_db.get(t) or {}).get("articles"), start, tz) for t in roles}
    rows = {t: {"ticker": t, "role": roles[t], "newsCount": len(news[t]), "headlines": news[t],
                "state": "no_news" if not news[t] else "error", "severity": 0, "category": "none",
                "reason": "", "confidence": "", "flagged": [], "deal": False, "dealNote": "", "split": False, "splitNote": ""}
            for t in roles}
    with_news = [t for t in roles if news[t]]
    market = None
    if market_headlines is not None:
        market = {"date": rec["date"], "asOf": rec["runAt"], "status": "ok", "error": None, "riskLevel": None,
                  "summary": "", "events": [], "headlines": market_headlines, "flagged": []}
        if not market_headlines:
            market.update(status="error", error="no market headlines found")
        rec["market"] = market
    want_market = bool(market and market_headlines)
    want_events = bool(event_headlines)
    rec["takeovers"], rec["splitNews"] = [], []
    if with_news or want_market or want_events:
        if not api_key:
            rec.update(status="error", error="GEMINI_API_KEY is not set — add it to the .env file next to app.py and restart")
            if market:
                market.update(status="error", error=rec["error"])
        else:
            lines = []
            for t in with_news:
                lines.append(f"\n{t} ({'HELD' if roles[t] == 'held' else 'CANDIDATE'}):")
                lines += [f"  {i}. [{h['time']} ET, {h['provider']}] {h['title']}" for i, h in enumerate(news[t], 1)]
            if not with_news:
                lines.append('\n(no ticker has headlines in this window: return "results": [])')
            if want_market:
                lines.append("\n\nMARKET-WIDE headlines:")
                lines += [f"  M{i}. [{h['time']} ET, {h['provider']}] {h['title']}" for i, h in enumerate(market_headlines, 1)]
            if want_events:
                lines.append("\n\nM&A headlines:")
                lines += [f"  E{i}. [{h['time']} ET, {h['provider']}] {h['title']} (tags: {', '.join(h.get('tickers') or []) or 'none'})"
                          for i, h in enumerate(event_headlines, 1)]
            try:
                rec["geminiCalled"] = True
                out = call_gemini_json(api_key, NEWS_PROMPT + (MARKET_ADDON if want_market else "") + (EVENT_ADDON if want_events else ""),
                                       f"Now: {rec['runAt']}. Window: headlines {window_label}. Tickers and headlines:" + "\n".join(lines))
                if want_market:
                    m = out.get("market") or {}
                    lvl = str(m.get("riskLevel") or "").lower()
                    market.update(riskLevel=lvl if lvl in ("low", "elevated", "high") else None, summary=m.get("summary") or "",
                                  events=[e for e in (m.get("events") or []) if isinstance(e, dict)][:5],
                                  flagged=[i for i in (m.get("headlines") or []) if isinstance(i, int) and 1 <= i <= len(market_headlines)])
                    if not market["riskLevel"]:
                        market.update(status="error", error="Gemini returned no market read")
                if want_events:
                    for e in out.get("takeovers") or []:
                        if not isinstance(e, dict) or not (e.get("ticker") or e.get("company")):
                            continue
                        idx = [i for i in (e.get("headlines") or []) if isinstance(i, int) and 1 <= i <= len(event_headlines)]
                        rec["takeovers"].append({
                            "ticker": str(e.get("ticker") or "").upper(), "company": e.get("company") or "",
                            "acquirer": e.get("acquirer") or "", "pricePerShare": e.get("pricePerShare") or "",
                            "premium": e.get("premium") or "", "stage": str(e.get("stage") or "").lower(),
                            "note": e.get("note") or "", "headlines": [event_headlines[i - 1] for i in idx]})
                    for e in out.get("splits") or []:
                        if not isinstance(e, dict) or not (e.get("ticker") or e.get("company")):
                            continue
                        idx = [i for i in (e.get("headlines") or []) if isinstance(i, int) and 1 <= i <= len(event_headlines)]
                        kind = str(e.get("kind") or "").lower()
                        rec["splitNews"].append({
                            "ticker": str(e.get("ticker") or "").upper(), "company": e.get("company") or "",
                            "kind": kind if kind in ("forward", "reverse") else "", "ratio": e.get("ratio") or "",
                            "date": e.get("date") or "", "headlines": [event_headlines[i - 1] for i in idx]})
                for r in out.get("results", []):
                    row = rows.get(str(r.get("ticker", "")).upper()) or rows.get(r.get("ticker"))
                    if not row or not row["newsCount"]:
                        continue
                    sev = int(r.get("severity") or 0)
                    cat = (r.get("category") or "none").strip().lower()
                    neg = bool(r.get("negative")) and sev > 0 and cat not in ("none", "macro/sector")  # price moves / market news aren't company news
                    idx = [i for i in (r.get("headlines") or []) if isinstance(i, int) and 1 <= i <= row["newsCount"]]
                    row.update(state="negative" if neg else "clear", severity=sev if neg else 0,
                               category=cat, reason=r.get("reason") or "",
                               confidence=r.get("confidence") or "", flagged=idx,
                               deal=bool(r.get("deal")), dealNote=(r.get("dealNote") or "") if r.get("deal") else "",
                               split=bool(r.get("split")), splitNote=(r.get("splitNote") or "") if r.get("split") else "")
                for t in with_news:
                    if rows[t]["state"] == "error":
                        rows[t]["reason"] = "Gemini returned no verdict for this ticker"
            except Exception as e:
                log.error(f"[NewsCheck] Gemini FAIL: {e}")
                rec.update(status="error", error=str(e)[:300])
                if market:
                    market.update(status="error", error=rec["error"])
        if rec["status"] == "error":
            for t in with_news:
                if rows[t]["state"] == "error" and not rows[t]["reason"]:
                    rows[t]["reason"] = "Not analysed — " + rec["error"]
    order = {"negative": 0, "error": 1, "clear": 2, "no_news": 3}
    rec["tickers"] = sorted(rows.values(), key=lambda r: (order[r["state"]], -r["severity"], r["role"] != "held", r["ticker"]))
    rec["counts"] = {s: sum(r["state"] == s for r in rows.values()) for s in order}
    rec["counts"]["deal"] = sum(r["deal"] for r in rows.values())
    return rec


def fetch_market_headlines(queries, tz, max_age_h=MARKET_HEADLINE_MAX_AGE_H, limit=MAX_MARKET_HEADLINES):
    """Market-wide headlines from the last max_age_h hours (Yahoo search),
    with the tickers Yahoo tagged on each."""
    import yfinance as yf
    seen, items = set(), []
    for q in queries:
        try:
            for i in yf.Search(q, news_count=10).news or []:
                t = i.get("providerPublishTime")
                if not t or i.get("link") in seen or (time.time() - t) / 3600 > max_age_h:
                    continue
                seen.add(i.get("link"))
                items.append((t, {"title": i.get("title", ""), "url": i.get("link", ""), "provider": i.get("publisher", ""),
                                  "time": datetime.fromtimestamp(t, tz).strftime("%m-%d %H:%M"),
                                  "tickers": [x for x in (i.get("relatedTickers") or []) if "=" not in x][:6]}))
        except Exception as e:
            log.warning(f"[MarketRisk] search FAIL: {q} | {e}")
    items.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in items[:limit]]


def fetch_split_calendar(tz, market="US", back_days=7, ahead_days=60):
    """Announced stock splits from Yahoo's split calendar (exact ratio and
    pay date). market "US": plain symbols (BRK-B style allowed); "TSX": .TO.
    Forward split (e.g. 2-for-1) vs reverse (1-for-20): reverse splits are
    mostly distressed small caps — a warning, not an opportunity."""
    import yfinance as yf
    from datetime import date
    cal, out, offset = yf.Calendars(), [], 0
    start, end = date.today() - timedelta(days=back_days), date.today() + timedelta(days=ahead_days)
    while offset < 1000:
        df = cal.get_splits_calendar(start=start, end=end, limit=100, offset=offset)
        if df is None or df.empty:
            break
        for sym, r in df.iterrows():
            keep = sym.endswith(".TO") if market == "TSX" else "." not in sym
            if not keep:
                continue
            old, new = r.get("Old Share Worth"), r.get("Share Worth")
            try:
                old, new = float(old), float(new)
            except (TypeError, ValueError):
                continue
            if not old or not new or old == new:
                continue
            fwd = new > old
            fmt = lambda x: f"{x:g}"
            pay = r.get("Payable On")
            out.append({"ticker": sym, "company": r.get("Company") or "", "kind": "forward" if fwd else "reverse",
                        "ratio": f"{fmt(new / old if fwd else 1)}-for-{fmt(1 if fwd else old / new)}",
                        "payDate": pd_date(pay), "optionable": bool(r.get("Optionable"))})
        if len(df) < 100:
            break
        offset += 100
    out.sort(key=lambda x: x["payDate"] or "")
    return out


def pd_date(x):
    try:
        return x.strftime("%Y-%m-%d")
    except Exception:
        return str(x)[:10] if x else None


def review_losses(trades, news_db, check_runs, api_key, tz):
    """trades: losing outcome records (each with a unique "key"). Returns
    {key: review} for the ones Gemini answered (or an error stub)."""
    now = datetime.now(tz)
    payload, reviews = [], {}
    for o in trades:
        buy, sell = _parse_local(o.get("buyDate") or "", tz), _parse_local(o.get("sellDate") or "", tz)
        start = (buy - timedelta(days=1)) if buy else now - timedelta(days=7)
        heads = headlines_since((news_db.get(o["ticker"]) or {}).get("articles"), start, tz, end=sell or now)
        titles = {h["title"] for h in heads}
        for run in check_runs:  # flagged headlines from earlier news checks survive the 7-day news store
            for r in run.get("tickers", []):
                if r.get("ticker") != o["ticker"]:
                    continue
                for i in r.get("flagged", []):
                    h = (r.get("headlines") or [])[i - 1:i]
                    if h and h[0]["title"] not in titles and start.strftime("%Y-%m-%d") <= run.get("date", "") <= (sell or now).strftime("%Y-%m-%d"):
                        titles.add(h[0]["title"])
                        heads.append({**h[0], "time": run["date"][5:] + " " + h[0].get("time", "")})
        heads = heads[:MAX_HEADLINES_PER_TICKER]
        reviews[o["key"]] = {"key": o["key"], "ticker": o["ticker"], "buyDate": o.get("buyDate"), "sellDate": o.get("sellDate"),
                             "daysHeld": o.get("daysHeld"), "gainLossPct": o.get("gainLossPct"),
                             "indexReturnPct": o.get("spyReturnPct"), "excessReturnPct": o.get("excessReturnPct"),
                             "trigger": o.get("trigger"), "bucket": (o.get("factors") or {}).get("bucket"),
                             "headlines": heads, "newsCount": len(heads), "cause": None, "reason": "",
                             "confidence": "", "flagged": [], "reviewedAt": now.strftime("%Y-%m-%d %H:%M %Z")}
        r = reviews[o["key"]]
        payload.append(f"\nkey={o['key']} | {o['ticker']} | bucket {r['bucket']} | held {r['daysHeld']}d | return {r['gainLossPct']}% "
                       f"| index {r['indexReturnPct']}% | excess {r['excessReturnPct']}% | exit: {o.get('trigger')} — {(o.get('sellReason') or '')[:120]}"
                       + "".join(f"\n  {i}. [{h['time']}, {h['provider']}] {h['title']}" for i, h in enumerate(heads, 1))
                       + ("" if heads else "\n  (no stored headlines for the holding period)"))
    if not payload:
        return {}
    if not api_key:
        for r in reviews.values():
            r["reason"] = "Not reviewed — GEMINI_API_KEY is not set"
        return reviews
    try:
        out = call_gemini_json(api_key, LOSS_PROMPT, "Losing trades:" + "".join(payload))
        for x in out.get("results", []):
            r = reviews.get(str(x.get("key")))
            if not r:
                continue
            cause = str(x.get("cause") or "").lower()
            r.update(cause=cause if cause in ("company news", "sector", "market", "no news (technical)") else None,
                     reason=x.get("reason") or "", confidence=x.get("confidence") or "",
                     flagged=[i for i in (x.get("headlines") or []) if isinstance(i, int) and 1 <= i <= r["newsCount"]])
    except Exception as e:
        log.error(f"[LossReview] Gemini FAIL: {e}")
        for r in reviews.values():
            r["reason"] = "Not reviewed — " + str(e)[:200]
    return reviews
