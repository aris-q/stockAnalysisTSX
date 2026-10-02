"""Gemini negative-news check for the TradeAI account.

One batched Gemini call per run covers every ticker (holdings + today's
candidates), using ONLY headlines published today (ET). Runs at the end of
each news refresh. Every ticker gets a result — "no_news", "clear",
"negative" or "error" — so the UI can always show the full list.

Calls the REST endpoint with requests (no google-genai dependency). The
rest of the app's Gemini use stays disabled (ai_summary._call_gemini).
"""
import json
import logging
import time
from datetime import datetime

import requests

log = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
MAX_HEADLINES_PER_TICKER = 15

SYSTEM_PROMPT = """You screen stock news for a trading account. You get, per ticker, the headlines published TODAY (newest first, numbered), with publisher and time, and whether the account HOLDS it or it is a buy CANDIDATE. You only see headlines, not article bodies: judge only what a headline clearly says.

For EVERY ticker given, decide:
1. negative: is there company-specific negative news? Count: fraud/investigation, lawsuit, guidance cut or earnings miss, analyst downgrade or price-target cut, share offering/dilution, CEO/CFO exit, product failure/recall, regulatory block, lost major customer, credit downgrade, bankruptcy risk.
2. severity 0-5: 0 = none / only positive or neutral; 1-2 = minor or routine (one downgrade, small miss); 3 = material (guidance cut, offering, exec exit); 4-5 = serious (fraud, investigation, bankruptcy, delisting).
3. category: one of fraud/investigation, lawsuit, guidance/earnings, downgrade, dilution, management, product/regulatory, customer, credit, bankruptcy, macro/sector, none. Use macro/sector when the news is about the whole market or industry, not the company (those are severity <= 1).
4. headlines: the numbers of the headline(s) the verdict is based on ([] if none).
5. reason: <= 20 words.
6. confidence: low / medium / high (low when a headline is ambiguous).

Rules: a price move by itself ("stock falls", "slips", "down 12% this quarter", "underperforms") is NOT negative news — severity 0 unless the headline names a company-specific cause from the list above; an investor's short position or bearish opinion piece is at most severity 1; ignore listicles and "stocks to watch" pieces; ignore headlines that mention the ticker but are really about another company; count a repeated story once; do not predict prices.

Reply with JSON only: {"results":[{"ticker":"...","negative":true|false,"severity":0,"category":"...","headlines":[1],"reason":"...","confidence":"..."}]} with one entry per ticker given."""


def todays_headlines(articles, today, tz):
    """Articles whose pubDate falls on `today` (YYYY-MM-DD) in tz, newest first."""
    out = []
    for a in articles or []:
        p = a.get("pubDate")
        if not p:
            continue
        try:
            d = datetime.fromisoformat(p.replace("Z", "+00:00")).astimezone(tz)
        except Exception:
            continue
        if d.strftime("%Y-%m-%d") == today:
            out.append({"title": a.get("title", ""), "url": a.get("url", ""),
                        "provider": a.get("provider", ""), "time": d.strftime("%H:%M")})
    out.sort(key=lambda x: x["time"], reverse=True)
    return out[:MAX_HEADLINES_PER_TICKER]


def _call_gemini_json(api_key, user_prompt):
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
    }
    last = None
    for attempt in range(3):
        r = requests.post(GEMINI_URL, headers={"x-goog-api-key": api_key}, json=body, timeout=120)
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


def run_check(roles, news_db, api_key, tz, trigger):
    """roles: {ticker: "held"|"candidate"}. Returns the full result record."""
    now = datetime.now(tz)
    today = now.strftime("%Y-%m-%d")
    rec = {"runId": str(int(time.time() * 1000)), "runAt": now.strftime("%Y-%m-%d %H:%M %Z"),
           "date": today, "trigger": trigger, "model": GEMINI_MODEL, "status": "ok", "error": None,
           "geminiCalled": False, "tickers": []}
    news = {t: todays_headlines((news_db.get(t) or {}).get("articles"), today, tz) for t in roles}
    rows = {t: {"ticker": t, "role": roles[t], "newsCount": len(news[t]), "headlines": news[t],
                "state": "no_news" if not news[t] else "error", "severity": 0, "category": "none",
                "reason": "", "confidence": "", "flagged": []}
            for t in roles}
    with_news = [t for t in roles if news[t]]
    if with_news:
        if not api_key:
            rec.update(status="error", error="GEMINI_API_KEY is not set — add it to the .env file and restart")
        else:
            lines = []
            for t in with_news:
                lines.append(f"\n{t} ({'HELD' if roles[t] == 'held' else 'CANDIDATE'}):")
                lines += [f"  {i}. [{h['time']} ET, {h['provider']}] {h['title']}" for i, h in enumerate(news[t], 1)]
            try:
                rec["geminiCalled"] = True
                out = _call_gemini_json(api_key, f"Today is {today}. Tickers and today's headlines:" + "\n".join(lines))
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
                               confidence=r.get("confidence") or "", flagged=idx)
                for t in with_news:
                    if rows[t]["state"] == "error":
                        rows[t]["reason"] = "Gemini returned no verdict for this ticker"
            except Exception as e:
                log.error(f"[NewsCheck] Gemini FAIL: {e}")
                rec.update(status="error", error=str(e)[:300])
        if rec["status"] == "error":
            for t in with_news:
                if rows[t]["state"] == "error" and not rows[t]["reason"]:
                    rows[t]["reason"] = "Not analysed — " + rec["error"]
    order = {"negative": 0, "error": 1, "clear": 2, "no_news": 3}
    rec["tickers"] = sorted(rows.values(), key=lambda r: (order[r["state"]], -r["severity"], r["role"] != "held", r["ticker"]))
    rec["counts"] = {s: sum(r["state"] == s for r in rows.values()) for s in order}
    return rec
