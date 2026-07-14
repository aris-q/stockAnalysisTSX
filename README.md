# stockAnalysisTSX

TSX Composite build of [stock-analysis](../stock-analysis), same design and pipeline, two differences:

1. **Universe**: Dream Scan runs against the S&P/TSX Composite Index (~220 tickers,
   `fetch_tsx_composite_tickers()` in `fetcher.py`) instead of S&P 500 + Nasdaq 100 + TSX 60.
   Gainers are CDN-only. TSX 60 is also available.
2. **No Gemini / AI scoring**: `ai_summary._call_gemini` is a no-op — it returns `""`
   instead of calling out, no API key needed. Every caller already degrades gracefully on
   a parse failure, so the pipeline shape (Identify → Fetch → **AI Analyze** → Recommend),
   every endpoint, and the "AI Analyze" button are unchanged. The prompt that *would* have
   been sent is still built and shown in the candidate detail panel — only the network
   call and the score/signal/trajectory fields are skipped. Ranking runs entirely on dream
   score + technicals (`bucketWeights` set to `dream: 1.0, ai: 0.0` in every bucket).

## Setup

```
python -m venv venv
venv\Scripts\activate        # or: source venv/bin/activate
pip install -r requirements.txt
python app.py                # http://127.0.0.1:5060
```

Optional `.env` for FRED_API_KEY / NEWSAPI_KEY (Macro tab, News features) — not required
to run Dream Scan, Identify, Fetch, or Recommend.

## Starter watchlist

`SHOP.TO`, `RY.TO`, `SU.TO`, `ABX.TO` — swap freely from the Watchlist tab or `watchlist.json`.

## `minBuyComposite` is untuned

Set to 50 (vs. 60 in the US repo) as a starting point, since the composite ceiling now
comes from dream score + technical bonuses only. Retune once a few days of real TSX Dream
Scan output show the actual score distribution — see `purchaseMatrix.minBuyComposite` in
the Scoring Matrix panel.
