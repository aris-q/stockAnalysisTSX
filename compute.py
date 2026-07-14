# 20260429
import logging

log = logging.getLogger(__name__)

KNOWN_SECTORS = {
    "GOLD": "Basic Materials",
    "KGC": "Basic Materials",
    "NEM": "Basic Materials",
    "AEM": "Basic Materials",
}

BANK_SECTORS = {"Financial Services", "Banks - Diversified", "Banks - Regional"}

def compute_bank_metrics(stock):
    ticker = stock.get("ticker")
    industry = stock.get("industry", "")
    sector = stock.get("sector", "")
    
    if sector not in BANK_SECTORS and "Bank" not in industry:
        return None

    annual_income = stock.get("annualIncome", [])
    annual_balance = stock.get("annualBalance", [])

    try:
        net_income = annual_income[0].get("Net Income") if annual_income else None
        equity_0 = annual_balance[0].get("Stockholders Equity") if len(annual_balance) > 0 else None
        equity_1 = annual_balance[1].get("Stockholders Equity") if len(annual_balance) > 1 else None
        avg_equity = (equity_0 + equity_1) / 2 if equity_0 and equity_1 else equity_0

        roe = round(net_income / avg_equity, 4) if net_income and avg_equity else None

        total_debt = annual_balance[0].get("Total Debt") if annual_balance else None
        total_assets = annual_balance[0].get("Total Assets") if annual_balance else None
        loan_to_deposit = round(total_debt / total_assets, 4) if total_debt and total_assets else None

        checklist = [
            {
                "metric": "ROE",
                "threshold": "> 12%",
                "value": f"{round(roe*100,1)}%" if roe else None,
                "pass": roe > 0.12 if roe else None,
                "auto": True
            },
            {
                "metric": "Loan-to-Deposit (proxy)",
                "threshold": "< 90%",
                "value": f"{round(loan_to_deposit*100,1)}%" if loan_to_deposit else None,
                "pass": loan_to_deposit < 0.90 if loan_to_deposit else None,
                "auto": True
            },
            {
                "metric": "CET1",
                "threshold": "> 10%",
                "value": None,
                "pass": None,
                "auto": False
            },
            {
                "metric": "LCR",
                "threshold": "> 100%",
                "value": None,
                "pass": None,
                "auto": False
            },
            {
                "metric": "NPL Ratio",
                "threshold": "< 2%",
                "value": None,
                "pass": None,
                "auto": False
            },
        ]
        log.info(f"Bank metrics {ticker} | ROE:{roe} LTD:{loan_to_deposit}")
        return checklist

    except Exception as e:
        log.error(f"Bank metrics FAIL: {ticker} | {e}")
        return None

def fix_sector(ticker, sector):
    corrected = KNOWN_SECTORS.get(ticker.upper(), sector)
    if corrected != sector:
        log.warning(f"Sector corrected for {ticker}: '{sector}' → '{corrected}'")
    return corrected


def pct_change(current, previous):
    try:
        if current is not None and previous and previous != 0:
            return round((current - previous) / abs(previous), 4)
    except Exception:
        pass
    return None


def compute_period_changes(periods, field, label):
    result = {}
    try:
        q0 = periods[0].get(field) if len(periods) > 0 else None
        q1 = periods[1].get(field) if len(periods) > 1 else None
        q4 = periods[4].get(field) if len(periods) > 4 else None

        result[f"{label}_qoq"] = pct_change(q0, q1)
        result[f"{label}_yoy"] = pct_change(q0, q4)

        if result[f"{label}_qoq"] is not None:
            log.info(f"  {label} QoQ: {result[f'{label}_qoq']}")
        if result[f"{label}_yoy"] is not None:
            log.info(f"  {label} YoY: {result[f'{label}_yoy']}")
    except Exception as e:
        log.error(f"compute_period_changes FAIL: {label} | {e}")
    return result


def compute_annual_yoy(periods, field, label):
    result = {}
    try:
        y0 = periods[0].get(field) if len(periods) > 0 else None
        y1 = periods[1].get(field) if len(periods) > 1 else None
        y2 = periods[2].get(field) if len(periods) > 2 else None

        result[f"{label}_yoy_1"] = pct_change(y0, y1)
        result[f"{label}_yoy_2"] = pct_change(y1, y2)

        if result[f"{label}_yoy_1"] is not None:
            log.info(f"  {label} YoY Y0vsY1: {result[f'{label}_yoy_1']}")
        if result[f"{label}_yoy_2"] is not None:
            log.info(f"  {label} YoY Y1vsY2: {result[f'{label}_yoy_2']}")
    except Exception as e:
        log.error(f"compute_annual_yoy FAIL: {label} | {e}")
    return result


def compute_metrics(stock):
    ticker = stock.get("ticker")
    sector = stock.get("sector")

    annual_income = stock.get("annualIncome", [])
    quarterly_income = stock.get("quarterlyIncome", [])
    annual_cashflow = stock.get("annualCashflow", [])
    quarterly_cashflow = stock.get("quarterlyCashflow", [])
    annual_balance = stock.get("annualBalance", [])
    quarterly_balance = stock.get("quarterlyBalance", [])
    corrected_sector = fix_sector(ticker, sector)
    bank_checklist = compute_bank_metrics({**stock, "sector": corrected_sector})

    log.info(f"--- Computing: {ticker} ---")

    def margin(income):
        rev = income.get("Total Revenue")
        ni = income.get("Net Income")
        return round(ni / rev, 4) if rev and ni and rev > 0 else None

    annual_margin = margin(annual_income[0]) if annual_income else None
    quarterly_margin = margin(quarterly_income[0]) if quarterly_income else None

    total_cash = annual_balance[0].get("Cash And Cash Equivalents") if annual_balance else None
    total_debt = annual_balance[0].get("Total Debt") if annual_balance else None
    net_cash = (total_cash - total_debt) if total_cash is not None and total_debt is not None else None

    latest_fcf = quarterly_cashflow[0].get("Free Cash Flow") if quarterly_cashflow else None
    latest_capex = quarterly_cashflow[0].get("Capital Expenditure") if quarterly_cashflow else None
    latest_ocf = quarterly_cashflow[0].get("Operating Cash Flow") if quarterly_cashflow else None

    quarterly_changes = {}
    quarterly_changes.update(compute_period_changes(quarterly_income, "Total Revenue", "revenue"))
    quarterly_changes.update(compute_period_changes(quarterly_income, "Net Income", "netIncome"))
    quarterly_changes.update(compute_period_changes(quarterly_income, "Gross Profit", "grossProfit"))
    quarterly_changes.update(compute_period_changes(quarterly_income, "Operating Income", "operatingIncome"))
    quarterly_changes.update(compute_period_changes(quarterly_cashflow, "Free Cash Flow", "fcf"))
    quarterly_changes.update(compute_period_changes(quarterly_cashflow, "Operating Cash Flow", "ocf"))
    quarterly_changes.update(compute_period_changes(quarterly_balance, "Total Debt", "totalDebt"))

    annual_changes = {}
    annual_changes.update(compute_annual_yoy(annual_income, "Total Revenue", "revenue"))
    annual_changes.update(compute_annual_yoy(annual_income, "Net Income", "netIncome"))
    annual_changes.update(compute_annual_yoy(annual_income, "Gross Profit", "grossProfit"))
    annual_changes.update(compute_annual_yoy(annual_income, "Operating Income", "operatingIncome"))
    annual_changes.update(compute_annual_yoy(annual_cashflow, "Free Cash Flow", "fcf"))
    annual_changes.update(compute_annual_yoy(annual_cashflow, "Capital Expenditure", "capex"))
    annual_changes.update(compute_annual_yoy(annual_balance, "Total Debt", "totalDebt"))

    log.info(f"{ticker} | annual_margin:{annual_margin} q_margin:{quarterly_margin} net_cash:{net_cash} fcf:{latest_fcf}")

    return {
        **stock,
        "sector": corrected_sector,
        "annualProfitMargin": annual_margin,
        "quarterlyProfitMargin": quarterly_margin,
        "netCash": net_cash,
        "latestFCF": latest_fcf,
        "latestCapex": latest_capex,
        "latestOCF": latest_ocf,
        "quarterlyChanges": quarterly_changes,
        "annualChanges": annual_changes,
        "bankChecklist": bank_checklist,

    }


def rank_peers(watchlist):
    def rank_by(key, label):
        valid = [s for s in watchlist if s.get(key) is not None]
        ranked = sorted(valid, key=lambda x: x[key], reverse=True)
        for i, stock in enumerate(ranked):
            stock[f"{label}Rank"] = i + 1
            log.info(f"{label} rank #{i+1}: {stock['ticker']} ({stock[key]})")

    rank_by("annualProfitMargin", "margin")
    rank_by("netCash", "netCash")
    rank_by("marketCap", "marketCap")
    rank_by("latestFCF", "fcf")

    return watchlist


def process_watchlist(watchlist):
    log.info("=== Phase 3: Computing Metrics ===")
    computed = []
    for stock in watchlist:
        try:
            result = compute_metrics(stock)
            computed.append(result)
            log.info(f"Computed OK: {stock['ticker']}")
        except Exception as e:
            log.error(f"Compute FAIL: {stock['ticker']} | {e}")
            computed.append(stock)
    try:
        ranked = rank_peers(computed)
    except Exception as e:
        log.error(f"Rank FAIL: {e}")
        ranked = computed
    log.info(f"=== Phase 3: Done | total:{len(ranked)} tickers:{[s['ticker'] for s in ranked]} ===")
    return ranked





#old
# import logging

# log = logging.getLogger(__name__)

# KNOWN_SECTORS = {
#     "GOLD": "Basic Materials",
#     "KGC": "Basic Materials",
#     "NEM": "Basic Materials",
#     "AEM": "Basic Materials",
# }

# BANK_SECTORS = {"Financial Services", "Banks - Diversified", "Banks - Regional"}

# def compute_bank_metrics(stock):
#     ticker = stock.get("ticker")
#     industry = stock.get("industry", "")
#     sector = stock.get("sector", "")
    
#     if sector not in BANK_SECTORS and "Bank" not in industry:
#         return None

#     annual_income = stock.get("annualIncome", [])
#     annual_balance = stock.get("annualBalance", [])

#     try:
#         net_income = annual_income[0].get("Net Income") if annual_income else None
#         equity_0 = annual_balance[0].get("Stockholders Equity") if len(annual_balance) > 0 else None
#         equity_1 = annual_balance[1].get("Stockholders Equity") if len(annual_balance) > 1 else None
#         avg_equity = (equity_0 + equity_1) / 2 if equity_0 and equity_1 else equity_0

#         roe = round(net_income / avg_equity, 4) if net_income and avg_equity else None

#         total_debt = annual_balance[0].get("Total Debt") if annual_balance else None
#         total_assets = annual_balance[0].get("Total Assets") if annual_balance else None
#         loan_to_deposit = round(total_debt / total_assets, 4) if total_debt and total_assets else None

#         checklist = [
#             {
#                 "metric": "ROE",
#                 "threshold": "> 12%",
#                 "value": f"{round(roe*100,1)}%" if roe else None,
#                 "pass": roe > 0.12 if roe else None,
#                 "auto": True
#             },
#             {
#                 "metric": "Loan-to-Deposit (proxy)",
#                 "threshold": "< 90%",
#                 "value": f"{round(loan_to_deposit*100,1)}%" if loan_to_deposit else None,
#                 "pass": loan_to_deposit < 0.90 if loan_to_deposit else None,
#                 "auto": True
#             },
#             {
#                 "metric": "CET1",
#                 "threshold": "> 10%",
#                 "value": None,
#                 "pass": None,
#                 "auto": False
#             },
#             {
#                 "metric": "LCR",
#                 "threshold": "> 100%",
#                 "value": None,
#                 "pass": None,
#                 "auto": False
#             },
#             {
#                 "metric": "NPL Ratio",
#                 "threshold": "< 2%",
#                 "value": None,
#                 "pass": None,
#                 "auto": False
#             },
#         ]
#         log.info(f"Bank metrics {ticker} | ROE:{roe} LTD:{loan_to_deposit}")
#         return checklist

#     except Exception as e:
#         log.error(f"Bank metrics FAIL: {ticker} | {e}")
#         return None

# def fix_sector(ticker, sector):
#     corrected = KNOWN_SECTORS.get(ticker.upper(), sector)
#     if corrected != sector:
#         log.warning(f"Sector corrected for {ticker}: '{sector}' → '{corrected}'")
#     return corrected


# def pct_change(current, previous):
#     try:
#         if current is not None and previous and previous != 0:
#             return round((current - previous) / abs(previous), 4)
#     except Exception:
#         pass
#     return None


# def compute_period_changes(periods, field, label):
#     result = {}
#     try:
#         q0 = periods[0].get(field) if len(periods) > 0 else None
#         q1 = periods[1].get(field) if len(periods) > 1 else None
#         q4 = periods[4].get(field) if len(periods) > 4 else None

#         result[f"{label}_qoq"] = pct_change(q0, q1)
#         result[f"{label}_yoy"] = pct_change(q0, q4)

#         if result[f"{label}_qoq"] is not None:
#             log.info(f"  {label} QoQ: {result[f'{label}_qoq']}")
#         if result[f"{label}_yoy"] is not None:
#             log.info(f"  {label} YoY: {result[f'{label}_yoy']}")
#     except Exception as e:
#         log.error(f"compute_period_changes FAIL: {label} | {e}")
#     return result


# def compute_annual_yoy(periods, field, label):
#     result = {}
#     try:
#         y0 = periods[0].get(field) if len(periods) > 0 else None
#         y1 = periods[1].get(field) if len(periods) > 1 else None
#         y2 = periods[2].get(field) if len(periods) > 2 else None

#         result[f"{label}_yoy_1"] = pct_change(y0, y1)
#         result[f"{label}_yoy_2"] = pct_change(y1, y2)

#         if result[f"{label}_yoy_1"] is not None:
#             log.info(f"  {label} YoY Y0vsY1: {result[f'{label}_yoy_1']}")
#         if result[f"{label}_yoy_2"] is not None:
#             log.info(f"  {label} YoY Y1vsY2: {result[f'{label}_yoy_2']}")
#     except Exception as e:
#         log.error(f"compute_annual_yoy FAIL: {label} | {e}")
#     return result


# def compute_metrics(stock):
#     ticker = stock.get("ticker")
#     sector = stock.get("sector")

#     annual_income = stock.get("annualIncome", [])
#     quarterly_income = stock.get("quarterlyIncome", [])
#     annual_cashflow = stock.get("annualCashflow", [])
#     quarterly_cashflow = stock.get("quarterlyCashflow", [])
#     annual_balance = stock.get("annualBalance", [])
#     quarterly_balance = stock.get("quarterlyBalance", [])
#     bank_checklist = compute_bank_metrics({**stock, "sector": corrected_sector})

#     log.info(f"--- Computing: {ticker} ---")

#     def margin(income):
#         rev = income.get("Total Revenue")
#         ni = income.get("Net Income")
#         return round(ni / rev, 4) if rev and ni and rev > 0 else None

#     annual_margin = margin(annual_income[0]) if annual_income else None
#     quarterly_margin = margin(quarterly_income[0]) if quarterly_income else None

#     total_cash = annual_balance[0].get("Cash And Cash Equivalents") if annual_balance else None
#     total_debt = annual_balance[0].get("Total Debt") if annual_balance else None
#     net_cash = (total_cash - total_debt) if total_cash is not None and total_debt is not None else None

#     latest_fcf = quarterly_cashflow[0].get("Free Cash Flow") if quarterly_cashflow else None
#     latest_capex = quarterly_cashflow[0].get("Capital Expenditure") if quarterly_cashflow else None
#     latest_ocf = quarterly_cashflow[0].get("Operating Cash Flow") if quarterly_cashflow else None

#     quarterly_changes = {}
#     quarterly_changes.update(compute_period_changes(quarterly_income, "Total Revenue", "revenue"))
#     quarterly_changes.update(compute_period_changes(quarterly_income, "Net Income", "netIncome"))
#     quarterly_changes.update(compute_period_changes(quarterly_income, "Gross Profit", "grossProfit"))
#     quarterly_changes.update(compute_period_changes(quarterly_income, "Operating Income", "operatingIncome"))
#     quarterly_changes.update(compute_period_changes(quarterly_cashflow, "Free Cash Flow", "fcf"))
#     quarterly_changes.update(compute_period_changes(quarterly_cashflow, "Operating Cash Flow", "ocf"))
#     quarterly_changes.update(compute_period_changes(quarterly_balance, "Total Debt", "totalDebt"))

#     annual_changes = {}
#     annual_changes.update(compute_annual_yoy(annual_income, "Total Revenue", "revenue"))
#     annual_changes.update(compute_annual_yoy(annual_income, "Net Income", "netIncome"))
#     annual_changes.update(compute_annual_yoy(annual_income, "Gross Profit", "grossProfit"))
#     annual_changes.update(compute_annual_yoy(annual_income, "Operating Income", "operatingIncome"))
#     annual_changes.update(compute_annual_yoy(annual_cashflow, "Free Cash Flow", "fcf"))
#     annual_changes.update(compute_annual_yoy(annual_cashflow, "Capital Expenditure", "capex"))
#     annual_changes.update(compute_annual_yoy(annual_balance, "Total Debt", "totalDebt"))

#     corrected_sector = fix_sector(ticker, sector)

#     log.info(f"{ticker} | annual_margin:{annual_margin} q_margin:{quarterly_margin} net_cash:{net_cash} fcf:{latest_fcf}")

#     return {
#         **stock,
#         "sector": corrected_sector,
#         "annualProfitMargin": annual_margin,
#         "quarterlyProfitMargin": quarterly_margin,
#         "netCash": net_cash,
#         "latestFCF": latest_fcf,
#         "latestCapex": latest_capex,
#         "latestOCF": latest_ocf,
#         "quarterlyChanges": quarterly_changes,
#         "annualChanges": annual_changes,
#         "bankChecklist": bank_checklist,

#     }


# def rank_peers(watchlist):
#     def rank_by(key, label):
#         valid = [s for s in watchlist if s.get(key) is not None]
#         ranked = sorted(valid, key=lambda x: x[key], reverse=True)
#         for i, stock in enumerate(ranked):
#             stock[f"{label}Rank"] = i + 1
#             log.info(f"{label} rank #{i+1}: {stock['ticker']} ({stock[key]})")

#     rank_by("annualProfitMargin", "margin")
#     rank_by("netCash", "netCash")
#     rank_by("marketCap", "marketCap")
#     rank_by("latestFCF", "fcf")

#     return watchlist


# def process_watchlist(watchlist):
#     log.info("=== Phase 3: Computing Metrics ===")
#     computed = []
#     for stock in watchlist:
#         try:
#             result = compute_metrics(stock)
#             computed.append(result)
#             log.info(f"Computed OK: {stock['ticker']}")
#         except Exception as e:
#             log.error(f"Compute FAIL: {stock['ticker']} | {e}")
#             computed.append(stock)
#     try:
#         ranked = rank_peers(computed)
#     except Exception as e:
#         log.error(f"Rank FAIL: {e}")
#         ranked = computed
#     log.info(f"=== Phase 3: Done | total:{len(ranked)} tickers:{[s['ticker'] for s in ranked]} ===")
#     return ranked