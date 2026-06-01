"""PL698 — AWS/Azure Sev-1 Cluster - Long DDOG/NET"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Curated major AWS/Azure Sev-1 cluster periods (3+ significant incidents within 30 days)
# Based on well-documented major cloud outage clusters from public post-mortems and news
# Entry date = day of / day after the cluster becomes notable (3rd event in 30d window)
KNOWN_SEV1_CLUSTERS = [
    # AWS us-east-1 major outage Dec 7-10 2021 (multiple services)
    "2021-12-10",
    # AWS us-east-1 Oct 2021 major incident + Nov follow-ups
    "2021-11-01",
    # Azure Active Directory + compute outages Jul 2021
    "2021-07-22",
    # Azure multi-region outage Mar 2021
    "2021-03-16",
    # AWS us-east-1 Nov 2020 major outage
    "2020-11-25",
    # AWS Apr 2021 multiple Sev-1 incidents
    "2021-04-22",
    # Azure Jun 2023 multi-day DDoS + infra outage
    "2023-06-13",
    # CrowdStrike/Azure-triggered global outage Jul 2024
    "2024-07-19",
    # AWS us-east-1 + us-west-2 Dec 2023 issues
    "2023-12-04",
    # Azure/M365 outage Oct 2023
    "2023-10-26",
    # AWS us-east multiple incidents Jun 2022
    "2022-06-13",
    # Azure AD multi-region Sep 2022
    "2022-09-28",
    # Google Cloud + AWS issues Aug 2023
    "2023-08-15",
    # AWS Dec 2022 outage cluster
    "2022-12-15",
    # AWS Mar 2023 Kinesis/Lambda cluster
    "2023-03-07",
]


def main():
    sid = "PL698_cloud_sev1_long_ddog_net"
    # DDOG IPO: Sep 2019; NET IPO: Sep 2019
    try:
        px = load_prices(["DDOG", "NET", "XLK", "SPY"], start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    required = ["DDOG", "NET", "XLK", "SPY"]
    missing = [t for t in required if t not in px.columns or px[t].dropna().empty]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    px = px.ffill().dropna(subset=required)
    ret = daily_returns(px)
    ddog_r = ret["DDOG"]
    net_r = ret["NET"]
    xlk_r = ret["XLK"]
    spy_r = ret["SPY"]

    # Convert known event dates to trading dates
    triggers = []
    last_trigger = pd.Timestamp("2000-01-01")

    for date_str in sorted(KNOWN_SEV1_CLUSTERS):
        event_dt = pd.Timestamp(date_str)
        # Find nearest trading day on or after event
        candidates = ret.index[ret.index >= event_dt]
        if candidates.empty:
            continue
        trigger_date = candidates[0]
        # Debounce: no two triggers within 45 days
        if (trigger_date - last_trigger).days < 45:
            continue
        if trigger_date not in ret.index:
            continue
        triggers.append(trigger_date)
        last_trigger = trigger_date

    print(f"Sev-1 cluster events: {len(triggers)}")

    if len(triggers) < 3:
        return mark_failed(sid, f"insufficient events ({len(triggers)})")

    # Build PnL: long DDOG+NET 50/50, hedge 50% short XLK, hold 30 days
    hold = 30
    pnl = pd.Series(0.0, index=ret.index)
    events = []

    for entry_date in triggers:
        p = ret.index.get_loc(entry_date)
        ep = min(p + hold, len(ret))

        chunk_ddog = ddog_r.iloc[p:ep]
        chunk_net = net_r.iloc[p:ep]
        chunk_xlk = xlk_r.iloc[p:ep]

        # Long 0.5*DDOG + 0.5*NET, short 0.5*XLK
        chunk_pnl = 0.5 * chunk_ddog + 0.5 * chunk_net - 0.5 * chunk_xlk

        # 8% stop loss on gross basket
        gross_pnl = 0.5 * chunk_ddog + 0.5 * chunk_net
        cum_gross = (1 + gross_pnl).cumprod() - 1
        stop_hits = cum_gross[cum_gross < -0.08]
        if not stop_hits.empty:
            stop_i = cum_gross.index.get_loc(stop_hits.index[0])
            chunk_pnl = chunk_pnl.iloc[:stop_i + 1]

        n = len(chunk_pnl)
        pnl.iloc[p:p + n] += chunk_pnl.values[:n]

        basket_ret = float((1 + chunk_pnl).prod() - 1)
        ddog_ret = float((1 + chunk_ddog.iloc[:n]).prod() - 1)
        net_ret = float((1 + chunk_net.iloc[:n]).prod() - 1)
        sp_chunk = spy_r.iloc[p:ep]
        sp_ret = float((1 + sp_chunk).prod() - 1) if len(sp_chunk) > 0 else None

        events.append({
            "entry_date": str(entry_date.date()),
            "ddog_return": round(ddog_ret, 4),
            "net_return": round(net_ret, 4),
            "basket_return": round(basket_ret, 4),
            "spy_return": round(sp_ret, 4) if sp_ret is not None else None,
        })

    active = pnl[pnl != 0]
    print(f"Active PnL days: {len(active)}")
    if len(active) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active)})")

    m = compute_metrics(active, benchmark=spy_r, name="Cloud Sev-1 Cluster Long DDOG/NET vs XLK")

    rets = [e["basket_return"] for e in events]
    save_result(sid, m, extra={
        "rule": "Long DDOG+NET equal-weight 30 days after AWS/Azure Sev-1 cluster (3+ incidents in 30d), hedge 50% short XLK; 8% stop-loss",
        "mechanism": "Major cloud outages accelerate enterprise demand for observability (Datadog) and security (Cloudflare) solutions; companies increase monitoring/CDN budgets post-incident",
        "source": "yfinance DDOG, NET, XLK; AWS Service Health Dashboard, Azure Status History (curated event list)",
        "n_events": len(events),
        "avg_event_return": round(float(np.mean(rets)), 4) if rets else None,
        "event_win_rate": round(float(np.mean([r > 0 for r in rets])), 4) if rets else None,
        "events": events,
        "caveats": "Limited history (DDOG/NET IPO Sept 2019); curated event list may have selection bias; actual AWS/Azure status pages not scraped",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
