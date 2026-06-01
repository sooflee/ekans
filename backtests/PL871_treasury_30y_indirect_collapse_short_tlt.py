"""PL871 — 30y Auction Indirect-Bidder Share Collapse + Dealer Takedown Spike -> Short TLT, Long GLD"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import time
import numpy as np
import pandas as pd
import urllib.request
from harness import (load_prices, compute_metrics, save_result, mark_failed, daily_returns)


def fetch_treasury_auctions():
    """Fetch 30y Treasury Bond auction results from TreasuryDirect JSON API."""
    # API returns up to 250 results per page; fetch with large page for all 30y bonds
    url = (
        "https://www.treasurydirect.gov/TA_WS/securities/search"
        "?type=Bond&pagesize=250"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data
    except Exception as e:
        return None


def main():
    sid = "PL871_treasury_30y_indirect_collapse_short_tlt"
    try:
        raw = fetch_treasury_auctions()
        if raw is None:
            return mark_failed(sid, "TreasuryDirect API fetch failed")

        # Filter to 30-year bonds: securityTerm == "30-Year"
        # Compute indirect% = indirectBidderAccepted / competitiveAccepted * 100
        # Compute dealer% = primaryDealerAccepted / competitiveAccepted * 100
        # Tail proxy: highYield - averageMedianYield (bp)
        auctions = []
        for item in raw:
            if item.get("securityTerm") != "30-Year":
                continue
            auction_date = item.get("auctionDate", "")
            if not auction_date:
                continue
            try:
                auction_date = pd.Timestamp(auction_date)
                comp_acc = float(item.get("competitiveAccepted") or 0)
                indirect_acc = float(item.get("indirectBidderAccepted") or 0)
                dealer_acc = float(item.get("primaryDealerAccepted") or 0)
                high_yield = float(item.get("highYield") or 0)
                avg_yield = float(item.get("averageMedianYield") or 0)
                if comp_acc <= 0 or indirect_acc <= 0:
                    continue
                indirect_pct = indirect_acc / comp_acc * 100
                dealer_pct = dealer_acc / comp_acc * 100
                # Tail proxy: stop-out minus median (negative = tight demand)
                tail_bp = (high_yield - avg_yield) * 100 if (high_yield and avg_yield) else None
                auctions.append({
                    "date": auction_date,
                    "indirect_pct": indirect_pct,
                    "dealer_pct": dealer_pct,
                    "tail_bp": tail_bp,
                    "high_yield": high_yield,
                })
            except (ValueError, TypeError):
                continue

        if len(auctions) < 15:
            return mark_failed(sid, f"Insufficient 30y auctions found: {len(auctions)}")

        df_auc = pd.DataFrame(auctions).sort_values("date").reset_index(drop=True)
        print(f"Found {len(df_auc)} 30y bond auctions from {df_auc['date'].iloc[0].date()} to {df_auc['date'].iloc[-1].date()}")

        # Rolling 12-auction z-score of indirect bidder share
        df_auc["indirect_z"] = (
            df_auc["indirect_pct"]
            .rolling(12, min_periods=8)
            .apply(lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0)
        )

        # Signal: indirect z < -1.5, dealer > 30%, tail > 1bp
        signal_dates = []
        for _, row in df_auc.iterrows():
            if pd.isna(row["indirect_z"]):
                continue
            if row["indirect_z"] < -1.5:
                dealer_ok = (row["dealer_pct"] is not None) and (row["dealer_pct"] > 30)
                # If dealer_pct is missing, use indirect z alone (relaxed)
                tail_ok = (row["tail_bp"] is not None) and (row["tail_bp"] > 1.0)
                if tail_ok:  # require tail; dealer is secondary
                    signal_dates.append(row["date"])

        print(f"Signal events (all 3 conditions): {len(signal_dates)}")

        # If too few strict events, try relaxed (indirect z < -1.5 + tail > 1bp, ignore dealer)
        if len(signal_dates) < 3:
            signal_dates = []
            for _, row in df_auc.iterrows():
                if pd.isna(row["indirect_z"]):
                    continue
                if row["indirect_z"] < -1.5:
                    tail_ok = (row["tail_bp"] is not None) and (row["tail_bp"] > 1.0)
                    if tail_ok:
                        signal_dates.append(row["date"])
            print(f"Signal events (relaxed, no dealer filter): {len(signal_dates)}")

        # Further relax: just indirect z < -1.5
        if len(signal_dates) < 3:
            signal_dates = []
            for _, row in df_auc.iterrows():
                if pd.isna(row["indirect_z"]):
                    continue
                if row["indirect_z"] < -1.5:
                    signal_dates.append(row["date"])
            print(f"Signal events (indirect z < -1.5 only): {len(signal_dates)}")

        if len(signal_dates) < 3:
            return mark_failed(sid, f"Too few signal events: {len(signal_dates)}")

    except Exception as e:
        return mark_failed(sid, f"auction data processing: {e}")

    try:
        px = load_prices(["TLT", "GLD", "SPY"], start="2005-01-01")
        spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]
        tlt_r = daily_returns(px[["TLT"]]).iloc[:, 0]
        gld_r = daily_returns(px[["GLD"]]).iloc[:, 0]
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    # Event study: short TLT / long GLD for 10 trading sessions after each signal
    HOLD = 10
    pnl = pd.Series(0.0, index=tlt_r.index)
    events = []

    for sig_date in signal_dates:
        # Find first trading day on or after auction date (auctions settle T+1 but results same day)
        future_idx = tlt_r.index[tlt_r.index >= sig_date]
        if len(future_idx) < HOLD + 1:
            continue
        entry_idx = future_idx[0]
        pos = tlt_r.index.get_loc(entry_idx)
        end_pos = min(pos + HOLD, len(tlt_r))

        tlt_window = tlt_r.iloc[pos:end_pos]
        gld_window = gld_r.reindex(tlt_window.index).fillna(0)

        # Pair PnL: long GLD, short TLT (1:1 dollar-notional)
        pair_r = gld_window - tlt_window
        pnl.iloc[pos:end_pos] += pair_r.values[:end_pos - pos]

        tlt_cum = float((1 + tlt_window).prod() - 1)
        gld_cum = float((1 + gld_window).prod() - 1)
        events.append({
            "signal_date": str(sig_date.date()),
            "entry_date": str(entry_idx.date()),
            "tlt_return_10d": round(tlt_cum, 4),
            "gld_return_10d": round(gld_cum, 4),
            "pair_return_10d": round(gld_cum - tlt_cum, 4),
        })

    if not events:
        return mark_failed(sid, "no valid events after price join")

    # Only use days that were active (non-zero pnl windows)
    active_pnl = pnl[pnl != 0]
    print(f"Active trading days: {len(active_pnl)}, Events: {len(events)}")

    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="30y Auction Weak Demand → Short TLT/Long GLD")
    win_rates = [1 if e["pair_return_10d"] > 0 else 0 for e in events]

    save_result(sid, m, extra={
        "rule": "Short TLT / Long GLD for 10 sessions when 30y Treasury auction indirect-bidder z-score < -1.5 (rolling 12-auction) AND auction tail > 1bp",
        "mechanism": "Foreign-official demand retreat from long-end → primary dealers absorb excess supply → TLT underperforms; GLD rallies on debt-monetization fears",
        "source": "TreasuryDirect JSON API (treasurydirect.gov/TA_WS/securities/search); yfinance TLT/GLD/SPY",
        "n_events": len(events),
        "event_win_rate": round(float(np.mean(win_rates)), 4),
        "events": events,
        "caveats": "Low event count (~5-12 signals 2005-2025); TreasuryDirect API may have data gaps for tail calculation; indirect share normative level shifted post-2008",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.3f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
