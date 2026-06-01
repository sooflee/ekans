"""PL762_ecb_fed_rate_diff_uup_long — ECB-Fed Policy Rate Differential Widening >100bps -> Long UUP / Short FXE"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL762_ecb_fed_rate_diff_uup_long"

    # Load FRED rate data
    try:
        fred_fed = load_fred("DFEDTARU", start="2008-01-01")  # Fed funds upper bound
        fred_ecb = load_fred("ECBDFR", start="2008-01-01")    # ECB deposit facility rate
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")

    try:
        px = load_prices(["UUP", "FXE", "SPY"], start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty or "UUP" not in px.columns or "FXE" not in px.columns:
        return mark_failed(sid, "UUP or FXE price data unavailable")

    ret = daily_returns(px)
    uup_r = ret["UUP"].dropna()
    fxe_r = ret["FXE"].dropna()
    spy_r = ret["SPY"].dropna()

    # Build combined index of trading days
    common_idx = uup_r.index.intersection(fxe_r.index)
    uup_r = uup_r.reindex(common_idx)
    fxe_r = fxe_r.reindex(common_idx)

    # Build the Fed-ECB differential on daily frequency (forward-fill FRED series)
    fed_rate = fred_fed.squeeze().rename("fed")
    ecb_rate = fred_ecb.squeeze().rename("ecb")

    # Combine on a daily calendar, forward-fill
    daily_cal = pd.date_range(start="2008-01-01", end=common_idx[-1], freq="D")
    fed_daily = fed_rate.reindex(daily_cal).ffill()
    ecb_daily = ecb_rate.reindex(daily_cal).ffill()
    diff_daily = fed_daily - ecb_daily  # Fed rate minus ECB rate

    # Compute rolling 60-calendar-day change in differential
    diff_60d_chg = diff_daily - diff_daily.shift(60)

    # Align to trading days
    diff_td = diff_60d_chg.reindex(common_idx).ffill()

    if diff_td.dropna().empty:
        return mark_failed(sid, "no differential data after alignment")

    # Generate signals: when 60d change > +100bps (1.0 pct point)
    THRESHOLD = 1.0   # 100 bps
    EARLY_EXIT_THRESHOLD = 0.15  # 15 bps (narrowing signal)
    HOLD_DAYS = 40
    MIN_GAP_CAL = 60  # minimum 60 calendar days between entries

    pnl = pd.Series(0.0, index=common_idx)
    evts = []
    last_exit_date = pd.Timestamp("2000-01-01")
    i = 0

    while i < len(common_idx):
        td = common_idx[i]
        val = diff_td.get(td)
        if pd.isna(val):
            i += 1
            continue

        # Entry condition: 60d change > +100bps and min gap since last exit
        if val > THRESHOLD and (td - last_exit_date).days >= MIN_GAP_CAL:
            entry_pos = i
            entry_date = td

            # Exit loop
            exit_pos = min(entry_pos + HOLD_DAYS, len(common_idx))
            early_exit_reason = "hold_complete"

            for j in range(entry_pos + 1, exit_pos):
                exit_td = common_idx[j]
                curr_diff_chg = diff_td.get(exit_td)
                if not pd.isna(curr_diff_chg) and curr_diff_chg < EARLY_EXIT_THRESHOLD:
                    exit_pos = j
                    early_exit_reason = "early_exit_diff_narrowing"
                    break

            window_idx = common_idx[entry_pos:exit_pos]
            uu_window = uup_r.reindex(window_idx)
            fx_window = fxe_r.reindex(window_idx)

            # Long UUP + Short FXE (equal notional) = (uup_r - fxe_r) / 2
            combo_pnl = (uu_window - fx_window) / 2

            if len(combo_pnl) >= 5:
                pnl.iloc[entry_pos:exit_pos] = combo_pnl.values

                uu_cum = float((1 + uu_window).prod() - 1)
                fx_cum = float((1 + fx_window).prod() - 1)
                combo_cum = float((1 + combo_pnl).prod() - 1)
                spy_w = spy_r.reindex(window_idx)
                spy_cum = float((1 + spy_w).prod() - 1) if len(spy_w) > 0 else None

                evts.append({
                    "entry_date": str(entry_date.date()),
                    "exit_reason": early_exit_reason,
                    "n_days": len(combo_pnl),
                    "fed_ecb_diff_60d_chg": round(float(val), 4),
                    "uup_return": round(uu_cum, 4),
                    "fxe_short_return": round(-fx_cum, 4),
                    "combo_return": round(combo_cum, 4),
                    "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
                })

                last_exit_date = common_idx[exit_pos - 1]
                i = exit_pos
                continue

        i += 1

    print(f"Events: {len(evts)}")
    if not evts:
        return mark_failed(sid, "no valid events found")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="ECB-Fed Diff Widening -> Long UUP / Short FXE")
    returns_list = [e["combo_return"] for e in evts]

    save_result(sid, m, extra={
        "rule": "Long UUP / Short FXE equal-notional when Fed-ECB 60-day rate differential change > +100bps; hold 40 trading days or exit if differential narrows below +15bps",
        "mechanism": "Interest rate differential predicts currency direction via covered interest parity; widening Fed-ECB spread = USD appreciation vs EUR",
        "source": "FRED DFEDTARU, FRED ECBDFR; yfinance UUP, FXE, SPY",
        "n_events": len(evts),
        "avg_event_return": round(float(np.mean(returns_list)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in returns_list])), 4),
        "events": evts,
    })

    print(f"Done: {len(evts)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 0)*100:.1f}%")
    for e in evts:
        print(f"  Entry {e['entry_date']}: UUP={e['uup_return']:.1%} FXE_short={e['fxe_short_return']:.1%} combo={e['combo_return']:.1%} ({e['exit_reason']})")


if __name__ == "__main__":
    main()
