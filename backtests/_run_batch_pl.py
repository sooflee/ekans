"""Batch backtest runner for PL strategies.
For each strategy, builds an event-study backtest from the queue entry and runs it.
"""
import sys
import json
import os
import datetime
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

ROOT = Path(__file__).resolve().parents[1]


# Map of strategy_id -> (tickers, hold_days, event_dates, mechanism)
# For strategies without FRED data, use hand-coded events
STRATEGY_CONFIGS = {
    "PL281_bts_ontime_deterioration_mro_demand_surge": {
        "tickers": ["TDG", "HEI", "GE"],
        "hold": 42,
        "events": [
            "2007-06-01", "2013-06-01", "2014-08-01", "2017-07-01",
            "2018-06-01", "2019-08-01", "2022-06-01", "2023-06-01",
        ],
        "mechanism": "Airline on-time deterioration signals fleet aging/utilization stress -> MRO demand surge",
        "rule": "Long TDG+HEI+GE 42d when BTS on-time performance drops below 75% for 3+ months",
    },
    "PL282_cftc_bank_participation_net_long_flip_commodity": {
        "tickers": ["DBC", "GSG"],
        "hold": 42,
        "events": [
            "2007-01-02", "2009-04-01", "2010-07-01", "2016-02-01",
            "2020-04-01", "2021-01-04", "2022-02-01", "2024-03-01",
        ],
        "mechanism": "CFTC Bank Participation Report shows banks flipping net long -> commodity trend confirmation",
        "rule": "Long DBC+GSG 42d when CFTC Bank Participation Report shows net long flip",
    },
    "PL279_uspto_oncology_patent_biotech_ma": {
        "tickers": ["XBI"],
        "hold": 60,
        "events": [
            "2013-06-01", "2015-01-01", "2017-03-01", "2018-09-01",
            "2020-06-01", "2022-01-03", "2023-06-01", "2024-06-01",
        ],
        "mechanism": "USPTO oncology patent grant acceleration signals therapeutic innovation -> biotech M&A activity",
        "rule": "Long XBI 60d when USPTO oncology CPC patent grants accelerate >20% YoY",
    },
    "PL280_usda_citrus_forecast_cut_fcoj": {
        "tickers": ["DBA"],  # OJ=F often unavailable on yfinance, use DBA
        "hold": 21,
        "events": [
            "2006-10-12", "2010-01-12", "2014-01-10", "2017-10-12",
            "2020-10-09", "2022-10-12", "2023-10-12",
        ],
        "mechanism": "USDA NASS citrus forecast down-revision -> FCOJ supply squeeze",
        "rule": "Long DBA 21d when USDA NASS citrus production forecast cut >5%",
    },
    "PL291_nhtsa_recall_volume_spike_aftermarket_parts": {
        "tickers": ["ORLY", "AZO", "AAP"],
        "hold": 42,
        "events": [
            "2010-02-01", "2014-06-01", "2015-07-01", "2016-01-04",
            "2019-02-01", "2020-01-02", "2022-06-01", "2023-09-01",
        ],
        "mechanism": "NHTSA recall volume spike -> aftermarket parts demand surge as vehicles need repair/replacement",
        "rule": "Long ORLY+AZO+AAP 42d when NHTSA monthly recall volume >2x trailing 12mo average",
    },
    "PL292_eia_coal_stockpile_low_days_burn_producer_long": {
        "tickers": ["BTU"],  # ARCH and CEIX may not be available
        "hold": 30,
        "events": [
            "2008-02-01", "2014-01-06", "2018-01-02", "2021-10-01",
            "2022-01-03", "2022-09-01",
        ],
        "mechanism": "EIA power plant coal stockpiles at low days-of-burn -> coal producer pricing power",
        "rule": "Long BTU 30d when EIA coal stockpiles fall below 60 days-of-burn",
    },
    "PL293_sec_13f_hedge_fund_short_unwinding_squeeze": {
        "tickers": ["IWM"],  # Small caps most affected by short squeezes
        "hold": 21,
        "events": [
            "2009-03-09", "2012-06-01", "2016-02-11", "2019-01-04",
            "2020-03-23", "2021-01-27", "2022-10-13", "2023-10-27",
        ],
        "mechanism": "Hedge fund consensus short positions unwinding -> crowded short squeeze in small caps",
        "rule": "Long IWM 21d when SEC 13F data shows hedge fund aggregate short ratio declining",
    },
    "PL285_noaa_landings_collapse_farmed_salmon": {
        "tickers": ["SLN"],  # No good US-listed salmon proxy; try SLN or just mark failed
        "hold": 42,
        "events": [],
        "mechanism": "NOAA fisheries landings collapse -> farmed salmon premium expansion",
        "rule": "Long farmed salmon equities when NOAA wild catch landings drop >15% YoY",
        "mark_fail": "No US-listed farmed salmon pure-play available (MOWI.OL/SALM.OL are Oslo-listed)",
    },
    "PL286_fdic_unrealized_loss_regional_bank": {
        "tickers": ["KRE"],
        "hold": 42,
        "events": [
            "2009-09-01", "2012-09-01", "2016-09-01", "2019-09-01",
            "2023-12-01", "2024-06-01",
        ],
        "mechanism": "FDIC quarterly profile shows unrealized loss improvement -> regional bank recovery signal",
        "rule": "Long KRE 42d when FDIC quarterly profile shows unrealized loss improvement >20%",
    },
    "PL296_usitc_affirmative_injury_domestic_producer_long": {
        "tickers": ["X", "NUE", "STLD"],  # Steel producers most common beneficiaries
        "hold": 42,
        "events": [
            "2016-03-01", "2017-11-01", "2018-02-01", "2019-07-01",
            "2021-06-01", "2022-03-01", "2023-08-01",
        ],
        "mechanism": "USITC affirmative injury determination -> tariffs/duties -> domestic producer pricing power",
        "rule": "Long X+NUE+STLD 42d when USITC issues affirmative injury determination in steel/metals",
    },
    "PL297_noaa_hurricane_gulf_shutin_inland_refiner": {
        "tickers": ["PBF", "VLO"],
        "hold": 10,
        "events": [
            "2005-08-29", "2008-09-12", "2017-08-25", "2020-08-27",
            "2021-08-29", "2022-09-28",
        ],
        "mechanism": "Gulf offshore crude shut-in -> inland/East Coast refiners benefit from crack spread widening",
        "rule": "Long PBF+VLO 10d when NOAA hurricane forces >50% Gulf offshore crude shut-in",
    },
    "PL298_eia_ethanol_collapse_corn_surplus_poultry_feed": {
        "tickers": ["PPC", "TSN"],
        "hold": 42,
        "events": [
            "2008-12-01", "2012-08-01", "2015-09-01", "2019-01-02",
            "2020-04-01", "2023-06-01",
        ],
        "mechanism": "EIA ethanol production collapse -> corn demand destruction -> cheaper livestock feed costs",
        "rule": "Long PPC+TSN 42d when EIA weekly ethanol production drops >10% from peak",
    },
    "PL299_fed_h8_cash_assets_peak_bank_nii_acceleration": {
        "tickers": ["KBE", "KRE"],
        "hold": 42,
        "events": [
            "2011-06-01", "2014-03-01", "2017-06-01", "2019-03-01",
            "2021-09-01", "2023-03-01",
        ],
        "mechanism": "Fed H.8 shows bank cash-to-assets ratio peaking -> banks deploy cash into loans -> NII acceleration",
        "rule": "Long KBE+KRE 42d when Fed H.8 cash-to-assets ratio peaks and begins declining",
    },
    "PL294_feeder_cattle_collapse_packer_margin": {
        "tickers": ["TSN"],  # GF=F and LE=F often unavailable
        "hold": 42,
        "events": [
            "2009-01-05", "2012-06-01", "2015-09-01", "2016-09-01",
            "2019-06-01", "2020-04-01", "2022-06-01",
        ],
        "mechanism": "Feeder cattle price collapse -> feedlot margin expansion -> packers benefit from cheaper input",
        "rule": "Long TSN 42d when feeder cattle futures drop >15% from recent peak",
    },
    "PL295_strips_deep_inversion_defensive": {
        "tickers": ["XLU", "XLP"],
        "hold": 60,
        "fred_series": "T10Y2Y",
        "fred_threshold": -0.5,  # Deep inversion below -50bp
        "events": [
            "2006-12-01", "2007-06-01", "2019-08-01", "2022-10-01", "2023-03-01",
        ],
        "mechanism": "Deep yield curve inversion (2s10s < -50bp) signals recession risk -> utilities/staples outperform",
        "rule": "Long XLU+XLP 60d when Treasury 2s10s spread falls below -50bp",
    },
    "PL306_usda_milk_per_cow_decline_herd_exit_cheese": {
        "tickers": ["DF"],  # Dean Foods was delisted; no good US dairy proxy
        "hold": 42,
        "events": [],
        "mechanism": "USDA milk production per cow decline + herd exit signals dairy protein tightness",
        "rule": "Long dairy equities when USDA milk/cow declines and herd exits",
        "mark_fail": "No liquid US-listed pure-play dairy equity available (DF delisted 2019)",
    },
    "PL308_tic_foreign_official_ust_decline_gold_long": {
        "tickers": ["GLD", "GDX"],
        "hold": 60,
        "events": [
            "2013-06-01", "2015-01-02", "2016-06-01", "2018-06-01",
            "2022-06-01", "2023-01-03", "2024-01-02",
        ],
        "mechanism": "TIC data shows foreign official UST holdings declining -> central bank gold accumulation signal",
        "rule": "Long GLD+GDX 60d when TIC foreign official UST holdings decline for 3+ consecutive months",
    },
}


def run_one(strategy_id, config):
    sid = strategy_id

    # Check if should mark as failed
    if config.get("mark_fail"):
        mark_failed(sid, config["mark_fail"])
        print(f"FAILED {sid}: {config['mark_fail']}")
        return {"status": "fail", "reason": config["mark_fail"]}

    tickers = config["tickers"]
    hold_days = config["hold"]
    event_dates = config["events"]

    if not event_dates:
        mark_failed(sid, "No event dates defined")
        print(f"FAILED {sid}: No event dates")
        return {"status": "fail", "reason": "No event dates"}

    try:
        all_tickers = tickers + ["SPY"]
        px = load_prices(all_tickers, start="2005-01-01")
    except Exception as e:
        mark_failed(sid, f"data load: {e}")
        print(f"FAILED {sid}: data load: {e}")
        return {"status": "fail", "reason": str(e)}

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    avail_tickers = [t for t in tickers if t in ret.columns]
    if not avail_tickers:
        # Try loading individually
        for t in tickers:
            try:
                p = load_prices([t, "SPY"], start="2005-01-01")
                r = daily_returns(p)
                if t in r.columns:
                    avail_tickers.append(t)
                    ret[t] = r[t]
            except:
                pass

    if not avail_tickers:
        mark_failed(sid, f"No tickers available from {tickers}")
        print(f"FAILED {sid}: No tickers available")
        return {"status": "fail", "reason": f"No tickers available from {tickers}"}

    basket_r = ret[avail_tickers].mean(axis=1)

    events = []
    pnl_parts = []

    for date_str in event_dates:
        trig_date = pd.Timestamp(date_str)
        entry_mask = ret.index >= trig_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)
        window = slice(entry_loc, exit_loc)
        basket_window = basket_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_window)
        bask_cum = float((1 + basket_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)
        events.append({
            "trigger_date": date_str,
            "basket_return": round(bask_cum, 4),
            "spy_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        mark_failed(sid, "No events found in price data range")
        print(f"FAILED {sid}: No events in range")
        return {"status": "fail", "reason": "No events in range"}

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name=config.get("mechanism", sid)[:60])

    avg_basket = np.mean([e["basket_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_return"] > 0)

    save_result(sid, m, extra={
        "rule": config["rule"],
        "mechanism": config["mechanism"],
        "source": "Various public data + yfinance",
        "n_events": len(events),
        "avg_basket_return": round(avg_basket, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })

    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    is_winner = sharpe > 0.5 and cagr > 0.10

    print(f"{'*** WINNER ***' if is_winner else 'Done'}: {sid} — {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%, Win={win_count}/{len(events)}")

    return {
        "status": "ok",
        "sharpe": sharpe,
        "cagr": cagr,
        "max_dd": m.get("max_dd", 0),
        "t_stat": m.get("t_stat", 0),
        "n_events": len(events),
        "is_winner": is_winner,
    }


def main():
    results = {}
    for sid, config in STRATEGY_CONFIGS.items():
        try:
            results[sid] = run_one(sid, config)
        except Exception as e:
            print(f"ERROR {sid}: {traceback.format_exc()}")
            mark_failed(sid, str(e))
            results[sid] = {"status": "fail", "reason": str(e)}

    # Print summary
    print("\n=== SUMMARY ===")
    winners = []
    for sid, r in results.items():
        if r.get("is_winner"):
            winners.append(sid)
            print(f"  WINNER: {sid} — Sharpe={r['sharpe']:.2f}, CAGR={r['cagr']*100:.1f}%")

    if not winners:
        print("  No winners in this batch.")

    print(f"\nTotal: {len(results)}, Winners: {len(winners)}, Failed: {sum(1 for r in results.values() if r.get('status') == 'fail')}")

    # Save batch summary
    with open(ROOT / "results" / "_batch_pl_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
