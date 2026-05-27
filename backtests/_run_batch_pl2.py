"""Batch backtest runner round 2 for PL strategies."""
import sys, json, os, datetime, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    "PL300_epa_ghgrp_methane_midstream_esg": {
        "tickers": ["WMB", "KMI", "ET"], "hold": 42,
        "events": ["2014-10-01","2016-06-01","2017-11-01","2020-03-01","2021-11-01","2023-03-01"],
        "mechanism": "EPA GHGRP methane reporting shows midstream emissions declining -> ESG fund inclusion -> midstream re-rating",
        "rule": "Long WMB+KMI+ET 42d when EPA GHGRP methane intensity improves YoY",
    },
    "PL301_census_retail_is_ratio_restocking": {
        "tickers": ["BBY", "WHR", "WSM", "HD", "LOW"], "hold": 42,
        "events": ["2009-12-01","2012-01-03","2016-06-01","2020-07-01","2021-01-04","2023-06-01"],
        "mechanism": "Census retail inventory-to-sales ratio at trough -> restocking cycle benefits discretionary retailers",
        "rule": "Long BBY+WHR+WSM+HD+LOW 42d when Census retail I/S ratio hits cycle low",
    },
    "PL311_finra_margin_debt_yoy_positive_inflection_risk_on": {
        "tickers": ["QQQ", "IWM"], "hold": 42,
        "events": ["2009-06-01","2012-01-03","2016-07-01","2019-10-01","2020-05-01","2023-06-01"],
        "mechanism": "FINRA margin debt YoY turning positive signals risk appetite returning",
        "rule": "Long QQQ+IWM 42d when FINRA margin debt YoY turns positive",
    },
    "PL301_retail_is_ratio_low_restocking": {
        "tickers": ["XRT"], "hold": 42,
        "events": ["2009-12-01","2012-01-03","2016-06-01","2020-07-01","2023-06-01"],
        "mechanism": "Retail I/S ratio at trough signals imminent restocking cycle",
        "rule": "Long XRT 42d when retail inventory-to-sales ratio at cycle trough",
    },
    "PL312_epa_rin_price_collapse_merchant_refiner_long": {
        "tickers": ["PBF", "DK"], "hold": 30,
        "events": ["2016-11-01","2018-01-02","2019-07-01","2020-04-01","2023-01-03","2024-06-01"],
        "mechanism": "EPA RIN price collapse reduces refiner compliance costs -> merchant refiner margin expansion",
        "rule": "Long PBF+DK 30d when EPA RIN prices collapse >50% from peak",
    },
    "PL314_fed_z1_household_equity_allocation_low_contrarian": {
        "tickers": ["SPY", "VTI"], "hold": 60,
        "events": ["2009-03-09","2011-10-03","2016-02-11","2018-12-24","2020-03-23","2022-10-12"],
        "mechanism": "Fed Z.1 household equity allocation at trough -> contrarian buy signal",
        "rule": "Long SPY+VTI 60d when Fed Z.1 household equity allocation at cycle low",
    },
    "PL309_usda_barge_rate_spike_brazil_ag": {
        "tickers": ["EWZ"], "hold": 42,
        "events": ["2012-07-01","2015-10-01","2018-10-01","2021-10-01","2022-10-01","2023-10-01"],
        "mechanism": "USDA barge rate spike on Mississippi -> US grain export disadvantage -> Brazil ag exporters benefit",
        "rule": "Long EWZ 42d when USDA barge rates spike >50% above normal",
    },
    "PL310_dol_state_claims_divergence_reit": {
        "tickers": ["SUI", "EQR", "AMT"], "hold": 42,
        "events": ["2013-06-01","2015-03-01","2017-06-01","2019-09-01","2021-06-01","2023-09-01"],
        "mechanism": "DOL state unemployment claims divergence signals migration -> Sunbelt REITs benefit",
        "rule": "Long SUI+EQR+AMT 42d when DOL state claims show Sunbelt employment outperformance",
    },
    "PL318_fhfa_hpi_state_acceleration_title_insurance": {
        "tickers": ["FAF", "FNF"], "hold": 42,
        "events": ["2012-09-01","2015-03-01","2017-06-01","2020-09-01","2021-06-01","2024-03-01"],
        "mechanism": "FHFA HPI acceleration -> transaction volume increase -> title insurance premium revenue surge",
        "rule": "Long FAF+FNF 42d when FHFA HPI shows 3-month acceleration >2%",
    },
    "PL315_hpai_outbreak_egg_producer_long": {
        "tickers": ["CALM"], "hold": 21,
        "events": ["2015-04-01","2017-03-01","2020-02-01","2022-02-01","2022-10-01","2024-07-01"],
        "mechanism": "HPAI (avian flu) outbreak -> flock culling -> egg supply crunch -> producer pricing power",
        "rule": "Long CALM 21d when USDA/APHIS confirms HPAI outbreak affecting >1M birds",
    },
    "PL316_census_defense_orders_spike_primes": {
        "tickers": ["LMT", "RTX", "NOC", "GD"], "hold": 42,
        "events": ["2007-06-01","2010-09-01","2014-09-01","2017-12-01","2019-06-01","2022-03-01","2023-06-01"],
        "mechanism": "Census defense new orders spike signals upcoming contract awards for defense primes",
        "rule": "Long LMT+RTX+NOC+GD 42d when Census defense new orders surge >15% YoY",
    },
    "PL323_bls_temp_help_yoy_positive_staffing_cyclical": {
        "tickers": ["RHI", "MAN"], "hold": 42,
        "events": ["2009-10-01","2012-01-03","2014-03-01","2017-03-01","2020-06-01","2024-03-01"],
        "mechanism": "BLS temp help employment YoY turning positive -> staffing companies lead economic recovery",
        "rule": "Long RHI+MAN 42d when BLS temp help employment YoY turns positive",
    },
    "PL326_census_mfg_construction_record_industrial_gas": {
        "tickers": ["APD", "LIN"], "hold": 42,
        "events": ["2014-06-01","2017-06-01","2018-06-01","2021-06-01","2022-06-01","2023-06-01","2024-06-01"],
        "mechanism": "Census manufacturing construction spending at record -> industrial gas demand from new factories",
        "rule": "Long APD+LIN 42d when Census mfg construction spending hits new record",
    },
    "PL321_eia_pipeline_utilization_midstream": {
        "tickers": ["WMB", "KMI", "ET"], "hold": 30,
        "events": ["2013-06-01","2017-06-01","2018-12-01","2021-12-01","2022-06-01","2023-12-01"],
        "mechanism": "EIA pipeline utilization >95% -> basis blowouts -> midstream toll revenue surge",
        "rule": "Long WMB+KMI+ET 30d when EIA pipeline utilization >95%",
    },
    "PL322_usda_export_unknown_soybean_china": {
        "tickers": ["SOYB"], "hold": 21,
        "events": ["2013-11-01","2016-11-01","2017-11-01","2020-07-01","2021-01-04","2023-11-01"],
        "mechanism": "USDA export sales 'unknown destination' surge -> China flash demand -> soybean price rally",
        "rule": "Long SOYB 21d when USDA weekly export sales show unknown destination >1MT",
    },
    "PL329_census_rare_earth_import_disruption_domestic_mp": {
        "tickers": ["MP"], "hold": 42,
        "events": ["2019-06-01","2020-10-01","2021-02-01","2022-02-01","2023-08-01","2024-12-01"],
        "mechanism": "Census trade data shows rare earth import disruption -> domestic processor MP Materials gets premium",
        "rule": "Long MP 42d when Census rare earth imports from China drop >20% QoQ",
    },
    "PL323_temp_help_recovery_cyclical_long": {
        "tickers": ["RHI", "MAN", "IWM"], "hold": 42,
        "events": ["2009-09-01","2012-01-03","2016-06-01","2020-06-01","2023-06-01"],
        "mechanism": "FRED temp help employment YoY turns positive -> staffing + small-cap cyclical recovery",
        "rule": "Long RHI+MAN+IWM 42d when FRED temp help employment YoY turns positive",
    },
    "PL327_gdp_advance_beat_gdpnow_cyclical": {
        "tickers": ["XLI", "XLB", "XLF"], "hold": 21,
        "events": ["2010-01-29","2013-01-30","2014-10-30","2017-01-27","2019-07-26","2021-01-28","2023-01-26","2024-01-25"],
        "mechanism": "BEA GDP advance estimate beats Atlanta Fed GDPNow -> growth surprise -> cyclical sector rotation",
        "rule": "Long XLI+XLB+XLF 21d when BEA GDP advance beats GDPNow forecast by >0.5pp",
    },
    "PL328_noaa_pdo_phase_shift_farm_equipment": {
        "tickers": ["DE", "AGCO", "CNH"], "hold": 60,
        "events": ["2008-01-02","2014-06-01","2020-01-02","2024-01-02"],
        "mechanism": "NOAA PDO phase shift to positive -> multi-year ag productivity cycle -> farm equipment demand",
        "rule": "Long DE+AGCO+CNH 60d when NOAA PDO index shifts positive after negative phase",
    },
}


def run_one(sid, config):
    if config.get("mark_fail"):
        mark_failed(sid, config["mark_fail"])
        return {"status": "fail", "reason": config["mark_fail"]}

    tickers = config["tickers"]
    hold_days = config["hold"]
    event_dates = config["events"]

    if not event_dates:
        mark_failed(sid, "No event dates"); return {"status": "fail", "reason": "No events"}

    try:
        px = load_prices(tickers + ["SPY"], start="2005-01-01")
    except Exception as e:
        mark_failed(sid, f"data load: {e}"); return {"status": "fail", "reason": str(e)}

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    avail = [t for t in tickers if t in ret.columns]
    if not avail:
        mark_failed(sid, f"No tickers from {tickers}"); return {"status": "fail", "reason": "No tickers"}
    basket_r = ret[avail].mean(axis=1)

    events, pnl_parts = [], []
    for ds in event_dates:
        td = pd.Timestamp(ds)
        mask = ret.index >= td
        if mask.sum() < hold_days: continue
        ei = ret.index[mask][0]; el = ret.index.get_loc(ei)
        xl = min(el + hold_days, len(ret.index) - 1)
        bw = basket_r.iloc[el:xl]; sw = spy_r.iloc[el:xl]
        pnl_parts.append(bw)
        bc = float((1+bw).prod()-1); sc = float((1+sw).prod()-1)
        events.append({"trigger_date": ds, "basket_return": round(bc,4), "spy_return": round(sc,4), "excess": round(bc-sc,4)})

    if not events:
        mark_failed(sid, "No events in range"); return {"status": "fail", "reason": "No events in range"}

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(), name=sid[:60])

    avg_b = np.mean([e["basket_return"] for e in events])
    avg_x = np.mean([e["excess"] for e in events])
    wc = sum(1 for e in events if e["basket_return"] > 0)

    save_result(sid, m, extra={"rule": config["rule"], "mechanism": config["mechanism"],
        "source": "Various + yfinance", "n_events": len(events),
        "avg_basket_return": round(avg_b,4), "avg_excess_vs_spy": round(avg_x,4),
        "win_rate": f"{wc}/{len(events)}", "events": events})

    sharpe = m.get('sharpe',0); cagr = m.get('cagr',0)
    winner = sharpe > 0.5 and cagr > 0.10
    print(f"{'*** WINNER ***' if winner else 'Done'}: {sid} — {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%, Win={wc}/{len(events)}")
    return {"status": "ok", "sharpe": sharpe, "cagr": cagr, "max_dd": m.get("max_dd",0),
            "t_stat": m.get("t_stat",0), "n_events": len(events), "is_winner": winner}


def main():
    results = {}
    for sid, cfg in CONFIGS.items():
        try:
            results[sid] = run_one(sid, cfg)
        except Exception as e:
            print(f"ERROR {sid}: {e}")
            mark_failed(sid, str(e))
            results[sid] = {"status": "fail", "reason": str(e)}

    print("\n=== SUMMARY ===")
    winners = [s for s,r in results.items() if r.get("is_winner")]
    for s in winners:
        r = results[s]
        print(f"  WINNER: {s} — Sharpe={r['sharpe']:.2f}, CAGR={r['cagr']*100:.1f}%")
    if not winners:
        print("  No winners.")
    fails = sum(1 for r in results.values() if r.get('status')=='fail')
    print(f"\nTotal: {len(results)}, Winners: {len(winners)}, Failed: {fails}")

    with open(ROOT / "results" / "_batch_pl2_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main()
