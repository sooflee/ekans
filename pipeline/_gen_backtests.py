"""Generate backtest .py files for all 28 new strategies using a unified template."""
import json, os
from pathlib import Path

ROOT = Path("/Users/benson/Projects/ekans")

TEMPLATE = '''"""{signal_id} -- {name}
{rule}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "{signal_id}"
    
    # Load FRED data
    try:
        fred = load_fred({fred_series_list}, start="{fred_start}")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {{e}}")
    
    if fred is None or fred.empty:
        return mark_failed(sid, "FRED data empty")
    
{signal_logic}
    
    if len(trigger_dates) == 0:
        return mark_failed(sid, "no trigger events found")
    
    print(f"Trigger events: {{len(trigger_dates)}}")
    for d in trigger_dates:
        print(f"  {{d.date()}}")
    
    # Load prices
    try:
        px = load_prices({ticker_list}, start="{px_start}")
    except Exception as e:
        return mark_failed(sid, f"price load: {{e}}")
    
    ret = daily_returns(px)
    
    # Build basket
    trade_tickers = {trade_tickers}
    available = [t for t in trade_tickers if t in ret.columns]
    if len(available) == 0:
        return mark_failed(sid, f"no trade tickers available in data")
    
    basket_ret = ret[available].mean(axis=1)
    spy_ret = ret["SPY"] if "SPY" in ret.columns else basket_ret * 0
    
    hold_days = {hold_days}
    pnl_series = pd.Series(0.0, index=basket_ret.index)
    event_results = []
    
    for td in trigger_dates:
        entry_date = td + pd.offsets.MonthBegin(1) if not isinstance(td, pd.Timestamp) else td
        mask = basket_ret.index >= entry_date
        if mask.sum() < max(30, hold_days // 2):
            continue
        entry_idx = basket_ret.index[mask][0]
        pos = basket_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(basket_ret))
        if end_pos - pos < 30:
            continue
        
        event_rets = basket_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]
        
        spy_cumret = None
        spy_start = spy_ret.index.get_loc(entry_idx) if entry_idx in spy_ret.index else None
        if spy_start is not None:
            se = min(spy_start + hold_days, len(spy_ret))
            spy_cumret = float((1 + spy_ret.iloc[spy_start:se]).prod() - 1)
        
        event_results.append({{
            "trigger_date": str(td.date() if hasattr(td, 'date') else td),
            "return": round(cumret, 4),
            "spy_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        }})
    
    if len(event_results) == 0:
        return mark_failed(sid, "no valid events after price alignment")
    
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({{len(in_pos)}})")
    
    m = compute_metrics(in_pos, benchmark=spy_ret, name="{name}")
    rets_arr = [e["return"] for e in event_results]
    
    save_result(sid, m, extra={{
        "rule": "{rule_escaped}",
        "source": "FRED {fred_source}; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    }})
    print(f"Done: {{len(event_results)}} events, avg return={{np.mean(rets_arr)*100:.2f}}%")


if __name__ == "__main__":
    main()
'''

# Signal logic generators
def gen_yoy_above(params):
    s = params['series']
    thresh = params['threshold']
    consec = params.get('consecutive', 3)
    return f'''    series = fred["{s}"].dropna()
    if len(series) < 13:
        return mark_failed(sid, "insufficient data for YoY")
    yoy = series.pct_change(12) * 100
    yoy = yoy.dropna()
    
    above = yoy > {thresh}
    trigger_dates = []
    streak = 0
    fired = False
    for i in range(len(above)):
        if above.iloc[i]:
            streak += 1
            if streak >= {consec} and not fired:
                trigger_dates.append(yoy.index[i])
                fired = True
        else:
            streak = 0
            fired = False
'''

def gen_yoy_below(params):
    s = params['series']
    thresh = params['threshold']
    consec = params.get('consecutive', 2)
    return f'''    series = fred["{s}"].dropna()
    if len(series) < 13:
        return mark_failed(sid, "insufficient data for YoY")
    yoy = series.pct_change(12) * 100
    yoy = yoy.dropna()
    
    below = yoy < {thresh}
    trigger_dates = []
    streak = 0
    fired = False
    for i in range(len(below)):
        if below.iloc[i]:
            streak += 1
            if streak >= {consec} and not fired:
                trigger_dates.append(yoy.index[i])
                fired = True
        else:
            streak = 0
            fired = False
'''

def gen_yoy_turn_positive(params):
    s = params['series']
    neg = params.get('negative_months', 6)
    return f'''    series = fred["{s}"].dropna()
    if len(series) < 13:
        return mark_failed(sid, "insufficient data for YoY")
    yoy = series.pct_change({"4" if neg <= 4 else "12"}) * 100
    yoy = yoy.dropna()
    
    trigger_dates = []
    neg_count = 0
    fired = False
    for i in range(len(yoy)):
        if yoy.iloc[i] < 0:
            neg_count += 1
            fired = False
        elif yoy.iloc[i] >= 0 and neg_count >= {neg} and not fired:
            trigger_dates.append(yoy.index[i])
            fired = True
            neg_count = 0
        else:
            neg_count = 0
'''

def gen_yoy_reacceleration(params):
    s = params['series']
    hi = params['high_thresh']
    lo = params['low_thresh']
    lo_m = params['low_months']
    return f'''    series = fred["{s}"].dropna()
    if len(series) < 13:
        return mark_failed(sid, "insufficient data for YoY")
    yoy = series.pct_change(12) * 100
    yoy = yoy.dropna()
    
    trigger_dates = []
    low_count = 0
    fired = False
    for i in range(len(yoy)):
        if yoy.iloc[i] < {lo}:
            low_count += 1
            fired = False
        elif low_count >= {lo_m} and yoy.iloc[i] >= {hi} and not fired:
            trigger_dates.append(yoy.index[i])
            fired = True
            low_count = 0
        else:
            if yoy.iloc[i] >= {hi}:
                low_count = 0
'''

def gen_level_below(params):
    s = params['series']
    thresh = params['threshold']
    consec = params.get('consecutive', 6)
    ravg = params.get('rolling_avg', 1)
    return f'''    series = fred["{s}"].dropna()
    if len(series) < {ravg + 1}:
        return mark_failed(sid, "insufficient data")
    if {ravg} > 1:
        series = series.rolling({ravg}).mean().dropna()
    
    below = series < {thresh}
    trigger_dates = []
    streak = 0
    fired = False
    for i in range(len(below)):
        if below.iloc[i]:
            streak += 1
            if streak >= {consec} and not fired:
                trigger_dates.append(series.index[i])
                fired = True
        else:
            streak = 0
            fired = False
'''

def gen_level_above(params):
    s = params['series']
    thresh = params['threshold']
    return f'''    series = fred["{s}"].dropna()
    
    trigger_dates = []
    was_below = False
    for i in range(len(series)):
        if series.iloc[i] < {thresh}:
            was_below = True
        elif was_below and series.iloc[i] >= {thresh}:
            trigger_dates.append(series.index[i])
            was_below = False
'''

def gen_peak_then_decline(params):
    s = params['series']
    dp = params['decline_periods']
    return f'''    series = fred["{s}"].dropna()
    if len(series) < {dp + 2}:
        return mark_failed(sid, "insufficient data")
    
    trigger_dates = []
    for i in range({dp + 1}, len(series)):
        is_declining = all(series.iloc[i-j] < series.iloc[i-j-1] for j in range({dp}))
        if is_declining and series.iloc[i - {dp}] > series.iloc[i - {dp} - 1]:
            trigger_dates.append(series.index[i])
'''

def gen_level_cross_below(params):
    s = params['series']
    thresh = params['threshold']
    prior = params.get('prior_above', thresh + 100)
    pm = params.get('prior_months', 3)
    return f'''    series = fred["{s}"].dropna()
    
    trigger_dates = []
    above_count = 0
    fired = False
    for i in range(len(series)):
        if series.iloc[i] > {prior}:
            above_count += 1
            fired = False
        elif above_count >= {pm} and series.iloc[i] < {thresh} and not fired:
            trigger_dates.append(series.index[i])
            fired = True
            above_count = 0
        else:
            if series.iloc[i] < {thresh}:
                above_count = 0
'''

def gen_level_cross_above(params):
    s = params['series']
    thresh = params['threshold']
    pm = params.get('prior_below_months', 6)
    return f'''    series = fred["{s}"].dropna()
    
    trigger_dates = []
    below_count = 0
    fired = False
    for i in range(len(series)):
        if series.iloc[i] < {thresh}:
            below_count += 1
            fired = False
        elif below_count >= {pm} and series.iloc[i] >= {thresh} and not fired:
            trigger_dates.append(series.index[i])
            fired = True
            below_count = 0
        else:
            below_count = 0
'''

def gen_trough_bounce(params):
    s = params['series']
    level = params['trough_level']
    bounce = params['bounce_months']
    return f'''    series = fred["{s}"].dropna()
    
    trigger_dates = []
    for i in range({bounce + 1}, len(series)):
        # Check if series[i-bounce] was a local min below {level}
        if series.iloc[i - {bounce}] < {level}:
            is_trough = (series.iloc[i - {bounce}] <= series.iloc[i - {bounce} - 1])
            is_bouncing = all(series.iloc[i - j] > series.iloc[i - {bounce}] for j in range({bounce}))
            if is_trough and is_bouncing:
                trigger_dates.append(series.index[i])
'''

def gen_yoy_change_above(params):
    s = params['series']
    thresh = params['threshold']
    consec = params.get('consecutive', 3)
    return f'''    series = fred["{s}"].dropna()
    if len(series) < 13:
        return mark_failed(sid, "insufficient data")
    yoy_chg = series.diff(12)
    yoy_chg = yoy_chg.dropna()
    
    above = yoy_chg > {thresh}
    trigger_dates = []
    streak = 0
    fired = False
    for i in range(len(above)):
        if above.iloc[i]:
            streak += 1
            if streak >= {consec} and not fired:
                trigger_dates.append(yoy_chg.index[i])
                fired = True
        else:
            streak = 0
            fired = False
'''

def gen_ratio_rise(params):
    n = params['num']
    d = params['den']
    pct = params['rise_pct']
    return f'''    if "{n}" not in fred.columns or "{d}" not in fred.columns:
        return mark_failed(sid, "missing FRED series")
    
    # Align and compute ratio
    both = fred[["{n}", "{d}"]].dropna()
    if len(both) < 7:
        return mark_failed(sid, "insufficient data")
    ratio = both["{n}"] / both["{d}"]
    ratio = ratio.dropna()
    
    # Rolling 6-month low
    rolling_min = ratio.rolling(6, min_periods=3).min()
    pct_above = (ratio / rolling_min - 1) * 100
    
    trigger_dates = []
    fired = False
    for i in range(6, len(pct_above)):
        if pct_above.iloc[i] > {pct} and not fired:
            trigger_dates.append(ratio.index[i])
            fired = True
        elif pct_above.iloc[i] < {pct} / 2:
            fired = False
'''

def gen_ratio_below(params):
    n = params['num']
    d = params['den']
    thresh = params['threshold']
    return f'''    if "{n}" not in fred.columns or "{d}" not in fred.columns:
        return mark_failed(sid, "missing FRED series")
    
    # Handle different frequencies - forward fill quarterly GDP to align with daily Wilshire
    num = fred["{n}"].dropna()
    den = fred["{d}"].dropna()
    
    # Scale GDP to trillions if needed for Wilshire ratio
    if den.max() > 10000:
        den = den  # GDP in billions already
    
    # Resample both to monthly
    num_m = num.resample("M").last().dropna()
    den_m = den.resample("M").last().ffill().dropna()
    
    # Align
    idx = num_m.index.intersection(den_m.index)
    if len(idx) < 12:
        return mark_failed(sid, "insufficient aligned data")
    
    ratio = num_m.loc[idx] / den_m.loc[idx]
    ratio = ratio.dropna()
    # Normalize: for Wilshire 5000 / GDP, values are large. Use percentile approach.
    # If ratio typically > 100, it's Wilshire index / GDP in billions
    if ratio.median() > 10:
        ratio = ratio / 1000  # scale down
    
    trigger_dates = []
    was_above = False
    for i in range(len(ratio)):
        if ratio.iloc[i] > {thresh}:
            was_above = True
        elif was_above and ratio.iloc[i] < {thresh}:
            trigger_dates.append(ratio.index[i])
            was_above = False
'''

def gen_yoy_deceleration(params):
    s = params['series']
    decel = params['decel_pp']
    return f'''    series = fred["{s}"].dropna()
    if len(series) < 19:
        return mark_failed(sid, "insufficient data")
    yoy = series.pct_change(12) * 100
    yoy = yoy.dropna()
    
    # 6-month rolling peak of YoY
    rolling_peak = yoy.rolling(6, min_periods=3).max()
    decel = rolling_peak - yoy
    
    trigger_dates = []
    fired = False
    for i in range(6, len(decel)):
        if decel.iloc[i] > {decel} and not fired:
            trigger_dates.append(yoy.index[i])
            fired = True
        elif decel.iloc[i] < {decel} / 2:
            fired = False
'''

LOGIC_MAP = {
    "fred_yoy_above": gen_yoy_above,
    "fred_yoy_below": gen_yoy_below,
    "fred_yoy_turn_positive": gen_yoy_turn_positive,
    "fred_yoy_reacceleration": gen_yoy_reacceleration,
    "fred_level_below": gen_level_below,
    "fred_level_above": gen_level_above,
    "fred_peak_then_decline": gen_peak_then_decline,
    "fred_level_cross_below": gen_level_cross_below,
    "fred_level_cross_above": gen_level_cross_above,
    "fred_trough_bounce": gen_trough_bounce,
    "fred_yoy_change_above": gen_yoy_change_above,
    "ratio_rise": gen_ratio_rise,
    "ratio_below": gen_ratio_below,
    "fred_yoy_deceleration": gen_yoy_deceleration,
}

# Strategy definitions (same as in develop script)
STRATEGY_DEFS = {
    "PL102": {"signal_id": "PL102_soybean_oil_ppi_food", "name": "Soybean Oil PPI Spike -> Long GIS+CPB", "rule": "Long GIS+CPB 126d when WPU02220301 YoY > +20% for 3 months", "tickers": ["GIS", "CPB", "SPY"], "fred_series": ["WPU02220301"], "hold_days": 126, "entry_logic": "fred_yoy_above", "entry_params": {"series": "WPU02220301", "threshold": 20, "consecutive": 3}},
    "PL103": {"signal_id": "PL103_corn_crush_margin_ethanol", "name": "Corn Crush Margin -> Long ADM+GPRE", "rule": "Long ADM+GPRE 126d when GASREGW/PCORNUS ratio rises >20% from 6mo low", "tickers": ["ADM", "GPRE", "SPY"], "fred_series": ["GASREGW", "PCORNUS"], "hold_days": 126, "entry_logic": "ratio_rise", "entry_params": {"num": "GASREGW", "den": "PCORNUS", "rise_pct": 20}},
    "PL104": {"signal_id": "PL104_wheat_ppi_collapse_restaurants", "name": "Wheat PPI Collapse -> Long DRI+MCD", "rule": "Long DRI+MCD 126d when WPU02110301 YoY < -15%", "tickers": ["DRI", "MCD", "SPY"], "fred_series": ["WPU02110301"], "hold_days": 126, "entry_logic": "fred_yoy_below", "entry_params": {"series": "WPU02110301", "threshold": -15, "consecutive": 2}},
    "PL107": {"signal_id": "PL107_semi_ppi_deflation_trough", "name": "Semi PPI Deflation Trough -> Long SMH", "rule": "Long SMH 126d when PCU33443344 YoY turns positive after 6+ months negative", "tickers": ["SMH", "SPY"], "fred_series": ["PCU33443344"], "hold_days": 126, "entry_logic": "fred_yoy_turn_positive", "entry_params": {"series": "PCU33443344", "negative_months": 6}},
    "PL108": {"signal_id": "PL108_tech_inventory_destock_distributors", "name": "Computer Inventory Destock -> Long ARW+AVT", "rule": "Long ARW+AVT 126d when A33SNO YoY turns positive after 6+ months decline", "tickers": ["ARW", "AVT", "SPY"], "fred_series": ["A33SNO"], "hold_days": 126, "entry_logic": "fred_yoy_turn_positive", "entry_params": {"series": "A33SNO", "negative_months": 6}},
    "PL109": {"signal_id": "PL109_health_employment_hospitals", "name": "Health Employment Surge -> Long HCA+THC", "rule": "Long HCA+THC 126d when CES6562000101 YoY > +3% for 3 months", "tickers": ["HCA", "THC", "SPY"], "fred_series": ["CES6562000101"], "hold_days": 126, "entry_logic": "fred_yoy_above", "entry_params": {"series": "CES6562000101", "threshold": 3, "consecutive": 3}},
    "PL110": {"signal_id": "PL110_rx_ppi_inflation_pbm", "name": "Rx PPI Inflation -> Long CI+UNH", "rule": "Long CI+UNH 126d when PCU325411325411 YoY > +5% for 3 months", "tickers": ["CI", "UNH", "SPY"], "fred_series": ["PCU325411325411"], "hold_days": 126, "entry_logic": "fred_yoy_above", "entry_params": {"series": "PCU325411325411", "threshold": 5, "consecutive": 3}},
    "PL111": {"signal_id": "PL111_medtech_spending_recovery", "name": "Health Store Sales Recovery -> Long SYK+ISRG", "rule": "Long SYK+ISRG 126d when S4423SM YoY > +5% after 6+ months below +2%", "tickers": ["SYK", "ISRG", "SPY"], "fred_series": ["S4423SM"], "hold_days": 126, "entry_logic": "fred_yoy_reacceleration", "entry_params": {"series": "S4423SM", "high_thresh": 5, "low_thresh": 2, "low_months": 6}},
    "PL112": {"signal_id": "PL112_vehicle_age_auto_parts", "name": "Vehicle Sales Drought -> Long AZO+ORLY", "rule": "Long AZO+ORLY 252d when TOTALSA 12mo avg < 14M SAAR for 6 months", "tickers": ["AZO", "ORLY", "SPY"], "fred_series": ["TOTALSA"], "hold_days": 252, "entry_logic": "fred_level_below", "entry_params": {"series": "TOTALSA", "threshold": 14, "consecutive": 6, "rolling_avg": 12}},
    "PL113": {"signal_id": "PL113_motor_vehicle_ip_suppliers", "name": "Motor Vehicle IP Recovery -> Long BWA+LEA", "rule": "Long BWA+LEA 126d when IPG3361T3S YoY turns positive after 6+ months negative", "tickers": ["BWA", "LEA", "SPY"], "fred_series": ["IPG3361T3S"], "hold_days": 126, "entry_logic": "fred_yoy_turn_positive", "entry_params": {"series": "IPG3361T3S", "negative_months": 6}},
    "PL114": {"signal_id": "PL114_cre_delinquency_trough_reits", "name": "CRE Delinquency Trough -> Long BXP+SPG", "rule": "Long BXP+SPG 252d when DRCLACBS peaks then declines 2 quarters", "tickers": ["BXP", "SPG", "SPY"], "fred_series": ["DRCLACBS"], "hold_days": 252, "entry_logic": "fred_peak_then_decline", "entry_params": {"series": "DRCLACBS", "decline_periods": 2}},
    "PL115": {"signal_id": "PL115_cmbs_spread_regional_banks", "name": "HY Spread Compression -> Long KRE", "rule": "Long KRE 126d when BAMLHE0A0HYM2 drops below 400 after 3+ months above 500", "tickers": ["KRE", "SPY"], "fred_series": ["BAMLHE0A0HYM2"], "hold_days": 126, "entry_logic": "fred_level_cross_below", "entry_params": {"series": "BAMLHE0A0HYM2", "threshold": 400, "prior_above": 500, "prior_months": 3}},
    "PL116": {"signal_id": "PL116_nonres_construction_industrial_reits", "name": "Nonres Construction Inflection -> Long PLD", "rule": "Long PLD 252d when TLNRESCONS YoY turns positive after 6+ months negative", "tickers": ["PLD", "SPY"], "fred_series": ["TLNRESCONS"], "hold_days": 252, "entry_logic": "fred_yoy_turn_positive", "entry_params": {"series": "TLNRESCONS", "negative_months": 6}},
    "PL117": {"signal_id": "PL117_michigan_expectations_luxury", "name": "Michigan Expectations Trough -> Long RL+TPR", "rule": "Long RL+TPR 126d when MICH hits trough below 60 then bounces 3 months", "tickers": ["RL", "TPR", "SPY"], "fred_series": ["MICH"], "hold_days": 126, "entry_logic": "fred_trough_bounce", "entry_params": {"series": "MICH", "trough_level": 60, "bounce_months": 3}},
    "PL118": {"signal_id": "PL118_consumer_confidence_travel", "name": "CB Present Situation Surge -> Long MAR+EXPE", "rule": "Long MAR+EXPE 126d when CSCICP03USM665S YoY change > +10 for 3 months", "tickers": ["MAR", "EXPE", "SPY"], "fred_series": ["CSCICP03USM665S"], "hold_days": 126, "entry_logic": "fred_yoy_change_above", "entry_params": {"series": "CSCICP03USM665S", "threshold": 10, "consecutive": 3}},
    "PL119": {"signal_id": "PL119_freight_rate_collapse_retailers", "name": "Import Price Collapse -> Long COST+TGT", "rule": "Long COST+TGT 126d when IR (import prices) YoY < -5% for 3 months", "tickers": ["COST", "TGT", "SPY"], "fred_series": ["IR"], "hold_days": 126, "entry_logic": "fred_yoy_below", "entry_params": {"series": "IR", "threshold": -5, "consecutive": 3}},
    "PL120": {"signal_id": "PL120_import_price_recovery_traders", "name": "Import Price Recovery -> Long ADM+BG", "rule": "Long ADM+BG 126d when IR YoY turns positive after 6+ months negative", "tickers": ["ADM", "BG", "SPY"], "fred_series": ["IR"], "hold_days": 126, "entry_logic": "fred_yoy_turn_positive", "entry_params": {"series": "IR", "negative_months": 6}},
    "PL121": {"signal_id": "PL121_import_surge_warehouse_reits", "name": "Real Goods Import Surge -> Long PLD+STAG", "rule": "Long PLD+STAG 126d when IMPGS YoY > +8% for 3 months", "tickers": ["PLD", "STAG", "SPY"], "fred_series": ["IMPGS"], "hold_days": 126, "entry_logic": "fred_yoy_above", "entry_params": {"series": "IMPGS", "threshold": 8, "consecutive": 3}},
    "PL122": {"signal_id": "PL122_hdd_spike_utilities", "name": "Nat Gas Price Spike -> Long XLU", "rule": "Long XLU 126d when MHHNGSP crosses above $4 from below", "tickers": ["XLU", "SPY"], "fred_series": ["MHHNGSP"], "hold_days": 126, "entry_logic": "fred_level_above", "entry_params": {"series": "MHHNGSP", "threshold": 4}},
    "PL123": {"signal_id": "PL123_insurance_cpi_reinsurers", "name": "Insurance CPI Acceleration -> Long RNR+ACGL", "rule": "Long RNR+ACGL 252d when CUSR0000SEHC YoY > +8% for 6 months", "tickers": ["RNR", "ACGL", "SPY"], "fred_series": ["CUSR0000SEHC"], "hold_days": 252, "entry_logic": "fred_yoy_above", "entry_params": {"series": "CUSR0000SEHC", "threshold": 8, "consecutive": 6}},
    "PL124": {"signal_id": "PL124_defense_spending_primes", "name": "Defense Spending Acceleration -> Long LMT+RTX+NOC", "rule": "Long LMT+RTX+NOC 252d when FDEFX YoY > +5% for 2 quarters", "tickers": ["LMT", "RTX", "NOC", "SPY"], "fred_series": ["FDEFX"], "hold_days": 252, "entry_logic": "fred_yoy_above", "entry_params": {"series": "FDEFX", "threshold": 5, "consecutive": 2}},
    "PL125": {"signal_id": "PL125_state_local_construction_engineers", "name": "State/Local Construction Surge -> Long PWR+MTZ", "rule": "Long PWR+MTZ 252d when TLPBLCONS YoY > +10% for 3 months", "tickers": ["PWR", "MTZ", "SPY"], "fred_series": ["TLPBLCONS"], "hold_days": 252, "entry_logic": "fred_yoy_above", "entry_params": {"series": "TLPBLCONS", "threshold": 10, "consecutive": 3}},
    "PL126": {"signal_id": "PL126_fed_nondefense_invest_it", "name": "Fed Nondefense Investment -> Long ACN+LDOS", "rule": "Long ACN+LDOS 252d when A782RX1Q020SBEA YoY > +5% for 2 quarters", "tickers": ["ACN", "LDOS", "SPY"], "fred_series": ["A782RX1Q020SBEA"], "hold_days": 252, "entry_logic": "fred_yoy_above", "entry_params": {"series": "A782RX1Q020SBEA", "threshold": 5, "consecutive": 2}},
    "PL127": {"signal_id": "PL127_nfci_easing_small_cap_value", "name": "NFCI Easing From Tight -> Long IWN", "rule": "Long IWN 252d when NFCI drops below 0 after 3+ months above +0.5", "tickers": ["IWN", "SPY"], "fred_series": ["NFCI"], "hold_days": 252, "entry_logic": "fred_level_cross_below", "entry_params": {"series": "NFCI", "threshold": 0, "prior_above": 0.5, "prior_months": 3}},
    "PL128": {"signal_id": "PL128_m2_velocity_commodities", "name": "M2 Velocity Inflection -> Long DJP", "rule": "Long DJP 252d when M2V YoY turns positive after 4+ quarters decline", "tickers": ["DJP", "SPY"], "fred_series": ["M2V"], "hold_days": 252, "entry_logic": "fred_yoy_turn_positive", "entry_params": {"series": "M2V", "negative_months": 4}},
    "PL129": {"signal_id": "PL129_term_premium_positive_xlf", "name": "Term Premium Positive -> Long XLF", "rule": "Long XLF 252d when THREEFYTP10 crosses above 0 after 6+ months below", "tickers": ["XLF", "SPY"], "fred_series": ["THREEFYTP10"], "hold_days": 252, "entry_logic": "fred_level_cross_above", "entry_params": {"series": "THREEFYTP10", "threshold": 0, "prior_below_months": 6}},
    "PL130": {"signal_id": "PL130_wilshire_gdp_mean_reversion", "name": "Wilshire/GDP Low -> Long SPY 12mo", "rule": "Long SPY 252d when WILL5000INDFC/GDP ratio drops below 1.2", "tickers": ["SPY"], "fred_series": ["WILL5000INDFC", "GDP"], "hold_days": 252, "entry_logic": "ratio_below", "entry_params": {"num": "WILL5000INDFC", "den": "GDP", "threshold": 1.2}},
    "PL131": {"signal_id": "PL131_oer_deceleration_tlt", "name": "OER Deceleration -> Long TLT", "rule": "Long TLT 126d when CUSR0000SEHC YoY decelerates >2pp from 6mo peak", "tickers": ["TLT", "SPY"], "fred_series": ["CUSR0000SEHC"], "hold_days": 126, "entry_logic": "fred_yoy_deceleration", "entry_params": {"series": "CUSR0000SEHC", "decel_pp": 2}},
}

generated = 0
for iid, sdef in STRATEGY_DEFS.items():
    sid = sdef['signal_id']
    fpath = ROOT / "backtests" / f"{sid}.py"
    
    logic_fn = LOGIC_MAP.get(sdef['entry_logic'])
    if not logic_fn:
        print(f"ERROR: Unknown logic type {sdef['entry_logic']} for {sid}")
        continue
    
    signal_logic = logic_fn(sdef['entry_params'])
    
    trade_tickers = [t for t in sdef['tickers'] if t != 'SPY']
    
    code = TEMPLATE.format(
        signal_id=sid,
        name=sdef['name'],
        rule=sdef['rule'],
        rule_escaped=sdef['rule'].replace('"', '\\"'),
        fred_series_list=repr(sdef['fred_series']),
        fred_start="1990-01-01",
        signal_logic=signal_logic,
        ticker_list=repr(sdef['tickers']),
        px_start="1995-01-01",
        trade_tickers=repr(trade_tickers),
        hold_days=sdef['hold_days'],
        fred_source=", ".join(sdef['fred_series']),
    )
    
    with open(fpath, "w") as f:
        f.write(code)
    
    generated += 1
    print(f"Generated: {fpath.name}")

print(f"\nGenerated {generated} backtest files")
