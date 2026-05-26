"""
Batch develop 28 new ideas into strategies, create backtest files, run them.
"""
import json, os, sys, datetime as dt
from pathlib import Path

ROOT = Path("/Users/benson/Projects/ekans")

# ---- Step 1: Load ideas, develop new ones ----
with open(ROOT / "pipeline/ideas_queue.json") as f:
    ideas = json.load(f)

with open(ROOT / "pipeline/strategies_queue.json") as f:
    strategies = json.load(f)

new_ideas = [i for i in ideas if i['status'] == 'new']
print(f"Found {len(new_ideas)} new ideas to develop")

# Strategy definitions for each new idea
STRATEGY_DEFS = {
    "PL102": {
        "signal_id": "PL102_soybean_oil_ppi_food",
        "name": "Soybean Oil PPI Spike -> Long GIS+CPB (Hedged Food)",
        "category": "C",
        "rule": "Use FRED WPU02220301 (soybean oil PPI). When YoY > +20% for 3+ months, input costs surge for packaged food. Paradoxically, hedged players raise prices with lag. Long GIS+CPB equal-weight 126 trading days. They pass through costs to consumers.",
        "tickers": ["GIS", "CPB", "SPY"],
        "fred_series": ["WPU02220301"],
        "hold_days": 126,
        "entry_logic": "fred_yoy_above",
        "entry_params": {"series": "WPU02220301", "threshold": 20, "consecutive": 3},
    },
    "PL103": {
        "signal_id": "PL103_corn_crush_margin_ethanol",
        "name": "Corn Crush Margin Expansion -> Long ADM+GPRE",
        "category": "C",
        "rule": "Compute corn crush proxy: FRED GASREGW (gasoline) / FRED PCORNUS (corn). When ratio rises >20% from 6mo low: long ADM+GPRE 126d. Ethanol margins expand when gasoline rises relative to corn input.",
        "tickers": ["ADM", "GPRE", "SPY"],
        "fred_series": ["GASREGW", "PCORNUS"],
        "hold_days": 126,
        "entry_logic": "ratio_rise",
        "entry_params": {"num": "GASREGW", "den": "PCORNUS", "rise_pct": 20},
    },
    "PL104": {
        "signal_id": "PL104_wheat_ppi_collapse_restaurants",
        "name": "Wheat PPI Collapse -> Long DRI+MCD",
        "category": "C",
        "rule": "Use FRED WPU02110301 (wheat flour PPI). When YoY < -15% for 2+ months: long DRI+MCD 126 days. Restaurants benefit from lower flour costs with margin expansion lag.",
        "tickers": ["DRI", "MCD", "SPY"],
        "fred_series": ["WPU02110301"],
        "hold_days": 126,
        "entry_logic": "fred_yoy_below",
        "entry_params": {"series": "WPU02110301", "threshold": -15, "consecutive": 2},
    },
    "PL107": {
        "signal_id": "PL107_semi_ppi_deflation_trough",
        "name": "Semiconductor PPI Deflation Trough -> Long SMH",
        "category": "C",
        "rule": "Use FRED PCU33443344 (semiconductor PPI). When YoY turns positive after 6+ months negative: long SMH 126 days. Price recovery signals demand inflection.",
        "tickers": ["SMH", "SPY"],
        "fred_series": ["PCU33443344"],
        "hold_days": 126,
        "entry_logic": "fred_yoy_turn_positive",
        "entry_params": {"series": "PCU33443344", "negative_months": 6},
    },
    "PL108": {
        "signal_id": "PL108_tech_inventory_destock_distributors",
        "name": "Computer Inventory Destock -> Long ARW+AVT",
        "category": "C",
        "rule": "Use FRED A33SNO (computer/electronic inventories). When YoY turns positive after 6+ months of decline (destock complete): long ARW+AVT 126 days. Restocking cycle benefits distributors.",
        "tickers": ["ARW", "AVT", "SPY"],
        "fred_series": ["A33SNO"],
        "hold_days": 126,
        "entry_logic": "fred_yoy_turn_positive",
        "entry_params": {"series": "A33SNO", "negative_months": 6},
    },
    "PL109": {
        "signal_id": "PL109_health_employment_hospitals",
        "name": "Health Employment Surge -> Long HCA+THC",
        "category": "C",
        "rule": "Use FRED CES6562000101 (healthcare employment). When YoY > +3% for 3 months: long HCA+THC 126 days. Employment surge = volume growth for hospitals.",
        "tickers": ["HCA", "THC", "SPY"],
        "fred_series": ["CES6562000101"],
        "hold_days": 126,
        "entry_logic": "fred_yoy_above",
        "entry_params": {"series": "CES6562000101", "threshold": 3, "consecutive": 3},
    },
    "PL110": {
        "signal_id": "PL110_rx_ppi_inflation_pbm",
        "name": "Rx Drug PPI Inflation -> Long CI+UNH",
        "category": "C",
        "rule": "Use FRED PCU325411325411 (pharma PPI). When YoY > +5% for 3 months: long CI+UNH 126 days. Drug price inflation = higher PBM spread income.",
        "tickers": ["CI", "UNH", "SPY"],
        "fred_series": ["PCU325411325411"],
        "hold_days": 126,
        "entry_logic": "fred_yoy_above",
        "entry_params": {"series": "PCU325411325411", "threshold": 5, "consecutive": 3},
    },
    "PL111": {
        "signal_id": "PL111_medtech_spending_recovery",
        "name": "Health Store Sales Recovery -> Long SYK+ISRG",
        "category": "C",
        "rule": "Use FRED S4423SM (health care store sales). When YoY > +5% after 6+ months below +2%: long SYK+ISRG 126 days.",
        "tickers": ["SYK", "ISRG", "SPY"],
        "fred_series": ["S4423SM"],
        "hold_days": 126,
        "entry_logic": "fred_yoy_reacceleration",
        "entry_params": {"series": "S4423SM", "high_thresh": 5, "low_thresh": 2, "low_months": 6},
    },
    "PL112": {
        "signal_id": "PL112_vehicle_age_auto_parts",
        "name": "Vehicle Sales Drought -> Long AZO+ORLY",
        "category": "C",
        "rule": "Use FRED TOTALSA (vehicle sales). When trailing 12mo avg drops below 14M SAAR for 6+ months (aging fleet): long AZO+ORLY 252 days. Fewer new cars = more repairs on old fleet.",
        "tickers": ["AZO", "ORLY", "SPY"],
        "fred_series": ["TOTALSA"],
        "hold_days": 252,
        "entry_logic": "fred_level_below",
        "entry_params": {"series": "TOTALSA", "threshold": 14, "consecutive": 6, "rolling_avg": 12},
    },
    "PL113": {
        "signal_id": "PL113_motor_vehicle_ip_suppliers",
        "name": "Motor Vehicle IP Recovery -> Long BWA+LEA",
        "category": "C",
        "rule": "Use FRED IPG3361T3S (motor vehicle IP). When YoY turns positive after 6+ months negative: long BWA+LEA 126 days. Production recovery drives OEM supplier revenue.",
        "tickers": ["BWA", "LEA", "SPY"],
        "fred_series": ["IPG3361T3S"],
        "hold_days": 126,
        "entry_logic": "fred_yoy_turn_positive",
        "entry_params": {"series": "IPG3361T3S", "negative_months": 6},
    },
    "PL114": {
        "signal_id": "PL114_cre_delinquency_trough_reits",
        "name": "CRE Delinquency Trough -> Long BXP+SPG",
        "category": "C",
        "rule": "Use FRED DRCLACBS (CRE delinquency, quarterly). When rate peaks and drops for 2 consecutive quarters: long BXP+SPG 252 days. Credit improvement = REIT recovery.",
        "tickers": ["BXP", "SPG", "SPY"],
        "fred_series": ["DRCLACBS"],
        "hold_days": 252,
        "entry_logic": "fred_peak_then_decline",
        "entry_params": {"series": "DRCLACBS", "decline_periods": 2},
    },
    "PL115": {
        "signal_id": "PL115_cmbs_spread_regional_banks",
        "name": "HY Spread Compression -> Long KRE",
        "category": "C",
        "rule": "Use FRED BAMLHE0A0HYM2 (HY OAS). When OAS drops below 400bps after being above 500 for 3+ months: long KRE 126 days. Credit easing benefits CRE-exposed regionals.",
        "tickers": ["KRE", "SPY"],
        "fred_series": ["BAMLHE0A0HYM2"],
        "hold_days": 126,
        "entry_logic": "fred_level_cross_below",
        "entry_params": {"series": "BAMLHE0A0HYM2", "threshold": 400, "prior_above": 500, "prior_months": 3},
    },
    "PL116": {
        "signal_id": "PL116_nonres_construction_industrial_reits",
        "name": "Nonres Construction Inflection -> Long PLD",
        "category": "C",
        "rule": "Use FRED TLNRESCONS (nonresidential construction). When YoY turns positive after 6+ months negative: long PLD 252 days. Construction recovery = warehouse/industrial demand.",
        "tickers": ["PLD", "SPY"],
        "fred_series": ["TLNRESCONS"],
        "hold_days": 252,
        "entry_logic": "fred_yoy_turn_positive",
        "entry_params": {"series": "TLNRESCONS", "negative_months": 6},
    },
    "PL117": {
        "signal_id": "PL117_michigan_expectations_luxury",
        "name": "Michigan Expectations Trough -> Long RL+TPR",
        "category": "C",
        "rule": "Use FRED MICH (Michigan expectations). When index hits local min and < 60, then bounces for 3 months: long RL+TPR 126 days. Consumer confidence recovery = luxury spending recovery.",
        "tickers": ["RL", "TPR", "SPY"],
        "fred_series": ["MICH"],
        "hold_days": 126,
        "entry_logic": "fred_trough_bounce",
        "entry_params": {"series": "MICH", "trough_level": 60, "bounce_months": 3},
    },
    "PL118": {
        "signal_id": "PL118_consumer_confidence_travel",
        "name": "CB Present Situation Surge -> Long MAR+EXPE",
        "category": "C",
        "rule": "Use FRED CSCICP03USM665S (CB Present Situation). When YoY > +10 points for 3 months: long MAR+EXPE 126 days. Strong present-situation reads = travel/leisure spending boom.",
        "tickers": ["MAR", "EXPE", "SPY"],
        "fred_series": ["CSCICP03USM665S"],
        "hold_days": 126,
        "entry_logic": "fred_yoy_change_above",
        "entry_params": {"series": "CSCICP03USM665S", "threshold": 10, "consecutive": 3},
    },
    "PL119": {
        "signal_id": "PL119_freight_rate_collapse_retailers",
        "name": "Freight Rate Collapse -> Long COST+TGT",
        "category": "C",
        "rule": "Use FRED DHLSFRTI (freight friction index, if available) or proxy via import prices IR. When YoY < -20%: long COST+TGT 126 days. Lower shipping costs = retail margin expansion.",
        "tickers": ["COST", "TGT", "SPY"],
        "fred_series": ["IR"],
        "hold_days": 126,
        "entry_logic": "fred_yoy_below",
        "entry_params": {"series": "IR", "threshold": -5, "consecutive": 3},
    },
    "PL120": {
        "signal_id": "PL120_import_price_recovery_traders",
        "name": "Import Price Recovery -> Long ADM+BG",
        "category": "C",
        "rule": "Use FRED IR (import price index). When YoY turns positive after 6+ months negative: long ADM+BG 126 days. Rising import prices = higher trading margins for commodity houses.",
        "tickers": ["ADM", "BG", "SPY"],
        "fred_series": ["IR"],
        "hold_days": 126,
        "entry_logic": "fred_yoy_turn_positive",
        "entry_params": {"series": "IR", "negative_months": 6},
    },
    "PL121": {
        "signal_id": "PL121_import_surge_warehouse_reits",
        "name": "Real Goods Import Surge -> Long PLD+STAG",
        "category": "C",
        "rule": "Use FRED IMPGS (real goods imports). When YoY > +8% for 3 months: long PLD+STAG 126 days. Import surge = warehouse demand surge.",
        "tickers": ["PLD", "STAG", "SPY"],
        "fred_series": ["IMPGS"],
        "hold_days": 126,
        "entry_logic": "fred_yoy_above",
        "entry_params": {"series": "IMPGS", "threshold": 8, "consecutive": 3},
    },
    "PL122": {
        "signal_id": "PL122_hdd_spike_utilities",
        "name": "Gas-Weighted HDD Spike -> Long XLU",
        "category": "C",
        "rule": "Use FRED GASPRICE or proxy via natural gas price MHHNGSP. When winter (Nov-Feb) natural gas price > $4: long XLU for 126 days. Cold winters boost regulated utility revenues.",
        "tickers": ["XLU", "SPY"],
        "fred_series": ["MHHNGSP"],
        "hold_days": 126,
        "entry_logic": "fred_level_above",
        "entry_params": {"series": "MHHNGSP", "threshold": 4},
    },
    "PL123": {
        "signal_id": "PL123_insurance_cpi_reinsurers",
        "name": "Property Insurance CPI Acceleration -> Long RNR+ACGL",
        "category": "C",
        "rule": "Use FRED CUSR0000SEHC (CPI household insurance). When YoY > +8% for 6 months: long RNR+ACGL 252 days. Premium inflation = underwriting profit expansion for reinsurers.",
        "tickers": ["RNR", "ACGL", "SPY"],
        "fred_series": ["CUSR0000SEHC"],
        "hold_days": 252,
        "entry_logic": "fred_yoy_above",
        "entry_params": {"series": "CUSR0000SEHC", "threshold": 8, "consecutive": 6},
    },
    "PL124": {
        "signal_id": "PL124_defense_spending_primes",
        "name": "Federal Defense Spending Acceleration -> Long LMT+RTX+NOC",
        "category": "C",
        "rule": "Use FRED FDEFX (federal defense spending, quarterly). When YoY > +5% for 2 consecutive quarters: long LMT+RTX+NOC 252 days.",
        "tickers": ["LMT", "RTX", "NOC", "SPY"],
        "fred_series": ["FDEFX"],
        "hold_days": 252,
        "entry_logic": "fred_yoy_above",
        "entry_params": {"series": "FDEFX", "threshold": 5, "consecutive": 2},
    },
    "PL125": {
        "signal_id": "PL125_state_local_construction_engineers",
        "name": "State/Local Construction Surge -> Long PWR+MTZ",
        "category": "C",
        "rule": "Use FRED TLPBLCONS (state/local public construction). When YoY > +10% for 3 months: long PWR+MTZ 252 days. Public works spending = engineering/construction backlog.",
        "tickers": ["PWR", "MTZ", "SPY"],
        "fred_series": ["TLPBLCONS"],
        "hold_days": 252,
        "entry_logic": "fred_yoy_above",
        "entry_params": {"series": "TLPBLCONS", "threshold": 10, "consecutive": 3},
    },
    "PL126": {
        "signal_id": "PL126_fed_nondefense_invest_it",
        "name": "Federal Nondefense Investment -> Long ACN+LDOS",
        "category": "C",
        "rule": "Use FRED A782RX1Q020SBEA (federal nondefense investment, quarterly). When YoY > +5% for 2 quarters: long ACN+LDOS 252 days.",
        "tickers": ["ACN", "LDOS", "SPY"],
        "fred_series": ["A782RX1Q020SBEA"],
        "hold_days": 252,
        "entry_logic": "fred_yoy_above",
        "entry_params": {"series": "A782RX1Q020SBEA", "threshold": 5, "consecutive": 2},
    },
    "PL127": {
        "signal_id": "PL127_nfci_easing_small_cap_value",
        "name": "NFCI Easing From Tight -> Long IWN",
        "category": "C",
        "rule": "Use FRED NFCI (financial conditions). When NFCI drops below 0 after being above +0.5 for 3+ months: long IWN 252 days. Financial conditions easing = small-cap value recovery.",
        "tickers": ["IWN", "SPY"],
        "fred_series": ["NFCI"],
        "hold_days": 252,
        "entry_logic": "fred_level_cross_below",
        "entry_params": {"series": "NFCI", "threshold": 0, "prior_above": 0.5, "prior_months": 3},
    },
    "PL128": {
        "signal_id": "PL128_m2_velocity_commodities",
        "name": "M2 Velocity Inflection -> Long DJP",
        "category": "C",
        "rule": "Use FRED M2V (M2 velocity, quarterly). When M2V turns positive YoY after 4+ quarters of decline: long DJP 252 days. Velocity inflection = money flowing into real economy = commodity demand.",
        "tickers": ["DJP", "SPY"],
        "fred_series": ["M2V"],
        "hold_days": 252,
        "entry_logic": "fred_yoy_turn_positive",
        "entry_params": {"series": "M2V", "negative_months": 4},
    },
    "PL129": {
        "signal_id": "PL129_term_premium_positive_xlf",
        "name": "Term Premium Positive -> Long XLF",
        "category": "C",
        "rule": "Use FRED THREEFYTP10 (10Y term premium). When term premium crosses above 0 after 6+ months below 0: long XLF 252 days. Positive term premium = steeper curve = bank NIM expansion.",
        "tickers": ["XLF", "SPY"],
        "fred_series": ["THREEFYTP10"],
        "hold_days": 252,
        "entry_logic": "fred_level_cross_above",
        "entry_params": {"series": "THREEFYTP10", "threshold": 0, "prior_below_months": 6},
    },
    "PL130": {
        "signal_id": "PL130_wilshire_gdp_mean_reversion",
        "name": "Wilshire/GDP Low -> Long SPY 12mo",
        "category": "C",
        "rule": "Use FRED WILL5000INDFC and GDP. Compute ratio. When ratio drops below 1.2 (undervaluation): long SPY 252 days. Buffett indicator mean-reversion from depressed levels.",
        "tickers": ["SPY"],
        "fred_series": ["WILL5000INDFC", "GDP"],
        "hold_days": 252,
        "entry_logic": "ratio_below",
        "entry_params": {"num": "WILL5000INDFC", "den": "GDP", "threshold": 1.2},
    },
    "PL131": {
        "signal_id": "PL131_oer_deceleration_tlt",
        "name": "OER Deceleration -> Long TLT",
        "category": "C",
        "rule": "Use FRED CUSR0000SEHC (CPI shelter/OER proxy). When YoY decelerates by >2pp from 6mo peak: long TLT 126 days. Shelter disinflation = lower CPI = bond rally.",
        "tickers": ["TLT", "SPY"],
        "fred_series": ["CUSR0000SEHC"],
        "hold_days": 126,
        "entry_logic": "fred_yoy_deceleration",
        "entry_params": {"series": "CUSR0000SEHC", "decel_pp": 2},
    },
}

# ---- Develop ideas and create strategies ----
now_str = dt.datetime.utcnow().isoformat() + "+00:00"
existing_signal_ids = {s.get('signal_id') for s in strategies}

new_strategies = []
for idea in new_ideas:
    iid = idea['idea_id']
    if iid not in STRATEGY_DEFS:
        print(f"WARNING: No strategy def for {iid}, skipping")
        continue
    
    sdef = STRATEGY_DEFS[iid]
    
    # Check reject conditions
    if idea.get('bt_feasibility', 0) < 4:
        idea['status'] = 'rejected'
        idea['reject_reason'] = f"bt_feasibility {idea.get('bt_feasibility')} < 4"
        continue
    
    # Mark as developed
    idea['status'] = 'developed'
    
    # Create strategy if not already exists
    if sdef['signal_id'] in existing_signal_ids:
        print(f"Strategy {sdef['signal_id']} already exists, skipping")
        continue
    
    strat = {
        "strategy_id": iid,
        "idea_id": iid,
        "created_at": now_str,
        "status": "ready",
        "signal_id": sdef['signal_id'],
        "name": sdef['name'],
        "category": sdef.get('category', 'C'),
        "rule": sdef['rule'],
        "tickers": sdef['tickers'],
        "data_sources_concrete": {
            "prices": f"yfinance {', '.join(sdef['tickers'])}",
            "fundamental": f"FRED {', '.join(sdef['fred_series'])}"
        },
        "originality": idea.get('originality', 7),
        "bt_feasibility": idea.get('bt_feasibility', 5),
    }
    strategies.append(strat)
    new_strategies.append(strat)
    print(f"Developed: {iid} -> {sdef['signal_id']}")

# Save updated ideas and strategies
with open(ROOT / "pipeline/ideas_queue.json", "w") as f:
    json.dump(ideas, f, indent=2)

with open(ROOT / "pipeline/strategies_queue.json", "w") as f:
    json.dump(strategies, f, indent=2)

print(f"\nDeveloped {len(new_strategies)} new strategies, total strategies now: {len(strategies)}")
print(f"Ready strategies: {[s['signal_id'] for s in strategies if s['status'] == 'ready']}")
