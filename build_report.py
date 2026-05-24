"""
Builds a single self-contained index.html from results/*.json + research/*.md.

Output: /Users/benson/Projects/ekans/index.html ready to commit + serve on GitHub Pages.
"""

import json
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESEARCH = ROOT / "research"
OUT = ROOT / "full_catalog.html"


# ---------- catalog metadata ----------
# Map signal IDs (the JSON `signal_id` field) → category, full name, asset, horizon,
# originality, backtest-feasibility, short description, and source citation.
# Pulled from the research/01*.md files; trimmed for the table.

CATALOG = {
    # A — calendar / seasonal
    "A01_halloween": ("A","Halloween / Sell-in-May","equities","6mo",3,5,"Long SPY Nov-Apr; cash May-Oct.","Bouman-Jacobsen AER 2002; Zhang-Jacobsen 2021"),
    "A02_turn_of_month": ("A","Turn-of-Month","equities","~4d/mo",5,5,"Long SPY last day of month + first 3.","Ariel JFE 1987; McConnell-Xu 2008"),
    "A03_santa_claus": ("A","Santa Claus Rally","equities","7d",2,5,"Long SPY last 5 Dec + first 2 Jan trading days.","Hirsch 1972"),
    "A04_first_five_days": ("A","First Five Days of January","equities","12mo",3,5,"If SPY Jan day 1-5 > 0, long SPY rest of year.","Hirsch"),
    "A05_january_small_caps": ("A","January Small-Cap Effect","equities","1mo",3,5,"Long IWM during January only.","Classic tax-loss harvesting"),
    "A06_fomc_even_week": ("A","FOMC Even-Week Effect","equities","biweekly",7,5,"Long SPY only during even weeks of FOMC cycle.","Cieslak-Morse-Vissing-Jorgensen JF 2019"),
    "A07_pre_fomc_drift": ("A","Pre-FOMC Drift","equities","2d",6,5,"Long SPY T-1 close to T close on FOMC days.","Lucca-Moench JF 2015"),
    "A08_opex_week": ("A","Monthly OPEX Week","equities","1wk/mo",6,5,"Long SPY Fri-before-OPEX through OPEX Fri.","Quantpedia; delta-hedge unwind"),
    "A09_sunshine": ("A","Sunshine Effect (NYC weather)","equities","daily",7,4,"Index returns higher on sunny days.","Hirshleifer-Shumway JF 2003"),
    "A10_lunar_phase": ("A","Lunar Phase","equities","15d",9,5,"Long during new-moon ±7d; short during full-moon ±7d.","Yuan-Zheng-Zhu JEF 2006"),
    "A11_dst_anomaly": ("A","DST Anomaly (Monday after)","equities","1d",8,5,"Short SPY Monday after DST change.","Kamstra-Kramer-Levi AER 2000"),
    "A12_treasury_auction": ("A","Treasury Auction Cycle","bonds","5d",8,4,"Auction announcement → yield drift.","Lou-Yan-Zhang RFS 2013"),
    "A13_presidential_year3": ("A","Presidential Cycle Year 3","equities","1yr",4,5,"Long SPY only year 3 of every presidential term.","Hirsch; Beyer JPM"),
    "A14_btc_halving": ("A","BTC Halving Cycle","crypto","multi-year",2,5,"Long BTC 6mo before halving through 18mo after.","Halving narrative"),
    "A15_halloween_tom_combo": ("A","Halloween × Turn-of-Month combo","equities","various",5,5,"Long SPY only when both A01 and A02 active.","Stack of A01+A02"),
    "A16_seasonality_heston_sadka": ("A","Heston-Sadka Same-Month Seasonality","equities","monthly",7,5,"Hold SPY in months whose 10y trailing avg return > 0.","Heston-Sadka JFE 2008; Keloharju et al 2016"),
    # B — vol / options
    "B01_vix_term_structure": ("B","VIX Term Structure (VIX/VIX3M)","equities","days-wks",4,5,"Long SPY when ratio < 0.92; flat above 1.0.","VIX and More"),
    "B02_vix_30_contrarian": ("B","VIX > 30 Contrarian Buy","equities","1-3mo",3,5,"Buy SPY when VIX > 30 having been < 20 within 60d.","Whaley"),
    "B03_vvix_vix_ratio": ("B","VVIX/VIX Ratio","equities","1-4wk",5,5,"Short SPY 20d when VVIX/VIX > 6.5 and VIX < 18.","CBOE VVIX"),
    "B05_putcall_ratio": ("B","Put/Call Ratio Extremes","equities","1-4wk",2,5,"10d MA P/C ratio > 0.75 → long; < 0.50 → flat.","Sentimentrader"),
    # C — cross-asset / macro
    "C01_lumber_gold": ("C","Lumber/Gold (Gayed)","equities/bonds","13wk",4,4,"Long SPY when lumber > gold 13wk; else TLT.","Gayed 2015 SSRN"),
    "C02_utilities_spy": ("C","Utilities/SPY Ratio (Gayed)","equities","1mo",3,5,"Cash 4 weeks when XLU outpaces SPY.","Gayed-Atilgan 2014"),
    "C03_copper_gold": ("C","Copper/Gold → TLT","bonds","1-6mo",5,5,"Short TLT when copper/gold rises while 10Y flat.","Hua-Wang JIFMIM 2023"),
    "C04_gold_silver": ("C","Gold/Silver Ratio Mean-Reversion","commodities","6-24mo",4,5,"Long silver/short gold when ratio > 90; reverse < 50.","Long-run mean ~65"),
    "C05_faber_gtaa": ("C","Faber GTAA 10-mo MA","multi-asset","monthly",2,5,"Hold each of 5 assets only above 10-mo SMA.","Faber 2007 SSRN"),
    "C06_dual_momentum": ("C","Dual Momentum (Antonacci GEM)","equities/bonds","monthly",3,5,"Pick higher 12m return: SPY vs ACWX; else AGG.","Antonacci 2013"),
    "C07_accel_dual_momentum": ("C","Accelerating Dual Momentum","equities/bonds","monthly",5,5,"Avg of 1/3/6m returns picks SPY/SCZ/TLT.","EngineeredPortfolio 2018"),
    "C08_yield_curve_10y2y": ("C","Yield Curve Inversion (10Y-2Y)","equities/bonds","yrs",3,5,"Reduce equity 12mo after T10Y2Y crosses below 0.","Estrella-Hardouvelis 1991"),
    "C09_yield_curve_10y3m": ("C","Yield Curve Inversion (10Y-3M)","equities/bonds","yrs",4,5,"Same with T10Y3M (Estrella-Mishkin probit).","Estrella-Mishkin 1998"),
    "C10_hy_oas_regime": ("C","HY OAS Credit Spread Regime","equities/credit","1-12mo",3,5,"De-risk when HY OAS rises 100bp from 6m low + > 500bp.","Verdad Capital research"),
    "C11_move_vix_ratio": ("C","MOVE/VIX Ratio","cross-asset","2-8wk",6,4,"Cash when MOVE/VIX > 6 sustained 5+d.","FinTwit (Ksidial); CFA Inst 2025"),
    "C12_acm_term_premium": ("C","ACM 10Y Term Premium","bonds/equities","yrs",7,5,"Short TLT when ACM term premium turns positive.","NY Fed Adrian-Crump-Moench"),
    "C13_tips_breakeven": ("C","TIPS Breakeven → Commodities","commodities","1-3mo",5,5,"Long DBC when 10y breakeven rises > 60d MA + 20bp.","FRED T10YIE"),
    "C14_tsmom": ("C","Time-Series Momentum (TSMOM)","multi-asset","1-12mo",3,4,"12m positive → long that asset across SPY/TLT/GLD/DBC/EFA/EEM.","Moskowitz-Ooi-Pedersen JFE 2012"),
    "C15_trend_following": ("C","Multi-horizon Trend (1+3+12m)","multi-asset","1-12mo",3,4,"Blend of 1/3/12m return signs across basket.","Hurst-Ooi-Pedersen JPM 2017"),
    "C16_currency_carry": ("C","G4 Currency Carry","FX","months",3,4,"Long high-rate / short low-rate FX (4 currencies).","Lustig-Verdelhan AER 2007"),
    "C17_commodity_backwardation": ("C","Commodity Backwardation/Roll Yield","commodities","monthly",4,3,"Long backwardated / short contangoed commodities.","Erb-Harvey FAJ 2006"),
    "C18_big_mac_ppp": ("C","Big Mac PPP FX Mean-Reversion","FX","2-5yr",3,5,"Long undervalued / short overvalued PPP currencies.","Cumby NBER 1996"),
    # D — equity factor
    "D01_value_hml": ("D","Value (HML)","equities","mo-yrs",1,5,"Long high B/M; short low B/M (Ken French daily).","Fama-French 1993"),
    "D02_momentum_umd": ("D","Cross-Sectional Momentum (UMD)","equities","1-12mo",1,5,"Long past 12m winners; short losers (Ken French).","Jegadeesh-Titman JF 1993"),
    "D03_52w_high_momentum": ("D","52-Week-High Momentum","equities","mo",5,5,"Long stocks near 52w high; short far below (75-name basket).","George-Hwang JF 2004"),
    "D04_industry_momentum": ("D","Industry Momentum","equities","1-12mo",4,5,"Long top 5 / short bottom 5 industries by 6m return.","Moskowitz-Grinblatt JF 1999"),
    "D05_max_lottery": ("D","MAX / Lottery Effect","equities","1mo",5,5,"Long low-MAX(5) / short high-MAX(5) prior month (basket).","Bali-Cakici-Whitelaw JFE 2011"),
    "D06_idio_vol": ("D","Idiosyncratic Vol Puzzle","equities","1mo",4,4,"Long low-idio-vol / short high-idio-vol (basket).","Ang et al JF 2006"),
    "D09_bab": ("D","Betting Against Beta (BAB)","equities","mo",2,4,"Long low-beta / short high-beta (basket).","Frazzini-Pedersen JFE 2014"),
    "D11_short_interest": ("D","Short Interest","equities","mo",4,3,"Top short-interest decile shorts (FINRA bulk).","Boehmer-Jones-Zhang JF 2008"),
    "D14_net_issuance": ("D","Net Stock Issuance","equities","1-3yr",4,4,"Long repurchasers / short issuers (basket, since 2016).","Pontiff-Woodgate JF 2008"),
    "D17_coskewness": ("D","Crash Risk / Coskewness","equities","mo",7,4,"Long high-(neg)coskewness / short low.","Harvey-Siddique JF 2000"),
    # E — event-driven
    "E01_form4_cluster": ("E","Form 4 Cluster Insider Buying","equities","6-12mo",3,5,"≥3 insider buys ≥$25k within 10d → long 252d.","Lakonishok-Lee RFS 2001; Cohen-Malloy-Pomorski JF 2012"),
    "E05_friday_8k_drift": ("E","Friday-After-Close 8-K Drift","equities","1-2mo",6,5,"Short stocks whose 8-Ks file after 16:00 ET Fri.","DellaVigna-Pollet JF 2009"),
    "E06_item_402_restate": ("E","Item 4.02 Non-Reliance Restatements","equities","6-12mo",4,5,"Short on 8-K Item 4.02 (restatement) filing.","Hennes-Leone-Miller TAR 2008"),
    "E07_activist_13d": ("E","Schedule 13D Activist Drift","equities","3-18mo",3,5,"Long target after activist 13D (Elliott/Starboard/etc).","Brav et al JF 2008"),
    "E09_ipo_lockup": ("E","IPO Lockup Expiry Short","equities","5d",3,4,"Short recent IPOs T-5 to T+5 around 180-day lockup.","Field-Hanka JF 2001"),
    "E10_spinoff_drift": ("E","Spin-Off Drift (Greenblatt)","equities","12-36mo",4,4,"Buy SpinCo at first regular-way trade; hold 24mo.","Cusatis-Miles-Woolridge JFE 1993"),
    "E11_pelosi_mirror": ("E","Pelosi Disclosed-Trade Mirror","equities","3-12mo",2,4,"Mirror Pelosi PTRs within 45-day disclosure window.","Unusual Whales; Ziobrowski 2004"),
    "E13_sp500_inclusion": ("E","S&P 500 Inclusion Drift","equities","days-wks",5,4,"Buy adds on announcement; T0..T+10 CAR.","Shleifer JF 1986; Greenwood-Sammon 2024 NBER"),
    "NEW01_uspto_trademark_ls": ("E","USPTO Trademark Filings L/S","equities","12mo",7,5,"Long top tercile / short bottom tercile by TM filings.","Mgmt Science via UCLA Anderson Review"),
    "NEW02_director_interlock": ("E","Director Network Interlock Trades","equities","3-12mo",8,4,"Long when interlocked director buys other-co stock.","Director Network JBF 2020"),
    # F — sentiment / breadth / TA classics
    "F01_aaii_bull_bear": ("F","AAII Bull-Bear Extremes","equities","4-12wk",2,5,"Long after bull-bear < -20 for 2 wks; flat if > +30.","AAII survey"),
    "F02_naaim_exposure": ("F","NAAIM Exposure Extremes","equities","4-12wk",4,5,"Long after 4wk MA NAAIM < 30; flat > 90.","NAAIM"),
    "F03_margin_debt_yoy": ("F","Margin Debt YoY Change","equities","3-12mo",5,5,"Reduce equity when FINRA margin debt YoY < -20%.","Jesse Felder; FINRA monthly"),
    "F06_td_sequential": ("F","DeMark TD Sequential","equities","days-wks",5,5,"9-bar setup completion → fade direction.","Tom DeMark"),
    "F08_rsi2_connors": ("F","RSI(2) Mean-Reversion (Connors)","equities","days",5,5,"Buy SPY when RSI(2) < 5 & close > 200d SMA; exit > 5d SMA.","Connors 2009"),
    "F09_golden_cross": ("F","Golden Cross 50/200 SMA","equities","mo-yrs",1,5,"Long SPY when 50d > 200d SMA.","Universal TA"),
    "F10_macd_divergence": ("F","MACD Bullish Divergence","equities","days-wks",3,5,"Buy when price prints 20d low but MACD higher low.","Universal TA"),
    "F11_bb_squeeze": ("F","Bollinger Band Squeeze Breakout","equities","days-wks",3,5,"BB width at 6m low → trade breakout direction.","John Bollinger"),
    "F13_buy_the_dip": ("F","Buy-the-Dip vs DCA (anti-signal)","equities","yrs",6,5,"AQR test: simple DCA vs wait-for-10%-dip.","AQR 2025"),
    # G — alt-data web
    "G01_google_trends_recession": ("G","Google Trends 'recession' spike","equities","1-6mo",6,4,"Long SPY 6mo after trends z > 1.5σ.","Preis-Moat-Stanley Nature 2013"),
    "G02_wikipedia_pageviews": ("G","Wikipedia Pageview MoM L/S","equities","1mo",7,5,"Long top decile / short bottom decile MoM Δ views.","Pyun 2024; Behrendt-Zimmermann"),
    "G03_wsb_reddit_velocity": ("G","WSB Reddit Mention Velocity","equities","1-15d",5,4,"Contrarian short after top WSB mention spike.","Bradley/Hanousek/Jame/Xiao MS 2024"),
    "G07_warn_act_layoffs": ("G","WARN Act Layoff Notices","equities","1-6mo",7,4,"Short stocks after large CA WARN notice.","Goldman 'WARNing Signs' 2023"),
    "G10_china_port_throughput": ("G","China Port Throughput (IMF PortWatch)","China/global","1-3mo",6,3,"FXI vs SPY rotation by Shanghai/Ningbo/Shenzhen z-score.","IMF Cerdeiro et al 2020"),
    "G12_hdd_natgas": ("G","HDD/CDD vs Natural Gas","nat gas","1-2wk",4,5,"Long NG when forecast HDD rises > 5% wk/wk.","Mu Energy Econ 2007"),
    # H — crypto
    "H01_perp_funding": ("H","Perp Funding Rate Extremes (BTC)","crypto","1-30d",4,5,"Short BTC perp when funding z > 2; long when < -2.","Coinglass community"),
    "H02_stablecoin_supply_ratio": ("H","Stablecoin Supply Ratio (SSR)","crypto","wks-mo",5,4,"Long BTC when SSR oscillator < -0.8.","Glassnode SSR"),
    "H03_coinbase_premium": ("H","Coinbase Premium Index","crypto","1-30d",5,4,"Long BTC when 24h MA premium > +0.05% for 3d.","CryptoQuant"),
    "H04_exchange_netflow": ("H","Exchange Netflow","crypto","3-30d",4,3,"Sell BTC on positive 7d netflow z > +1.5.","CryptoQuant"),
    "H05_hash_ribbons": ("H","Hash Ribbons (Edwards)","crypto","3-12mo",6,5,"Long BTC when 30d hashrate MA crosses above 60d MA.","Charles Edwards"),
    "H06_mvrv_zscore": ("H","MVRV Z-Score","crypto","6-24mo",3,5,"Long BTC when z < 0; flat when z > 5.","BitcoinMagazinePro / Glassnode"),
    "H07_puell_multiple": ("H","Puell Multiple","crypto","6-24mo",4,5,"Long BTC when Puell < 0.5; flat when > 3.","D Puell"),
    "H08_nupl": ("H","NUPL (Net Unrealized Profit/Loss)","crypto","6-24mo",4,4,"Long BTC when NUPL < 0; reduce > 0.75.","Glassnode"),
    "H09_etf_flows": ("H","Spot BTC ETF Net Flows","crypto","1-30d",6,5,"Long BTC when 5d sum of ETF flows > $1B.","Farside Investors"),
    "H10_coin_days_destroyed": ("H","Coin Days Destroyed","crypto","1-12mo",6,4,"Distribution when 7d CDD > 3x 90d MA + > 5y HODL drops.","Glassnode"),
    "H11_btc_correlation_regime": ("H","BTC-NDX/BTC-Gold Correlation Regime","crypto/cross","mo",5,5,"Regime-size BTC: half when tech-corr; full when gold-corr.","Cross-asset regime"),
    "H12_cot_commercial": ("H","COT Commercial Hedger Extremes","commod/FX","4-12wk",5,5,"Long when commercials > 90th pct long.","Bohl-Sulewski 2023"),
    # I — pattern TA (BT8 versions)
    "I01_fvg_fill": ("I","ICT Fair Value Gap fill","equities","days-wks",3,4,"Long after pullback closes into bullish FVG zone.","ICT (M. Huddleston)"),
    "I02_order_block": ("I","ICT Order Block Retest","equities","days-wks",3,4,"Long on retest of OB after >1.5xATR up-impulse.","ICT"),
    "I03_liquidity_sweep": ("I","Liquidity Sweep Reversal (fade)","equities","days-wks",4,5,"Short on day high > prior 20d high but close < it.","ICT/SMC"),
    "I04_premium_discount": ("I","Premium/Discount Bias + RSI(2)","equities","wks",4,5,"Connors RSI(2) only when price < 50% of 60d range.","ICT-rebranded"),
    "I05_wyckoff_spring": ("I","Wyckoff Spring (daily)","equities","wks-mo",4,5,"Long after break below 30d range low + reclaim on volume.","Wyckoff 1930s"),
    "I06_seiden_zone": ("I","Sam Seiden Supply/Demand Zone","equities","days-wks",5,4,"Long after tight base preceded by drop + followed by rally.","Sam Seiden OTA"),
    "I07_fib_pullback": ("I","0.618 Fibonacci Pullback","equities","wks",2,4,"Long on 50-61.8% retracement in uptrend.","Universal TA"),
    # alt-data wildcards (BT7's overlapping IDs - kept distinct via filename suffix)
    "I04_trademark_filings_ls": ("E","USPTO Trademark Filings (proof-of-concept)","equities","12mo",7,5,"Hand-curated high vs low trademark filers.","UCLA Anderson Review"),
    "I06_lunar_january_ashares": ("J","Lunar January Effect (Chinese A-shares)","China","1mo",8,5,"Long ASHR only during first lunar month.","Liang-Liu-Zebedee SSRN 4209010"),
    "I07_geomagnetic_kp": ("J","Geomagnetic Storm Exit","equities","1-10d",9,5,"Exit SPY 10d when 5d Kp index in top decile.","Krivelyova-Robotti Atlanta Fed"),
    "I10_containerboard_pulse": ("J","Containerboard Recession Pulse","equities","3-12mo",7,4,"FRED IPG322S → IWM regime overlay.","FreightWaves; Goldman"),
    "I12_enso_ag_commodities": ("J","ENSO Ag Commodity Pair","commodities","6mo+",7,5,"SOYB vs CORN by NOAA ONI El Niño/La Niña.","NOAA ONI; Ubilava UC Davis"),
    "I14_lny_pre_holiday_drift": ("J","Lunar New Year Asian Pre-Holiday Drift","Asia eq","5-15d",7,5,"Long EWH+EWS 10 sessions ending LNY-eve.","Yuan-Gupta JIFMIM"),
    # J — curiosity / null
    "J03_super_bowl": ("J","Super Bowl Indicator","equities","12mo",2,5,"NFC win → long SPY (debunked null test).","Krueger-Kennedy JF 1990"),
    # K — geopolitical / policy
    "G1_gprt_defense": ("K","GPRT Threats → Defense Drift","defense eq","30-90d",6,5,"Long ITA+primes basket on GPR Threats > 1.5σ for 5d.","Caldara-Iacoviello GPR"),
    "G10_ndaa_defense": ("K","NDAA Passage Defense Drift","defense eq","10-30d",7,5,"Long ITA T-10 to T+3 around NDAA signature.","Stat-arb folklore"),
    "G14_prefomc_regime": ("K","Pre-FOMC + Dovish DGS2 Filter","equities","2d",6,5,"Pre-FOMC drift conditional on dovish 2y rate move.","Lucca-Moench + 2024 regime work"),
    "G15_election_vix": ("K","US Election VIX Ramp (VIXY)","vol","30-180d",4,5,"Long VIXY T-120 to T-5 before US presidential election.","St Louis Fed"),
    "G17_boj_surprise": ("K","BoJ Surprise YCC Tweak","FX/JPY","1-5d",7,4,"Short USDJPY 5d after hawkish BoJ surprise.","Forex.com 2023"),
    "G18_pboc_rrr_fxi": ("K","PBOC RRR Cut → Hang Seng","China eq","1-10d",6,5,"Long FXI 10d after PBOC RRR cut.","Investing.com"),
    "G19_npc_two_sessions": ("K","China NPC Two-Sessions Drift","China eq","10-15d",8,5,"Short FXI into NPC; flip long 10d post-close.","SCMP"),
    "G22_em_election_fx": ("K","EM Election Surprise FX Fade","EM FX","10-30d",7,4,"Fade T0→T+1 EM-FX gap from election surprise.","VanEck EM commentary"),
    "G23_gpr_tail_spy": ("K","GPR Tail-Spike Contrarian BUY SPY","equities","30-90d",6,5,"Long SPY when GPR > 3σ AND USREC=0.","Federal Reserve IFDP 1222"),
}


# ---------- load results ----------
def load_results():
    results = []
    for fp in sorted(RESULTS.glob("*.json")):
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception as e:
            print(f"skip {fp}: {e}")
            continue

        sid = fp.stem
        meta = CATALOG.get(sid)
        if not meta:
            base = "_".join(sid.split("_")[:3])
            meta = CATALOG.get(base)
        if not meta:
            # Fall back to inferring from JSON extras + filename
            cat_letter = sid[0] if sid and sid[0].isalpha() else "?"
            # Derive a display name from the suffix after the prefix
            parts = sid.split("_", 1)
            name_part = parts[1].replace("_", " ").title() if len(parts) > 1 else sid
            rule_or_mech = data.get("rule") or data.get("mechanism") or ""
            src_str = data.get("source") or ""
            meta = (cat_letter, name_part, "?", "?", 0, 0, rule_or_mech[:120], src_str[:120])

        cat, name, asset, horizon, orig, bt_feas, desc, src = meta
        results.append({
            "id": sid,
            "cat": cat,
            "name": name,
            "asset": asset,
            "horizon": horizon,
            "originality": orig,
            "bt_feasibility": bt_feas,
            "desc": desc,
            "source": src,
            "status": data.get("status", "ok"),
            "sharpe": data.get("sharpe"),
            "cagr": data.get("cagr"),
            "max_dd": data.get("max_dd"),
            "calmar": data.get("calmar"),
            "hit_rate": data.get("hit_rate"),
            "t_stat": data.get("t_stat"),
            "ann_vol": data.get("ann_vol"),
            "n_days": data.get("n_days"),
            "start": data.get("start"),
            "end": data.get("end"),
            "bench_cagr": data.get("bench_cagr"),
            "excess_cagr": data.get("excess_cagr"),
            "bench_sharpe": data.get("bench_sharpe"),
            "fail_reason": data.get("reason") if data.get("status") == "fail" else None,
        })
    return results


def fmt_pct(x, dp=2):
    if x is None: return ""
    try:
        return f"{x*100:.{dp}f}%"
    except Exception:
        return ""

def fmt_num(x, dp=2):
    if x is None: return ""
    try:
        return f"{x:.{dp}f}"
    except Exception:
        return ""


CAT_NAMES = {
    "A": "Calendar / Seasonal",
    "B": "Volatility / Options-Implied",
    "C": "Cross-Asset Macro / Rotation",
    "D": "Equity Factor / Cross-Sectional",
    "E": "Event-Driven / Corporate Actions",
    "F": "Sentiment / Breadth / Classic TA",
    "G": "Alt-Data / Web",
    "H": "Crypto On-Chain / Structural",
    "I": "Retail Pattern TA",
    "J": "Behavioral / Pop-Culture / Curiosity",
    "K": "Policy / Geopolitical",
}


def render_html(results):
    # Sort by Sharpe desc with None last
    def sort_key(r):
        s = r.get("sharpe")
        return (s is None, -(s or 0))

    sorted_by_sharpe = sorted(results, key=sort_key)
    top10 = [r for r in sorted_by_sharpe if r.get("sharpe") is not None][:10]

    # group by category
    by_cat = {}
    for r in results:
        by_cat.setdefault(r["cat"], []).append(r)

    # build JSON for client-side table
    table_data = []
    for r in results:
        row = {
            "id": r["id"],
            "cat": r["cat"],
            "name": r["name"],
            "asset": r["asset"],
            "horizon": r["horizon"],
            "orig": r["originality"],
            "btf": r["bt_feasibility"],
            "desc": r["desc"],
            "source": r["source"],
            "status": r["status"],
            "sharpe": r["sharpe"],
            "cagr": r["cagr"],
            "max_dd": r["max_dd"],
            "hit": r["hit_rate"],
            "t": r["t_stat"],
            "n_days": r["n_days"],
            "bench_cagr": r["bench_cagr"],
        }
        table_data.append(row)

    table_json = json.dumps(table_data, default=str)

    top_rows = ""
    for r in top10:
        top_rows += f"""<tr>
            <td><span class="badge cat-{r['cat']}">{r['cat']}</span></td>
            <td><strong>{escape(r['name'])}</strong><br><small>{escape(r['id'])}</small></td>
            <td>{escape(r['asset'])}</td>
            <td class="r mono">{fmt_num(r['sharpe'])}</td>
            <td class="r mono">{fmt_pct(r['cagr'])}</td>
            <td class="r mono">{fmt_pct(r['max_dd'])}</td>
            <td class="r mono">{fmt_num(r['t_stat'])}</td>
            <td class="r mono">{fmt_pct(r['hit_rate'])}</td>
            <td><small>{escape(r['source'])}</small></td>
        </tr>"""

    cat_overview_rows = ""
    for catkey, catname in CAT_NAMES.items():
        rs = by_cat.get(catkey, [])
        n = len(rs)
        n_ok = sum(1 for r in rs if r["status"] == "ok")
        n_fail = sum(1 for r in rs if r["status"] == "fail")
        sharpes = [r["sharpe"] for r in rs if r.get("sharpe") is not None]
        median_sh = sorted(sharpes)[len(sharpes)//2] if sharpes else None
        max_sh = max(sharpes) if sharpes else None
        cat_overview_rows += f"""<tr>
            <td><span class="badge cat-{catkey}">{catkey}</span></td>
            <td>{escape(catname)}</td>
            <td class="r">{n}</td>
            <td class="r">{n_ok}</td>
            <td class="r">{n_fail}</td>
            <td class="r mono">{fmt_num(median_sh) if median_sh is not None else "—"}</td>
            <td class="r mono">{fmt_num(max_sh) if max_sh is not None else "—"}</td>
        </tr>"""

    return TEMPLATE.replace("__TOP_ROWS__", top_rows)\
                   .replace("__CAT_OVERVIEW__", cat_overview_rows)\
                   .replace("__TABLE_DATA__", table_json)\
                   .replace("__N_TOTAL__", str(len(results)))\
                   .replace("__N_OK__", str(sum(1 for r in results if r["status"] == "ok")))\
                   .replace("__N_FAIL__", str(sum(1 for r in results if r["status"] == "fail")))


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trading Signals Catalog — Backtest Results</title>
<style>
:root {
  --bg: #0f1419;
  --bg-2: #1a1f2e;
  --bg-3: #232a3d;
  --fg: #e6e9ef;
  --fg-dim: #9ca3af;
  --accent: #7dd3fc;
  --good: #34d399;
  --bad: #f87171;
  --warn: #fbbf24;
  --border: #2d3548;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  --cat-A: #a78bfa; --cat-B: #fb7185; --cat-C: #34d399; --cat-D: #fbbf24;
  --cat-E: #60a5fa; --cat-F: #f472b6; --cat-G: #a3e635; --cat-H: #fb923c;
  --cat-I: #94a3b8; --cat-J: #c084fc; --cat-K: #2dd4bf;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0; background: var(--bg); color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 15px; line-height: 1.55;
}
header {
  padding: 60px 24px 36px; max-width: 1280px; margin: 0 auto;
  border-bottom: 1px solid var(--border);
}
h1 { font-size: 2.1em; margin: 0 0 8px; letter-spacing: -0.02em; }
h2 { font-size: 1.5em; margin: 48px 0 16px; letter-spacing: -0.01em; }
h3 { font-size: 1.15em; margin: 28px 0 12px; }
.sub { color: var(--fg-dim); margin-top: 6px; }
.container { max-width: 1280px; margin: 0 auto; padding: 0 24px 80px; }
.stats-row { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 28px; }
.stat-box {
  background: var(--bg-2); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px 20px; min-width: 140px;
}
.stat-box .v { font-size: 1.7em; font-weight: 600; font-family: var(--mono); }
.stat-box .l { color: var(--fg-dim); font-size: 0.85em; margin-top: 2px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
table th, table td {
  text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border);
  vertical-align: top;
}
table th {
  background: var(--bg-2); position: sticky; top: 0; z-index: 1;
  font-weight: 600; cursor: pointer; user-select: none;
}
table th:hover { background: var(--bg-3); }
table tr:hover td { background: rgba(125,211,252,0.04); }
table .r { text-align: right; }
table .mono { font-family: var(--mono); font-size: 13px; }
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 4px;
  font-size: 12px; font-weight: 600; font-family: var(--mono);
  color: var(--bg);
}
.cat-A { background: var(--cat-A); } .cat-B { background: var(--cat-B); }
.cat-C { background: var(--cat-C); } .cat-D { background: var(--cat-D); }
.cat-E { background: var(--cat-E); } .cat-F { background: var(--cat-F); }
.cat-G { background: var(--cat-G); } .cat-H { background: var(--cat-H); }
.cat-I { background: var(--cat-I); } .cat-J { background: var(--cat-J); }
.cat-K { background: var(--cat-K); }
.controls {
  display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  margin: 20px 0; padding: 16px; background: var(--bg-2);
  border: 1px solid var(--border); border-radius: 8px;
}
.controls input, .controls select {
  background: var(--bg-3); border: 1px solid var(--border); color: var(--fg);
  padding: 8px 12px; border-radius: 6px; font-size: 14px; font-family: inherit;
}
.controls input { width: 240px; }
.controls label { color: var(--fg-dim); font-size: 13px; margin-right: 4px; }
.scroll-x { overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); }
.good { color: var(--good); }
.bad { color: var(--bad); }
.warn { color: var(--warn); }
.pos { color: var(--good); font-weight: 600; }
.neg { color: var(--bad); font-weight: 600; }
.dim { color: var(--fg-dim); }
a { color: var(--accent); }
.caveat {
  background: rgba(251,191,36,0.08); border-left: 3px solid var(--warn);
  padding: 16px 20px; border-radius: 4px; margin: 20px 0;
}
.caveat strong { color: var(--warn); }
ul { padding-left: 22px; }
li { margin: 4px 0; }
code { background: var(--bg-3); padding: 1px 5px; border-radius: 3px; font-family: var(--mono); font-size: 13px; }
.footer { text-align: center; color: var(--fg-dim); margin-top: 60px; padding: 24px; font-size: 13px; }
.bar-cell { position: relative; }
.bar { position: absolute; left: 0; top: 0; bottom: 0; background: rgba(125,211,252,0.12); z-index: 0; }
.bar.neg { background: rgba(248,113,113,0.12); }
.bar-cell span { position: relative; z-index: 1; }
</style>
</head>
<body>

<header>
  <h1>Trading Signals Catalog &amp; Backtest Report</h1>
  <p class="sub">Comprehensive survey of trading signals from retail social media, academic literature, FinTwit, alternative data, crypto on-chain, and geopolitical event-driven sources. Backtested on free data (yfinance, FRED, public APIs) where feasible.</p>
  <div class="stats-row">
    <div class="stat-box"><div class="v">__N_TOTAL__</div><div class="l">signals backtested</div></div>
    <div class="stat-box"><div class="v">__N_OK__</div><div class="l">implemented OK</div></div>
    <div class="stat-box"><div class="v">__N_FAIL__</div><div class="l">data unavailable</div></div>
    <div class="stat-box"><div class="v">~199</div><div class="l">in raw catalog</div></div>
    <div class="stat-box"><div class="v">11</div><div class="l">signal categories</div></div>
  </div>
</header>

<div class="container">

<h2>Reading this report (read first)</h2>
<div class="caveat">
<strong>Honest caveats up front:</strong>
<ul>
<li><strong>Most signals do not generate alpha after honest backtesting.</strong> This is the expected outcome. Genuine, repeatable, easy-to-implement alpha is rare. The Sharpe distribution below is mostly between -0.5 and +0.7, dominated by buy-and-hold equivalents.</li>
<li><strong>Free-data limitations bite.</strong> FRED truncates some series to 3 years without an API key, paid alt-data (GEX, Glassdoor, satellite, App Store) was marked failed, EDGAR full-text scraping is slow, and CryptoQuant/Glassnode block free access. Signals marked <code>status: fail</code> with a <code>reason</code> were tried but lacked data.</li>
<li><strong>Survivorship bias warning (Category D).</strong> The equity factor cross-sectional tests (D03/D05/D06/D09/D14/D17) used a hand-curated 75-name liquid US large-cap universe. Every name in the basket survives to 2026. The classic L/S anomalies flipped <em>negative</em> because shorting today's mega-cap winners (NVDA/AAPL/META) is brutal. A true CRSP point-in-time test would likely flip these back positive. Read these results as risk-factor demonstrations, not alpha estimates.</li>
<li><strong>Ken French factor results (D01, D02, D04) are clean</strong> because they use Ken French's daily factor library directly (free, no survivorship issue).</li>
<li><strong>"Originality" is rough.</strong> 1 = textbook / every CFA knows it; 10 = obscure / not in standard quant playbook. Use it to find unfamiliar ideas, not as a reliability score.</li>
<li><strong>t-stat ≠ profitable strategy.</strong> Several signals (F08 RSI(2), A07 pre-FOMC) show real per-trade edge (t&gt;2.5) but the strategy is mostly in cash and earns less absolute return than buy-and-hold. Hit rate and t-stat measure event-level edge; CAGR is the wealth measure.</li>
<li><strong>Backtests sit on closing prices with no slippage, no borrow cost, no taxes.</strong> Short-only strategies in particular will deteriorate materially in live trading.</li>
</ul>
</div>

<h2>Top 10 by Sharpe</h2>
<p class="dim">Sorted by Sharpe; comparison to benchmark (buy-and-hold of underlying) in the rightmost column when available. Caveats above apply.</p>
<div class="scroll-x">
<table>
<thead><tr><th>Cat</th><th>Signal</th><th>Asset</th><th>Sharpe</th><th>CAGR</th><th>MaxDD</th><th>t-stat</th><th>Hit</th><th>Source</th></tr></thead>
<tbody>
__TOP_ROWS__
</tbody>
</table>
</div>

<h2>Category overview</h2>
<div class="scroll-x">
<table>
<thead><tr><th>Cat</th><th>Name</th><th>N</th><th>OK</th><th>Fail</th><th>Median Sharpe</th><th>Max Sharpe</th></tr></thead>
<tbody>
__CAT_OVERVIEW__
</tbody>
</table>
</div>

<h2>Master signal table</h2>
<p class="dim">All signals with backtest results. Click column headers to sort. Filter by category or text-search.</p>

<div class="controls">
  <label>Filter:</label>
  <input id="search" type="text" placeholder="search name, ID, source...">
  <label>Category:</label>
  <select id="catfilter">
    <option value="">all</option>
    <option value="A">A – Calendar</option>
    <option value="B">B – Vol</option>
    <option value="C">C – Cross-asset</option>
    <option value="D">D – Equity factor</option>
    <option value="E">E – Event-driven</option>
    <option value="F">F – Sentiment/Breadth</option>
    <option value="G">G – Alt-data</option>
    <option value="H">H – Crypto</option>
    <option value="I">I – Pattern TA</option>
    <option value="J">J – Curiosity</option>
    <option value="K">K – Policy/Geopolitical</option>
  </select>
  <label>Status:</label>
  <select id="statusfilter">
    <option value="">all</option>
    <option value="ok">ok</option>
    <option value="fail">fail</option>
  </select>
  <label>Min Sharpe:</label>
  <select id="minsharpe">
    <option value="">any</option>
    <option value="0">≥ 0</option>
    <option value="0.5">≥ 0.5</option>
    <option value="0.7">≥ 0.7</option>
    <option value="1.0">≥ 1.0</option>
  </select>
</div>

<div class="scroll-x">
<table id="signals">
<thead><tr>
  <th data-key="cat">Cat</th>
  <th data-key="id">ID</th>
  <th data-key="name">Signal</th>
  <th data-key="asset">Asset</th>
  <th data-key="horizon">Horizon</th>
  <th data-key="orig" class="r">Orig</th>
  <th data-key="btf" class="r">BT-feas</th>
  <th data-key="sharpe" class="r">Sharpe</th>
  <th data-key="cagr" class="r">CAGR</th>
  <th data-key="max_dd" class="r">MaxDD</th>
  <th data-key="t" class="r">t-stat</th>
  <th data-key="hit" class="r">Hit</th>
  <th data-key="status">Status</th>
</tr></thead>
<tbody id="tbody"></tbody>
</table>
</div>

<h2>Methodology</h2>
<p>The research process ran in 4 phases:</p>
<ol>
<li><strong>Mining (Phase 1).</strong> Nine parallel research agents covered: TikTok/retail-trader social, academic literature, trading forums + FinTwit, alternative data / political / event-driven, crypto on-chain + macro, behavioral / pop-culture, geopolitical / war / sanctions / defense, creative wildcard, and global government policy. Each agent returned 15–37 signals in a structured format with sources.</li>
<li><strong>Consolidation (Phase 2).</strong> Deduplicated to ~199 raw entries, sorted into 11 categories (A–K), graded each on originality (1–10) and backtest-feasibility (1–5).</li>
<li><strong>Backtesting (Phase 3).</strong> Eight parallel backtest worker agents implemented every signal feasible with free data, using a shared harness (yfinance, FRED, pandas-datareader). Each signal gets a JSON result with Sharpe, CAGR, MaxDD, t-stat, hit rate, plus assumptions noted in the Python file. Signals that couldn't be implemented are marked <code>status: fail</code> with a reason citing the literature.</li>
<li><strong>Reporting (Phase 4).</strong> This single HTML file, generated from <code>build_report.py</code> + the results/ directory.</li>
</ol>

<h2>Headline findings</h2>
<ul>
<li><strong>A14 BTC halving cycle</strong> is the largest absolute-return signal (Sharpe 1.20, +14pp excess CAGR vs BTC buy-and-hold, t=4.93) — but N=3 fully observed halvings. Classic small-N narrative risk.</li>
<li><strong>H06 MVRV proxy</strong> slightly beats BTC buy-and-hold on Sharpe (0.97 vs 0.90) — the only crypto on-chain signal that does.</li>
<li><strong>F03 Margin Debt YoY</strong> (Sharpe 0.72, CAGR 11.46%) and <strong>C02 Utilities/SPY (Gayed)</strong> (Sharpe 0.74, t=3.77) are the best macro-conditional equity timers we tested.</li>
<li><strong>C07 Accelerating Dual Momentum</strong> (Sharpe 0.73) and <strong>F09 Golden Cross</strong> (Sharpe 0.69, t=3.55) match or beat SPY on risk-adjusted basis with materially shallower drawdowns — good defensive variants but no CAGR uplift.</li>
<li><strong>F08 RSI(2) Connors</strong> (t=2.50) and <strong>A07 Pre-FOMC drift</strong> (t=2.49) have real per-event edge but are out-of-market too often to compete with buy-and-hold CAGR.</li>
<li><strong>E09 IPO Lockup expiry short</strong> (t=2.10, 61% hit rate, +3.15% over 10 days) is the cleanest event-driven hit.</li>
<li><strong>I03 Liquidity sweep fade</strong> is significantly NEGATIVE (t=-2.26): SPY breakouts above 20-day highs that close back below tend to follow through up, not fade. Important falsification of an ICT/SMC core idea.</li>
<li><strong>G-14 Pre-FOMC + dovish DGS2 filter</strong> reverse-finding: adding the dovish-regime filter DESTROYS the edge (Sharpe drops from 0.49 to 0.07). Literature's regime story is wrong direction in our sample.</li>
<li><strong>G-15 US Election VIX ramp</strong> is catastrophic: -9% CAGR, -86% DD via VIXY roll-cost drag. Realized vol bump is real; the equity instruments to express it are not.</li>
<li><strong>D-category equity factors</strong> using a survivorship-biased basket all flipped negative on the L/S — read as caveat-laden, not as factor death.</li>
</ul>

<h2>Honest disclaimers</h2>
<ul>
<li>Past performance is not predictive. Most published anomalies decay materially post-publication (McLean-Pontiff RFS 2016, ~32% decay average).</li>
<li>Free-data backtests overstate edge: no transaction costs, no slippage, no borrow cost, no taxes, no fund-level liquidity constraints.</li>
<li>Several "wins" above ride on a small number of events (BTC halvings: N=3; election VIX: N=4; BoJ surprises: N=3). Small-N findings are narrative-fitting more than statistical evidence.</li>
<li>The basket / universe shortcut in Category D is acknowledged.</li>
<li>Some failed signals (Glassdoor, satellite parking, GEX, App Store, EDGAR search traffic) may have real edge but require paid data.</li>
</ul>

<h2>Source materials</h2>
<ul>
<li>Code: <code>backtests/*.py</code> (one file per signal, runnable standalone)</li>
<li>Raw research outputs: <code>research/01[a-i]_*.md</code></li>
<li>Master catalog: <code>research/master_signals.md</code></li>
<li>Backtest metrics: <code>results/*.json</code></li>
<li>Shared harness: <code>backtests/harness.py</code></li>
</ul>

<div class="footer">
Generated from <code>build_report.py</code>. Free data only (yfinance, FRED, public APIs). Sources cited per signal in the master table.
</div>

</div>

<script>
const DATA = __TABLE_DATA__;

function fmtPct(x, dp=2) {
  if (x === null || x === undefined || isNaN(x)) return "";
  return (x*100).toFixed(dp) + "%";
}
function fmtNum(x, dp=2) {
  if (x === null || x === undefined || isNaN(x)) return "";
  return Number(x).toFixed(dp);
}
function colorVal(v, kind="sharpe") {
  if (v === null || v === undefined || isNaN(v)) return "";
  if (kind === "sharpe" || kind === "t") {
    if (v >= 1.0) return "pos";
    if (v >= 0.5) return "pos";
    if (v > 0) return "";
    if (v <= -0.5) return "neg";
    return "neg";
  }
  if (kind === "cagr") {
    if (v >= 0.05) return "pos";
    if (v <= -0.05) return "neg";
    return "";
  }
  if (kind === "dd") {
    if (v <= -0.5) return "neg";
    if (v <= -0.25) return "warn";
    return "";
  }
  return "";
}

let currentSort = { key: "sharpe", dir: -1 };

function render() {
  const q = document.getElementById("search").value.toLowerCase();
  const cf = document.getElementById("catfilter").value;
  const sf = document.getElementById("statusfilter").value;
  const ms = parseFloat(document.getElementById("minsharpe").value);

  let rows = DATA.filter(r => {
    if (cf && r.cat !== cf) return false;
    if (sf && r.status !== sf) return false;
    if (!isNaN(ms) && (r.sharpe === null || r.sharpe === undefined || r.sharpe < ms)) return false;
    if (q) {
      const hay = (r.name + " " + r.id + " " + r.source + " " + r.desc + " " + r.asset).toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  rows.sort((a, b) => {
    const k = currentSort.key;
    const av = a[k], bv = b[k];
    const an = av === null || av === undefined;
    const bn = bv === null || bv === undefined;
    if (an && bn) return 0;
    if (an) return 1;
    if (bn) return -1;
    if (typeof av === "string") return currentSort.dir * av.localeCompare(bv);
    return currentSort.dir * (av - bv);
  });

  const tbody = document.getElementById("tbody");
  tbody.innerHTML = rows.map(r => {
    const sh = r.sharpe, cg = r.cagr, dd = r.max_dd, tt = r.t, hi = r.hit;
    const status = r.status === "fail"
      ? `<span class="bad">FAIL</span>`
      : `<span class="dim">ok</span>`;
    return `<tr>
      <td><span class="badge cat-${r.cat}">${r.cat}</span></td>
      <td class="mono dim">${r.id}</td>
      <td><strong>${r.name}</strong><br><small class="dim">${r.desc} · <em>${r.source}</em></small></td>
      <td>${r.asset}</td>
      <td>${r.horizon}</td>
      <td class="r mono">${r.orig}</td>
      <td class="r mono">${r.btf}</td>
      <td class="r mono ${colorVal(sh,'sharpe')}">${fmtNum(sh)}</td>
      <td class="r mono ${colorVal(cg,'cagr')}">${fmtPct(cg)}</td>
      <td class="r mono ${colorVal(dd,'dd')}">${fmtPct(dd)}</td>
      <td class="r mono ${colorVal(tt,'t')}">${fmtNum(tt)}</td>
      <td class="r mono">${fmtPct(hi,1)}</td>
      <td>${status}</td>
    </tr>`;
  }).join("");
}

document.querySelectorAll("#signals th[data-key]").forEach(th => {
  th.addEventListener("click", () => {
    const k = th.dataset.key;
    if (currentSort.key === k) currentSort.dir *= -1;
    else { currentSort.key = k; currentSort.dir = (k === "name" || k === "cat" || k === "id" || k === "asset" || k === "horizon" || k === "status") ? 1 : -1; }
    render();
  });
});

["search","catfilter","statusfilter","minsharpe"].forEach(id => {
  document.getElementById(id).addEventListener("input", render);
});

render();
</script>

</body>
</html>
"""


def main():
    results = load_results()
    print(f"Loaded {len(results)} results.")
    html = render_html(results)
    OUT.write_text(html)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
