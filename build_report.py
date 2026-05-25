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
    # build JSON for client-side cards
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
            "start": r["start"],
            "end": r["end"],
            "bench_cagr": r["bench_cagr"],
            "fail_reason": r["fail_reason"],
        }
        table_data.append(row)

    table_json = json.dumps(table_data, default=str)

    return TEMPLATE.replace("__TABLE_DATA__", table_json)\
                   .replace("__N_TOTAL__", str(len(results)))\
                   .replace("__N_OK__", str(sum(1 for r in results if r["status"] == "ok")))\
                   .replace("__N_FAIL__", str(sum(1 for r in results if r["status"] == "fail")))


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Full Signal Catalog</title>
<style>
:root {
  --bg: #0f1419; --bg-2: #1a1f2e; --bg-3: #232a3d;
  --fg: #e6e9ef; --fg-dim: #9ca3af;
  --accent: #7dd3fc; --good: #34d399; --bad: #f87171; --warn: #fbbf24;
  --border: #2d3548;
  --mono: "SF Mono", "Fira Code", "Cascadia Code", monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --cat-A: #a78bfa; --cat-B: #fb7185; --cat-C: #34d399; --cat-D: #fbbf24;
  --cat-E: #60a5fa; --cat-F: #f472b6; --cat-G: #a3e635; --cat-H: #fb923c;
  --cat-I: #94a3b8; --cat-J: #c084fc; --cat-K: #2dd4bf;
}
* { box-sizing: border-box; margin: 0; }
body {
  background: var(--bg); color: var(--fg);
  font: 14px/1.55 var(--sans);
}
a { color: var(--accent); }
.pos { color: var(--good); font-weight: 600; }
.neg { color: var(--bad); font-weight: 600; }
.warn { color: var(--warn); }
.dim { color: var(--fg-dim); }
.good { color: var(--good); }
.bad { color: var(--bad); }

/* ---------- Layout ---------- */
.layout {
  display: grid;
  grid-template-columns: 260px 1fr;
  max-width: 1400px;
  margin: 0 auto;
  gap: 0;
  align-items: start;
  min-height: 100vh;
}

/* ---------- Sidebar ---------- */
.sidebar {
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
  background: var(--bg-2);
  border-right: 1px solid var(--border);
  padding: 16px 14px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.sidebar::-webkit-scrollbar { width: 4px; }
.sidebar::-webkit-scrollbar-thumb { background: var(--bg-3); border-radius: 2px; }
.sb-title {
  font-size: 14px; font-weight: 700; letter-spacing: -0.01em;
  display: flex; align-items: baseline; gap: 8px;
}
.sb-title .count {
  font-size: 11px; font-weight: 400; color: var(--fg-dim);
  background: var(--bg-3); padding: 1px 7px; border-radius: 8px;
  font-family: var(--mono);
}
.sb-label {
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--fg-dim); margin-bottom: 5px;
}
.sb-group { display: flex; flex-direction: column; gap: 4px; }

/* Search */
#search {
  width: 100%; padding: 7px 10px; font-size: 13px; font-family: inherit;
  background: var(--bg-3); border: 1px solid var(--border); color: var(--fg);
  border-radius: 6px; outline: none;
}
#search:focus { border-color: var(--accent); }
#search::placeholder { color: var(--fg-dim); }

/* Category pills */
.cat-pills { display: flex; flex-wrap: wrap; gap: 4px; }
.cat-pill {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 4px 8px; border-radius: 12px; font-size: 11px;
  background: var(--bg-3); border: 1px solid var(--border);
  color: var(--fg); cursor: pointer; font-family: inherit;
  transition: background 0.12s, border-color 0.12s;
}
.cat-pill:hover { border-color: var(--fg-dim); }
.cat-pill.active { background: rgba(125,211,252,0.15); border-color: var(--accent); color: var(--accent); font-weight: 600; }
.cat-pill .swatch {
  display: inline-block; width: 8px; height: 8px; border-radius: 2px;
}
.pill-count {
  font-family: var(--mono); font-size: 10px; color: var(--fg-dim);
  margin-left: 2px;
}
.cat-pill.active .pill-count { color: var(--accent); }

/* Status toggle */
.status-toggle {
  display: flex; gap: 0; background: var(--bg-3); border-radius: 6px;
  padding: 3px; border: 1px solid var(--border);
}
.status-toggle button {
  flex: 1; padding: 5px 8px; font-size: 11px; font-weight: 600;
  background: transparent; color: var(--fg-dim); border: none;
  border-radius: 4px; cursor: pointer; font-family: inherit;
}
.status-toggle button:hover { color: var(--fg); }
.status-toggle button.active {
  background: var(--bg-2); color: var(--accent);
  box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}

/* Sliders */
.slider-group {
  display: flex; flex-direction: column; gap: 5px;
  padding: 8px 10px; background: var(--bg-3); border-radius: 6px;
  border: 1px solid var(--border);
}
.slider-group label {
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 10px; color: var(--fg-dim); text-transform: uppercase;
  letter-spacing: 0.07em;
}
.slider-group label span {
  color: var(--accent); font-family: var(--mono); text-transform: none;
  letter-spacing: 0; font-size: 12px; font-weight: 600;
}
.slider-group input[type="range"] {
  width: 100%; height: 4px; -webkit-appearance: none; appearance: none;
  background: var(--bg-2); border-radius: 2px; cursor: pointer;
}
.slider-group input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none;
  width: 14px; height: 14px; border-radius: 50%;
  background: var(--accent); cursor: pointer; border: 2px solid var(--bg-2);
}
.slider-group input[type="range"]::-moz-range-thumb {
  width: 14px; height: 14px; border-radius: 50%;
  background: var(--accent); cursor: pointer; border: 2px solid var(--bg-2);
}

/* Sort buttons */
.sort-btns {
  display: flex; flex-wrap: wrap; gap: 4px;
}
.sort-btn {
  padding: 4px 8px; font-size: 10px; font-weight: 600;
  background: var(--bg-3); border: 1px solid var(--border); color: var(--fg-dim);
  border-radius: 4px; cursor: pointer; font-family: inherit;
  transition: background 0.12s, border-color 0.12s;
}
.sort-btn:hover { color: var(--fg); border-color: var(--fg-dim); }
.sort-btn.active { color: var(--accent); border-color: var(--accent); }
.sort-btn.active::after { content: " \25BC"; font-size: 8px; }
.sort-btn.active.asc::after { content: " \25B2"; font-size: 8px; }

/* Reset button */
#reset-btn {
  padding: 7px 0; font-size: 12px; font-weight: 600;
  background: var(--bg-3); border: 1px solid var(--border); color: var(--fg-dim);
  border-radius: 6px; cursor: pointer; font-family: inherit; text-align: center;
}
#reset-btn:hover { color: var(--accent); border-color: var(--accent); }

/* Sidebar summary counts */
.sb-stats {
  display: flex; gap: 8px; font-size: 11px; color: var(--fg-dim);
  font-family: var(--mono);
}
.sb-stats .v { font-weight: 600; color: var(--fg); }
.sb-stats .ok-v { color: var(--good); font-weight: 600; }
.sb-stats .fail-v { color: var(--bad); font-weight: 600; }

/* ---------- Main ---------- */
main {
  min-width: 0;
  padding: 16px 20px 60px;
  display: flex;
  flex-direction: column;
}

/* Summary row */
.summary-row {
  display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
  margin-bottom: 12px; font-size: 12px; color: var(--fg-dim);
}
.summary-row .v { font-weight: 600; font-family: var(--mono); }
.summary-row .all { color: var(--fg); }
.summary-row .ok { color: var(--good); }
.summary-row .fail { color: var(--bad); }
#visible-count { color: var(--accent); font-weight: 600; font-family: var(--mono); }

/* ---------- Cards ---------- */
#cards-container {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.signal-card {
  padding: 8px 14px;
  border-radius: 6px;
  margin: 2px 0;
  cursor: pointer;
  background: var(--bg-2);
  border: 1px solid var(--border);
  transition: background 0.12s, border-color 0.12s;
  overflow: hidden;
}
.signal-card:hover {
  border-color: var(--fg-dim);
  background: var(--bg-3);
}
.signal-card.expanded {
  border-color: var(--accent);
  background: var(--bg-2);
  cursor: default;
}
.signal-card.failed {
  opacity: 0.55;
}
.signal-card.failed:hover {
  opacity: 0.8;
}
.signal-card.failed.expanded {
  opacity: 0.85;
}

/* Collapsed row */
.card-collapsed {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 29px;
}
.card-swatch {
  width: 6px; height: 22px; border-radius: 2px; flex-shrink: 0;
}
.card-name {
  font-size: 13px; font-weight: 600; flex: 1; min-width: 0;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.card-inline-metrics {
  display: flex; gap: 12px; align-items: center; flex-shrink: 0;
  font-family: var(--mono); font-size: 12px;
}
.card-inline-metrics .lbl {
  color: var(--fg-dim); font-size: 10px; margin-right: 3px;
}
.card-status-badge {
  font-size: 10px; font-weight: 700; padding: 2px 6px;
  border-radius: 3px; font-family: var(--mono);
}
.card-status-badge.ok {
  background: rgba(52,211,153,0.12); color: var(--good);
}
.card-status-badge.fail {
  background: rgba(248,113,113,0.15); color: var(--bad);
}

/* Expanded details */
.card-details {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.3s ease;
}
.signal-card.expanded .card-details {
  max-height: 600px;
}
.card-details-inner {
  padding: 12px 0 6px 16px;
}

/* Metrics grid inside expanded */
.card-metrics {
  display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;
}
.card-m {
  background: var(--bg-3); padding: 5px 10px; border-radius: 5px;
  border: 1px solid var(--border); min-width: 68px;
}
.card-m .mv { font: 600 14px var(--mono); }
.card-m .ml { color: var(--fg-dim); font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em; }

/* Description sections */
.card-section {
  margin: 8px 0;
}
.card-section-label {
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--fg-dim); margin-bottom: 3px;
}
.card-section-text {
  font-size: 13px; color: var(--fg); line-height: 1.5;
}
.card-section-text.rule {
  background: var(--bg-3); border-left: 3px solid var(--accent);
  padding: 8px 12px; border-radius: 4px; font-family: var(--mono); font-size: 12px;
}
.card-section-text.fail-reason {
  background: rgba(248,113,113,0.07); border-left: 3px solid var(--bad);
  padding: 8px 12px; border-radius: 4px;
}
.card-meta {
  display: flex; gap: 16px; flex-wrap: wrap; font-size: 11px; color: var(--fg-dim);
  margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border);
}
.card-meta span { display: inline-flex; align-items: center; gap: 4px; }
.card-meta .mlbl { font-weight: 600; color: var(--fg-dim); }

.badge {
  display: inline-block; padding: 2px 7px; border-radius: 4px;
  font-size: 11px; font-weight: 600; font-family: var(--mono); color: var(--bg);
}
.cat-A { background: var(--cat-A); } .cat-B { background: var(--cat-B); }
.cat-C { background: var(--cat-C); } .cat-D { background: var(--cat-D); }
.cat-E { background: var(--cat-E); } .cat-F { background: var(--cat-F); }
.cat-G { background: var(--cat-G); } .cat-H { background: var(--cat-H); }
.cat-I { background: var(--cat-I); } .cat-J { background: var(--cat-J); }
.cat-K { background: var(--cat-K); }

.footer {
  text-align: center; color: var(--fg-dim); margin-top: 32px;
  padding: 16px; font-size: 12px;
}

/* ---------- Responsive ---------- */
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  .sidebar {
    position: static; height: auto; border-right: none;
    border-bottom: 1px solid var(--border);
  }
  .card-inline-metrics { gap: 8px; }
}
</style>
</head>
<body>

<div class="layout">

  <!-- ===== Sidebar ===== -->
  <aside class="sidebar">
    <div class="sb-title">Full Catalog <span class="count">__N_TOTAL__</span></div>
    <div class="sb-stats">
      <span><span class="v">__N_TOTAL__</span> total</span>
      <span><span class="ok-v">__N_OK__</span> ok</span>
      <span><span class="fail-v">__N_FAIL__</span> fail</span>
    </div>

    <div class="sb-group">
      <div class="sb-label">Search</div>
      <input id="search" type="text" placeholder="name, ID, source...">
    </div>

    <div class="sb-group">
      <div class="sb-label">Category</div>
      <div class="cat-pills" id="cat-pills">
        <button class="cat-pill active" data-cat=""><span class="swatch" style="background:var(--fg-dim)"></span>All <span class="pill-count" id="cnt-all">__N_TOTAL__</span></button>
        <button class="cat-pill" data-cat="A"><span class="swatch" style="background:var(--cat-A)"></span>A <span class="pill-count" id="cnt-A"></span></button>
        <button class="cat-pill" data-cat="B"><span class="swatch" style="background:var(--cat-B)"></span>B <span class="pill-count" id="cnt-B"></span></button>
        <button class="cat-pill" data-cat="C"><span class="swatch" style="background:var(--cat-C)"></span>C <span class="pill-count" id="cnt-C"></span></button>
        <button class="cat-pill" data-cat="D"><span class="swatch" style="background:var(--cat-D)"></span>D <span class="pill-count" id="cnt-D"></span></button>
        <button class="cat-pill" data-cat="E"><span class="swatch" style="background:var(--cat-E)"></span>E <span class="pill-count" id="cnt-E"></span></button>
        <button class="cat-pill" data-cat="F"><span class="swatch" style="background:var(--cat-F)"></span>F <span class="pill-count" id="cnt-F"></span></button>
        <button class="cat-pill" data-cat="G"><span class="swatch" style="background:var(--cat-G)"></span>G <span class="pill-count" id="cnt-G"></span></button>
        <button class="cat-pill" data-cat="H"><span class="swatch" style="background:var(--cat-H)"></span>H <span class="pill-count" id="cnt-H"></span></button>
        <button class="cat-pill" data-cat="I"><span class="swatch" style="background:var(--cat-I)"></span>I <span class="pill-count" id="cnt-I"></span></button>
        <button class="cat-pill" data-cat="J"><span class="swatch" style="background:var(--cat-J)"></span>J <span class="pill-count" id="cnt-J"></span></button>
        <button class="cat-pill" data-cat="K"><span class="swatch" style="background:var(--cat-K)"></span>K <span class="pill-count" id="cnt-K"></span></button>
      </div>
    </div>

    <div class="sb-group">
      <div class="sb-label">Status</div>
      <div class="status-toggle" id="status-toggle">
        <button class="active" data-status="">All</button>
        <button data-status="ok">OK</button>
        <button data-status="fail">Fail</button>
      </div>
    </div>

    <div class="sb-group">
      <div class="slider-group">
        <label>Min CAGR <span id="cagr-val">0%</span></label>
        <input type="range" id="min-cagr" min="0" max="100" value="0" step="1">
      </div>
    </div>

    <div class="sb-group">
      <div class="slider-group">
        <label>Min Sharpe <span id="sharpe-val">any</span></label>
        <input type="range" id="min-sharpe" min="-10" max="30" value="-10" step="1">
      </div>
    </div>

    <div class="sb-group">
      <div class="sb-label">Sort by</div>
      <div class="sort-btns" id="sort-btns">
        <button class="sort-btn active" data-sort="sharpe">Sharpe</button>
        <button class="sort-btn" data-sort="cagr">CAGR</button>
        <button class="sort-btn" data-sort="t">t-stat</button>
        <button class="sort-btn" data-sort="name">Name</button>
        <button class="sort-btn" data-sort="cat">Cat</button>
        <button class="sort-btn" data-sort="max_dd">MaxDD</button>
      </div>
    </div>

    <button id="reset-btn">Reset filters</button>
  </aside>

  <!-- ===== Main ===== -->
  <main>
    <div class="summary-row">
      <span>Showing <span id="visible-count">__N_TOTAL__</span> of <span class="v all">__N_TOTAL__</span> signals</span>
      <span><span class="v ok">__N_OK__</span> ok</span>
      <span><span class="v fail">__N_FAIL__</span> fail</span>
    </div>

    <div id="cards-container"></div>

    <div class="footer">
      Generated from <code>build_report.py</code>. Free data only (yfinance, FRED, public APIs).
    </div>
  </main>

</div>

<script>
const DATA = __TABLE_DATA__;

// ---- Category metadata ----
const CAT_NAMES = {
  A:"Calendar",B:"Vol/Options",C:"Cross-Asset",D:"Equity Factor",
  E:"Event-Driven",F:"Sentiment/TA",G:"Alt-Data",H:"Crypto",
  I:"Pattern TA",J:"Curiosity",K:"Policy/Geo"
};
const CAT_COLORS = {
  A:"var(--cat-A)",B:"var(--cat-B)",C:"var(--cat-C)",D:"var(--cat-D)",
  E:"var(--cat-E)",F:"var(--cat-F)",G:"var(--cat-G)",H:"var(--cat-H)",
  I:"var(--cat-I)",J:"var(--cat-J)",K:"var(--cat-K)"
};

// ---- Compute per-category counts ----
(function initCounts() {
  const counts = {};
  DATA.forEach(r => { counts[r.cat] = (counts[r.cat] || 0) + 1; });
  Object.keys(CAT_NAMES).forEach(c => {
    const el = document.getElementById("cnt-" + c);
    if (el) el.textContent = counts[c] || 0;
  });
})();

// ---- Formatting helpers ----
function fmtPct(x, dp) {
  if (dp === undefined) dp = 2;
  if (x === null || x === undefined || isNaN(x)) return "—";
  return (x * 100).toFixed(dp) + "%";
}
function fmtNum(x, dp) {
  if (dp === undefined) dp = 2;
  if (x === null || x === undefined || isNaN(x)) return "—";
  return Number(x).toFixed(dp);
}
function colorCls(v, kind) {
  if (v === null || v === undefined || isNaN(v)) return "";
  if (kind === "sharpe" || kind === "t") {
    return v >= 0.5 ? "pos" : v <= -0.5 ? "neg" : v < 0 ? "neg" : "";
  }
  if (kind === "cagr") {
    return v >= 0.05 ? "pos" : v <= -0.05 ? "neg" : "";
  }
  if (kind === "dd") {
    return v <= -0.5 ? "neg" : v <= -0.25 ? "warn" : "";
  }
  return "";
}

// ---- State ----
let currentSort = { key: "sharpe", dir: -1 };
let activeCat = "";
let activeStatus = "";
let expandedIds = new Set();

// ---- Sidebar wiring ----
document.querySelectorAll("#cat-pills .cat-pill").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#cat-pills .cat-pill").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    activeCat = btn.dataset.cat;
    render();
  });
});

document.querySelectorAll("#status-toggle button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#status-toggle button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    activeStatus = btn.dataset.status;
    render();
  });
});

const cagrSlider = document.getElementById("min-cagr");
const cagrVal = document.getElementById("cagr-val");
cagrSlider.addEventListener("input", () => {
  const v = parseInt(cagrSlider.value);
  cagrVal.textContent = v === 0 ? "0%" : v + "%";
  render();
});

const sharpeSlider = document.getElementById("min-sharpe");
const sharpeVal = document.getElementById("sharpe-val");
sharpeSlider.addEventListener("input", () => {
  const v = parseInt(sharpeSlider.value);
  sharpeVal.textContent = v <= -10 ? "any" : (v / 10).toFixed(1);
  render();
});

document.getElementById("search").addEventListener("input", render);

// Sort buttons
document.querySelectorAll("#sort-btns .sort-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const k = btn.dataset.sort;
    if (currentSort.key === k) {
      currentSort.dir *= -1;
    } else {
      currentSort.key = k;
      currentSort.dir = (k === "name" || k === "cat" || k === "id") ? 1 : -1;
    }
    document.querySelectorAll("#sort-btns .sort-btn").forEach(b => {
      b.classList.remove("active", "asc");
    });
    btn.classList.add("active");
    if (currentSort.dir === 1) btn.classList.add("asc");
    render();
  });
});

// Reset
document.getElementById("reset-btn").addEventListener("click", () => {
  document.getElementById("search").value = "";
  activeCat = "";
  document.querySelectorAll("#cat-pills .cat-pill").forEach(b => b.classList.remove("active"));
  document.querySelector('#cat-pills .cat-pill[data-cat=""]').classList.add("active");
  activeStatus = "";
  document.querySelectorAll("#status-toggle button").forEach(b => b.classList.remove("active"));
  document.querySelector('#status-toggle button[data-status=""]').classList.add("active");
  cagrSlider.value = 0; cagrVal.textContent = "0%";
  sharpeSlider.value = -10; sharpeVal.textContent = "any";
  currentSort = { key: "sharpe", dir: -1 };
  document.querySelectorAll("#sort-btns .sort-btn").forEach(b => b.classList.remove("active", "asc"));
  document.querySelector('#sort-btns .sort-btn[data-sort="sharpe"]').classList.add("active");
  expandedIds.clear();
  render();
});

// ---- Build card HTML ----
function buildCard(r) {
  const isExp = expandedIds.has(r.id);
  const isFail = r.status === "fail";
  const cls = "signal-card" + (isExp ? " expanded" : "") + (isFail ? " failed" : "");
  const swatchColor = CAT_COLORS[r.cat] || "var(--fg-dim)";

  // Collapsed row
  const shText = r.sharpe != null ? fmtNum(r.sharpe) : "—";
  const cgText = r.cagr != null ? fmtPct(r.cagr, 1) : "—";
  const shCls = colorCls(r.sharpe, "sharpe");
  const cgCls = colorCls(r.cagr, "cagr");
  const statusBadge = isFail
    ? '<span class="card-status-badge fail">FAIL</span>'
    : '<span class="card-status-badge ok">OK</span>';

  let collapsed = '<div class="card-collapsed">' +
    '<div class="card-swatch" style="background:' + swatchColor + '"></div>' +
    '<span class="card-name">' + escHtml(r.name) + '</span>' +
    '<div class="card-inline-metrics">' +
      '<span><span class="lbl">Sh</span><span class="' + shCls + '">' + shText + '</span></span>' +
      '<span><span class="lbl">CAGR</span><span class="' + cgCls + '">' + cgText + '</span></span>' +
      statusBadge +
    '</div>' +
  '</div>';

  // Expanded details
  let details = '<div class="card-details"><div class="card-details-inner">';

  // Metrics row
  details += '<div class="card-metrics">';
  const metrics = [
    ["Sharpe", fmtNum(r.sharpe), colorCls(r.sharpe, "sharpe")],
    ["CAGR", fmtPct(r.cagr), colorCls(r.cagr, "cagr")],
    ["MaxDD", fmtPct(r.max_dd), colorCls(r.max_dd, "dd")],
    ["t-stat", fmtNum(r.t), colorCls(r.t, "t")],
    ["Hit", fmtPct(r.hit, 1), ""],
    ["BenchCAGR", fmtPct(r.bench_cagr), ""],
  ];
  metrics.forEach(([label, val, cls]) => {
    details += '<div class="card-m"><div class="ml">' + label + '</div><div class="mv ' + cls + '">' + val + '</div></div>';
  });
  details += '</div>';

  // Rule / description
  if (r.desc) {
    details += '<div class="card-section"><div class="card-section-label">Rule / Mechanism</div>' +
      '<div class="card-section-text rule">' + escHtml(r.desc) + '</div></div>';
  }

  // Fail reason
  if (isFail && r.fail_reason) {
    details += '<div class="card-section"><div class="card-section-label">Fail Reason</div>' +
      '<div class="card-section-text fail-reason">' + escHtml(r.fail_reason) + '</div></div>';
  }

  // Source
  if (r.source) {
    details += '<div class="card-section"><div class="card-section-label">Source</div>' +
      '<div class="card-section-text">' + escHtml(r.source) + '</div></div>';
  }

  // Meta row
  details += '<div class="card-meta">';
  details += '<span><span class="mlbl">ID:</span> ' + escHtml(r.id) + '</span>';
  details += '<span><span class="mlbl">Asset:</span> ' + escHtml(r.asset) + '</span>';
  details += '<span><span class="mlbl">Horizon:</span> ' + escHtml(r.horizon) + '</span>';
  if (r.n_days) details += '<span><span class="mlbl">N days:</span> ' + r.n_days + '</span>';
  if (r.start) details += '<span><span class="mlbl">Start:</span> ' + escHtml(r.start) + '</span>';
  if (r.end) details += '<span><span class="mlbl">End:</span> ' + escHtml(r.end) + '</span>';
  details += '</div>';

  details += '</div></div>';

  return '<div class="' + cls + '" data-id="' + escHtml(r.id) + '">' + collapsed + details + '</div>';
}

function escHtml(s) {
  if (!s) return "";
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

// ---- Render ----
function render() {
  const q = document.getElementById("search").value.toLowerCase();
  const minCagr = parseInt(cagrSlider.value) / 100;
  const minSharpeRaw = parseInt(sharpeSlider.value);
  const minSharpe = minSharpeRaw <= -10 ? -Infinity : minSharpeRaw / 10;

  let rows = DATA.filter(r => {
    if (activeCat && r.cat !== activeCat) return false;
    if (activeStatus && r.status !== activeStatus) return false;
    if (minCagr > 0 && (r.cagr === null || r.cagr === undefined || r.cagr < minCagr)) return false;
    if (minSharpe > -Infinity && (r.sharpe === null || r.sharpe === undefined || r.sharpe < minSharpe)) return false;
    if (q) {
      const hay = (r.name + " " + r.id + " " + (r.source||"") + " " + (r.desc||"") + " " + r.asset).toLowerCase();
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

  document.getElementById("visible-count").textContent = rows.length;

  const container = document.getElementById("cards-container");
  container.innerHTML = rows.map(r => buildCard(r)).join("");

  // Attach click handlers
  container.querySelectorAll(".signal-card").forEach(card => {
    card.addEventListener("click", (e) => {
      const id = card.dataset.id;
      if (expandedIds.has(id)) {
        expandedIds.delete(id);
        card.classList.remove("expanded");
      } else {
        expandedIds.add(id);
        card.classList.add("expanded");
      }
    });
  });
}

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
