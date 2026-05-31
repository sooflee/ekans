"""Loop 1 Idea Scout — Hydrogen economy & DOE-funded transitions.

Three ideas:
  1. 45V Treasury final-rule three-pillar relaxation -> Texas/PJM ERCOT-tied
     green-H2 projects unlock -> APD on-site H2 contract gross-margin tailwind
     and CMI Accelera electrolyzer order book inflection. Industrial gas /
     industrials.
  2. Japan GX Promotion Act + Green Innovation Fund H2/NH3 supply-chain CfD
     auction round -> JERA/Kobe Steel ammonia co-firing offtake commitments at
     Hekinan/Kobe units -> US ammonia merchant exporters (CF Industries) and
     low-carbon ammonia producers (LIN via Blue Point JV exposure) benefit on
     Pacific-basin ATR-blue-NH3 netbacks; commodity-equity pair structure.
  3. COUNTER-SIGNAL vs. crowded long-PLUG/BE convergence: SEC EDGAR ATM-equity
     issuance velocity (S-3 ASR sales-agreement utilization + 424B5 take-down
     prints) for cash-burning hydrogen pure-plays. When PLUG/BE/FCEL/BLDP burn
     rates require >$150M new equity in any rolling 90d window, dilution
     cascade compresses multiples ~25-40% historically (PLUG Aug-23, Jan-24
     prints). Short basket; flags exactly when 45V/H2Hub bullish narrative
     peaks.
"""
import sys
sys.path.insert(0, "/Users/benson/Projects/ekans/pipeline")
from queue_io import append_ideas, heartbeat

new_ideas = [
    {
        "name": "45V Treasury Final-Rule Three-Pillar Relaxation -> ERCOT-Tied Green H2 FIDs Unlock -> APD On-Site Margin + CMI Accelera Order Inflection",
        "category": "C",
        "asset_class": "single_name_equity",
        "horizon": "6-16 weeks",
        "thesis": "The 45V production-tax-credit final rule (Treasury / IRS) defines three-pillar additionality, temporal-matching, and deliverability rules that determine which green-H2 projects qualify for the $3/kg credit. The Jan-2024 NPRM was widely viewed as too strict (hourly matching from 2028, strict additionality) and stalled FID on ~80% of announced US electrolyzer projects per BNEF tracking. When Treasury publishes the final rule and relaxes any of (i) hourly to annual matching window deferral, (ii) deliverability region carve-outs for ERCOT, or (iii) grandfathering rules for projects with existing PPAs, a wave of paused projects in ERCOT and PJM (NextEra/Air Products Wilbarger, ExxonMobil Baytown, OCI Beaumont blue/green-NH3 retrofits) re-enter FID. Air Products (APD) holds the deepest on-site H2 contract pipeline (Neom, La Porte, Massena) and is the marginal seller of merchant H2 to refiners; FID restart converts ~$15B announced backlog from option-value to contracted revenue. Cummins (CMI) Accelera segment sells PEM electrolyzers and is the only US-listed industrial with a credible 1+ GW pipeline. The 45V final-rule print is a discrete, dateable event; rule text relaxation can be measured by hour-vs-annual matching language and ERCOT carve-out language within hours of publication via Federal Register XML.",
        "causal_chain": [
            "1. Treasury/IRS publish 45V final rule in Federal Register (free XML)",
            "2. Parse rule text for three-pillar language: temporal matching window (hourly vs annual through 2028+), additionality (new vs existing clean gen), deliverability region definitions (single ERCOT zone vs sub-region)",
            "3. Score relaxation index: +1 each for (a) annual matching extended >=24 months, (b) ERCOT deliverability treated as single region, (c) grandfathering of pre-2024 PPAs",
            "4. If relaxation score >=2, paused green-H2 projects in ERCOT/PJM (BNEF tracker shows ~12 GW announced, ~2 GW FID'd) move from FEED to FID within 60-90 days",
            "5. APD electrolyzer offtake contracts (Wilbarger TX, Neom co-investment) re-rate from option value to contracted EBITDA — sell-side upgrades within 2-4 weeks",
            "6. CMI Accelera order book inflects: PEM electrolyzer purchase orders surge as ERCOT projects unlock; segment revenue guide raised at next print",
            "7. Long APD + CMI basket for 6-16 weeks; pair with short PLUG/BE leg to isolate beneficiaries from cash-burning pure-plays (APD/CMI are EBITDA-positive, PLUG/BE are not)"
        ],
        "data_sources": [
            "Federal Register API XML for Treasury/IRS 45V final rule (free, federalregister.gov)",
            "DOE OCED H2Hub milestone disclosures (free, energy.gov/oced)",
            "BNEF 1H2026 Hydrogen Market Outlook FID tracker (paywall; free abstract via Bloomberg.com)",
            "yfinance for APD, CMI, LIN, PLUG, BE daily OHLCV",
            "SEC EDGAR for APD/CMI 8-K hydrogen-segment disclosures"
        ],
        "originality": 8,
        "bt_feasibility": 4,
        "source_reference": "Treasury IRS 45V NPRM (Dec 2023) + comment letters from Constellation, NextEra, Air Products; BNEF 1H2026 Hydrogen Market Outlook; CMI Accelera Q4-2025 earnings transcript on electrolyzer order book",
        "research_phase": "pipeline_scout",
        "dedup_key": "treasury_45v_finalrule_threepillar_apd_cmi"
    },
    {
        "name": "Japan GX Green Innovation Fund H2/NH3 Supply-Chain CfD Auction Round -> JERA Ammonia Co-Firing Offtake -> CF Industries / LIN Blue-NH3 Pacific Netback Long",
        "category": "C",
        "asset_class": "single_name_equity",
        "horizon": "8-20 weeks",
        "thesis": "Japan's GX Promotion Act (FY2024) authorizes a 15-year Contract-for-Difference auction administered by METI/NEDO for low-carbon H2 and ammonia supply at premium-vs-grey prices. The Green Innovation Fund disburses milestone-based grants to JERA/IHI/Kobe Steel for ammonia co-firing at Hekinan Unit 4 (20% NH3 by FY2027) and Mitsubishi Heavy's 100%-NH3 gas-turbine demo at Takasago. Each CfD auction round (semi-annual cadence, METI press releases free) awards 10-year USD-denominated offtake premiums of $50-150/ton ammonia above grey-NH3 reference. US Gulf Coast ATR-blue-ammonia producers (CF Industries via the Blue Point JV with JERA/Mitsui, OCI Beaumont, Nutrien Geismar) and merchant low-carbon H2 producers (Linde via Blue Point) win export contracts because Gulf Coast NG feedstock + CCS economics undercut Middle East ATR-NH3 on landed-Yokohama basis. When METI announces auction-round results, the implied Pacific-basin blue-NH3 contract price moves the netback for Gulf exporters; CF's contracted-volume disclosure (10-K Schedule II) lags by 2-3 months, giving a trade window. Long CF + LIN basket on auction-result announcements with awarded volume >0.5 Mtpa; short Yara (YAR.OL) as a beta hedge — Yara's European green-NH3 strategy faces 45V-disadvantaged economics vs Gulf blue.",
        "causal_chain": [
            "1. METI/NEDO publishes GX H2/NH3 CfD auction-round results (free, meti.go.jp press release Japanese + English)",
            "2. Parse results for: awarded ammonia volume (Mtpa), CfD premium ($/ton over JKM-linked reference), supplier-country mix",
            "3. Estimate Pacific-basin blue-NH3 contracted-price floor: reference + CfD premium = effective JERA/Kobe pay-price",
            "4. Cross-check awarded suppliers vs US Gulf project pipeline: CF Blue Point (Donaldsonville LA), OCI Beaumont, Nutrien Geismar, ExxonMobil Baytown",
            "5. If awarded Gulf-export volume >0.5 Mtpa, CF contracted-EBITDA visibility expands 2-3 years forward; sell-side urea-segment models lag by 4-8 weeks",
            "6. Long CF + LIN basket on auction-result day (T+0 publication); short Yara (YAR.OL) as cost-curve loser (EU green-NH3 LCOA >Gulf blue by ~$200/ton)",
            "7. Exit at next CF 10-Q when contracted-volume disclosure hits SEC filing"
        ],
        "data_sources": [
            "METI GX Promotion Act CfD auction-round press releases (free, meti.go.jp)",
            "NEDO Green Innovation Fund H2/NH3 project disbursement tracker (free, nedo.go.jp)",
            "JERA Hekinan Unit-4 NH3 co-firing milestone disclosures (free, jera.co.jp IR)",
            "EIA Weekly Natural Gas storage + spot Henry Hub for blue-NH3 feedstock cost",
            "yfinance for CF, LIN, NTR, YAR.OL daily OHLCV"
        ],
        "originality": 9,
        "bt_feasibility": 3,
        "source_reference": "METI GX Promotion Act (May 2023) and CfD auction framework guidance (FY2024); JERA Hekinan ammonia co-firing roadmap; IEA Ammonia Trade Outlook 2025; ICIS Blue Ammonia Pacific Basin Netback Analysis 2025",
        "research_phase": "pipeline_scout",
        "dedup_key": "japan_gx_cfd_ammonia_cf_lin_pacific_netback"
    },
    {
        "name": "Hydrogen Pure-Play ATM-Equity Issuance Velocity from SEC 424B5 Prints -> Dilution Cascade -> Short PLUG/BE/BLDP/FCEL Basket (Counter to Long-H2 Convergence)",
        "category": "K",
        "asset_class": "single_name_equity",
        "horizon": "2-8 weeks",
        "thesis": "PLUG, BE, BLDP, and FCEL share two structural features: (a) negative operating cash flow with sub-12-month liquidity runways at current burn rates, and (b) active S-3 ASR shelf registrations with at-the-market (ATM) sales agreements with Morgan Stanley / B. Riley / etc. SEC EDGAR exposes ATM-equity take-downs in real time via 424B5 prospectus supplements and 10-Q 'shares issued under sales agreement' disclosures. When the trailing 90-day ATM issuance for any of these names exceeds 5% of float OR aggregate cumulative dilution-adjusted equity raised exceeds $150M, the stock has historically traded -25 to -40% over the following 4-8 weeks as the marginal share supply overwhelms incremental positive narrative. This is a counter-signal precisely because it fires when bullish H2 catalysts (45V final rule, H2Hub milestones, JERA awards in the other two ideas) drive PLUG/BE share-prices higher — managements opportunistically tap the ATM into rallies, which then caps and reverses the move. Existing ideas PL-doe_h2hub_milestone_electrolyzer and PL-doe_hydrogen_shot_electrolyzer_cost_milestone are crowded longs on this same basket; this counter-signal supplies the hedge leg. Trade: short PLUG/BE/BLDP/FCEL basket sized by float-adjusted issuance-velocity z-score; pair with long APD or LIN as positive-FCF industrial-gas beta hedge (which keeps the long-H2 directional view but isolates the dilution-vulnerable subset). Backtest the 424B5 print -> 30/60d forward return on each name 2020-2025 (PLUG has >12 ATM print events with documented -20 to -45% drawdowns each).",
        "causal_chain": [
            "1. Daily scrape SEC EDGAR full-text search for 424B5 filings + 10-Q ATM disclosures for PLUG, BE, BLDP, FCEL, NEL.OL, ITM.L",
            "2. Track cumulative shares issued under each active sales-agreement; compute trailing-90d ATM issuance as % of pre-period float",
            "3. Cross-check 10-Q cash + investments vs trailing-4Q operating cash burn = runway months",
            "4. Signal fires when (runway <12 months) AND (trailing-90d ATM issuance >5% of float) AND (trailing-30d stock return >+15% — i.e., management tapping into rally)",
            "5. Mechanism: incremental share supply at ~5-10% of ADV daily overhang depresses bid/ask for 4-8 weeks; convertible holders delta-hedge by shorting",
            "6. Short basket; size each leg inversely to liquidity-adjusted-volatility; hedge with long APD or LIN at 30% notional",
            "7. Exit at runway-extending event (strategic investor, JV, project-finance close) or at -25% target on basket"
        ],
        "data_sources": [
            "SEC EDGAR full-text search API for 424B5 and 10-Q filings (free, efts.sec.gov)",
            "SEC EDGAR XBRL facts for shares outstanding + cash equivalents per 10-Q",
            "yfinance for PLUG, BE, BLDP, FCEL, APD, LIN daily OHLCV and float",
            "FINRA daily short-volume reports (free) for cross-check on existing crowded short"
        ],
        "originality": 8,
        "bt_feasibility": 5,
        "source_reference": "PLUG 2021-2025 ATM-equity print history (multiple 424B5 supplements with Morgan Stanley sales agreement); BE 2023 ATM utilization (B. Riley sales agreement); academic literature on SEO/ATM dilution drift (Loughran-Ritter 1995; Brophy-Ouimet-Sialm 2009 on PIPE/ATM equity issuance returns)",
        "research_phase": "pipeline_scout",
        "dedup_key": "edgar_atm_424b5_h2_pureplay_dilution_short",
        "counter_signal": True,
        "counters": "long_hydrogen_h2hub"
    },
]

assigned = append_ideas(new_ideas)
print("Assigned IDs:", assigned)
heartbeat("idea_scout")
print("Heartbeat updated.")
