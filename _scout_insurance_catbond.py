import sys
sys.path.insert(0, '/Users/benson/Projects/ekans/pipeline')
from queue_io import append_ideas, heartbeat

new_ideas = [
    {
        "name": "Artemis Cat-Bond Secondary Spread Spike → Reinsurer Premium Hardening Long RNR",
        "category": "G",
        "asset_class": "single_name_equity",
        "horizon": "4-12 weeks",
        "thesis": (
            "When cat-bond secondary market spreads spike (>50bps in 30 days) without a corresponding major "
            "loss event, it signals risk appetite withdrawal from ILS investors, forcing primary insurers to "
            "cede more risk to traditional reinsurers at harder prices. RenaissanceRe (RNR) and Axis Capital (AXS) "
            "capture the premium dislocation as cat-bond capacity exits, expanding their underwriting margins "
            "into the next renewal cycle."
        ),
        "causal_chain": [
            "1. Artemis weekly cat-bond market reports show secondary spread widening >50bps over 30 days",
            "2. ILS / cat-bond fund redemptions or model updates reduce available capacity",
            "3. Primary insurers (State Farm, Allstate, Citizens) face higher retrocession costs or capacity gaps",
            "4. They increase cession ratios to traditional reinsurers, accepting higher premium rates",
            "5. Reinsurer technical margin (rate adequacy) improves; expected combined ratio falls",
            "6. Analysts revise RNR/AXS EPS estimates upward 4-8 weeks before quarterly results",
            "7. RNR and AXS stocks re-rate on improved pricing power signal"
        ],
        "data_sources": [
            "Artemis.bm weekly cat-bond market update (free HTML scrape, spread indices)",
            "Swiss Re ILS market newsletter (quarterly PDF)",
            "SEC EDGAR 10-Q: reinsurer net premium written YoY delta",
            "yfinance: RNR, AXS daily prices"
        ],
        "originality": 8,
        "bt_feasibility": 3,
        "source_reference": (
            "Artemis.bm cat-bond market update; Swiss Re sigma ILS 2024; "
            "Froot & O'Connell (1999) The pricing of US catastrophe reinsurance"
        ),
        "research_phase": "pipeline_scout",
        "dedup_key": "artemis_catbond_spread_widening_reinsurer_premium",
    },
    {
        "name": "CMS MLR Rebate Trigger Breach → MA Insurer EPS Drag Short HUM/ELV",
        "category": "H",
        "asset_class": "single_name_equity",
        "horizon": "2-8 weeks",
        "thesis": (
            "The ACA/Medicare Advantage 85% MLR floor requires insurers to rebate excess profits when "
            "the medical loss ratio falls below threshold. When BLS CPI medical care services rises >0.4% MoM "
            "for two consecutive months and insurer 10-Q disclosures show MLR tracking above 89%, "
            "EPS estimates must be cut sharply. HUM and ELV are most exposed due to their high MA membership "
            "mix; the signal fires before the next earnings print."
        ),
        "causal_chain": [
            "1. BLS CPI medical care services sub-index rises >0.4% MoM for 2 consecutive months",
            "2. CMS releases quarterly MA enrollment data showing membership mix shift toward higher-acuity members",
            "3. Insurer 10-Q interim MLR disclosure or analyst MLR tracker crosses 89%",
            "4. Street consensus EPS must be revised down; buy-side models flag MLR compliance risk",
            "5. HUM and ELV gap down on elevated MLR guidance revision at next earnings",
            "6. Short signal enters on CPI medical-care spike; exit on next earnings release"
        ],
        "data_sources": [
            "BLS CPI medical care services (FRED series CUSR0000SAM)",
            "CMS MA enrollment data: cms.gov statistics-trends-and-reports",
            "SEC EDGAR 10-Q: HUM, ELV quarterly MLR disclosures",
            "yfinance: HUM, ELV daily prices"
        ],
        "originality": 7,
        "bt_feasibility": 4,
        "source_reference": (
            "ACA 42 CFR Part 158 MLR reporting rule; CMS 2024 MA rate notice; "
            "Deutsche Bank health insurance MLR tracker note Oct 2023"
        ),
        "research_phase": "pipeline_scout",
        "dedup_key": "cms_mlr_rebate_trigger_ma_insurer_short",
        "counter_signal": True,
        "counters": "long_SPY,long_quality",
    },
    {
        "name": "Reinsurer ROE Cycle Peak + Price-to-Book Above 1.5x → Counter Short RNR/AXS",
        "category": "K",
        "asset_class": "single_name_equity",
        "horizon": "8-26 weeks",
        "thesis": (
            "Reinsurance stocks historically mean-revert sharply when trailing ROE exceeds 20% and "
            "price-to-book surpasses 1.5x simultaneously, because pricing cycles attract capital inflows "
            "(new Bermuda startups, ILS fund ramp) that compress margins within 2-4 quarters. This "
            "counter-signal shorts RNR and AXS when both conditions are met and catastrophe activity "
            "remains below the 10-year average, signalling a soft-market transition ahead."
        ),
        "causal_chain": [
            "1. Track RNR/AXS P/B ratio via yfinance (quarterly book value from 10-Q vs. market price)",
            "2. Confirm trailing 12-month ROE > 20% from SEC EDGAR filings",
            "3. Check NOAA NCEI billion-dollar disaster count trailing 12 months vs. 10-year average — must be below average",
            "4. Low cat activity + high ROE signals the calm cycle that historically attracts new reinsurance capital",
            "5. Artemis sidecar/startup data confirms new ILS/cat-bond capacity entering the market",
            "6. New capacity compresses renewal rate increases at Jun or Jan renewal; combined ratio guidance rises",
            "7. RNR/AXS re-rate toward 1.0x book; enter short when P/B > 1.5x and cat losses below avg"
        ],
        "data_sources": [
            "yfinance: RNR, AXS price and book value per share",
            "SEC EDGAR 10-K/10-Q: reinsurer ROE and book value per share",
            "NOAA NCEI billion-dollar disaster database (free CSV at ncei.noaa.gov)",
            "Artemis.bm: new ILS sidecar and startup capital announcements"
        ],
        "originality": 8,
        "bt_feasibility": 3,
        "source_reference": (
            "Cummins & Weiss (2009) Convergence of insurance and financial markets; "
            "AM Best reinsurance market hardening/softening cycle report 2023; "
            "Berger Cummins & Tennyson (1992) reinsurance underwriting cycle"
        ),
        "research_phase": "pipeline_scout",
        "dedup_key": "reinsurer_roe_peak_book_premium_short_rnr",
        "counter_signal": True,
        "counters": "long_SPY,long_quality",
    },
]

assigned_ids = append_ideas(new_ideas)
print("Assigned IDs:", assigned_ids)

heartbeat("idea_scout")
print("Heartbeat sent.")
