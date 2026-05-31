"""Loop 1 idea scout — cannabis & regulated-substance industries.

Focuses on:
- DEA Schedule III rescheduling / Federal Register milestones (cannabis catalyst)
- NIAAA per-capita alcohol consumption decline (long-term staples short)
- CDC NYTS youth nicotine prevalence + FDA Deeming Rule enforcement (tobacco)

Avoids the existing 'state_cannabis_ballot_initiative_mso_reit' (PL???) and
'AI4_glp1_spirits_short' backtest territory.
"""
import sys
sys.path.insert(0, "/Users/benson/Projects/ekans/pipeline")
from queue_io import append_ideas, heartbeat

new_ideas = [
    {
        # Idea 1 — Cannabis policy catalyst (long MSO + MSOS)
        "name": "DEA Federal Register Rescheduling Milestone -> MSO Equity Rerating",
        "category": "F",
        "asset_class": "single_name_equity",
        "horizon": "2-8 weeks",
        "thesis": (
            "Cannabis MSOs (CURLF, GTBIF, TCNNF, MSOS ETF) trade at deep discounts "
            "to peer consumer-cyclicals because of IRC 280E tax burden (~70% effective tax) "
            "and federal Schedule I status. The DEA's rescheduling process emits discrete, "
            "free Federal Register docket events: NPRM publication, comment period close, "
            "ALJ hearing scheduling, final rule publication. Each milestone produces a "
            "10-30% rerating in the basket as 280E removal probability gets repriced. "
            "Trade the docket event windows because retail flow lags institutional read of "
            "Federal Register filings by 1-3 trading days."
        ),
        "causal_chain": [
            "1. DEA publishes a procedural milestone in Federal Register (free via federalregister.gov API + Reginfo.gov)",
            "2. Milestone shifts implied rescheduling probability (Kalshi/Polymarket contracts move on the same news)",
            "3. Schedule III status removes IRC 280E, restoring ~$0.30 of EBITDA per $1 revenue for vertically integrated MSOs",
            "4. Institutional cannabis funds and arb desks lift MSOS/CURLF/GTBIF/TCNNF within hours",
            "5. Retail flow (Reddit, cannabis subs) follows over 3-10 trading days, completing the rerating leg",
            "6. Mean reversion if no follow-up milestone within 30 days as probability fade sets in"
        ],
        "data_sources": [
            "Federal Register API (free, federalregister.gov/developers/api/v1)",
            "Reginfo.gov OIRA dashboard for OMB review of DEA rules",
            "yfinance for CURLF, GTBIF, TCNNF, MSOS, TLRY, CGC",
            "Kalshi / Polymarket public odds API (free, for probability gauge)"
        ],
        "originality": 8,
        "bt_feasibility": 4,
        "source_reference": (
            "HHS Aug 2023 rescheduling recommendation; DEA NPRM May 2024 (89 FR 44597); "
            "Viridian Capital cannabis equity reports on 280E impact; cannabis industry "
            "analyst notes on milestone trade structure"
        ),
        "research_phase": "pipeline_scout",
        "dedup_key": "dea_federal_register_rescheduling_mso_rerate",
    },
    {
        # Idea 2 — Counter-signal: alcohol consumption decline, short staples
        "name": "NIAAA Per-Capita Alcohol Consumption Decline -> Short Spirits/Beer Majors",
        "category": "C",
        "asset_class": "single_name_equity",
        "horizon": "3-9 months",
        "thesis": (
            "NIAAA's Apparent Per Capita Alcohol Consumption series (free annual + state "
            "monthly tax-receipt proxies) has rolled over since 2022 as Gen Z consumes "
            "~30% less alcohol than millennials at the same age and 'sober curious' / "
            "cannabis substitution accelerates. STZ, DEO, TAP, BUD, SAM all guide volume "
            "growth that anchors on the prior decade's flat trend, not the 2022+ decline. "
            "When state-level alcohol excise receipts (free, monthly from state DORs) print "
            ">5% YoY decline for 3+ consecutive months across the top-10 states, sell-side "
            "models update with a 1-2 quarter lag. Short STZ/DEO/SAM/BUD basket vs long XLP "
            "as the counter-leg captures the staples convergence trade. This is a defensive "
            "counter-signal versus long_staples and long_consumer convergence."
        ),
        "causal_chain": [
            "1. State Department of Revenue alcohol excise receipts published monthly (free, top-10 states)",
            "2. Aggregate 12-month rolling state receipts confirm NIAAA national volume trend with 3-month lead",
            "3. Cross-check with Beer Institute / DISCUS quarterly shipments (free press release data)",
            "4. When 3-month YoY decline > 5% across >=6 of top-10 states, volume miss is structural not cyclical",
            "5. STZ/DEO/SAM/BUD guidance based on prior decade trend overshoots actual demand",
            "6. Earnings revisions cycle compresses multiples; short basket outperforms XLP long leg by 4-9% over 6 months"
        ],
        "data_sources": [
            "NIAAA Apparent Per Capita Alcohol Consumption (free, niaaa.nih.gov/alcohols-effects-health/alcohol-topics/alcohol-facts-and-statistics/apparent-per-capita-alcohol-consumption)",
            "State DOR alcohol excise tax receipts (CA, TX, NY, FL, PA, IL, OH, GA, NC, MI -- free)",
            "Beer Institute monthly shipments (free press release)",
            "DISCUS economic briefing (free annual + quarterly)",
            "yfinance STZ, DEO, TAP, BUD, SAM, XLP"
        ],
        "originality": 7,
        "bt_feasibility": 4,
        "source_reference": (
            "NIAAA Surveillance Report #119 (2022 declining trend); IWSR US Beverage Alcohol "
            "Review showing -3% volume 2023; Gallup 2023 youth alcohol survey (sub-30 "
            "consumption -30% vs 2002); Berenberg sell-side note on staples vol miss risk"
        ),
        "research_phase": "pipeline_scout",
        "dedup_key": "niaaa_alcohol_volume_decline_staples_short",
        "counter_signal": True,
        "counters": "long_staples,long_consumer",
    },
    {
        # Idea 3 — Tobacco / nicotine regulatory pressure
        "name": "CDC NYTS Youth Nicotine + FDA Deeming Rule Enforcement -> Tobacco Margin Compression",
        "category": "F",
        "asset_class": "single_name_equity",
        "horizon": "2-6 months",
        "thesis": (
            "FDA publishes Marketing Granted Orders (MGOs) and Marketing Denial Orders (MDOs) "
            "for tobacco products under the Deeming Rule (free via FDA Tobacco Product "
            "Marketing Orders database). MDO clusters on flavored vape/e-cig SKUs remove "
            "~20-35% of US e-cig category volume from market within 30 days as retailers "
            "destock to avoid civil penalties. MO (Njoy), BTI (Vuse), and Altria/PMI cigarette "
            "volumes face two competing forces: (a) MDO-driven category contraction lifts "
            "cigarette mix back to MO/BTI (margin positive), (b) CDC NYTS annual youth nicotine "
            "prevalence trigger leads to FDA proposed nicotine reduction cap rule, which "
            "would impair cigarette pricing power. Trade is: long MO/BTI on MDO cluster events "
            "(short-term mix lift) AND short MO/BTI on CDC NYTS prevalence > 9% threshold "
            "(triggers FDA nicotine cap activity within 90 days). This is two distinct "
            "regulatory-event signals on the same names with opposite signs."
        ),
        "causal_chain": [
            "1. FDA publishes Marketing Denial Order (MDO) list weekly (free, fda.gov/tobacco-products)",
            "2. When >=5 MDOs hit flavored e-cig SKUs in a 30-day window, distributors destock to avoid penalties",
            "3. E-cig category volume falls 20-35%, cigarette category gains 1-2% as substitution",
            "4. MO/BTI Q+1 earnings beat on cigarette volume mix (long signal, 8-12 weeks)",
            "5. Separately: CDC NYTS annual report (Oct release) -- youth current nicotine use crosses 9%",
            "6. FDA opens Advance Notice of Proposed Rulemaking (ANPRM) on nicotine cap (historical precedent: 2018 ANPRM, 2022 reproposal)",
            "7. MO/BTI multiples compress 8-15% on ANPRM publication as pricing-power thesis breaks (short signal)"
        ],
        "data_sources": [
            "FDA Tobacco Product Marketing Orders database (free, ctp-tpms.fda.gov)",
            "Federal Register API for ANPRM / NPRM publications",
            "CDC National Youth Tobacco Survey annual (free, cdc.gov/tobacco/data-statistics/surveys/nyts)",
            "Nielsen tobacco scanner data (paid, but quarterly press summaries are free in MO/BTI earnings decks)",
            "yfinance MO, PM, BTI, JUUL-adjacent names"
        ],
        "originality": 8,
        "bt_feasibility": 3,
        "source_reference": (
            "FDA Deeming Rule (81 FR 28973); FDA Premarket Tobacco Applications enforcement "
            "memos 2023-2025; CDC NYTS 2024 report (youth e-cig prevalence 7.8%); "
            "Truth Initiative tracking of MDO impact on category volume"
        ),
        "research_phase": "pipeline_scout",
        "dedup_key": "fda_mdo_cluster_cdc_nyts_tobacco_margin_dual_signal",
    },
]

assigned = append_ideas(new_ideas)
heartbeat("idea_scout")
print("ASSIGNED:", assigned)
