"""Loop 1 Idea Scout — 2026-05-29 — NTSB / FAA / NHTSA / USCG transportation safety event clusters."""
import sys
sys.path.insert(0, "/Users/benson/Projects/ekans/pipeline")
from queue_io import append_ideas, heartbeat

new_ideas = [
    {
        # Idea 1 — NTSB Major Investigation launch on a single named Class I rail (event-driven,
        # distinct from PL525 which uses FRA rolling derailment rate). Long survivor / short named carrier.
        "name": "NTSB Major Investigation Launch on Named Class I Rail -> Carrier-Specific Overhang (Short Named Rail / Long Peer Basket)",
        "category": "F",
        "asset_class": "single_name_equity",
        "horizon": "3-10 weeks",
        "thesis": "When NTSB publicly designates a 'major investigation' on a specific Class I rail derailment (an explicit NTSB.gov press release, distinct from routine FRA Office of Safety reporting), the named carrier faces a 6-18 month regulatory overhang: STB-Class I service-quality scrutiny, EPA/state environmental liability claims, and shipper contract re-bidding. The market reacts on the headline day but underprices the multi-quarter overhang. PL525 uses the rolling DERAILMENT RATE; this is an EVENT-driven trade on the NTSB major-investigation press release specifically (NTSB issues only 12-25 of these annually). Trade pair: short the named carrier (e.g., NSC after E. Palestine 2023, CSX after Sandusky, UNP after Texas), long an equal-weight basket of the other four Class I rails (UNP/CSX/NSC/CP/CNI ex-named) over 3-10 weeks until either preliminary report (45-day) or first quarterly print.",
        "causal_chain": [
            "1. NTSB issues press release at ntsb.gov/news/press-releases naming a Class I rail and designating 'major investigation' (DCA/RRD docket prefix; ~12-25 events/yr historically)",
            "2. Within 48 hours: state Attorney General opens inquiry, EPA Region office issues Unilateral Administrative Order (UAO) for cleanup at named carrier's expense",
            "3. Class I peer carriers route around named carrier's affected corridor for 30-90 days, gaining incremental ton-miles",
            "4. STB (Surface Transportation Board) opens service-quality docket on named carrier; shipper complaints filed publicly on stb.gov",
            "5. Named carrier's next 10-Q discloses incremental cleanup/legal accrual (NSC accrued $1.1B+ after E. Palestine within 2 quarters)",
            "6. Sell-side rail analysts (Stifel Burk, BofA Ossenbeck) cut named carrier OR while raising peers; pair trade captures 4-9% relative spread",
            "7. Pair entry on NTSB press-release day: short named carrier 1.5%, long equal-weight basket of remaining four Class I rails 1.5%; hold 3-10 weeks until first 10-Q or NTSB preliminary report",
        ],
        "data_sources": [
            "NTSB press releases at ntsb.gov/news/press-releases (free, RSS available)",
            "NTSB CAROL docket search at data.ntsb.gov for major-investigation classification",
            "FRA Office of Safety Analysis equipment-cause incident reports (free, monthly)",
            "STB filings at stb.gov dockets (free)",
            "EPA ECHO database for UAO orders (free)",
            "yfinance (UNP, CSX, NSC, CP, CNI)",
        ],
        "originality": 7,
        "bt_feasibility": 4,
        "source_reference": "NTSB Railroad Investigation Manual; NTSB press releases 2010-2025; NSC E. Palestine OH 2023-02-03 (NSC -10% in 2 weeks, peer rails +3-5%); CSX Sandusky OH 2007; UNP San Antonio 2004 events; Stifel J. Burk rail-safety notes 2023-2024",
        "research_phase": "pipeline_scout",
        "dedup_key": "ntsb_major_investigation_named_rail_pair",
    },
    {
        # Idea 2 — NTSB hazmat tank-car finding driving DOT-105 chlorine/ammonia retrofit demand at TRN/GBX (BUILDERS).
        # Distinct from existing PHMSA pipeline (DOT-117 crude). Different chemical, different tank-car class, different mechanism.
        "name": "NTSB Hazmat Tank-Car Finding on DOT-105 Chlorine/Ammonia Cars -> PHMSA Retrofit Rule -> Trinity/Greenbrier Builder Backlog Surge",
        "category": "F",
        "asset_class": "single_name_equity",
        "horizon": "8-24 weeks",
        "thesis": "When an NTSB investigation report on a hazmat rail derailment cites specific design deficiencies on DOT-105 pressure tank cars (anhydrous ammonia, chlorine, LPG) — e.g., head shield gauge, thermal protection, valve bonnet failure — PHMSA historically follows with a Notice of Proposed Rulemaking (NPRM) within 6-18 months mandating fleet-wide retrofits. The current North American DOT-105 fleet is ~30K cars; a retrofit mandate at $25-60K/car creates a $750M-1.8B retrofit backlog over 3-4 years. Trinity Industries (TRN) and Greenbrier (GBX) are the two surviving North American tank-car builders post-ARI shutdown, plus their lease-fleet arms (TRIP and GBXL) capture ratebook uplift on retrofitted cars. Distinct from existing PHMSA pipeline incident → tank-car play (PL — phmsa_pipeline_incident_crude_rail_tank_car), which covers DOT-117 CRUDE-BY-RAIL cars from pipeline-incident-driven crude-by-rail rebound. This is the DOT-105 PRESSURE car retrofit cycle driven by NTSB chemical-rail findings — entirely different chemical, different tank-car class, different regulatory pathway.",
        "causal_chain": [
            "1. NTSB Railroad Modal investigation publishes final report at ntsb.gov/investigations citing DOT-105 design contribution to release (e.g., 'inadequate thermal protection,' 'head shield failure,' 'valve bonnet bolt fatigue')",
            "2. NTSB issues formal Safety Recommendation R-XX-XX to PHMSA and FRA at ntsb.gov/safety/safety-recs (free, indexed)",
            "3. PHMSA Office of Hazardous Materials Safety opens rulemaking docket on regulations.gov within 6-18 months (NPRM)",
            "4. Final rule mandates retrofit timeline (typical 5-10 year phase-in, accelerated for chlorine/anhydrous ammonia per HM-251 precedent)",
            "5. Tank-car lessors (GATX, TRIP, GBXL) commit retrofit capex; tank-car builders (TRN, GBX) book order surge for new compliant cars and retrofit kits",
            "6. TRN and GBX quarterly backlog rises >10% over 2-4 quarters; FY revenue guidance raised",
            "7. Trade entry on NTSB Safety Recommendation release date: long basket {TRN 1.5%, GBX 1.5%} hold 8-24 weeks until NPRM publication or first backlog beat; hedge with short basket {UNP 0.5%, NSC 0.5%} as marginal-cost passthrough is delayed",
        ],
        "data_sources": [
            "NTSB safety recommendations database at ntsb.gov/safety/safety-recs (free, RSS)",
            "NTSB CAROL investigation docket downloads (free)",
            "PHMSA rulemaking dockets on regulations.gov (free)",
            "FRA equipment cause codes E-series (free)",
            "Railway Supply Institute (RSI) tank-car fleet age/composition reports (free)",
            "yfinance (TRN, GBX, GATX, UNP, NSC)",
        ],
        "originality": 8,
        "bt_feasibility": 3,
        "source_reference": "NTSB Hazmat Accident Reports 2005-2024 (Graniteville SC 2005 chlorine, Minot ND 2002 anhydrous ammonia, Paulsboro NJ 2012 vinyl chloride, E. Palestine 2023 vinyl chloride); PHMSA HM-251 final rule 2015 on crude-oil tank cars as precedent retrofit-rule pathway; TRN/GBX 10-K backlog disclosures 2015-2024",
        "research_phase": "pipeline_scout",
        "dedup_key": "ntsb_hazmat_dot105_retrofit_trn_gbx_builder",
    },
    {
        # Idea 3 — COUNTER-SIGNAL — NHTSA monthly recall mass-release CLUSTER hitting multiple OEMs simultaneously.
        # Counters generic "long autos / long XLY consumer cyclical" convergence. Multi-OEM short basket.
        "name": "NHTSA Monthly Recall Mass-Release Cluster Hitting 3+ Major OEMs -> Auto Sector Counter-Signal (Short F/GM/STLA Basket vs Long XLY-ex-Auto Proxy)",
        "category": "F",
        "asset_class": "single_name_equity",
        "horizon": "4-10 weeks",
        "thesis": "When the NHTSA Recall RSS (nhtsa.gov/recalls) publishes a single-month cluster where THREE OR MORE major OEMs (Ford F, GM, Stellantis STLA, Toyota TM, Honda HMC) each release a recall affecting >250K vehicles, the simultaneous cluster signals (a) shared supplier defect propagation (typical: Takata, ZF, Continental, Bosch electronics) or (b) a regulatory cycle pulling all OEMs into compliance accruals at once. PL20 covers a SINGLE OEM EA opening; PL — `nhtsa_recall_volume_spike_aftermarket_parts` plays the AFTERMARKET LONG side on total recall volume. This counter-signal plays the MULTI-OEM SHORT side on a cluster-month event, hedged with long XLY-equivalent ex-auto (e.g., long HD or LOW as consumer cyclical proxy without auto exposure). Counters the generic 'long autos / long XLY consumer cyclical' convergence already in the catalog. Auto OEMs trade down 4-9% over 4-10 weeks as warranty accrual updates flow into the next two 10-Q prints simultaneously across the sector, vs. only a single OEM in PL20.",
        "causal_chain": [
            "1. NHTSA recall RSS at nhtsa.gov/recalls publishes recalls in real-time; aggregate at calendar-month granularity",
            "2. Monthly screen: count distinct major OEMs (F, GM, STLA, TM, HMC) with at least one recall affecting >250K vehicles in the month",
            "3. Cluster signal fires when month_count >= 3 (historical base rate: ~2-4 cluster months per year, e.g., Takata 2016-2018, Hyundai/Kia ICE fire 2019-2020, Stellantis ABS 2023)",
            "4. Within 4-8 weeks, named OEMs disclose incremental warranty/recall accrual in next 10-Q; consensus EPS revisions tick down 2-6%",
            "5. Shared supplier (Takata, ZF, Continental) defect signals supplier-level liability claims; chain extends to aftermarket parts demand (handled by aftermarket long ideas)",
            "6. NHTSA may open Engineering Analysis on the shared component; STLA/F/GM each face Q2-Q4 recall expansion",
            "7. Trade: short equal-weight basket {F, GM, STLA} 3% gross, long {HD, LOW} 2% gross as consumer-cyclical hedge ex-auto; hold 4-10 weeks until next earnings season prints accruals",
        ],
        "data_sources": [
            "NHTSA Recalls RSS at nhtsa.gov/recalls (free, real-time)",
            "NHTSA datasets at nhtsa.gov/research-data/data-sets (downloadable CSVs of all recall campaigns including affected populations)",
            "OEM 10-Q warranty reserve disclosures (free via SEC EDGAR)",
            "yfinance (F, GM, STLA, TM, HMC, HD, LOW)",
        ],
        "originality": 7,
        "bt_feasibility": 4,
        "source_reference": "NHTSA recall campaign datasets 2005-2025; Takata airbag recall cluster 2016-2019 (affected F, GM, TM, HMC simultaneously, all OEM stocks underperformed XLY by 6-12% over 6 months); Hyundai/Kia engine fire recall cluster 2019-2020; Stellantis ABS/electrical recall cluster 2023; auto OEM warranty reserve disclosure analysis (J.D. Power, Reuters Auto Recalls)",
        "research_phase": "pipeline_scout",
        "dedup_key": "nhtsa_monthly_multi_oem_recall_cluster_counter_auto",
        "counter_signal": True,
        "counters": "long_autos,long_XLY",
    },
]

assigned_ids = append_ideas(new_ideas)
print("Assigned:", assigned_ids)
heartbeat("idea_scout")
print("Heartbeat updated for idea_scout")
