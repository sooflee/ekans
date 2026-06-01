import sys
sys.path.insert(0, '/Users/benson/Projects/ekans/pipeline')
from queue_io import append_ideas, heartbeat

new_ideas = [
    {
        "name": "ADP Small-Business Payroll Decel -> Consumer-Cyclical Short (DRI)",
        "category": "G",
        "asset_class": "single_name_equity",
        "horizon": "4-8 weeks",
        "thesis": (
            "ADP National Employment Report small-business component (1-49 employees) "
            "is a leading indicator for discretionary consumer spending: small-biz payrolls "
            "lag large-biz by ~3 weeks and skew toward service, retail, and hospitality workers. "
            "When 3-month YoY growth in ADP small-biz payrolls turns negative, full-service "
            "restaurant revenue follows 4-6 weeks later, compressing DARDEN (DRI) same-store "
            "sales guidance and pressuring the stock."
        ),
        "causal_chain": [
            "ADP publishes monthly NER; small-biz (<50 employees) payroll growth goes negative YoY for 2+ consecutive months",
            "Hospitality/food-service workers are disproportionately employed at small businesses; income shock hits lower-quintile consumers first",
            "Full-service restaurant traffic (OpenTable covers) declines 4-6 weeks after payroll miss",
            "DARDEN (DRI) guidance risk rises; consensus same-store sales estimates get cut ahead of quarterly print",
            "Short DRI or buy DRI puts ~4 weeks after ADP small-biz signal fires"
        ],
        "data_sources": [
            "ADP National Employment Report (monthly, adpemploymentreport.com)",
            "OpenTable dining reservations index (public weekly)",
            "FRED PAYEMS for cross-validation",
            "DRI SEC filings (10-Q SSS guidance)"
        ],
        "originality": 7,
        "bt_feasibility": 4,
        "source_reference": (
            "ADP Research Institute Small Business Report series; "
            "Bernstein 2023 note on low-income consumer spending sensitivity to payroll timing"
        ),
        "research_phase": "pipeline_scout",
        "dedup_key": "adp_small_biz_payroll_decel_consumer_cyclical",
    },
    {
        "name": "Continuing Claims 4-Wk MA Inflection -> HYG Consumer-Sector Short (Counter-Signal)",
        "category": "H",
        "asset_class": "rates_bonds",
        "horizon": "6-12 weeks",
        "thesis": (
            "When 4-week moving average of continuing unemployment claims inflects upward by >=5% "
            "from a 12-month trough, consumer-sector high-yield credit spreads widen 60-120 bps "
            "within 8 weeks, as analysts reprice refinancing risk for leveraged restaurant, retail, "
            "and hotel names. This is a counter-signal against the current long-consumer-equity "
            "and long-HYG consensus: the labor market softening reaches HY credit ~6 weeks before "
            "equity analysts downgrade."
        ),
        "causal_chain": [
            "DOL weekly UI data: continuing claims 4-week MA rises >=5% from trailing 12-month low",
            "Continuing claims spike concentrated in leisure, hospitality, and retail NAICS codes (DOL state-level breakdown)",
            "Consumer-facing leveraged issuers see rising interest coverage risk; CLO managers flag retail/restaurant exposure",
            "HY consumer-sector OAS widens 60-120 bps within 8 weeks",
            "Express short HYG or long HYG puts; tighter expression: short consumer HY names via CDX"
        ],
        "data_sources": [
            "DOL weekly claims (FRED: CCSA for continuing, IC4WSA for 4-week MA)",
            "DOL state-level claims by industry (ETA 5159 reports)",
            "ICE BofA US High Yield Consumer Sector Index (FRED BAMLHY series)",
            "FINRA TRACE for HY bond spread proxies"
        ],
        "originality": 7,
        "bt_feasibility": 4,
        "source_reference": (
            "Gilchrist & Zakrajsek (2012) credit-spread leading indicator; "
            "Moody's Default & Recovery Database: consumer-sector HY defaults lag claims inflections by 1-2 quarters"
        ),
        "research_phase": "pipeline_scout",
        "dedup_key": "continuing_claims_ma_inflection_hy_consumer_short",
        "counter_signal": True,
        "counters": "long_HYG,long_consumer_equity",
    },
    {
        "name": "WARN Act Tech-Sector Cluster -> H-1B Dependent Employer Attrition -> KELYA Short",
        "category": "G",
        "asset_class": "single_name_equity",
        "horizon": "8-16 weeks",
        "thesis": (
            "Large-scale WARN Act tech layoffs reduce the H-1B dependent employer roster: when an "
            "employer lays off >50 US workers they must seek additional attestations for H-1B renewals "
            "(INA 212(n)(2)(C)(ii)), and many simply allow visa slots to lapse. "
            "Staffing firms that place H-1B workers (Kelly Services KELYA, ManpowerGroup MAN) "
            "see a double-hit: (1) placement volume falls as tech clients pause hiring, and "
            "(2) H-1B worker placement premiums compress as supply of visa-holding candidates rises. "
            "A 13-week rolling WARN Act tech-sector layoff count above 15,000 is the trigger."
        ),
        "causal_chain": [
            "DOL WARN Act database: rolling 13-week tech-sector (NAICS 51xx/54xx) layoffs exceed 15,000 employees",
            "USCIS LCA filings for H-1B cap-exempt renewals drop in same NAICS codes (iCert portal, public quarterly)",
            "H-1B dependent staffing firms (KELYA, MAN) lose both volume and margin on tech placements; utilization rates fall",
            "Quarterly billings guidance from KELYA/MAN misses consensus; sell-side cuts estimates",
            "Short KELYA or MAN 8-12 weeks after WARN trigger; exit before earnings if signal has dissipated"
        ],
        "data_sources": [
            "DOL WARN Act notice database (public CSV, updated monthly, dol.gov)",
            "USCIS iCert LCA performance data (quarterly public disclosure)",
            "KELYA and MAN quarterly earnings transcripts (SEC EDGAR)",
            "BLS JOLTS professional/business services quits and hires (monthly)"
        ],
        "originality": 8,
        "bt_feasibility": 3,
        "source_reference": (
            "INA Section 212(n)(2)(C) H-1B dependent employer layoff attestation rules; "
            "Bound Khanna & Morales (2017) NBER WP on H-1B and tech labor market dynamics"
        ),
        "research_phase": "pipeline_scout",
        "dedup_key": "warn_tech_cluster_h1b_dependent_it_staffing_short",
    },
]

assigned = append_ideas(new_ideas)
print("Assigned IDs:", assigned)

heartbeat("idea_scout")
print("Heartbeat OK")
