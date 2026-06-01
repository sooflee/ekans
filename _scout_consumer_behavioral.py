import sys
sys.path.insert(0, '/Users/benson/Projects/ekans/pipeline')
from queue_io import append_ideas, heartbeat

new_ideas = [
    {
        "name": "OpenTable Seated Diners YoY Decline -> Darden Restaurants Short",
        "category": "G",
        "asset_class": "single_name_equity",
        "horizon": "4-8 weeks",
        "thesis": (
            "OpenTable publishes free weekly seated-diners-YoY data for US restaurants. "
            "A rolling 4-week decline in seated diners (below -5% YoY) flags softening "
            "casual dining traffic roughly 6 weeks before same-restaurant-sales guidance "
            "cuts hit earnings. Darden (DRI), with ~1,900 Olive Garden and LongHorn "
            "locations heavily dependent on walk-in casual traffic, is the highest-revenue-"
            "concentration single name exposed."
        ),
        "causal_chain": [
            "OpenTable weekly seated diners YoY drops below -5% for 4 consecutive weeks",
            "Casual dining traffic softness confirmed by multi-week trend (filtering noise from holidays/weather)",
            "Darden Olive Garden and LongHorn cover ~40% of revenue and are most sensitive to casual-dining walk-in traffic",
            "Management typically guides down same-restaurant-sales at next quarterly earnings, 5-8 weeks after data signals",
            "Short DRI 4-8 weeks ahead of earnings; cover after guidance cut"
        ],
        "data_sources": [
            "OpenTable seated diners trends: https://www.opentable.com/state-of-industry",
            "Darden quarterly earnings transcripts (IR site)",
            "FRED: restaurant industry employment as corroborating signal",
            "yfinance: DRI price history"
        ],
        "originality": 7,
        "bt_feasibility": 4,
        "source_reference": (
            "OpenTable State of the Restaurant Industry weekly data (free, public); "
            "Darden IR press releases; Kimes & Beard (2013) on reservation data as demand lead."
        ),
        "research_phase": "pipeline_scout",
        "dedup_key": "opentable_seated_diners_yoy_decline_dri_short",
    },
    {
        "name": "Google Trends Value-Search Dual Surge -> CMG Short (Consumer Squeeze Counter)",
        "category": "F",
        "asset_class": "single_name_equity",
        "horizon": "3-6 weeks",
        "thesis": (
            "When Google Trends search volume for 'dollar menu' AND 'cheap groceries' "
            "simultaneously spike above their trailing 52-week mean by >=1.5 standard "
            "deviations, it signals a broad consumer wallet squeeze that "
            "disproportionately hurts premium fast-casual names. Chipotle (CMG), with "
            "average check ~$15 and no value menu, is the most exposed single QSR to a "
            "value-seeking rotation. Counter-signal to long-consumer-discretionary thesis."
        ),
        "causal_chain": [
            "Google Trends weekly index for 'dollar menu' and 'cheap groceries' both rise >=1.5 stdev above 52-week mean simultaneously",
            "Dual-keyword surge signals aggregate consumer purchasing power stress validated across income cohorts",
            "Value-seeking rotation accelerates traffic away from premium fast-casual (avg ticket $14-16) toward QSR value tiers ($5-8)",
            "CMG lacks a value menu and has above-average check sensitivity; similar traffic declines led to -3 to -5% SSS misses in 2022-23",
            "Short CMG 3-6 weeks before next earnings; exit if trends normalize before then"
        ],
        "data_sources": [
            "Google Trends API (free, public): https://trends.google.com - keywords: dollar menu, cheap groceries, cheap eats",
            "CMG quarterly same-restaurant-sales filings (SEC EDGAR 10-Q)",
            "BLS real personal consumption expenditures for food away from home (FRED: DTXRCX066SFRBDAL proxy)",
            "yfinance: CMG price history"
        ],
        "originality": 8,
        "bt_feasibility": 4,
        "source_reference": (
            "Choi & Varian (2012) Predicting the Present with Google Trends; "
            "Da, Engelberg & Gao (2011) on search-based investor attention; "
            "CMG 10-Q SSS sensitivity disclosures; restaurant traffic analysis by Black Box Intelligence."
        ),
        "research_phase": "pipeline_scout",
        "dedup_key": "google_trends_value_search_dual_surge_cmg_short_counter",
        "counter_signal": True,
        "counters": "long_XLY,long_consumer_discretionary",
    },
    {
        "name": "CarGurus Days-on-Market Surge -> CarMax GPU Compression Short",
        "category": "G",
        "asset_class": "single_name_equity",
        "horizon": "6-12 weeks",
        "thesis": (
            "CarGurus publishes free daily/weekly used-vehicle market data including "
            "median days-on-market (DOM) and retail listing price trends. A "
            "3-consecutive-week rise in national used-car median DOM above the trailing "
            "6-month seasonal norm signals used inventory overhang, which compresses "
            "dealer gross profit per unit (GPU). CarMax (KMX), the largest US independent "
            "used-car retailer, has near-1:1 sensitivity of quarterly GPU to used DOM -- "
            "every +5 day DOM increase historically precedes a $100-200/unit GPU decline "
            "within 1-2 quarters."
        ),
        "causal_chain": [
            "CarGurus used-car median days-on-market rises for 3+ consecutive weeks above 6-month seasonal norm",
            "Rising DOM signals inventory glut: dealers cannot move stock at current ask prices",
            "Dealers reduce listing prices to clear inventory, compressing wholesale and retail margins simultaneously",
            "CarMax GPU (gross profit per used unit) reflects market price declines within 1-2 quarters with ~6-week lag",
            "KMX earnings miss on GPU; consensus estimate revisions follow",
            "Short KMX 6-10 weeks before earnings when DOM signal fires; cover post-earnings"
        ],
        "data_sources": [
            "CarGurus Market Insights (free public data): https://www.cargurus.com/Cars/market-trends",
            "CarMax quarterly earnings releases (IR site): GPU by segment",
            "Manheim Used Vehicle Value Index (Cox Automotive, monthly): corroborating wholesale signal",
            "yfinance: KMX price history"
        ],
        "originality": 8,
        "bt_feasibility": 4,
        "source_reference": (
            "CarGurus Market Insights free weekly data; CarMax 10-Q GPU disclosure history; "
            "Cox Automotive Manheim index; NADA Used Car Guide seasonal DOM patterns."
        ),
        "research_phase": "pipeline_scout",
        "dedup_key": "cargurus_days_on_market_surge_kmx_gpu_short",
    },
]

assigned_ids = append_ideas(new_ideas)
print("Assigned:", assigned_ids)

heartbeat("idea_scout")
print("Heartbeat sent.")
