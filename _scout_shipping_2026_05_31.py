import sys
sys.path.insert(0, '/Users/benson/Projects/ekans/pipeline')
from queue_io import append_ideas, heartbeat

new_ideas = [
    {
        'name': 'STB Weekly Train Speed Collapse -> CSX Long / BNSF-Proxy Short Pair',
        'category': 'K',
        'asset_class': 'single_name_equity',
        'horizon': '4-10 weeks',
        'thesis': (
            'The Surface Transportation Board publishes weekly Class I railroad performance metrics '
            'including average train speed and terminal dwell time. When BNSF train speed drops >1 '
            'standard deviation below its 52-week mean while CSX/UNP remain above their means, shippers '
            'divert time-sensitive intermodal traffic toward better-performing railroads. '
            'Because BRK.B is the pure-play proxy for BNSF, a sustained STB service gap predicts '
            'relative revenue share shift: long CSX vs short BRK.B (rail is ~17% of BRK earnings).'
        ),
        'causal_chain': [
            'STB posts weekly Class I railroad service metrics (train speed, terminal dwell)',
            'BNSF train speed falls >1 stdev below 52-week mean for 3+ consecutive weeks',
            'Large shippers exercise contract optionality or spot-book away from BNSF toward CSX/UNP',
            'CSX revenue ton-miles and intermodal volumes accelerate on diverted traffic',
            'CSX quarterly OR improves relative to consensus while BNSF-proxy BRK.B rail segment lags',
            'CSX re-rates upward on earnings surprise; BRK.B rail contribution drags'
        ],
        'data_sources': [
            'STB Weekly Rail Service Data: https://www.stb.gov/reports-data/rail-service-data/',
            'AAR Weekly Railroad Traffic: https://www.aar.org/data-center/railroads-traffic/',
            'yfinance: CSX, BRK-B for price series',
        ],
        'originality': 7,
        'bt_feasibility': 3,
        'source_reference': (
            'STB Emergency Service Order 2022 filings; AAR Freight Rail Performance Tracker; '
            'FRA Rail Performance Measures quarterly release'
        ),
        'research_phase': 'pipeline_scout',
        'dedup_key': 'stb_speed_gap_class1_pair_csx_brkb',
    },
    {
        'name': 'ACP Panamax Grain Vessel Wait Time Spike -> CBOT Corn/Soy Basis Long',
        'category': 'G',
        'asset_class': 'commodity',
        'horizon': '2-6 weeks',
        'thesis': (
            'The Panama Canal Authority publishes real-time vessel queue and average wait times by '
            'vessel type. When Panamax-tier wait times exceed 10 days, US Gulf-origin grain shipments '
            'to Asia face 2-3 week delivery delays, disrupting merchandiser hedging windows and widening '
            'the US Gulf export basis. This creates a near-term short-covering squeeze in CBOT corn '
            'and soybean futures as buyers seek spot alternatives. Distinct from existing ACP slot '
            'auction → MATX (Jones Act) and ACP → LNG rerouting ideas in the catalog.'
        ),
        'causal_chain': [
            'ACP vessel queue dashboard shows Panamax wait time crossing 10-day threshold',
            'Asian grain importers cannot rely on on-the-water grain arriving on schedule',
            'Importers re-bid for available spot corn/soy or switch to Brazilian origin',
            'CBOT front-month corn/soy futures rise as US export competitiveness briefly improves on forced re-hedging',
            'After 4-6 weeks, bottleneck clears and basis reverts; exit signal = ACP wait time back below 5 days',
        ],
        'data_sources': [
            'Panama Canal Authority vessel queue dashboard: https://www.micanaldepanama.com/en/business/vessel-booking/',
            'ACP Transit Statistics monthly: https://www.micanaldepanama.com/en/about-us/statistics/',
            'USDA GATS export inspections weekly (corn, soybeans): https://apps.fas.usda.gov/gats/',
            'CBOT ZC and ZS futures via yfinance',
        ],
        'originality': 7,
        'bt_feasibility': 3,
        'source_reference': (
            'ACP Gatun Lake Level Weekly Bulletins; USDA AMS Grain Transportation Report weekly; '
            'Fed Reserve Chicago note on canal drought and grain markets 2023'
        ),
        'research_phase': 'pipeline_scout',
        'dedup_key': 'acp_panamax_wait_grain_basis_zc_zs',
    },
    {
        'name': 'Drewry WCI Spot Rate Spike -> Short Import Retailers (XRT Counter-Signal)',
        'category': 'F',
        'asset_class': 'sector_etf',
        'horizon': '6-12 weeks',
        'thesis': (
            'The existing catalog has PL119 (freight rate collapse -> long import retailers) but lacks '
            'a counter-mechanism. When Drewry WCI Asia-to-US composite spot rate spikes >40% in 8 weeks '
            '(historically triggered by Red Sea diversions, port congestion, or pre-tariff pull-forward), '
            'import-dependent specialty retailers face a 2-quarter gross margin squeeze with 6-12 week lag '
            'as booked container spot rates flow through COGS. XRT historically underperforms SPY by '
            '8-12% over the following two quarters. Counter to existing long SPY/equity bias.'
        ),
        'causal_chain': [
            'Drewry WCI Asia-US composite spot rate rises >40% over 8-week rolling window',
            'Specialty retailers that booked 6-month forward containers at spot see COGS guidance raised',
            'Retailers issue gross margin warnings in next earnings cycle (6-10 week lag)',
            'XRT corrects as consensus EPS estimates cut across discretionary retail names',
            'Hedge: short XRT, long XLP (staples less import-exposed, pass-through pricing power)'
        ],
        'data_sources': [
            'Drewry World Container Index weekly: https://www.drewry.co.uk/supply-chain-advisors/supply-chain-expertise/world-container-index-assessed-by-drewry',
            'Freightos Baltic Index (FBX): https://fbx.freightos.com/',
            'yfinance: XRT, XLP for price series',
        ],
        'originality': 7,
        'bt_feasibility': 4,
        'source_reference': (
            'Drewry WCI weekly historical data 2016-present; '
            'FRB San Francisco Supply Chain Disruptions and Inflation 2022; '
            'Goldman Sachs retail sector freight cost sensitivity analysis 2022'
        ),
        'research_phase': 'pipeline_scout',
        'dedup_key': 'drewry_wci_spike_short_xrt_import_retailers',
        'counter_signal': True,
        'counters': 'long_SPY',
    },
]

assigned_ids = append_ideas(new_ideas)
print('Assigned:', assigned_ids)

heartbeat('idea_scout')
print('Heartbeat sent.')
