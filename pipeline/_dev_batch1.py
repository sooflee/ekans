import sys
sys.path.insert(0, '/Users/benson/Projects/ekans/pipeline')
from queue_io import append_strategies, update_idea_status, heartbeat
from datetime import datetime, timezone

strategies = [
    {
        'strategy_id': 'PL768',
        'idea_id': 'PL768',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'ready',
        'signal_id': 'PL768_jmmc_dispersion_bno_long',
        'name': 'JMMC Analyst Dispersion -> BNO Long Pre-Meeting',
        'category': 'G',
        'asset_class': 'commodity',
        'horizon': '3-10 days',
        'rule': (
            'Using OPEC JMMC meeting calendar (fixed 4-6 weeks in advance), identify meeting date M. '
            'On day M-5 (5 trading days before meeting), go long BNO (Brent ETF). '
            'Exit at close of day M+3 (3 trading days after meeting). '
            'Only enter if pre-meeting Reuters/S&P Platts analyst survey shows stddev of expected '
            'OPEC output change >= 300 kb/d (high dispersion threshold). '
            'Cap position size at 2% NAV. '
            'Hard stop-loss: exit early if BNO falls more than 3% from entry.'
        ),
        'tickers': ['BNO', 'BZ=F', 'SPY'],
        'entry_conditions': [
            'JMMC meeting date identifiable 5+ trading days in advance from OPEC.org calendar',
            'Pre-meeting analyst survey dispersion (stddev of expected quota change) >= 300 kb/d',
            'Enter long BNO at close on day M-5'
        ],
        'exit_conditions': [
            'Exit at close on day M+3 (3 trading days after meeting)',
            'Hard stop: exit early if BNO closes -3% below entry price',
            'Force exit if meeting is cancelled or postponed to unknown date'
        ],
        'data_sources_concrete': {
            'prices': 'BNO (iPath Bloomberg Brent Crude ETN, yfinance); BZ=F (Brent front-month futures, yfinance)',
            'fundamental': 'none - OPEC JMMC calendar from opec.org; Reuters/S&P Platts survey dispersion from press archives'
        },
        'backtest_approach': (
            'Event-study: identify all JMMC meeting dates 2020-2025 (~12-15 events). '
            'For each event, compute BNO return from M-5 close to M+3 close. '
            'Separate into high-dispersion (stddev >= 300 kb/d) vs low-dispersion sub-groups. '
            'Key metric: mean return and hit rate for high-dispersion events. '
            'Benchmark vs buy-and-hold BNO over same date windows. '
            'Analog dates: 2022-10-05 OPEC+ surprise 2mb/d cut, 2023-04-02 surprise voluntary cuts, '
            '2024-06 baseline quota extension. '
            'Note: survey dispersion data requires manual annotation from press archives; '
            'backtester should use meeting surprise (actual vs median consensus) as proxy if needed.'
        ),
        'known_events': [
            '2022-10-05',
            '2023-04-02',
            '2023-11-26',
            '2024-06-02',
            '2024-11-03',
            '2025-02-02',
            '2025-05-04'
        ],
        'implementation_notes': (
            'Survey dispersion data is not programmatically free; backtester should treat meeting as '
            'high dispersion if |actual output change - median pre-meeting forecast| > 200 kb/d as proxy. '
            'BZ=F (futures) can substitute for BNO for longer history. '
            'JMMC calendar is advisory; actual OPEC+ ministerial meetings may differ - cross-reference.'
        ),
        'originality': 7,
        'bt_feasibility': 4
    },
    {
        'strategy_id': 'PL769',
        'idea_id': 'PL769',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'ready',
        'signal_id': 'PL769_opec_cut_stng_long_fro_short',
        'name': 'OPEC+ Surprise Cut -> Long STNG / Short FRO Product-vs-Crude Tanker Pair',
        'category': 'J',
        'asset_class': 'single_name_equity',
        'horizon': '2-6 weeks',
        'rule': (
            'On the day OPEC+ announces a surprise voluntary production cut (actual cut > 200 kb/d beyond '
            'consensus expectation), initiate a market-neutral pair trade: '
            'Go long STNG (Scorpio Tankers, product tanker fleet) and short FRO (Frontline, VLCC/crude fleet) '
            'in equal dollar notional. '
            'Hold for 30 calendar days or until STNG/FRO spread reverts to pre-event level, whichever first. '
            'Exit at 30-day mark or if pair spread moves adverse by >8% from entry.'
        ),
        'tickers': ['STNG', 'FRO', 'INSW', 'SPY'],
        'entry_conditions': [
            'OPEC+ official press release announces aggregate voluntary cut exceeding 200 kb/d above Bloomberg/Reuters median consensus',
            'Enter at next trading day open (T+1) after announcement date',
            'Equal dollar notional: long STNG, short FRO'
        ],
        'exit_conditions': [
            'Exit both legs at 30-calendar-day mark from entry',
            'Stop-loss: exit if STNG underperforms FRO by more than 8% from entry (pair spread -8%)',
            'Profit target: exit if STNG outperforms FRO by 15% from entry (pair spread +15%)'
        ],
        'data_sources_concrete': {
            'prices': 'STNG (Scorpio Tankers, yfinance); FRO (Frontline plc, yfinance); INSW (International Seaways, supplemental); SPY (benchmark)',
            'fundamental': 'none - OPEC announcement dates from opec.org press releases; Baltic BDTI/BCTI weekly summaries free at balticexchange.com'
        },
        'backtest_approach': (
            'Event-study on OPEC+ surprise cuts 2018-2025. Identify announcement dates where actual cut '
            'exceeded median pre-meeting Reuters/Platts consensus by >200 kb/d. '
            'For each event, compute STNG - FRO pair return (long-short, equal notional) from T+1 open '
            'over T+1 to T+30 calendar days. '
            'Compare vs control periods (OPEC+ meetings with no surprise). '
            'Analog dates: 2023-04-03 (day after 2023-04-02 voluntary cut announcement), '
            '2022-10-06 (day after Oct-2022 2mb/d cut), 2020-04-13 (historic 9.7mb/d cut). '
            'Dollar-neutral so no net beta exposure; isolate the product-vs-crude tanker divergence.'
        ),
        'known_events': [
            '2020-04-13',
            '2020-06-06',
            '2022-10-06',
            '2023-04-03',
            '2023-11-27',
            '2024-06-03',
            '2025-05-05'
        ],
        'implementation_notes': (
            'STNG IPO was ~2010 but thin liquidity pre-2018; use 2018 onward for clean history. '
            'FRO listed on NYSE since 2019 as restructured entity (post-merger with Golden Ocean spin); '
            'pre-2019 FRO data on yfinance may not reflect the current entity - treat cautiously. '
            'INSW can serve as alternative crude/dirty tanker short leg if FRO data is spotty. '
            'This is a counter-signal to the consensus long-oil-equities trade on OPEC cuts. '
            'Pair returns will be volatile on thin tanker equity liquidity; size accordingly (<1.5% NAV each leg).'
        ),
        'originality': 9,
        'bt_feasibility': 4
    }
]

append_strategies(strategies)
print('Strategies appended.')

update_idea_status('PL768', 'developed')
print('PL768 marked developed.')

update_idea_status('PL769', 'developed')
print('PL769 marked developed.')

heartbeat('strategy_developer')
print('Heartbeat sent.')
