import sys
sys.path.insert(0, '/Users/benson/Projects/ekans/pipeline')
from queue_io import append_strategies, update_idea_status
from datetime import datetime, timezone

strategies = [
    {
        'strategy_id': 'PL954',
        'idea_id': 'PL954',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'ready',
        'signal_id': 'PL954_mida_ee_approvals_ewm_long',
        'name': 'Malaysia MIDA E&E Investment Surge -> EWM Long vs SOXX Pair',
        'category': 'F',
        'asset_class': 'single_name_equity',
        'horizon': '6-12 weeks',
        'rule': (
            'When MIDA quarterly E&E-sector approved investment (MYR or USD equivalent) '
            'exceeds the trailing 4-quarter rolling mean by >= 1.5 standard deviations, '
            'go long EWM and short SOXX in equal notional (1:1 pair) at the next open following '
            'the MIDA report release (typically published 4-6 weeks after quarter-end). '
            'Hold for 10 calendar weeks (50 trading days). '
            'Exit early if EWM/SOXX ratio drawdown exceeds 8% from entry.'
        ),
        'tickers': ['EWM', 'SOXX', 'SPY'],
        'entry_conditions': [
            'MIDA quarterly E&E approved investment exceeds trailing-4Q rolling mean by >= 1.5 stdev',
            'Entry at next open after MIDA report publication date',
            'Long EWM, Short SOXX in equal notional'
        ],
        'exit_conditions': [
            'Hold 50 trading days (10 calendar weeks)',
            'Stop-loss: EWM/SOXX ratio drawdown >= 8% from entry'
        ],
        'data_sources_concrete': {
            'prices': 'EWM, SOXX, SPY via yfinance',
            'fundamental': 'MIDA Quarterly Investment Performance Report (mida.gov.my, free PDF); manually extract E&E approvals column 2010-2025'
        },
        'backtest_approach': (
            'Event-study PnL: identify MIDA E&E approval surprise dates from quarterly PDF releases '
            '(approx Mar, Jun, Sep, Dec each year). Compute trailing-4Q mean and stdev of E&E approvals. '
            'Flag events where approval > mean + 1.5*stdev. '
            'Measure EWM-SOXX pair return over next 50 trading days from event date. '
            'Expected ~8-12 events 2010-2025. Report mean/median pair return, hit rate, Sharpe on pair leg.'
        ),
        'known_events': [
            '2010-Q1 post-GFC E&E recovery surge',
            '2013-Q2 Penang corridor Phase 2 approvals',
            '2017-Q4 Intel Penang expansion approval',
            '2022-Q2 post-COVID E&E capex rebound',
            '2023-Q3 advanced packaging surge (CoWoS supply push)'
        ],
        'implementation_notes': (
            'Inari Amertron (0166.KL) and Unisem (5005.KL) are on Bursa and have yfinance coverage '
            'but thin volume; EWM is the liquid US-listed proxy (~35% E&E weight). '
            'KLIP listed in idea sources is NOT Malaysia exposure - ignore it. '
            'MIDA PDFs require manual extraction or a simple PDF table parser; '
            'backtester should note that MIDA reports are sometimes restated. '
            'bt_feasibility=3: data extraction effort is moderate but doable; '
            'only ~8-12 signal events in 15 years limits statistical power.'
        ),
        'originality': 8,
        'bt_feasibility': 3
    },
    {
        'strategy_id': 'PL955',
        'idea_id': 'PL955',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'ready',
        'signal_id': 'PL955_gso_vn_vehicle_exports_vnm_long',
        'name': 'Vietnam GSO HS-8703 Vehicle Export Acceleration -> Long VNM vs Short F+GM (EV Counter-Signal)',
        'category': 'D',
        'asset_class': 'equities',
        'horizon': '4-8 weeks',
        'rule': (
            'When Vietnam GSO monthly HS-8703 (passenger vehicles) export value shows '
            '2-month rolling YoY growth acceleration > 1 standard deviation above trailing '
            '12-month mean growth, go long VNM and short an equal-weighted basket of F + GM '
            '(50% each, equal notional to VNM position) at the next open after GSO release. '
            'Hold for 6 calendar weeks (30 trading days). '
            'Exit early if VNM vs (F+GM avg) relative drawdown exceeds 10% from entry.'
        ),
        'tickers': ['VNM', 'F', 'GM', 'VFS', 'SPY'],
        'entry_conditions': [
            'Vietnam GSO monthly HS-8703 export value 2-month rolling YoY growth > trailing-12M mean + 1 stdev',
            'Entry at next open following GSO monthly statistical release (typically released 29th of following month)',
            'Long VNM, Short equal-weighted F+GM'
        ],
        'exit_conditions': [
            'Hold 30 trading days (6 calendar weeks)',
            'Stop-loss: VNM vs F+GM basket relative drawdown >= 10% from entry'
        ],
        'data_sources_concrete': {
            'prices': 'VNM, F, GM, VFS via yfinance; SPY as benchmark',
            'fundamental': 'Vietnam GSO Monthly Statistical Information (gso.gov.vn), Motor Vehicle Exports table HS-8703'
        },
        'backtest_approach': (
            'Event-study PnL: build monthly time series of Vietnam HS-8703 vehicle export value (USD) '
            'from GSO monthly releases 2010-2025 (annual PDFs freely available; monthly since 2018). '
            'Compute 2-month rolling YoY growth; flag months where growth > trailing-12M mean + 1 stdev. '
            'Measure VNM minus (F+GM)/2 excess return over next 30 trading days from signal date. '
            'VFS only has data from 2021 so use VNM as primary long. '
            'Counter-signal: explicitly structured as a counter to long-US-autos thesis.'
        ),
        'known_events': [
            '2022-07 VinFast production ramp announcement, Vietnam exports surge',
            '2023-08 VinFast US IPO month, export surge precursor',
            '2024-Q1 VinFast US delivery push, GSO HS-8703 spike'
        ],
        'implementation_notes': (
            'This is a counter_signal strategy (counters: long_US_autos, long_F_GM); '
            'VNM and F/GM as broad basket targets are appropriate per the spec. '
            'VFS (VinFast) IPO was Aug 2023; include as supplemental long alongside VNM for post-2023 window. '
            'GSO monthly data is available in Vietnamese; key table is motor vehicle exports by HS code. '
            'Pre-2018 monthly granularity is lower; use quarterly data for 2010-2017 period. '
            'bt_feasibility=3: Vietnamese PDF parsing is effort-intensive; '
            'Cox Automotive free monthly summary provides US EV inventory cross-check.'
        ),
        'originality': 8,
        'bt_feasibility': 3
    }
]

append_strategies(strategies)
print('Batch 1 strategies appended successfully')

update_idea_status('PL954', 'developed')
print('PL954 marked developed')

update_idea_status('PL955', 'developed')
print('PL955 marked developed')
