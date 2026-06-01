import sys
sys.path.insert(0, '/Users/benson/Projects/ekans/pipeline')
from queue_io import append_strategies, update_idea_status, heartbeat
from datetime import datetime, timezone

strategies = [
    {
        'strategy_id': 'PL890',
        'idea_id': 'PL774',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'ready',
        'signal_id': 'PL774_pla_optempo_ewt_short_lmt_rtx_long',
        'name': 'PLA Taiwan Strait OPTEMPO Surge -> Short EWT, Long LMT/RTX Pair',
        'category': 'G',
        'asset_class': 'single_name_equity',
        'horizon': '5-20 trading days',
        'rule': (
            'Trigger: Taiwan MND daily PLA activity count (aircraft + naval vessels crossing median line) '
            '5-day rolling z-score > +2.0 stdev vs trailing 252-day baseline, AND a named PLA exercise '
            '(Joint Sword-type) officially announced by Eastern Theater Command. '
            'On the day both conditions are met, initiate: short EWT (iShares MSCI Taiwan ETF) '
            'and long equal-weight LMT + RTX basket (Lockheed Martin + Raytheon Technologies). '
            'Dollar-neutral: $1 short EWT per $0.50 long LMT + $0.50 long RTX. '
            'Exit on PLA exercise stand-down announcement OR US-PRC senior-level de-escalation '
            'communique, OR at 20 trading days, whichever comes first. '
            'Stop: exit entire pair if EWT rises 4% from entry (short squeeze risk).'
        ),
        'tickers': ['EWT', 'LMT', 'RTX', 'XAR', 'ITA', 'SPY'],
        'entry_conditions': [
            'Taiwan MND 5-day PLA activity z-score > +2.0 stdev vs trailing 252-day window',
            'Named PLA exercise officially announced by PLA Eastern Theater Command (Xinhua/CCTV)',
            'Enter short EWT + long LMT/RTX (equal-weight, dollar-neutral) at next-day open'
        ],
        'exit_conditions': [
            'Exit on PLA Eastern Theater Command exercise-conclusion announcement',
            'Exit on US-PRC senior-level de-escalation readout (State Dept / White House)',
            'Exit at 20-trading-day maximum hold period',
            'Stop-loss: exit if EWT rises 4% from entry price (covers short leg)'
        ],
        'data_sources_concrete': {
            'prices': 'EWT (iShares MSCI Taiwan ETF, yfinance); LMT (Lockheed Martin, yfinance); RTX (Raytheon Technologies, yfinance); XAR/ITA (defense ETFs, benchmark); SPY (broad benchmark)',
            'fundamental': 'FRED: none. Event dates from Taiwan MND PLA activity tracker (mnd.gov.tw); named exercises from PLA Eastern Theater Command Xinhua archive.'
        },
        'backtest_approach': (
            'Event-study on Taiwan Strait escalation episodes 2020-2025. '
            'Key analog dates: 2022-08-02 Pelosi visit (Joint Sword-precursor drills, 5-day window ~Aug 4-9), '
            '2024-05-23 Joint Sword-2024A (post-Lai Ching-te inauguration), '
            '2021-06 ADIZ incursion cluster (25+ aircraft), '
            '2023-04 (post-Tsai-McCarthy meeting drills). '
            'For each event, compute: (1) EWT return from T+1 to T+N (N=5,10,20), '
            '(2) LMT/RTX equal-weight return over same window, '
            '(3) pair return (long LMT+RTX, short EWT). '
            'Control: non-escalation OPEC+ calendar periods same length. '
            'EWT z-score proxy: use daily ADIZ incursion counts published by Taiwan MND; '
            'backtester can hardcode 5-10 known event dates if scraping is infeasible.'
        ),
        'known_events': [
            '2021-06-15',
            '2022-08-04',
            '2023-04-10',
            '2024-05-23'
        ],
        'implementation_notes': (
            'Taiwan MND data is publicly available but requires web scraping of mnd.gov.tw tables; '
            'backtester should hardcode the 4 known event onset dates above for the initial event study. '
            'Only 4 clean events in the 2020-2025 window - results will have wide confidence intervals; '
            'treat as directional evidence not statistical significance. '
            'EWT FX effect: TWD/USD move is embedded in EWT returns (both equity + FX channel). '
            'LMT history is clean from 2020; RTX was formed 2020-04-03 (Raytheon/UTC merger). '
            'Use RTX data from 2020-04-06 onward; for pre-merger proxy use RTN (old Raytheon). '
            'Defense sector re-rate may lag 10-15 days as sell-side updates FMS pipeline estimates.'
        ),
        'originality': 8,
        'bt_feasibility': 4
    },
    {
        'strategy_id': 'PL775',
        'idea_id': 'PL775',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'ready',
        'signal_id': 'PL775_taiwan_cable_cluster_irdm_asts_ewt',
        'name': 'Taiwan Strait Subsea Cable Severance Cluster -> Long IRDM/ASTS, Short EWT',
        'category': 'G',
        'asset_class': 'single_name_equity',
        'horizon': '10-60 trading days',
        'rule': (
            'Trigger: 2 or more Taiwan-region submarine cable cuts within a 14-day window '
            'confirmed by TeleGeography or Taiwan MODA, AND Chunghwa Telecom declares '
            'restoration ETA > 60 days (vessel-access blocked). '
            'On trigger day, enter: long IRDM (Iridium) + ASTS (AST SpaceMobile) equal-weight, '
            'short EWT (iShares MSCI Taiwan ETF). '
            'Dollar sizing: $1 EWT short per $0.50 IRDM + $0.50 ASTS long. '
            'Exit when TeleGeography reports cable RFS (return-to-full-service) on the severed cable '
            'OR PLAN exclusion zone lifted OR at 60-calendar-day maximum, whichever comes first. '
            'Stop: exit if EWT rises 5% from entry or IRDM+ASTS basket falls 7% from entry.'
        ),
        'tickers': ['IRDM', 'ASTS', 'EWT', 'CHT', 'SPY'],
        'entry_conditions': [
            '>=2 Taiwan-region cable severance events within 14 calendar days per TeleGeography/MODA',
            'Chunghwa Telecom or MODA confirms restoration ETA >60 days (repair vessel blocked)',
            'Enter long IRDM+ASTS, short EWT at next-day open after trigger confirmation'
        ],
        'exit_conditions': [
            'Exit on TeleGeography cable RFS update for the affected cable(s)',
            'Exit on confirmed PLAN exclusion-zone lift or repair vessel access granted',
            'Maximum hold: 60 calendar days',
            'Stop-loss: exit if EWT +5% from entry or long basket -7% from entry'
        ],
        'data_sources_concrete': {
            'prices': 'IRDM (Iridium Communications, yfinance); ASTS (AST SpaceMobile, yfinance); EWT (iShares MSCI Taiwan ETF, yfinance); CHT (Chunghwa Telecom ADR, yfinance); SPY (benchmark)',
            'fundamental': 'none - TeleGeography Submarine Cable Map (free, manual); Taiwan MODA press releases; Chunghwa Telecom (CHT) IR press releases'
        },
        'backtest_approach': (
            'Event-study with limited analog set 2023-2025 due to novel gray-zone pattern. '
            'Known cluster events: 2023-02-02 Matsu #2/#3 cable cut (restoration ~50 days), '
            '2024-12 Taiwan-Penghu cable damage cluster (if confirmed by TeleGeography). '
            'For each event, compute IRDM return, ASTS return (from 2022 IPO), and EWT return '
            'from T+1 to T+10, T+20, T+60. '
            'Pair return = (IRDM+ASTS equal-weight) - EWT. '
            'ASTS has data only from Jan 2022; use IRDM-only for pre-2022 events as proxy. '
            'Note: event count is very small (2-3 events); backtester should present '
            'individual event P&L rather than statistical aggregates. '
            'Chunghwa Telecom (CHT) can serve as additional long leg for cable-disruption demand.'
        ),
        'known_events': [
            '2023-02-02',
            '2024-12-01'
        ],
        'implementation_notes': (
            'ASTS listed Jan 2022 (SPAC merger); limited history means only post-2022 events usable for ASTS. '
            'Backtester should run IRDM-only variant (longer history) as primary and IRDM+ASTS as secondary. '
            'TeleGeography data requires manual lookup - backtester hardcodes the 2 known event dates. '
            'Very sparse event set (2 confirmed clusters) - treat as feasibility study, not robust backtest. '
            'CHT (Chunghwa Telecom ADR) may not react strongly as it has full restoration backup capacity; '
            'include for completeness. '
            'This strategy is partially exploratory; a third cluster event (2025 or beyond) would '
            'significantly improve statistical confidence.'
        ),
        'originality': 8,
        'bt_feasibility': 3
    }
]

append_strategies(strategies)
print('Strategies appended.')

update_idea_status('PL774', 'developed')
print('PL774 marked developed.')

update_idea_status('PL775', 'developed')
print('PL775 marked developed.')

heartbeat('strategy_developer')
print('Heartbeat sent.')
