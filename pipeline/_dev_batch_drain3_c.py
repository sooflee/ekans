import sys
sys.path.insert(0, '/Users/benson/Projects/ekans/pipeline')
from queue_io import append_strategies, update_idea_status
from datetime import datetime, timezone

strategies = [
    {
        'strategy_id': 'PL968',
        'idea_id': 'PL968',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'ready',
        'signal_id': 'PL968_fslr_backlog_asp_cancellation_short',
        'name': 'FSLR Backlog ASP/Spot Spread > 200% + Cancellation Rate Inflection -> Short FSLR (Counter-Signal)',
        'category': 'G',
        'asset_class': 'single_name_equity',
        'horizon': '6-12 weeks',
        'rule': (
            'When (1) global solar module spot price (PVInfoLink mono-PERC) sustains below $0.12/W '
            'for >= 8 consecutive weeks AND (2) FSLR most-recent 10-Q reported backlog ASP is >= $0.28/W '
            '(implying spread > 133%), AND (3) FSLR reported trailing-12-month cancellation rate '
            'exceeds 10% OR FSLR book-to-bill < 0.90 in most recent quarter, '
            'then go short FSLR at next open. '
            'Hold for 10 calendar weeks (50 trading days). '
            'Stop-loss: cover if FSLR rises more than 12% from entry. '
            'Profit target: cover if FSLR falls 25% from entry.'
        ),
        'tickers': ['FSLR', 'ENPH', 'SEDG', 'SPY'],
        'entry_conditions': [
            'PVInfoLink module spot price < $0.12/W for >= 8 consecutive weeks',
            'FSLR 10-Q backlog ASP >= $0.28/W (spot-to-ASP spread > 133%)',
            'FSLR trailing-12M cancellation rate > 10% OR book-to-bill < 0.90 per most recent quarter',
            'Enter short FSLR at next open after all three conditions met'
        ],
        'exit_conditions': [
            'Hold 50 trading days (10 calendar weeks)',
            'Stop-loss: cover if FSLR +12% from entry',
            'Profit target: cover if FSLR -25% from entry'
        ],
        'data_sources_concrete': {
            'prices': 'FSLR via yfinance; ENPH and SEDG as sector comparables; SPY as benchmark',
            'fundamental': (
                'PVInfoLink monthly ASP report (pvinsights.com/pvinsight-monthly-report, free summary); '
                'BloombergNEF Solar Module Price Index (summary data free via BNEF weekly email); '
                'FSLR 10-Q backlog table (EDGAR, filed quarterly): backlog MWdc, contracted ASP $/W, '
                'bookings, cancellations, book-to-bill ratio'
            )
        },
        'backtest_approach': (
            'Event-study using FSLR 10-Q filing dates as signal evaluation points (4 per year, 2012-2025). '
            'For each 10-Q: extract reported backlog ASP, book-to-bill, cancellation rate. '
            'Cross-reference with contemporaneous PVInfoLink spot price (use quarterly BNEF/PVInfoLink '
            'summary for historical reconstruction; spot below $0.10-0.12 was reached in late 2023 and 2024). '
            'Flag quarters where all three conditions trigger. '
            'Measure FSLR return from 10-Q filing date over next 50 trading days vs SPY. '
            'Counter-signal against long-FSLR thesis: confirm negative excess return during high-spread quarters. '
            'Supplement with option-implied vol analysis around earnings (put spread entry for elevated vol events).'
        ),
        'known_events': [
            '2023-Q3 module spot approached $0.13/W (PVInfoLink), first stress point for FSLR ASP',
            '2024-Q1 module spot reached $0.09/W (BNEF), FSLR backlog ASP at $0.31/W',
            '2024-Q2 FSLR first cancellation rate disclosure uptick',
            '2024-Q4 module spot $0.085/W (PVInfoLink Apr 2026 levels noted in source_reference)',
            '2026-Q1 FSLR earnings call backlog renegotiation discussion cited by scout'
        ],
        'implementation_notes': (
            'This is a counter_signal strategy (counters: long_FSLR, long_US_solar_fab). '
            'FSLR broad-basket targets are intentionally NOT used here - the mechanism implicates '
            'FSLR specifically via its unique long-dated contracted backlog model; other solar names '
            '(ENPH, SEDG) are module-price beneficiaries or are not backlog-model businesses. '
            'PVInfoLink historical spot data pre-2020 may require manual reconstruction from trade press; '
            'BloombergNEF provides free quarterly summaries sufficient for signal dating. '
            'FSLR 10-Q backlog ASP has been disclosed since ~2013; treat pre-2013 as unavailable. '
            'Cancellation rate was not always disclosed before 2021; use book-to-bill < 0.90 as '
            'primary proxy for 2013-2021 period. '
            'bt_feasibility=4: all data free or reconstructible; FSLR specific-name target is correct.'
        ),
        'originality': 8,
        'bt_feasibility': 4
    },
    {
        'strategy_id': 'PL969',
        'idea_id': 'PL969',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'ready',
        'signal_id': 'PL969_ntsb_major_investigation_rail_short',
        'name': 'NTSB Major Rail Derailment Investigation -> Short Named Class I Carrier vs Long Peer Basket',
        'category': 'G',
        'asset_class': 'single_name_equity',
        'horizon': '4-12 weeks',
        'rule': (
            'When NTSB opens a major investigation (formal docket with DCA/RRD prefix, Go-Team dispatched) '
            'into a Class I rail derailment within 7 days of the incident, '
            'short the named carrier at next open (T+1) after the NTSB docket publication date, '
            'and simultaneously go long an equal-weighted basket of the other US-listed Class I carriers '
            '(if NSC is named: long UNP+CSX+CP+CNI equally weighted; if UNP: long NSC+CSX+CP+CNI; etc.). '
            'Hold for 60 trading days (12 calendar weeks). '
            'Stop-loss: exit if named carrier outperforms peer basket by more than 8% from entry. '
            'Exit if named carrier underperforms peer basket by more than 20% (take profit).'
        ),
        'tickers': ['NSC', 'UNP', 'CSX', 'CP', 'CNI', 'SPY'],
        'entry_conditions': [
            'NTSB opens major investigation docket (DCA/RRD prefix) for a Class I rail derailment',
            'Go-Team dispatch confirmed on NTSB press release (ntsb.gov/news)',
            'Named carrier identified from NTSB incident report',
            'Entry at T+1 open (next trading day after NTSB major investigation announcement)',
            'Short named carrier, Long equal-weighted basket of other US-listed Class I carriers'
        ],
        'exit_conditions': [
            'Hold 60 trading days (12 calendar weeks)',
            'Stop-loss: exit if named carrier outperforms peer basket by >= 8%',
            'Profit target: exit if named carrier underperforms peer basket by >= 20%',
            'Force exit at next earnings if carrier reports > $500M reserve charge (confirms thesis; harvest gains)'
        ],
        'data_sources_concrete': {
            'prices': 'NSC, UNP, CSX, CP, CNI via yfinance (all have 16+ years of data); SPY as benchmark',
            'fundamental': (
                'NTSB Surface Accident Database / CAROL Query (data.ntsb.gov/carol-main-public, free); '
                'filter to RailroadAccident mode, major-investigation flag; '
                'cross-reference with NTSB press releases at ntsb.gov/news; '
                'FRA Office of Railroad Safety emergency orders (railroads.dot.gov, free); '
                'SEC 10-Q ASC 450 loss-contingency reserve disclosures for named carrier'
            )
        },
        'backtest_approach': (
            'Event-study: identify all NTSB major rail investigation docket openings 2010-2025 '
            'where the named carrier is one of NSC, UNP, CSX, CP, or CNI. '
            'Primary analog: NSC East Palestine (NTSB RRD23MR005, docket opened ~Feb 10, 2023). '
            'Other candidates: '
            '  - UNP Loup City NE derailment (2021) '
            '  - CSX Howard Siding derailment (2017) '
            '  - NSC Graniteville SC (2005 - pre-data window) '
            '  - Various FRA-reportable NTSB investigations 2010-2022. '
            'For each event: compute named-carrier return minus equal-weighted-peer-basket return '
            'from T+1 open over 60 trading days. '
            'Normalize by sector beta (regress on SPY). '
            'East Palestine: NSC underperformed UNP/CSX basket by ~15% over 6 months per idea thesis. '
            'Report: N events, mean pair return, hit rate, Sharpe.'
        ),
        'known_events': [
            '2023-02-10 NTSB RRD23MR005 East Palestine OH (NSC Norfolk Southern)',
            '2021-09 UNP Loup City NE derailment (NTSB investigation)',
            '2017-02 CSX investigation after Howard Siding collision',
            '2016-01 NS derailment NTSB investigation',
            '2015-05 Amtrak Philadelphia (operated on NSC track, NSC affected)',
            '2013-07 MMA Lac-Megantic (Montreal Maine & Atlantic, not Class I - control period)'
        ],
        'implementation_notes': (
            'Counter-signal to long_industrials, long_rails: this is a carrier-specific short, not a sector short. '
            'Peer basket includes only US-listed Class I carriers (CP and CNI are Canadian-listed with US ADRs; '
            'use CP and CNI via yfinance which has both). '
            'East Palestine is the cleanest analog with full financial disclosure available. '
            'Pre-2015 event sample may be thin; backtester should note if N < 5. '
            'Small sample caveat: Class I major NTSB investigations are rare (1-3 per decade at severity level '
            'triggering significant equity impact). '
            'If NTSB investigation database search returns < 3 clear events, supplement with FRA '
            'major-accident investigations where carrier-specific reserve charges exceeded $100M. '
            'bt_feasibility=4: all data sources are free and programmatic; primary analog well-documented.'
        ),
        'originality': 7,
        'bt_feasibility': 4
    }
]

append_strategies(strategies)
print('Batch 3 strategies (PL968 + PL969) appended successfully')

update_idea_status('PL968', 'developed')
print('PL968 marked developed')

update_idea_status('PL969', 'developed')
print('PL969 marked developed')
