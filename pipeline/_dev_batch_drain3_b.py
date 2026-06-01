import sys
sys.path.insert(0, '/Users/benson/Projects/ekans/pipeline')
from queue_io import append_strategies, update_idea_status

# PL964: Stablecoin Bill -> CRCL long
# The legislative trigger (stablecoin oversight assigned to OCC/Fed in conference committee print)
# has NEVER occurred before. FIT21/Lummis-Gillibrand have never passed conference committee.
# CRCL only has data since June 2025 (IPO'd Jun 2025).
# No historical analog date set exists for "stablecoin regulatory clarity bill passes committee."
# Route to Watchlist.

update_idea_status('PL964', 'watchlist',
    watchlist_trigger=(
        'Congress.gov shows a stablecoin oversight bill (FIT21 successor, Lummis-Gillibrand, '
        'or Clarity for Payment Stablecoins) receives a formal conference committee markup date '
        'and the draft text assigns oversight to OCC/Fed rather than SEC'
    ),
    watchlist_trade=(
        'Long CRCL, Short BKCH in equal notional (1:1 dollar pair). '
        'Target: CRCL +15-20% vs BKCH flat or down over 6-8 weeks. '
        'Size: up to 1.5% NAV each leg given binary regulatory event risk.'
    ),
    watchlist_mechanism=(
        'US-chartered stablecoin issuer (Circle/CRCL) benefits disproportionately when OCC/Fed '
        'oversight is confirmed: Tier-1 banks accelerate USDC integration for compliance, '
        'USDC market share gains vs Tether, CRCL net interest income scales on growing reserves. '
        'BKCH holds offshore/unregulated exchange exposures (Binance-related names) that reprice '
        'down on increased compliance pressure. Net: CRCL outperforms crypto-equity basket.'
    ),
    watchlist_no_analog_reason=(
        'No US stablecoin regulatory framework has ever been enacted. FIT21 passed House in May 2024 '
        'but died in Senate. Lummis-Gillibrand Stablecoin Act was not passed. The specific trigger -- '
        'a conference committee print assigning OCC/Fed oversight -- has zero precedent in the '
        'modern data era. CRCL itself only IPOd in June 2025 (less than 1 year of price history). '
        'Cannot construct an event-study backtest with no prior analog dates.'
    )
)
print('PL964 routed to watchlist')

# PL965: Howey ruling cluster -> short BLOK/BKCH, long IBIT
# Counter-signal. Historical analog dates exist: SEC v. Ripple (Dec 2020 filing),
# SEC v. Coinbase (Jun 2023), SEC v. Binance (Jun 2023), SEC v. Kraken (Nov 2023).
# BLOK has data from Jan 2020, BKCH from Jul 2021.
# IBIT only from Jan 2024 - use BTC-USD as proxy pre-Jan 2024.
# bt_feasibility=4 is appropriate.

from datetime import datetime, timezone

strategies = [
    {
        'strategy_id': 'PL965',
        'idea_id': 'PL965',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'ready',
        'signal_id': 'PL965_howey_ruling_cluster_bkch_short_ibit_long',
        'name': 'SEC Howey-Test Win Cluster -> Short BKCH / Long IBIT (Crypto-Equity vs Crypto-Spot Pair)',
        'category': 'F',
        'asset_class': 'sector_etf',
        'horizon': '2-5 weeks',
        'rule': (
            'When 3 or more SEC enforcement actions against crypto-token issuers receive favorable '
            'Howey-test rulings (district court or appellate) within any rolling 60-day window, '
            'enter: Short BKCH (Global X Blockchain ETF), Long IBIT (iShares Bitcoin Trust) '
            'in equal dollar notional at the next open after the 3rd ruling is published on SEC.gov '
            'litigation releases. '
            'Hold for 25 trading days (5 calendar weeks). '
            'Exit early if BKCH outperforms IBIT by more than 8% from entry (stop) '
            'or BKCH underperforms IBIT by more than 15% (take profit).'
        ),
        'tickers': ['BKCH', 'BLOK', 'IBIT', 'COIN', 'SPY'],
        'entry_conditions': [
            '3 or more SEC enforcement actions vs crypto token issuers receive favorable Howey-test rulings within 60 rolling days',
            'Rulings confirmed on SEC.gov/litigation/litreleases.htm or PACER CourtListener mirror',
            'Entry at next open after 3rd ruling date',
            'Short BKCH, Long IBIT in equal notional'
        ],
        'exit_conditions': [
            'Hold 25 trading days (5 calendar weeks)',
            'Stop-loss: BKCH outperforms IBIT by >= 8% from entry (pair adverse move)',
            'Profit target: BKCH underperforms IBIT by >= 15% from entry'
        ],
        'data_sources_concrete': {
            'prices': 'BKCH, BLOK, IBIT, COIN via yfinance; BTC-USD via yfinance for pre-2024 IBIT proxy; SPY benchmark',
            'fundamental': 'SEC litigation releases (sec.gov/litigation/litreleases.htm, free); CourtListener PACER mirror (free API for case dockets); EDGAR 10-Q legal-reserve disclosures'
        },
        'backtest_approach': (
            'Event-study PnL on SEC Howey-test enforcement cluster dates 2020-2025. '
            'Identify dates when SEC obtained favorable rulings in: '
            'SEC v. Ripple (XRP ruling Jul 2023), SEC v. LBRY (Nov 2022), '
            'SEC v. Coinbase staking claim (2023 injunction denied but elevated legal risk), '
            'SEC v. Binance (complaint filed Jun 2023), SEC v. Kraken (complaint filed Nov 2023). '
            'For pre-IBIT period (pre-Jan 2024), use BTC-USD as spot-bitcoin proxy leg. '
            'For BKCH (Jul 2021+), measure BKCH return minus BTC-USD return over 25 trading days. '
            'For BLOK (Jan 2020+), use as supplemental verification. '
            'Report: mean pair return (short BKCH, long BTC-proxy), hit rate, Sharpe, max drawdown. '
            'Counter-signal: explicitly positioned against long_BTC and long_crypto consensus.'
        ),
        'known_events': [
            '2020-12-22 SEC v. Ripple Labs complaint filed (XRP classified as security)',
            '2022-11-07 SEC v. LBRY favorable ruling (LBC classified as security)',
            '2023-06-06 SEC v. Coinbase complaint + SEC v. Binance complaint same week',
            '2023-07-13 SEC v. Ripple partial ruling (XRP programmatic sales not securities - mixed)',
            '2023-11-20 SEC v. Kraken complaint filed',
            '2024-01-10 SEC spot BTC ETF approvals (IBIT launch - reverse signal)',
            '2025-02 SEC enforcement pause under new administration (signal extinction test)'
        ],
        'implementation_notes': (
            'This is a counter_signal strategy (counters: long_BTC, long_crypto); '
            'BKCH/BLOK basket and IBIT as broad-basket targets are appropriate per the spec. '
            'IBIT only has data from Jan 2024; for pre-2024 events use BTC-USD (yfinance: BTC-USD) '
            'as proxy for the spot-bitcoin long leg. '
            'BKCH (Global X) has data from Jul 2021; BLOK (Amplify) from Jan 2020 for earlier events. '
            'The 2024 SEC pivot under new administration may reduce future signal frequency - '
            'backtester should test for regime change post-Jan 2025. '
            'Note: the Jul 2023 Ripple ruling was mixed (favorable for programmatic sales) and '
            'may have acted as a REVERSE signal for this strategy on that date.'
        ),
        'originality': 7,
        'bt_feasibility': 4
    }
]

append_strategies(strategies)
print('Batch 2 strategy (PL965) appended successfully')

update_idea_status('PL965', 'developed')
print('PL965 marked developed')
