# Phase 1AA — Niche Calendar + Structural Signals
# 10 signals. Dedup: all NEW; AA-4/5/6/8/9 are variants of related catalog entries but materially different.

### AA-1 XBI Quarterly Cap Rebalance Drift
- **Rule**: Long bottom-decile XBI constituents / short top-decile in 5 trading days before quarterly rebalance (3rd Fri of Mar/Jun/Sep/Dec); unwind on rebalance day.
- **Data**: ssga.com/spdr-sp-biotech-etf-xbi daily holdings CSV
- **CAGR**: 12-18%; Originality 9

### AA-2 ARKK Monthly Trade Front-Run
- **Rule**: At T+1 open after ARK Invest nightly trade email shows new buy > 0.5% of fund AUM, long that stock; hold until ARK stops buying (~3-7 sessions).
- **Data**: ark-funds.com/trade-notifications + cathiesark.com
- **CAGR**: 15-25%; Originality 8

### AA-3 MSTR Premium-to-mNAV Mean Reversion ⭐
- **Rule**: Short MSTR / long IBIT (dollar-neutral) when MSTR mcap > 2.5x BTC holdings mcap; unwind below 1.5x.
- **Data**: strategy.com/purchases + coingecko + yfinance MSTR
- **Mechanism**: MSTR premium compresses when retail mania cools or ATM dilution; 2.5x+ has reverted within weeks every cycle
- **CAGR**: 18-30% (25-40% per occurrence; 2-4 occurrences/yr)
- **Originality**: 9

### AA-4 TIPS Auction Breakeven Pre-Drift
- **Rule**: Short 10Y breakeven (long TIP / short IEF) from T-3 to T-1 of each 10Y TIPS auction (5/yr); close at auction result.
- **Data**: treasurydirect.gov + FRED T10YIE
- **CAGR**: 10-14%; Originality 8
- **Dedup**: Variant of A12 Treasury auction (which we failed). TIPS-specific 5/yr schedule with breakeven mechanic.

### AA-5 FOMC Minutes 3-Week Hawkish Echo
- **Rule**: Short 2Y futures (or long TBT) intraday 9:30-16:00 ET on FOMC Minutes release day (3 weeks post-FOMC, 8/yr).
- **Data**: federalreserve.gov/monetarypolicy/fomccalendars + ZT futures
- **Mechanism**: Minutes consistently reveal more hawkish dissent than the consensus statement
- **CAGR**: 10-15%; Originality 8

### AA-6 Post-Quad-Witching Monday Reversal
- **Rule**: At close of quad-witching Friday (3rd Fri Mar/Jun/Sep/Dec): buy SPY if down on week, short if up; exit Monday close.
- **Data**: yfinance SPY OHLC
- **CAGR**: 10-13%; Originality 8

### AA-7 JPM Healthcare Conference Biotech Pre-Drift
- **Rule**: Long XBI from 2nd Monday of December through close of 2nd Friday of January (JPM Healthcare Conference week).
- **Data**: jpmorgan.com/insights/business/business-growth/healthcare-conference + XBI
- **Mechanism**: Biotech CEOs front-load M&A and partnership reveals into JPM conference
- **CAGR**: 12-18%; Originality 8

### AA-8 MSCI Quarterly Index Review Add-Drift
- **Rule**: At close on MSCI Quarterly Review announcement (Feb/May/Aug/Nov, ~10 trading days before effective), long announced MSCI ACWI/EAFE adds; exit effective-date close.
- **Data**: msci.com/our-solutions/indexes/index-review + index-review-calendar
- **Mechanism**: ~$2T MSCI passive AUM mechanical purchasing
- **CAGR**: 13-20%; Originality 8
- **Dedup**: Distinct from E13 S&P 500 inclusion (decayed). MSCI EM/EAFE markets less arbitraged in US-centric catalogs.

### AA-9 MTUM Semi-Annual Reconstitution Cliff
- **Rule**: Short projected MTUM deletions (bottom decile of 6/12M momentum among current holdings) from T-5 to T-0 of semi-annual rebalance (last Mon May & Nov).
- **Data**: iShares MTUM holdings + MSCI USA Momentum methodology
- **CAGR**: 11-16%; Originality 9

### AA-10 Coinbase Listing Roadmap Front-Run
- **Rule**: When token added to Coinbase "assets under consideration" roadmap, buy that token at next CEX with availability; exit 24h after Coinbase spot trading goes live.
- **Data**: coinbase.com/blog/category/coinbase_roadmap + @CoinbaseAssets twitter
- **CAGR**: 20-40%; Originality 8
