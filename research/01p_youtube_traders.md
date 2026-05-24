# Phase 1P — YouTube / Podcast Trader-Sourced Signals
# Returned by hunt agent. 12 signals from Karsan, Green, Gromen, Howell, Constan, Johnson, Cowen, Lee, Steno Larsen.

### P-1 Karsan Vanna-Charm Pre-OPEX Window → Long SPX
- **Creator**: Cem Karsan / Kai Volatility — Top Traders Unplugged ep. 268
- **Rule**: Long SPX from Monday close preceding monthly OPEX week through OPEX Wednesday open.
- **Asset**: SPY / ES futures / 1w ATM SPY calls
- **Horizon**: ~7 trading sessions per month
- **Data**: CBOE OPEX calendar (3rd Fri of month) + SPY closes
- **Mechanism**: Dealer short-gamma + charm/vanna unwind into expiration → mechanical equity bid.
- **CAGR**: 5-8% standalone (per Karsan + systematicindividualinvestor.com)
- **Originality**: 9

### P-2 Karsan Post-OPEX Window of Weakness → Short SPX
- **Creator**: Cem Karsan
- **Rule**: Short SPX from OPEX Wednesday close through Tuesday after monthly OPEX.
- **Asset**: SPY puts
- **Horizon**: 4 trading days
- **Data**: OPEX calendar (flip side of P-1)
- **Mechanism**: Vanna/charm support disappears morning of OPEX → lowest dealer-flow tailwind = risk surfaces.
- **CAGR**: Asymmetric, small losses + occasional big hits
- **Originality**: 8

### P-3 Karsan Quarterly Melt-Up (10 days before quarterly OPEX)
- **Creator**: Cem Karsan
- **Rule**: Long SPX at close 10 trading days before quarterly (Mar/Jun/Sep/Dec) OPEX; exit at quarterly OPEX open.
- **Asset**: SPY
- **Horizon**: 2 weeks × 4 per year
- **Mechanism**: Quarterly contracts have 3× monthly OI → larger charm/vanna unwind. December has additional recollateralization tailwind.
- **CAGR**: ~3-4% from this rule alone
- **Originality**: 8

### P-4 Mike Green Passive Redemption Inflection → Hedge SPX/QQQ
- **Creator**: Mike Green / Logica — Excess Returns, MacroVoices 1525
- **Rule**: When monthly net flows into VTSAX flip negative on 3-month rolling basis, reduce SPX/QQQ exposure.
- **Asset**: SPY/QQQ hedge
- **Horizon**: 3-12 months structural
- **Data**: ICI long-term mutual fund flows monthly (ici.org/research/stats)
- **Mechanism**: ~95% of 401(k) contributions go to passive; multiplier $1 → $5-17 of market cap. Boomer redemptions are structural sell signal.
- **CAGR**: Asymmetric crash insurance
- **Originality**: 10

### P-5 Gromen True Interest Expense > 100% → Long Gold
- **Creator**: Luke Gromen / FFTT — MacroVoices 1352
- **Rule**: When (gross interest + Social Security + Medicare) / TTM tax receipts > 100%, overweight GLD vs TLT.
- **Asset**: GLD long / TLT short
- **Horizon**: 12-24 months structural
- **Data**: fiscaldata.treasury.gov Monthly Treasury Statement
- **Mechanism**: TIE > 100% → Fed cannot raise rates without bankrupting Treasury → real-rate suppression → gold as no one's liability.
- **CAGR**: Gromen targets $10-12K gold from $2.5-3K = 25-40% CAGR if thesis plays
- **Originality**: 9

### P-6 Howell Global Liquidity 13-Week Lead → Long BTC
- **Creator**: Michael Howell / CrossBorder Capital
- **Rule**: Long BTC when 13-week change in CrossBorder Global Liquidity Index is positive AND accelerating.
- **Asset**: BTC / IBIT
- **Horizon**: 13-week structural moves
- **Data**: Proxy with Fed + ECB + BOJ + PBOC summed in USD divided by global GDP; 13w change
- **Mechanism**: BTC is most liquidity-sensitive asset; risk-on flow lags 13 weeks due to intermediation.
- **CAGR**: ~70% in Howell-style backtest 2020-2024 vs ~55% BTC B&H
- **Originality**: 8

### P-7 Constan Coupon-vs-Bill Issuance Mix
- **Creator**: Andy Constan / Damped Spring
- **Rule**: When Treasury refunding announcement shifts bill-share above 15-20% historical norm by >3pp, long TLT + SPX.
- **Asset**: TLT + SPX
- **Horizon**: 1 quarter
- **Data**: Treasury Quarterly Refunding Statements (1st Wed of Feb/May/Aug/Nov)
- **Mechanism**: More bills = less long-duration supply for market to absorb → supports bonds and risk assets ("ATI" - Activist Treasury Issuance).
- **CAGR**: ~150bp easing equivalent in 2023-24 per Constan
- **Originality**: 9

### P-8 Brent Johnson DXY > 105 Defensive Switch
- **Creator**: Brent Johnson / Santiago Capital
- **Rule**: When DXY closes above 105 for 5 consecutive sessions, rotate from EEM into UUP + short GDX.
- **Asset**: EEM short / UUP long
- **Horizon**: Weeks-months
- **Data**: FRED DTWEXBGS (broad dollar proxy) or yfinance ^DXY
- **Mechanism**: 85-105 is manageable band; above 105 breaks foreign USD-debt sovereigns → dollar squeeze (Milkshake).
- **CAGR**: Asymmetric; ~3-5x/decade triggers
- **Originality**: 8

### P-9 Cowen Bitcoin Risk Metric Band Switch
- **Creator**: Benjamin Cowen / Into the Cryptoverse
- **Rule**: Accumulate BTC at risk-metric ≤ 0.4; sell at ≥ 0.75; flat between.
- **Asset**: BTC
- **Horizon**: Multi-year cycle
- **Data**: TradingView open-source script; z-score of log-price vs log-regression trend
- **Mechanism**: 4-year cycle accumulation/distribution zones via mean-reversion z-scoring.
- **CAGR**: Mixed — EdgePhase replication found marginally underperformed simple 200-WMA
- **Originality**: 8

### P-10 Tom Lee Multi-Theme Confluence (Granny Shots)
- **Creator**: Tom Lee / Fundstrat
- **Rule**: Hold US large-cap stocks in ≥ 2 of 7 themes (millennials, labor, energy/cyber, easing FC, style tilt, seasonality, PMI). Rebalance quarterly.
- **Asset**: GRNY ETF or replicated basket
- **Horizon**: Quarterly rebalance
- **Data**: grannyshots.com quarterly holdings
- **CAGR**: GRNY +30% vs SPY +17% in 2024 (one year)
- **Originality**: 8
- **Caveat**: < 3y live track record

### P-11 Karsan Year-End Recollateralization
- **Creator**: Cem Karsan
- **Rule**: If SPX YTD through Oct 31 > +5%, long SPX Nov 1 to Jan 31; else stand aside.
- **Asset**: SPY
- **Horizon**: 3 months conditional
- **Mechanism**: Gains create new equity collateral → re-leveraged into year-end + structured-product issuance heaviest into Dec OPEX.
- **CAGR**: Conditional outperforms unconditional Santa rally by ~3pp/yr
- **Originality**: 8

### P-12 Steno Larsen Global Liquidity Nowcast
- **Creator**: Andreas Steno Larsen / Steno Research
- **Rule**: When 4-week change in summed [Fed + ECB + BOJ + PBOC + SNB] balance sheets (USD) flips from negative to positive, overweight SPX + BTC for 8 weeks.
- **Asset**: SPX + BTC
- **Horizon**: 8 weeks
- **Data**: Fed H.4.1, ECB weekly financial statement, BOJ assets, PBOC monthly, SNB weekly
- **Mechanism**: Liquidity tide + growth-stable = outsized equity/BTC returns. Captures non-Fed liquidity.
- **Originality**: 8
