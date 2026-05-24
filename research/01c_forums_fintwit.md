# Phase 1C — Trading Forums + FinTwit Signals
# Returned by research agent. Cross-asset rotation, vol regime, sentiment.

### Lumber/Gold Ratio (Gayed)
- **What it is**: Ratio of front-month lumber futures to gold, used as a risk-on/risk-off macro switch.
- **Asset class**: US equities vs Treasuries (rotation)
- **Horizon**: 13-week swings
- **Entry/exit rule**: Compute 13-week % change in lumber and gold. If lumber outperforms gold, long equities; if gold outperforms, rotate into long Treasuries (TLT). Rebalance weekly.
- **Where heard / source**: Michael Gayed, "Lumber: Worth Its Weight in Gold" (2015, SSRN). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2604248
- **Reported edge**: Sharpe ~0.7 in-sample; OOS post-2015 much weaker.
- **Originality (1-10)**: 3
- **Backtest-feasibility free data (1-5)**: 4
- **Notes**: Lumber contract redesigned 2022; pre/post fragile.

### Utilities/S&P 500 Ratio (Gayed)
- **What it is**: When defensive utilities outperform broad market on 4-week basis, historically precedes higher equity volatility.
- **Asset class**: US equities (timing)
- **Horizon**: 1 month
- **Entry/exit rule**: 4-week return of XLU minus 4-week return of SPY. If XLU > SPY, reduce equity exposure for next 4 weeks.
- **Where heard / source**: Gayed & Atilgan (2014). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2517910
- **Reported edge**: CAGR uplift ~2-3% on rotation.
- **Originality**: 3
- **Backtest-feasibility**: 5
- **Notes**: Crowded since publication.

### Faber GTAA 10-Month Moving Average
- **What it is**: Monthly close above 10-month SMA = invested; below = cash. Diversified asset basket.
- **Asset class**: Multi-asset
- **Horizon**: Monthly rebalance, multi-year hold
- **Entry/exit rule**: For each of 5 asset classes (US stocks, foreign stocks, REITs, commodities, 10Y Treasuries), hold if last monthly close > 10-month SMA, else cash. Equal-weight "on" positions.
- **Where heard / source**: Mebane Faber (2007/2013). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461
- **Reported edge**: ~10% CAGR with half the drawdown of buy-and-hold; Sharpe ~0.7 vs 0.4.
- **Originality**: 2
- **Backtest-feasibility**: 5

### Dual Momentum (Antonacci GEM)
- **What it is**: Combines absolute (vs T-bills) and relative (US vs intl equities) momentum.
- **Asset class**: Equities vs bonds
- **Horizon**: Monthly rebalance
- **Entry/exit rule**: Monthly, compare 12-month returns of S&P 500 and MSCI ACWI ex-US. Pick higher. If winner's 12m return < T-bill return, hold AGG instead.
- **Where heard / source**: Gary Antonacci (2013). https://www.optimalmomentum.com/wp-content/uploads/2020/02/RiskPremiaHarvesting.pdf
- **Reported edge**: ~17% CAGR, Sharpe ~1.0 (1974-2013).
- **Originality**: 3
- **Backtest-feasibility**: 5

### Accelerating Dual Momentum (ADM)
- **What it is**: Variation using 1, 3, 6-month returns instead of 12-month, picking among SPY, SCZ, TLT.
- **Asset class**: Equities vs bonds
- **Horizon**: Monthly
- **Entry/exit rule**: Average of 1m, 3m, 6m total returns for SPY, SCZ, TLT. Hold top-ranked.
- **Where heard / source**: EngineeredPortfolio.com (2018). https://engineeredportfolio.com/2018/05/02/accelerating-dual-momentum-investing/
- **Reported edge**: ~22% CAGR, max DD ~18% (2003-2018).
- **Originality**: 5
- **Backtest-feasibility**: 5
- **Notes**: 2022 brutal (heavy TLT reliance).

### VIX Term Structure (VIX/VXV)
- **What it is**: Ratio of 1-month implied vol (VIX) to 3-month (VIX3M). Below 1 = contango; spikes above 1 = backwardation/stress.
- **Asset class**: US equities
- **Horizon**: Days to weeks
- **Entry/exit rule**: When VIX/VIX3M < 0.92, stay long. Cross above 1.0, reduce/hedge; re-enter below 0.95.
- **Where heard / source**: VIX and More blog (Bill Luby). https://vixandmore.blogspot.com/
- **Reported edge**: Anecdotal Sharpe ~1.0 on tactical SPY/cash filter.
- **Originality**: 4
- **Backtest-feasibility**: 5

### Dealer Gamma Positioning (GEX)
- **What it is**: Aggregate dealer gamma exposure from listed options OI. Long-gamma = vol-suppressing; short-gamma = trend-amplifying.
- **Asset class**: S&P 500
- **Horizon**: Days to a few weeks
- **Entry/exit rule**: GEX > +$2B per 1% move: fade extremes/sell straddles. GEX < 0: expect trend continuation, lean long vol.
- **Where heard / source**: SqueezeMetrics, "The Implied Order Book" (2017). https://squeezemetrics.com/download/The_Implied_Order_Book.pdf
- **Reported edge**: Realized vol ~2x higher on negative-GEX days.
- **Originality**: 6
- **Backtest-feasibility**: 2 (paid data)

### JPM Collar Roll
[same as Phase 1A signal #20 — dedupe in consolidation]

### VVIX/VIX Ratio
- **What it is**: Vol-of-vol vs VIX. Extreme highs suggest tail-hedging demand.
- **Asset class**: US equities / vol
- **Horizon**: 1-4 weeks
- **Entry/exit rule**: VVIX/VIX > 6.5 with VIX < 18 → warning, buy SPX puts. < 4.5 → vol exhausted, sell premium.
- **Where heard / source**: CBOE VVIX. https://www.cboe.com/tradable_products/vix/vvix/
- **Reported edge**: Hit-rate ~58% on subsequent 5-day VIX direction (anecdotal).
- **Originality**: 5
- **Backtest-feasibility**: 5
- **Notes**: Use rolling z-score, not absolute thresholds.

### SKEW Index Extremes
- **What it is**: CBOE SKEW measures relative price of OTM puts vs ATM in SPX.
- **Asset class**: US equities
- **Horizon**: 1-3 months
- **Entry/exit rule**: SKEW > 145 for 5+ days with VIX < 16: fade complacency, buy 2-3m SPX puts. SKEW < 120: tails cheap, own tails.
- **Where heard / source**: @choffstein. Newfound Research.
- **Reported edge**: Mixed; weak as standalone, useful in ensembles.
- **Originality**: 4
- **Backtest-feasibility**: 5

### Put/Call Ratio Extremes
- **What it is**: 10-day MA of CBOE equity-only put/call. Contrarian sentiment gauge.
- **Asset class**: US equities
- **Horizon**: 1-4 weeks
- **Entry/exit rule**: > 0.75 (bearish) → long SPY 20 sessions. < 0.50 (euphoria) → reduce.
- **Where heard / source**: Sentimentrader. https://sentimentrader.com/
- **Reported edge**: >2 sigma bearish followed by ~70% positive 1m returns since 2000.
- **Originality**: 2
- **Backtest-feasibility**: 5
- **Notes**: 2020-2021 retail call buying distorted the equity-only ratio; 0DTE complicates further.

### NAAIM Exposure Index Extremes
- **What it is**: Weekly survey of active manager equity exposure (0-200).
- **Asset class**: US equities
- **Horizon**: 4-12 weeks
- **Entry/exit rule**: 4-week NAAIM < 30 → long SPY 8 weeks. > 90 rising → hedge.
- **Where heard / source**: NAAIM. https://www.naaim.org/programs/naaim-exposure-index/
- **Reported edge**: <30 readings → ~+4-6% over 8 weeks vs ~+2% baseline.
- **Originality**: 4
- **Backtest-feasibility**: 5

### AAII Bull-Bear Spread
- **What it is**: Weekly retail survey of bullish vs bearish sentiment.
- **Asset class**: US equities
- **Horizon**: 4-12 weeks
- **Entry/exit rule**: (Bulls - Bears) < -20 for 2 weeks → long 12 weeks. > +30 for 3 weeks → fade.
- **Where heard / source**: AAII. https://www.aaii.com/sentimentsurvey
- **Reported edge**: 12m forward ~+13% after extreme bearishness vs ~+7% after bullishness.
- **Originality**: 2
- **Backtest-feasibility**: 5

### Copper/Gold Ratio for 10Y Yields
- **What it is**: Ratio of copper to gold tracks 10Y yield closely; divergences mean-revert.
- **Asset class**: Treasuries / rates
- **Horizon**: 1-6 months
- **Entry/exit rule**: 60-day z-score of (Copper/Gold) vs 10Y yield. Yield 1.5σ cheap → short TLT. 1.5σ rich → long TLT.
- **Where heard / source**: Gundlach; Quantpedia anomaly #492.
- **Reported edge**: 20-year correlation ~0.7 (visual).
- **Originality**: 4
- **Backtest-feasibility**: 5
- **Notes**: Broke 2020-2022 (China copper, CB gold buying).

### MOVE/VIX Ratio Divergence
- **What it is**: MOVE = Treasury implied vol; VIX = equity. Divergences presage cross-asset moves.
- **Asset class**: Cross-asset
- **Horizon**: 2-8 weeks
- **Entry/exit rule**: MOVE/VIX > 8 (rates panic, stocks calm) → buy SPX puts/VIX calls. < 4 → expect equity mean-reversion up.
- **Where heard / source**: @Ksidiii (Ambrus). https://twitter.com/Ksidiii
- **Reported edge**: Anecdotal; flagged 2022 ahead of equity drop.
- **Originality**: 6
- **Backtest-feasibility**: 4

### HY-IG Spread Z-Score
- **What it is**: Spread between HY and IG credit OAS. Risk-on/off via credit, often leads equities.
- **Asset class**: Credit + equity overlay
- **Horizon**: 1-3 months
- **Entry/exit rule**: FRED HY OAS minus IG OAS. 1-year z-score. Spread widens > 1σ from 60-day low → reduce equity 60 days. Tightens back below 60-day avg → re-enter.
- **Where heard / source**: Verdad Capital. https://verdadcap.com/archive
- **Reported edge**: Sharpe uplift ~0.2-0.3.
- **Originality**: 4
- **Backtest-feasibility**: 5 (FRED: BAMLH0A0HYM2, BAMLC0A0CM)

### Margin Debt YoY Change
- **What it is**: YoY change in FINRA margin debt. Extreme contractions correlate with bear markets.
- **Asset class**: US equities
- **Horizon**: 3-12 months
- **Entry/exit rule**: YoY < -20% → reduce equity 6 months. Turning back up through 0% → re-enter.
- **Where heard / source**: Jesse Felder. https://thefelderreport.com/
- **Reported edge**: Anecdotal; peaks at 2000, 2007, 2021 tops.
- **Originality**: 5
- **Backtest-feasibility**: 5

### Hindenburg Omen
- **What it is**: Confluence of NYSE breadth conditions (new highs + lows both >2.2% of issues, etc).
- **Asset class**: US equities
- **Horizon**: 1-3 months
- **Entry/exit rule**: When all 4 conditions trigger, hedge equity 40 trading days.
- **Where heard / source**: Jim Miekka (1995). McClellan Financial. https://www.mcoscillator.com/
- **Reported edge**: ~25-30% hit rate of "major crash" within 4 months — many false positives.
- **Originality**: 6
- **Backtest-feasibility**: 4

### Coppock Curve (Long-Term Bottoms)
- **What it is**: 10-month WMA of (14m ROC + 11m ROC). Cross up from below zero marks major equity bottoms.
- **Asset class**: US/global equities
- **Horizon**: Multi-year
- **Entry/exit rule**: Monthly: long when Coppock crosses up through zero from below; exit on next down-cross.
- **Where heard / source**: Edwin Coppock (1962, Barron's).
- **Reported edge**: Caught 1949, 1962, 1974, 1982, 2003, 2009, 2020. ~8 signals/century.
- **Originality**: 6
- **Backtest-feasibility**: 5
- **Notes**: Bottom-timer only.

### DeMark TD Sequential 9-13
- **What it is**: Counts of consecutive closes vs 4-bars-prior; "9 setup" and "13 countdown" are completion signals.
- **Asset class**: Any
- **Horizon**: Days to weeks (daily/weekly bars)
- **Entry/exit rule**: 9 consecutive closes > close 4 bars earlier = sell setup; then 13 closes (per countdown) = sell countdown — fade. Mirror for buys.
- **Where heard / source**: Tom DeMark. Wikipedia: https://en.wikipedia.org/wiki/DeMark_Indicators
- **Reported edge**: ~55-60% hit rate as reversal trigger (anecdotal).
- **Originality**: 5
- **Backtest-feasibility**: 5

### Bullish Percent Index (BPI) Reversals
- **What it is**: % of NYSE / S&P 500 stocks on P&F buy signals. Reversals from extremes signal regime change.
- **Asset class**: US equities (breadth)
- **Horizon**: 1-6 months
- **Entry/exit rule**: S&P 500 BPI < 30, reverses up by 6+ points → long 3 months. > 70, reverses down by 6+ → reduce.
- **Where heard / source**: Investors Intelligence. https://stockcharts.com/h-sc/ui?s=$BPSPX
- **Reported edge**: Anecdotal; used by Dorsey Wright.
- **Originality**: 5
- **Backtest-feasibility**: 4

### Google Trends "Recession" Spikes
- **What it is**: Search volume surges for distress terms mark capitulation (contrarian buy).
- **Asset class**: US equities
- **Horizon**: 1-6 months
- **Entry/exit rule**: 4-week MA of Google Trends "recession" > 1.5σ above 5-year mean → long SPY 6 months.
- **Where heard / source**: Preis/Moat/Stanley (Nature 2013). https://www.nature.com/articles/srep01684
- **Reported edge**: Sharpe ~1.16 in-sample 2004-2011; OOS weaker.
- **Originality**: 6
- **Backtest-feasibility**: 4
- **Notes**: Trends rebasing makes clean backtests tricky.

### Skyscraper Indicator
- **What it is**: Completion of record-breaking skyscraper has coincided with major business-cycle peaks (1908, 1929, 1973, 2000, 2010).
- **Asset class**: Global equities (macro)
- **Horizon**: 6-24 months
- **Entry/exit rule**: When ground broken on a future world's-tallest, gradual de-risking; min equity by topping-out date.
- **Where heard / source**: Andrew Lawrence, Dresdner Kleinwort (1999).
- **Reported edge**: N ~6-7 events in 120 years. No statistical power.
- **Originality**: 9
- **Backtest-feasibility**: 5
- **Notes**: Curiosity-only.

### Wikipedia Pageview Anomalies
- **What it is**: Surges in pageviews for finance-related Wikipedia articles ("Black Monday," "subprime") predict equity drawdowns.
- **Asset class**: US equities
- **Horizon**: 1-4 weeks
- **Entry/exit rule**: Composite z-score of basket of stress articles. z > 2 → reduce equity 20 sessions.
- **Where heard / source**: Moat/Curme et al (Nature 2013). https://www.nature.com/articles/srep01801
- **Reported edge**: ~141% cumulative on Dow strategy 2007-2012 (in-sample).
- **Originality**: 8
- **Backtest-feasibility**: 4
