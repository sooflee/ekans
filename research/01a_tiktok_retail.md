# Phase 1A — TikTok + Retail-Trader Signal Universe
# Returned by research agent. Spread of originality 1-8 (no genuine 9-10s found).

### 1. ICT Daily Fair Value Gap (FVG) Fill
- **What it is**: A "fair value gap" is a 3-candle imbalance where the wicks of candles 1 and 3 don't overlap, leaving an untraded zone. The rule is that price tends to revisit and partially fill these zones.
- **Asset class implied**: equities indices, FX, crypto (originally taught on NQ/ES)
- **Horizon**: days to weeks
- **Entry/exit rule**: On the daily chart, identify a bullish FVG (low of candle 3 > high of candle 1). When price pulls back into the FVG zone, go long; stop below the FVG low; target the recent swing high or the next opposing FVG.
- **Where heard**: Inner Circle Trader (Michael Huddleston) on YouTube; widely repackaged on TikTok including v.tpk-adjacent SMC creators. innercircletrader.net, edgeful.com.
- **Originality estimate (1-10)**: 3
- **Backtest-feasibility with free data (1-5)**: 4
- **Notes**: Edgeful's data shows ~60-63% of intraday FVGs remain unfilled, so the "must fill" mythology is overstated.

### 2. Bullish Order Block Retest (ICT)
- **What it is**: The last down-close candle before a strong up-impulse is the "order block." Price is expected to retrace to it and bounce.
- **Asset class implied**: any (indices, FX, crypto)
- **Horizon**: days to weeks on daily
- **Entry/exit rule**: After a daily candle that closes >1.5x ATR above a prior down-candle, mark the prior down-candle's range as the OB. Enter long on first retest of OB high; stop below OB low; target prior swing high.
- **Where heard**: ICT; phidiaspropfirm.com, trinitytrading.io.
- **Originality estimate**: 3
- **Backtest-feasibility**: 4
- **Notes**: Higher TF OBs reportedly achieve 75-85% hit rate in marketing materials — unverified.

### 3. Liquidity Sweep Reversal (Stop Hunt)
- **What it is**: Price spikes through a prior swing high/low to trigger stops, then closes back inside the range.
- **Asset class implied**: FX, crypto, futures
- **Horizon**: days to ~2 weeks
- **Entry/exit rule**: On daily, identify the prior 20-day high. If today's high > prior 20-day high but close < prior 20-day high, short next open; stop above sweep high; target the equilibrium (50%) of the prior range.
- **Where heard**: ICT/SMC universe (priceactionninja.com, FXNX).
- **Originality estimate**: 4
- **Backtest-feasibility**: 5
- **Notes**: Equivalent to "failed breakout" / Wyckoff Spring.

### 4. Premium/Discount Equilibrium Bias (ICT)
- **What it is**: Inside a defined dealing range, only buy in the discount half (<50%) and only sell in the premium half (>50%).
- **Asset class implied**: any
- **Horizon**: weeks
- **Entry/exit rule**: Compute 50% midpoint of last 60-day range. Only allow long entries when price < 50%; only short when price > 50%. Combine with another trigger (e.g., RSI<30).
- **Where heard**: ICT (thesimpleict.com, arongroups.co).
- **Originality estimate**: 4
- **Backtest-feasibility**: 5
- **Notes**: Mean reversion in a range, rebranded. Filter, not trigger.

### 5. Wyckoff Spring (Daily)
- **What it is**: After a multi-week trading range, price briefly breaks the range low on a single bar and reclaims it on high volume.
- **Asset class implied**: equities, crypto
- **Horizon**: weeks to months
- **Entry/exit rule**: Identify a 30+ day range. If daily low < range low but close > range low and daily volume > 1.5x 20-day average, go long next open. Stop below spring low. Target range high; trail with 20-day MA.
- **Where heard**: Wyckoff (1930s); trendspider.com, luxalgo.com; very common on TikTok crypto accounts.
- **Originality estimate**: 4
- **Backtest-feasibility**: 5

### 6. Sam Seiden Supply/Demand Zone "Set & Forget"
- **What it is**: Demand zone = tight base (1-3 candles) preceded by a sharp move down and followed by a sharp move up. Place a limit buy at the zone high.
- **Asset class implied**: any
- **Horizon**: days to weeks
- **Entry/exit rule**: Find daily pattern where (a) prior 3 candles dropped >2 ATR, (b) next 1-3 candles ranged within 0.5 ATR, (c) following candle rallied >2 ATR. Limit buy at base high; stop below base low; 3:1 R:R target.
- **Where heard**: Sam Seiden (Online Trading Academy).
- **Originality estimate**: 5
- **Backtest-feasibility**: 4
- **Notes**: Published win rates unaudited; many zones get blown through.

### 7. 0.618 Fibonacci Pullback in Uptrend
- **What it is**: In an established uptrend, price pulling back to 61.8% retracement of the prior leg is a high-probability long.
- **Asset class implied**: any
- **Horizon**: weeks
- **Entry/exit rule**: Identify last 50-day swing low to swing high. If 50-day MA is rising and price retraces into 50%-61.8% zone with a bullish daily engulfing candle, go long. Stop below swing low. Target 1.618 extension.
- **Where heard**: Universal TA; Trading Rush 100-trade test; Prechter's Elliott Wave.
- **Originality estimate**: 2
- **Backtest-feasibility**: 4

### 8. Bullish Butterfly Harmonic Pattern
- **What it is**: 5-point XABCD harmonic pattern: AB=0.786 XA, BC=0.382-0.886 AB, CD=1.272-1.618 BC, ending below X.
- **Asset class implied**: equities, FX
- **Horizon**: days to weeks
- **Entry/exit rule**: Long at point D; stop below 1.618 extension of XA; targets at 0.382 and 0.618 retrace of CD.
- **Where heard**: Scott Carney's "Harmonic Trading"; liberatedstocktrader.com claims 85% win rate.
- **Originality estimate**: 6
- **Backtest-feasibility**: 3
- **Notes**: Self-reported stats suspect; pattern definitions allow subjective leeway.

### 9. Elliott Wave 3 Entry
- **What it is**: After completed Wave 1 up and corrective Wave 2 down holds above Wave 1 origin, enter long expecting strongest Wave 3.
- **Asset class implied**: any
- **Horizon**: weeks to months
- **Entry/exit rule**: When price breaks above end of Wave 1, enter long. Stop at origin of Wave 1. Target 1.618 extension of Wave 1.
- **Where heard**: R.N. Elliott, Prechter.
- **Originality estimate**: 4
- **Backtest-feasibility**: 2 (wave labeling is subjective)

### 10. RSI(2) Mean Reversion (Larry Connors)
- **What it is**: 2-period RSI on daily; buy oversold (RSI<5), exit on cross above 5-day SMA, conditional on price above 200-day MA.
- **Asset class implied**: equities, indices
- **Horizon**: days (3-7 day average hold)
- **Entry/exit rule**: If close > 200-day SMA and RSI(2) < 5, buy at close. Exit when close > 5-day SMA.
- **Where heard**: Larry Connors, *Short Term Trading Strategies That Work*; QuantifiedStrategies.com.
- **Originality estimate**: 5
- **Backtest-feasibility**: 5
- **Notes**: ~9% annual return on SPY 1993-present, invested only 28% of time; mean-reversion edge has decayed since ~2010 in single names but persists in indices.

### 11. Golden Cross (50/200 SMA)
- **What it is**: Buy when 50-day SMA crosses above 200-day SMA; sell on opposite cross.
- **Asset class implied**: equities, indices
- **Horizon**: months to years
- **Entry/exit rule**: Long SPY when SMA50 > SMA200; flat when SMA50 < SMA200.
- **Where heard**: Universal TA.
- **Originality estimate**: 1
- **Backtest-feasibility**: 5
- **Notes**: ~79% win rate per tosindicators.com; trails buy-and-hold on raw returns but better risk-adjusted. Whipsaws in chop.

### 12. Bollinger Band Squeeze Breakout (Daily)
- **What it is**: When 20-day BB width contracts below a lookback minimum, expect volatility expansion; trade the breakout direction.
- **Asset class implied**: any
- **Horizon**: days to weeks
- **Entry/exit rule**: When BB(20,2) width is at a 6-month low, wait for daily close outside bands. Enter in breakout direction; stop on opposite band; target = 1x prior squeeze range.
- **Where heard**: John Bollinger.
- **Originality estimate**: 3
- **Backtest-feasibility**: 5

### 13. MACD Bullish Divergence
- **What it is**: Price makes a lower low but MACD histogram makes a higher low.
- **Asset class implied**: equities
- **Horizon**: days to weeks
- **Entry/exit rule**: On daily, when price prints a 20-day low but MACD histogram is above its value at the prior 20-day low, buy next open. Exit on MACD signal-line cross down.
- **Where heard**: Universal TA; TradingView EdgeTools 14M-test study.
- **Originality estimate**: 3
- **Backtest-feasibility**: 5
- **Notes**: EdgeTools study: +0.32pp/trade average edge on histogram divergence longs — economically marginal after costs.

### 14. Pre-FOMC Drift (Lucca-Moench)
- **What it is**: SPX earns abnormally large excess returns in the 24h before scheduled FOMC announcements.
- **Asset class implied**: equities (SPX)
- **Horizon**: 1-2 days
- **Entry/exit rule**: Buy SPY at close T-1 before FOMC; sell at announcement (2pm ET on FOMC day). 8 meetings/year.
- **Where heard**: Lucca & Moench (2015) NY Fed Staff Report.
- **Originality estimate**: 6
- **Backtest-feasibility**: 4
- **Notes**: >80% of equity premium 1994-2011. Follow-up (2020) finds drift essentially disappeared post-2015.

### 15. Santa Claus Rally
- **What it is**: SPX rallies during last 5 trading days of December plus first 2 of January.
- **Asset class implied**: equities (SPX)
- **Horizon**: 7 trading days
- **Entry/exit rule**: Buy SPY at close on 5th-to-last trading day of December; sell at close on 2nd trading day of January.
- **Where heard**: Yale Hirsch, Stock Trader's Almanac (1972).
- **Originality estimate**: 2
- **Backtest-feasibility**: 5
- **Notes**: ~1.3% average gain, positive ~80% of years.

### 16. First Five Days of January Indicator
- **What it is**: If SPX up over first 5 trading days of January, full year tends positive.
- **Asset class implied**: equities (SPX)
- **Horizon**: 12 months
- **Entry/exit rule**: If SPX cumulative Jan day 1-5 > 0, long SPY rest of year; if not, stay in cash.
- **Where heard**: Hirsch's Stock Trader's Almanac.
- **Originality estimate**: 3
- **Backtest-feasibility**: 5
- **Notes**: 85% positive when first 5 days are positive (median 15.9%); only 44% when negative.

### 17. Sell in May / Halloween Effect
- **What it is**: Equities deliver almost all returns in Nov-Apr; May-Oct returns near zero.
- **Asset class implied**: equities (broad)
- **Horizon**: 6 months on, 6 off
- **Entry/exit rule**: Long SPY from close of last trading day in October to close of last trading day in April; T-bills otherwise.
- **Where heard**: Bouman & Jacobsen (2002, AER); Zhang/Jacobsen 2018, 2021.
- **Originality estimate**: 2
- **Backtest-feasibility**: 5
- **Notes**: Found in 36/37 countries.

### 18. Turn-of-the-Month Effect (Lakonishok-Smidt)
- **What it is**: Equity excess returns concentrated in 4-day window: last trading day of month + first 3 of next month.
- **Asset class implied**: equities
- **Horizon**: 4 trading days/month
- **Entry/exit rule**: Long SPY at close of T-1, exit at close of T+3 of new month. Cash otherwise.
- **Where heard**: Lakonishok & Smidt (1988); Xu & McConnell (SSRN 917884).
- **Originality estimate**: 6
- **Backtest-feasibility**: 5
- **Notes**: ALL excess DJIA return 1897-2005 occurred in this window per Xu/McConnell.

### 19. Options Expiration Week Effect (Monthly OPEX)
- **What it is**: SPX returns the week of 3rd-Friday monthly options expiration are systematically positive.
- **Asset class implied**: equities indices
- **Horizon**: 1 week/month
- **Entry/exit rule**: Long SPY at close of Friday before monthly OPEX week; sell at close of OPEX Friday.
- **Where heard**: Quantpedia; QuantifiedStrategies.
- **Originality estimate**: 6
- **Backtest-feasibility**: 5
- **Notes**: Attributed to delta-hedge unwinding. April strongest; July/Jan weakest.

### 20. JPM Collar (JHEQX) Quarterly Pin
- **What it is**: JPMorgan's JHEQX fund rolls a ~$22B SPX collar at end of each quarter. Market makers' gamma hedging tends to "pin" SPX near short-put strike, then volatility expands after roll.
- **Asset class implied**: equities indices, vol
- **Horizon**: ~1-2 weeks around quarterly expiration
- **Entry/exit rule**: Long vol (VIX calls or long straddle) entered ~3 trading days before quarter-end OpEx; exit 5 days after roll.
- **Where heard**: SpotGamma, MenthorQ.
- **Originality estimate**: 7
- **Backtest-feasibility**: 2 (need historical JHEQX strikes)

### 21. Gamma Flip Level Regime Filter (SpotGamma)
- **What it is**: SPX above the dealer net-gamma "flip" level → vol-dampening, mean-reverting; below → vol-amplifying, trending.
- **Asset class implied**: equities indices
- **Horizon**: days to weeks
- **Entry/exit rule**: When SPX > Gamma Flip, run mean-reversion (e.g., RSI(2)) on SPY. When SPX < Gamma Flip, run trend-following instead.
- **Where heard**: SpotGamma; insiderfinance.io.
- **Originality estimate**: 8
- **Backtest-feasibility**: 1 (requires GEX data — paid)
- **Notes**: Not directional — regime filter. Genuinely novel relative to TikTok content.

### 22. Yield-Curve Inversion as Lagged Recession/Equity Signal
- **What it is**: When 10Y-2Y spread goes negative, recession follows on 6-24 month lag, but equities often rally first.
- **Asset class implied**: equities, bonds
- **Horizon**: months to years
- **Entry/exit rule**: When T10Y2Y first crosses below 0, set a 12-month timer; on month 12 post-inversion, reduce equity exposure to 50% until curve re-steepens above +50bps.
- **Where heard**: Estrella/Hardouvelis (1991); Chicago Fed letter 2018.
- **Originality estimate**: 3
- **Backtest-feasibility**: 5

### 23. Presidential Cycle Year 3
- **What it is**: Year 3 of US presidential term is by far the best equity year historically.
- **Asset class implied**: equities
- **Horizon**: 1 year
- **Entry/exit rule**: Hold 100% SPY during 3rd calendar year of every presidential term; revert to 60/40 or cash other years.
- **Where heard**: Stock Trader's Almanac; Beyer (J. Portfolio Management).
- **Originality estimate**: 4
- **Backtest-feasibility**: 5
- **Notes**: Average year-3 return ~16-17%, positive ~90% of cycles since 1933.

### 24. Post-Earnings Announcement Drift (PEAD)
- **What it is**: Stocks with positive earnings surprises continue drifting up ~60 days post-announcement; negative drift down.
- **Asset class implied**: equities (single names)
- **Horizon**: 1-3 months
- **Entry/exit rule**: SUE = (actual EPS - consensus) / std of estimates. Long top decile, short bottom, equal-weighted, rebalanced quarterly with 60-day hold.
- **Where heard**: Bernard & Thomas (1989, 1990).
- **Originality estimate**: 5
- **Backtest-feasibility**: 3
- **Notes**: ~5.1% over 3 months for long/short hedge portfolio. Shrunk but not vanished post-2010.

### 25. VIX Spike Above 30 Contrarian Buy
- **What it is**: VIX closing above 30 indicates panic; subsequent equity returns tend above-average.
- **Asset class implied**: equities indices, vol
- **Horizon**: 1-3 months
- **Entry/exit rule**: When VIX closes >30 having previously been <20 within last 60 days, buy SPY at next open. Exit when VIX closes <20 OR after 60 trading days.
- **Where heard**: Whaley papers; standard chart-school material.
- **Originality estimate**: 3
- **Backtest-feasibility**: 5
- **Notes**: VIX>50 historically marked durable lows (2008, 2020). Heavy left-skew risk.

### 26. AQR "Buy the Dip" Anti-Signal
- **What it is**: Meta-signal: waiting for a 5/10/20% pullback before deploying cash UNDERPERFORMS simple DCA in most calibrations.
- **Asset class implied**: equities indices
- **Horizon**: years
- **Entry/exit rule**: Counter-rule: do NOT condition deployment on pullback. Equal monthly DCA beats >60% of 196 "buy the dip" variants per AQR.
- **Where heard**: AQR (2025).
- **Originality estimate**: 6
- **Backtest-feasibility**: 5
- **Notes**: Null hypothesis benchmark.

## Agent notes on the two TikTok accounts
- **@deltatrendtrading (Thomas)**: Markets as "Ivy League quant finance education for free." Content is mostly meta/educational: codes and backtests strategies (e.g., MCMC-based in video 7402378014115335467), critiques other creators' setups. Accused of posting strategy videos without ample evidence and ulterior motives via a "1of1" partnership.
- **@v.tpk (V!)**: Account exists and posts finance content. No secondhand summaries, Reddit threads, YouTube reposts, or critiques found describing specific strategies taught. Would need direct viewing or a more specific seed.
