# Phase 1E — Crypto + Macro/Cross-Asset Signals
# Returned by research agent.

### 1. Perpetual Funding Rate Extreme (BTC/ETH)
- **What it is**: Funding rate paid by longs to shorts on perpetual swaps; persistent extremes indicate one-sided leverage that tends to reverse.
- **Asset class**: crypto
- **Horizon**: 1-30 days
- **Entry/exit rule**: 8h funding rate on Binance BTC/USDT perp; rolling 90-day z-score. Short perp when z > 2; long when z < -2 or funding deeply negative. Hold 5-10 days.
- **Data source**: free (Binance/Bybit public API, Coinglass, CryptoQuant free)
- **Source**: https://www.coinglass.com/funding
- **Reported edge**: Anecdotal: extreme +funding (>0.15% / 8h) preceded 10-30% corrections in 2021; deep -funding marked Mar 2020 bottom.
- **Originality**: 4
- **Backtest-feasibility**: 5
- **Notes**: Strongest combined with OI expansion and liquidation cluster.

### 2. Stablecoin Supply Ratio (SSR)
- **What it is**: BTC market cap / aggregate stablecoin market cap. Low SSR = stablecoin dry powder large relative to BTC = latent buying.
- **Asset class**: crypto
- **Horizon**: weeks to months
- **Entry/exit rule**: SSR Oscillator (200d SMA of SSR within Bollinger Bands(200,2)). Long BTC when oscillator < -0.8; reduce when > +0.8.
- **Data source**: free tier (Glassnode, DefiLlama)
- **Source**: https://academy.glassnode.com/indicators/stablecoin/ssr-stablecoin-supply-ratio
- **Reported edge**: Tagged macro lows in 2018, 2020, 2022.
- **Originality**: 5
- **Backtest-feasibility**: 4

### 3. Coinbase Premium Index
- **What it is**: Price spread of BTC on Coinbase (USD) vs Binance (USDT). Positive = US institutional spot demand outpacing global retail.
- **Asset class**: crypto
- **Horizon**: 1-30 days
- **Entry/exit rule**: 24h MA. Long bias when 24h MA > +0.05% sustained 3+ days; reduce/short when < -0.05%.
- **Data source**: free (CryptoQuant chart, Coinglass)
- **Source**: https://cryptoquant.com/asset/btc/chart/market-data/coinbase-premium-index
- **Reported edge**: Anecdotal; preceded 2020-2021 and 2024 ETF rallies.
- **Originality**: 5
- **Backtest-feasibility**: 4
- **Notes**: Diluted by ETF flows post-2024.

### 4. Exchange Netflow (BTC/ETH)
- **What it is**: Daily on-chain net flow into/out of exchange wallets.
- **Asset class**: crypto
- **Horizon**: 3-30 days
- **Entry/exit rule**: 7-day rolling sum of netflow z-scored over 365d. Bearish when z > +1.5; bullish when z < -1.5.
- **Data source**: free (CryptoQuant, Glassnode free tier)
- **Reported edge**: Spikes marked local tops (Nov 2021); outflows preceded Apr 2020, Jan 2023 rallies.
- **Originality**: 4
- **Backtest-feasibility**: 3
- **Notes**: Less reliable since 2024 due to ETF custody flows.

### 5. Hash Ribbons (Charles Edwards)
- **What it is**: Crossover of 30-day SMA of hashrate above 60-day SMA after miner capitulation.
- **Asset class**: crypto
- **Horizon**: 3-12 months
- **Entry/exit rule**: Long BTC when 30d hashrate MA crosses back above 60d MA after ≥14 days below. Exit on next miner capitulation or fixed target.
- **Data source**: free (Blockchain.com, mempool.space)
- **Source**: https://www.tradingview.com/script/kT7jIvqv-Hash-Ribbons/
- **Reported edge**: 14 signals since 2013, ~64% profitable; examples: Jan 2019 +56% in 90d, Mar 2020 +82%, Jul 2023 +25%.
- **Originality**: 6
- **Backtest-feasibility**: 5

### 6. MVRV Z-Score
- **What it is**: (Market Cap - Realized Cap) / stdev(Market Cap). Standardized unrealized P/L across all holders.
- **Asset class**: crypto
- **Horizon**: 6-24 months
- **Entry/exit rule**: Long when Z < 0 (accumulation); de-risk when Z > 6.85.
- **Data source**: free (Glassnode, BitcoinMagazinePro, Coin Metrics community)
- **Reported edge**: Marked every major cycle high within ~2 weeks (2013, 2017, 2021); bottoms 2015, 2018, 2022.
- **Originality**: 3
- **Backtest-feasibility**: 5
- **Notes**: ETF era may dampen signal.

### 7. Puell Multiple
- **What it is**: Daily USD value of newly issued BTC / 365d MA. Tracks miner revenue stress.
- **Asset class**: crypto
- **Horizon**: 6-24 months
- **Entry/exit rule**: Long when < 0.5 (miner capitulation); de-risk when > 3.0.
- **Data source**: free (BitcoinMagazinePro, Newhedge, Bitbo)
- **Reported edge**: Marked 2012, 2015, 2018, 2022 bottoms; > 4 marked 2013, 2017, 2021 tops.
- **Originality**: 4
- **Backtest-feasibility**: 5

### 8. NUPL (Net Unrealized Profit/Loss)
- **What it is**: (Market Cap - Realized Cap) / Market Cap.
- **Asset class**: crypto
- **Horizon**: 6-24 months
- **Entry/exit rule**: Scale into longs when NUPL < 0; scale out when > 0.75. Use LTH-NUPL for cleaner signal.
- **Data source**: free (Glassnode, LookIntoBitcoin)
- **Reported edge**: > 0.75 aligned with 5 historical tops; < 0 marked Jan 2015, Mar 2020, Nov 2022.
- **Originality**: 4
- **Backtest-feasibility**: 4
- **Notes**: Highly correlated with MVRV.

### 9. Spot BTC ETF Net Flows
- **What it is**: Daily aggregate creations/redemptions across US spot BTC ETFs (since Jan 2024).
- **Asset class**: crypto
- **Horizon**: 1-30 days
- **Entry/exit rule**: 5-day sum. Long bias when > $1B and trending; reduce when < -$500M for 3 days. Cross-check with IBIT premium/discount.
- **Data source**: free (Farside, CoinGlass, TheBlock, SoSoValue)
- **Reported edge**: Early ETF flows predictive of relative ranking by day 42-63 (CFBenchmarks); aggregate flows correlate with BTC spot.
- **Originality**: 6
- **Backtest-feasibility**: 5
- **Notes**: Short history (~16 months).

### 10. Coin Days Destroyed / Old-Supply Wake-up
- **What it is**: For each spent UTXO, BTC moved × dormancy in days. Spikes = long-term holders moving = distribution.
- **Asset class**: crypto
- **Horizon**: 1-12 months
- **Entry/exit rule**: 90d MA of CDD. Bearish when 7d CDD > 3x 90d MA AND > 5y HODL bucket shrinks. Bullish accumulation when CDD suppressed + > 2y supply growing.
- **Data source**: free (Glassnode, BitcoinMagazinePro HODL Waves)
- **Reported edge**: Spikes preceded each cycle top (late-2017, Apr/Nov-2021); > 5y HODL growth preceded next bull.
- **Originality**: 6
- **Backtest-feasibility**: 4
- **Notes**: Cleanest BTC-native long-cycle structural metric.

### 11. Halving Cycle Calendar
- **What it is**: Block subsidy halves every 210k blocks (~4 years). Returns concentrate 12-18 months post-halving.
- **Asset class**: crypto
- **Horizon**: multi-year
- **Entry/exit rule**: Overweight BTC from 6 months before halving until 18 months after; underweight 18-36 months after.
- **Data source**: free (any block explorer)
- **Reported edge**: 3 complete cycles; peaks ~12-18 months post-halving in 2013, 2017, 2021. Stock-to-flow failed 2024.
- **Originality**: 2
- **Backtest-feasibility**: 5
- **Notes**: N=3.

### 12. BTC-Nasdaq / BTC-Gold Correlation Regime
- **What it is**: Rolling correlations of BTC vs NDX, GLD, DXY. Shifts indicate tech-risk proxy vs digital gold.
- **Asset class**: crypto / cross-asset
- **Horizon**: months
- **Entry/exit rule**: 60d corr. Tech regime: corr(BTC,NDX) > 0.4 → size as high-beta tech. Digital gold: corr(BTC,GLD) > 0.3 AND corr(BTC,DXY) < -0.3 → size as macro hedge.
- **Data source**: free (Yahoo, FRED for DXY)
- **Reported edge**: 2025-2026 corr(BTC,NDX) hit +0.35 to +0.6 (3-year high); corr(BTC,DXY) flipped positive Mar 2026 per JPM.
- **Originality**: 5
- **Backtest-feasibility**: 5
- **Notes**: Sizing/regime tool, not directional.

---

### 13. 3m10y Yield Curve (Estrella-Mishkin Recession Probit)
- **What it is**: 10Y minus 3m Treasury yield, fed into EM probit for 12-month-ahead recession probability.
- **Asset class**: bonds / equities
- **Horizon**: 6-18 months
- **Entry/exit rule**: De-risk equities when 3m10y < 0 sustained 30+ days; equity risk arrives 6-18 months AFTER inversion BEGINS TO STEEPEN BACK. Re-risk when EM probit drops below 20% from prior high.
- **Data source**: free (FRED: T10Y3M)
- **Source**: Estrella-Mishkin 1998.
- **Reported edge**: Preceded every NBER recession since 1968, no false positives in 3m10y spec.
- **Originality**: 2
- **Backtest-feasibility**: 5

### 14. Lumber/Gold Ratio (Gayed)
[Already cataloged in Phase 1C — deduplicate in consolidation.]

### 15. Copper/Gold Ratio → 10Y Yield
- **What it is**: Ratio of copper to gold; leads 10Y UST yield.
- **Asset class**: bonds / commodities
- **Horizon**: 1-6 months
- **Entry/exit rule**: 60d change in copper/gold. Ratio rises >10% over 60d while 10Y yield is flat/falling → bias short duration. Reverse when ratio falls >10% with yields elevated.
- **Data source**: free (Yahoo: HG=F, GC=F; FRED: DGS10)
- **Source**: Hua & Wang JIFMIM 2023.
- **Reported edge**: Short-term informational lead in expansionary regimes; weakened post-2022 (China demand shock).
- **Originality**: 5
- **Backtest-feasibility**: 5

### 16. Gold/Silver Ratio Mean Reversion
- **What it is**: Ratio of gold/silver spot. Long-run mean ~65-70:1. Extremes mean-revert.
- **Asset class**: commodities
- **Horizon**: 6-24 months
- **Entry/exit rule**: Long silver/short gold when ratio > 90; reverse when < 50. Exit at 70. Use 5y z-score for robustness.
- **Data source**: free (Yahoo)
- **Reported edge**: > 100 only seen in 1991 recession, 2020 COVID, Apr 2025 — each followed by sharp reversion to <80 within 12 months.
- **Originality**: 4
- **Backtest-feasibility**: 5

### 17. COT — Commercial Hedger Extreme
- **What it is**: CFTC COT weekly: commercial hedgers vs managed money. Commercial extremes contrarian.
- **Asset class**: commodities / FX / bonds
- **Horizon**: 4-12 weeks
- **Entry/exit rule**: 3-year percentile of commercial net position. Long when > 90th percentile long; short when < 10th. Hold 4-12 weeks.
- **Data source**: free (CFTC.gov, Barchart, Quandl mirror)
- **Source**: Bohl/Sulewski 2023.
- **Reported edge**: Published backtests: SoFR/equity index/MSCI EAFE Sharpe 1.24-2.09 vs SPY 1.07.
- **Originality**: 5
- **Backtest-feasibility**: 5
- **Notes**: Tuesday-as-of, 3-day lag. Disaggregated report preferred over Legacy.

### 18. High Yield OAS Credit Spread (HY OAS)
- **What it is**: ICE BofA US HY Index OAS over Treasuries. Real-time pricing of sub-IG default risk.
- **Asset class**: bonds / equities
- **Horizon**: 1-12 months
- **Entry/exit rule**: De-risk when HY OAS rises 100bps from 6m low AND breaches 500bps. Re-risk when HY OAS rolls over from local peak by >100bps.
- **Data source**: free (FRED daily: BAMLH0A0HYM2)
- **Reported edge**: Clear in 2007-08, 2015-16 energy stress; lagged 2020 (single-month shock). > 800bps marked equity capitulation lows.
- **Originality**: 3
- **Backtest-feasibility**: 5

### 19. ACM 10Y Term Premium
- **What it is**: NY Fed Adrian-Crump-Moench decomposition of 10Y UST yield into expected rates + term premium.
- **Asset class**: bonds / equities
- **Horizon**: months to years
- **Entry/exit rule**: When ACM 10Y term premium rises > 50bps from 12m low (especially crossing from negative to positive), bias short duration. When it falls > 50bps with growth slowing, lean long duration.
- **Data source**: free (NY Fed, FRED mirrors)
- **Reported edge**: Recent rise from ~-100bps (2020) to ~+50bps (2025-26) coincides with bear-steepener regimes pressuring long-duration equities.
- **Originality**: 7
- **Backtest-feasibility**: 5

### 20. MOVE/VIX Ratio (Cross-Asset Vol)
- **What it is**: MOVE (Treasury implied vol) / VIX. High ratio = rates vol dominates; low = equity vol dominates.
- **Asset class**: bonds / equities / FX
- **Horizon**: weeks to months
- **Entry/exit rule**: Sustained ratio > 6 (vs ~4 mean) → rates-driven regime, bias defensive on duration-sensitive equities. Reversion below 4 → risk-on.
- **Data source**: free (Yahoo: ^MOVE, ^VIX)
- **Reported edge**: SOA Sep 2025 and CFA Institute Jul 2025: bond vol leads equity vol in volatile moments; Apr-2025 ratio spike preceded VIX catch-up by ~2 weeks.
- **Originality**: 7
- **Backtest-feasibility**: 4

### 21. TIPS Breakeven (10Y) — Inflation Expectations
- **What it is**: 10Y TIPS breakeven (T10YIE on FRED) as inflation expectations indicator.
- **Asset class**: commodities, TIPS
- **Horizon**: 1-3 months
- **Entry/exit rule**: Long broad commodities (DBC) when T10YIE > 60d MA AND Δ > +20bps over 30d.
- **Data source**: free (FRED: T10YIE)
- **Originality**: 5
- **Backtest-feasibility**: 5
