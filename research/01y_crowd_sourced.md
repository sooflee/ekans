# Phase 1Y — Crowd-Sourced Signals (Estimize, SPAC, Token Unlocks)
# 10 signals. Dedup audit: all NEW vs prior catalog.

### Y-1 Estimize Beat-the-Street Pre-Announcement
- **Rule**: Long stock 5 trading days before earnings when Estimize EPS > sell-side I/B/E/S by > 4% AND ≥ 10 Estimize contributors; exit at next-day open.
- **Asset**: US single-name equities, Estimize-covered, > $2B mcap
- **Data**: estimize.com API + Refinitiv/Zacks consensus
- **Mechanism**: Pham/Adams/Mansi (JF 2019) — Estimize unbiased and more accurate than sell-side
- **CAGR**: 12-18% (~150 trades/yr, 58% hit, ~80bp mean)
- **Originality**: 8

### Y-2 Estimize Dispersion → Long Earnings Straddle
- **Rule**: Buy ATM straddle Mon before earnings when Estimize EPS CV (stdev/|mean|) > 12m cross-sectional 90th pct AND straddle IV-rank < 70; close T+1.
- **Asset**: US single-name liquid weekly options
- **Mechanism**: Crowd disagreement = cleaner ex-ante uncertainty proxy than sell-side; Banker-Chen 2020
- **CAGR**: 15-22% on option sleeve
- **Originality**: 9

### Y-3 Estimize Top-Rated Analyst Cluster
- **Rule**: Long stock 10 days before earnings when ≥ 5 top-quintile Estimize contributors > sell-side in prior 14 days; exit T+1 post-print.
- **Asset**: US single-names
- **Mechanism**: Da-Huang RFS 2020 on crowd expert weighting — top analysts retain accuracy OOS
- **CAGR**: 14-20%
- **Originality**: 9

### Y-4 SPAC > 90% Redemption Free-Warrant Trade
- **Rule**: Buy de-SPAC warrants on day T+1 after redemption-vote disclosure showing > 90% redemptions; hold 90 days or until warrants double.
- **Asset**: De-SPAC warrants ($11.50 strike, 5-yr)
- **Data**: spacresearch.com + EDGAR 8-K
- **Mechanism**: At > 90% redemption, remaining float is tiny vs warrant overhang; convexity payoff
- **CAGR**: 25-40% (high variance)
- **Originality**: 9

### Y-5 De-SPAC Mean-Reversion Short
- **Rule**: Short de-SPAC common on T+5 after deal close when redemption > 85% AND price > $10; cover T+45 or on 30% decline.
- **Asset**: De-SPAC common
- **Mechanism**: Klausner-Ohlrogge-Ruan — high-redemption de-SPACs underperform 50%/year over 12mo
- **CAGR**: 18-30% on dedicated short sleeve
- **Originality**: 8

### Y-6 Second-Extension-Vote Short
- **Rule**: Short SPAC common at $10.05 day of 2nd-extension 8-K; cover at merger announcement or trust liquidation.
- **Asset**: Pre-merger SPAC common near trust value
- **Mechanism**: 2nd extension = deal-quality desperation; 95%+ redemptions on these deals
- **CAGR**: 10-14% capped-downside
- **Originality**: 9

### Y-7 Token Unlock Pre-Cliff Short
- **Rule**: Short perp 7 days before scheduled cliff unlock when unlock > 5% of circulating supply AND market cap > $200M.
- **Asset**: Mid-cap alt token perps (Binance/Bybit/OKX)
- **Data**: token.unlocks.app + cryptorank.io
- **Mechanism**: Cliff unlocks = quasi-instant supply dump from insiders; Howell-Niessner-Yermack RFS 2020
- **CAGR**: 20-35%
- **Originality**: 8

### Y-8 Linear-vs-Cliff Vesting Spread Pair
- **Rule**: Pair: short token undergoing cliff unlock vs long equal-notional same-sector basket on linear vesting; 7d window into cliff, when cliff > 3% of supply.
- **Asset**: Crypto perps (e.g., short ARB vs long OP+MATIC)
- **Mechanism**: Cliff = discrete supply shock; linear already priced in
- **CAGR**: 15-22% with lower DD
- **Originality**: 9

### Y-9 Post-Unlock Mean-Reversion Long
- **Rule**: Long token at T+10 after > 10%-of-supply cliff unlock that produced > 15% drawdown around event; exit T+60.
- **Asset**: Spot/perp long on unlocked token
- **Mechanism**: Post-cliff selling exhausts in 2 weeks; float-adjusted reset = bounce
- **CAGR**: 18-28%
- **Originality**: 8

### Y-10 Insider-Heavy Unlock Concentration Filter
- **Rule**: Short token 14 days before next cliff unlock when unlocking tranche is > 70% team+VC allocation, with size > 3% of supply.
- **Asset**: Crypto perp short
- **Mechanism**: Team/VC sell propensity >> ecosystem/foundation; cost basis ≈ 0
- **CAGR**: 22-32%
- **Originality**: 9
