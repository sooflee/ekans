# Phase 1M — Hunt Round 4: high-CAGR-potential signals
# Returned by hunt agent. 12 signals, originality 8-10, one-line rules, targeting >10% CAGR.

### M-1 ETH Staking Yield Compression
- **Rule**: Long ETH when 30d MA of consensus-layer staking APR drops below 2.8%.
- **Data**: beaconcha.in/api/v1/epoch/latest (validator count + effective balance)
- **Mechanism**: Falling APR = validator queue saturated = capital locked + new buyers must pay spot.
- **CAGR potential**: 20-30%
- **Originality**: 9

### M-2 DAI Savings Rate Jump
- **Rule**: Long ETH day after MakerDAO DSR is increased by >100 bp in a single governance vote.
- **Data**: api.makerburn.com/status + Etherscan free API for historical DSR events
- **Mechanism**: DSR hike = MakerDAO competing for stablecoin TVL → DeFi yield-chasing rotation bids ETH.
- **CAGR potential**: 15-25%
- **Originality**: 10

### M-3 stETH Discount Extreme
- **Rule**: Long ETH when Curve stETH/ETH pool price < 0.985 (>1.5% discount).
- **Data**: api.curve.fi + Curve subgraph (free)
- **Mechanism**: Deep stETH discounts = forced-liquidity events (Celsius June 2022, UST May 2022) = capitulation lows.
- **CAGR potential**: 25%+
- **Originality**: 10

### M-4 ETH/BTC Ratio Breakout
- **Rule**: Long ETH when ETH/BTC closes above trailing 200d MA after being below it for ≥60 consecutive days.
- **Data**: CoinGecko free API
- **Mechanism**: BTC leads cycles; ETH/BTC turn-up signals altseason rotation.
- **CAGR potential**: 15-20%
- **Originality**: 8

### M-5 TGA Cash Drain → Long SPY
- **Rule**: Long SPY when weekly change in Treasury General Account balance is more negative than -$100B.
- **Data**: FRED WTREGEN (weekly)
- **Mechanism**: TGA drawdown injects liquidity into banking system (reserves rise 1:1) → QE-like for risk assets.
- **CAGR potential**: 12-18%
- **Originality**: 9

### M-6 Fed Swap Line Drawdown → Short SPY
- **Rule**: Short SPY when total Fed central-bank liquidity swaps outstanding exceeds $10B (vs baseline ~$200M).
- **Data**: FRED SWPT (weekly H.4.1)
- **Mechanism**: Fed activates swap lines only during acute foreign-bank dollar funding stress (Mar 2020, Mar 2023 SVB/CS).
- **CAGR potential**: 15-20% (asymmetric crisis-only)
- **Originality**: 10

### M-7 Treasury Buyback Operation → Long TLT
- **Rule**: Long TLT for 5 trading days after any week where Treasury conducts regular buyback operation >$4B nominal coupons.
- **Data**: home.treasury.gov/data/treasury-securities-buybacks (CSV)
- **Mechanism**: Buybacks remove off-the-run duration supply → compresses term premium = direct duration bid. Program revived 2024.
- **CAGR potential**: 10-15%
- **Originality**: 10

### M-8 Russell Reconstitution Pre-Position
- **Rule**: Long IWM from 15 trading days before annual Russell reconstitution (last Friday of June) through that date.
- **Data**: lseg.com FTSE Russell schedule (deterministic)
- **Mechanism**: Indexers must buy projected additions in size → predictable upward pressure on R2000 constituents.
- **CAGR potential**: 10-12% (15 days/year basis)
- **Originality**: 8

### M-9 Refinery Utilization Crack → Long XLE
- **Rule**: Long XLE when weekly US refinery capacity utilization drops >4 percentage points WoW.
- **Data**: EIA API WPULEUS3
- **Mechanism**: Sharp drops = unplanned outages (Hurricane Ida 2021, Texas freeze 2021) → crack spreads spike, refiner margins rise.
- **CAGR potential**: 12-18%
- **Originality**: 8

### M-10 GLD AUM Outflow Capitulation
- **Rule**: Long GLD when GLD tonnes-held drops >3% in any rolling 20-day window.
- **Data**: spdrgoldshares.com/usa/historical-data (daily CSV)
- **Mechanism**: Sharp outflows = retail-paper capitulation that mechanically transfers gold from paper back to physical at discount = bottom signal.
- **CAGR potential**: 10-15%
- **Originality**: 9

### M-11 Conference Board LEI: Manufacturing Hours → Short SPY
- **Rule**: Short SPY when AWHMAN falls >0.4 hours over any 6-month window.
- **Data**: FRED AWHMAN (Average Weekly Hours of Production Workers, Manufacturing)
- **Mechanism**: Manufacturers cut overtime BEFORE layoffs → most reliable LEI sub-component for recession lead by 3-6 months.
- **CAGR potential**: 12-15% (drawdown avoidance)
- **Originality**: 9

### M-12 Hong Kong Stock Connect Northbound Flow
- **Rule**: Long FXI when trailing 10-day cumulative Northbound Stock Connect net buy exceeds RMB 50B.
- **Data**: hkex.com.hk Mutual-Market Stock-Connect Statistics Historical Monthly (daily CSV)
- **Mechanism**: Northbound = global institutional flow into mainland A-shares; sustained inflows lead H-share/ADR rallies.
- **CAGR potential**: 15-20%
- **Originality**: 9
