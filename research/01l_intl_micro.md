# Phase 1L — International + Microstructure Signals
# Returned by hunt agent. 15 signals, originality 8-9, one-line rules.

### L-1 Treasury Fail-to-Deliver Spike
- **Rule**: Long ZT/IEF when DTCC weekly Treasury FTD value exceeds 26-week 90th percentile.
- **Data**: dtcc.com/charts/treasury-fails-charts (scrape weekly)
- **Mechanism**: FTD spikes = collateral scarcity in repo → short-covering pulls cash Treasuries and futures basis richer.
- **Originality**: 9

### L-2 On-the-Run / Off-the-Run 10Y Spread
- **Rule**: Short SPY when on-the-run minus off-the-run 10Y yield spread widens >3bp above 60d mean.
- **Data**: FRED DGS10 + treasurydirect.gov CUSIP-level
- **Mechanism**: Widening OTR/OFTR = liquidity hoarding in most-liquid issue = flight-to-quality precursor (LTCM '98, Mar '20).
- **Originality**: 9

### L-3 Treasury Auction Direct Bidder Share Collapse
- **Rule**: Short TLT for 5 sessions after 10Y/30Y auction where Direct allotment falls below 5%.
- **Data**: treasurydirect.gov/auctions/results (parse XML/PDF)
- **Mechanism**: Direct bidders are real-money domestic; their absence forces dealers to warehouse paper → hedge by shorting futures.
- **Originality**: 9

### L-4 SNB Sight Deposits Surge
- **Rule**: Short EUR/CHF when SNB weekly sight deposits rise >2σ of 52w change.
- **Data**: data.snb.ch weekly Monday release
- **Mechanism**: Sight-deposit growth = SNB FX intervention; cessation removes the floor (1.20 break 2015).
- **Originality**: 9

### L-5 NRC Nuclear Capacity Outage Cluster
- **Rule**: Long NG futures when NRC daily reactor status shows >8 GW US nuclear capacity offline simultaneously.
- **Data**: nrc.gov/reading-rm/doc-collections/event-status/reactor-status/ (daily)
- **Mechanism**: Each GW lost requires ~0.15 Bcf/d substitute gas-fired generation; >8 GW outages front-run Henry Hub rallies before EIA storage confirms.
- **Originality**: 9

### L-6 SEC NT-10K / NT-10Q Late Filings
- **Rule**: Short any stock filing NT-10K or NT-10Q on the day of filing through the actual late filing.
- **Data**: SEC EDGAR full-text https://efts.sec.gov/LATEST/search-index?q=&forms=NT-10-K,NT-10-Q (daily)
- **Mechanism**: NT filings disclose inability to meet deadline → empirically correlates with auditor disputes, restatements, material weakness.
- **Originality**: 9
- **Note**: Almost mechanical — extremely promising

### L-7 DeFiLlama Stablecoin Chain Migration
- **Rule**: Long ETH when USDC supply on Ethereum L1 grows >5% WoW relative to total USDC across chains.
- **Data**: stablecoins.llama.fi/stablecoinchains (daily API)
- **Mechanism**: Stablecoin flow back to mainnet precedes ETH buying; users bridge home to deploy into ETH-denominated DeFi.
- **Originality**: 8

### L-8 Drewry World Container Index Acceleration
- **Rule**: Long XLI when Drewry WCI Shanghai-LA route posts 3 consecutive weekly increases >5%.
- **Data**: drewry.co.uk/supply-chain-advisors (Thursday weekly)
- **Mechanism**: Sustained transpacific freight rate spikes = US importer restocking demand → inventory builds + capex orders ~1 quarter later.
- **Originality**: 8

### L-9 Deribit BTC 25-Delta Risk Reversal Extreme
- **Rule**: Long BTC when Deribit 30-day 25-delta put-call skew exceeds +10 vol.
- **Data**: deribit.com/statistics/BTC + public API
- **Mechanism**: Extreme put skew = capitulation hedging by leveraged longs; once marginal hedger has paid up, downside gamma is sated.
- **Originality**: 8

### L-10 LME Cancelled Warrants Ratio
- **Rule**: Long copper (HG) when LME copper cancelled warrants exceed 40% of total on-warrant stocks.
- **Data**: lme.com/en/Market-data/Reports-and-data/Stocks (daily)
- **Mechanism**: Cancelled warrants = metal earmarked for physical withdrawal → end-user purchase commitments tighten float, push cash-3M into backwardation.
- **Originality**: 8

### L-11 USPTO Patent Grant Velocity (by assignee)
- **Rule**: Long stock when weekly USPTO grants exceed 52w avg by >2σ for 3 consecutive weeks.
- **Data**: bulkdata.uspto.gov weekly grant bulk files + assignee aggregation
- **Mechanism**: Grant velocity lags R&D by ~3 years, leads product commercialization by ~6-12 months — pipeline maturity not yet in analyst models.
- **Originality**: 9

### L-12 USDA Crop Progress Pollination Stress
- **Rule**: Long ZC corn when USDA Crop Progress "Good/Excellent" rating drops >5 ppt WoW during July pollination window.
- **Data**: usda.library.cornell.edu Monday afternoon Crop Progress (PDF)
- **Mechanism**: Pollination-window deterioration is irreversible for yield; reprices into Aug WASDE.
- **Originality**: 8

### L-13 NYSE TRIN Capitulation
- **Rule**: Long SPY at close of any day where NYSE TRIN closes above 2.5.
- **Data**: Stooq ^TRIN daily (free)
- **Mechanism**: TRIN >2.5 = down-volume dwarfs declining-issue count = forced liquidation panic, not orderly distribution.
- **Originality**: 8

### L-14 McClellan Oscillator Zero-Cross
- **Rule**: Long SPY when NYSE McClellan Oscillator crosses from below -50 up through zero.
- **Data**: Stooq ^ADV, ^DEC daily (compute 19d EMA(A-D) - 39d EMA(A-D))
- **Mechanism**: Zero-line cross from oversold = breadth-thrust launch of medium-term up-legs.
- **Originality**: 8

### L-15 Bank of England Reserve Account Drift
- **Rule**: Short GBP/USD when UK commercial bank reserves at BoE (Bankstats A1.1.1) decline 4 consecutive weekly prints.
- **Data**: bankofengland.co.uk/statistics/weekly-report (weekly)
- **Mechanism**: Falling reserves = active QT or DMO gilt issuance draining sterling liquidity → tightens financial conditions → GBP weakness as growth deteriorates.
- **Originality**: 9
