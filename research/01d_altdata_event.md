# Phase 1D — Alt-Data / Event-Driven / Political Signals
# Returned by research agent.

### Pelosi Disclosed-Trade Mirror
- **What it is**: Mirror Nancy Pelosi's STOCK Act PTRs within the 45-day legal disclosure window. Tracks predominantly long-dated tech call buys by her husband Paul Pelosi.
- **Asset class**: US single-name equities, mostly large-cap tech
- **Horizon**: 3-12 months
- **Entry/exit rule**: On the trading day after a Pelosi PTR appears on the House Clerk site, buy the disclosed underlying (for call purchases, buy the stock or a delta-weighted call); equal-weight portfolio rebalanced as new disclosures arrive; exit 12 months after entry.
- **Data source**: House Clerk Financial Disclosures (free, PDF), Quiver Quant, Capitol Trades (free web)
- **Source**: Unusual Whales 2022 Congressional Trading Report; Ziobrowski et al. JFQA 2004; NANC ETF.
- **Reported edge**: Unusual Whales claimed Pelosi +65% in 2023 vs SPY +24%; NANC ETF outperformed SPY by ~5% in first 12 months. Ziobrowski found senators earned ~85bps/mo abnormal 1993-1998.
- **Originality**: 2
- **Backtest-feasibility**: 4
- **Notes**: Heavy mega-cap tech tilt = largely a levered beta/tech factor bet. ETHICS Act ban risk.

### House Financial Services Committee Subset
- **What it is**: Subset of congressional trading restricted to House Financial Services / Senate Banking members (private info on bank reg, Fed nominations, crypto rules).
- **Asset class**: Banks, broker-dealers, fintech, crypto-adjacent
- **Horizon**: 1-6 months
- **Entry/exit rule**: Day after PTR by committee member, 5% position, hold 90 days, equal-weight.
- **Data source**: Capitol Trades, Quiver Quant, house.gov rosters
- **Source**: Tahoun & van Lent JAR 2019; Eggers & Hainmueller QJPS 2014.
- **Reported edge**: Tahoun found committee members' bailed-bank stocks earned outsized returns; Eggers found overall congressional portfolios underperformed.
- **Originality**: 4
- **Backtest-feasibility**: 4

### Form 4 Cluster Insider Buying
- **What it is**: ≥3 insiders at same firm buying open-market within 10 trading days = strong bullish signal.
- **Asset class**: US single names
- **Horizon**: 6-12 months
- **Entry/exit rule**: Trigger = ≥3 distinct Form 4 P transactions, each ≥$25k, within rolling 10 trading days. Buy t+1 close; hold 252 days.
- **Data source**: SEC EDGAR Form 4 XML feed (free)
- **Source**: Lakonishok & Lee RFS 2001; Cohen/Malloy/Pomorski JF 2012.
- **Reported edge**: Lakonishok-Lee ~4-6% annualized excess; cluster buys 10-12% 12m abnormal in CMP.
- **Originality**: 3
- **Backtest-feasibility**: 5
- **Notes**: Best in small/mid cap; large-cap edge compressed. Filter out 10% holder buys.

### Opportunistic vs Routine Insider Trades
- **What it is**: CMP split insiders into "routine" (same calendar month every year) and "opportunistic." Opportunistic carries the alpha.
- **Asset class**: US equities
- **Horizon**: 6-12 months
- **Entry/exit rule**: Classify insiders by historical pattern (≥3 trades in same calendar month over prior 3 years = routine). Only act on opportunistic. Long opportunistic buys, short sells; hold 6 months.
- **Data source**: SEC EDGAR (free)
- **Source**: Cohen/Malloy/Pomorski JF 2012.
- **Reported edge**: 82 bps/month alpha (~10% annualized).
- **Originality**: 5
- **Backtest-feasibility**: 4

### Form 144 Pre-Sale Notice Drift
- **What it is**: Form 144 is filed by affiliates announcing intent to sell. Filings cluster before negative news.
- **Asset class**: US equities, small/mid-cap
- **Horizon**: 1-3 months
- **Entry/exit rule**: Short at t+1 close after a Form 144 disclosing intent to sell >0.5% of shares outstanding by non-routine seller; cover at t+60.
- **Data source**: SEC EDGAR (electronic since April 2023)
- **Source**: Gao & Ma SSRN 2021; WSJ "Insiders Hit the Exits" reporting.
- **Reported edge**: Anecdotal; preceded Snap, Peloton, Coinbase drawdowns.
- **Originality**: 6
- **Backtest-feasibility**: 3 (clean machine-readable history only since 2023)

### 10b5-1 Plan Adoption Front-Running
- **What it is**: SEC 2023 amendments require disclosure of 10b5-1 plan adoptions in 10-Q/10-K. Insiders often adopt right before bad news.
- **Asset class**: US equities
- **Horizon**: 3-12 months
- **Entry/exit rule**: Parse 10-Q/10-K Item 408 for new 10b5-1 plans by CEO/CFO; short equal-weight at next open; cover 6 months.
- **Data source**: SEC EDGAR (free)
- **Source**: Larcker/Lynch/Tayan/Taylor Stanford 2021; SEC Final Rule 33-11138 (Dec 2022).
- **Reported edge**: Plans with <60-day cooling-off + sale within 90 days underperformed ~5% in 6 months.
- **Originality**: 7
- **Backtest-feasibility**: 3 (NLP needed; only post-2023 mandatory)
- **Notes**: 90-day cooling-off will weaken future signal.

### Friday-After-Close 8-K Drift
- **What it is**: 8-Ks filed Friday after 4pm ET (or before holidays) contain disproportionately bad news (DellaVigna-Pollet inattention).
- **Asset class**: US equities
- **Horizon**: 1-2 months
- **Entry/exit rule**: Short any stock whose 8-K is timestamped after 16:00 ET Friday (excluding scheduled 2.02 earnings); cover at t+30.
- **Data source**: SEC EDGAR 8-K index (free)
- **Source**: DellaVigna & Pollet JF 2009; Niessner SSRN 2015.
- **Reported edge**: 60-bp drift over 75 days; larger for bad-news items (2.06, 4.02, 5.02 departures).
- **Originality**: 6
- **Backtest-feasibility**: 5

### Item 4.02 Non-Reliance Restatements
- **What it is**: 8-K Item 4.02 ("non-reliance on previously issued financials") = canonical accounting-restatement red flag.
- **Asset class**: US equities
- **Horizon**: 6-12 months
- **Entry/exit rule**: Short at t+1 open after first 4.02; cover 12 months or at restated financials.
- **Data source**: SEC EDGAR item-tagged 8-K feed (free)
- **Source**: Hennes/Leone/Miller TAR 2008.
- **Reported edge**: Mean 3-day CAR -9%; further -15 to -20% over year for fraud-driven restatements.
- **Originality**: 4
- **Backtest-feasibility**: 5
- **Notes**: Borrow cost high post-announcement; micro-caps often un-shortable.

### Schedule 13D Activist Drift
- **What it is**: Activist filings (>5% with intent to influence) generate substantial post-filing drift.
- **Asset class**: US equities
- **Horizon**: 3-18 months
- **Entry/exit rule**: Buy at close of day after 13D filing by known activist (Elliott, Starboard, Trian, Pershing, ValueAct, etc.); hold 18 months or 13D/A exit.
- **Data source**: SEC EDGAR 13D (free)
- **Source**: Brav/Jiang/Partnoy/Thomas JF 2008; Bebchuk/Brav/Jiang Columbia Law Review 2015.
- **Reported edge**: ~7% abnormal in 20-day window around filing, no reversal over 5 years.
- **Originality**: 3
- **Backtest-feasibility**: 5

### 13F Famous-Fund Mirror
- **What it is**: Mirror new positions in 13F by concentrated managers within 45-day lag.
- **Asset class**: US equities
- **Horizon**: 3-12 months
- **Entry/exit rule**: Day after 13F appears, buy any NEW position >2% of portfolio; equal-weight; rebalance quarterly; sell when next 13F shows >50% cut.
- **Data source**: SEC EDGAR 13F-HR (free), WhaleWisdom
- **Source**: Cohen/Polk/Silli RFS 2010.
- **Reported edge**: "Best ideas" earn 39-82 bps/month alpha; naive mirror is flat-to-slightly-positive after costs.
- **Originality**: 2
- **Backtest-feasibility**: 5
- **Notes**: 45-day stale data is the killer. Shorts/swaps/non-US invisible.

### IPO Lockup Expiry Short
- **What it is**: 180-day lockup ending lets insiders sell; supply shock drives 1-3% drop.
- **Asset class**: US equities, recent IPOs
- **Horizon**: ~5 days around expiry
- **Entry/exit rule**: Short 5 days before lockup expiry; cover 5 days after; equal-weight.
- **Data source**: S-1/424B prospectus (EDGAR free), MarketBeat lockup calendar
- **Source**: Field & Hanka JF 2001; Bradley et al. 2001.
- **Reported edge**: -1.5% mean CAR in 5-day window, -3% for VC-backed (Field-Hanka).
- **Originality**: 3
- **Backtest-feasibility**: 4
- **Notes**: Crowded — borrow expensive. Edge shrunk to ~50 bps recently.

### Lazy Prices: 10-K Text Similarity
- **What it is**: When 10-K text differs materially from prior year, bad news disproportionately hidden in changes.
- **Asset class**: US equities
- **Horizon**: 12 months
- **Entry/exit rule**: Cosine similarity vs prior year; rank firms quarterly; long top quintile (most similar = "lazy"), short bottom (most changed); rebalance at 10-K filing date; hold 12 months.
- **Data source**: SEC EDGAR 10-K HTML (free)
- **Source**: Cohen/Malloy/Nguyen JF 2020.
- **Reported edge**: 30-60 bps/month L/S alpha (~4-7% annualized).
- **Originality**: 6
- **Backtest-feasibility**: 4

### Spin-Off Drift (Greenblatt)
- **What it is**: Newly spun-off companies sold indiscriminately (forced selling, index exclusion), then drift up.
- **Asset class**: US equities (SpinCo + RemainCo)
- **Horizon**: 12-36 months
- **Entry/exit rule**: Buy spin-off at first regular-way trade; hold 24 months; equal-weight basket.
- **Data source**: SEC EDGAR Form 10/10-12B (free)
- **Source**: Greenblatt 1997; Cusatis/Miles/Woolridge JFE 1993; McConnell/Ovtchinnikov JIM 2004.
- **Reported edge**: Cusatis et al: +30% 36-month excess return for SpinCos.
- **Originality**: 4
- **Backtest-feasibility**: 4

### Corporate Jet M&A Speculation
- **What it is**: Track FAA registration / ADS-B flights of acquirer corporate jets to target HQ city ahead of M&A.
- **Asset class**: US equities
- **Horizon**: 1-90 days
- **Entry/exit rule**: Acquirer-HQ-to-target-HQ flight with no public business rationale → buy target call options 1-3 months out; exit on announcement or 90 days.
- **Data source**: ADS-B Exchange API (free), FAA N-number registry (free)
- **Source**: Finer (2018-style methodology); Jiang/Qian/Yonker JFQA 2019; WSJ "Twitter Deal" 2022; Bloomberg "Pfizer-Allergan" 2015.
- **Reported edge**: Anecdotal; no clean academic alpha number.
- **Originality**: 8
- **Backtest-feasibility**: 2
- **Notes**: Post-2023 FAA Privacy ICAO Address program lets owners obscure tail numbers — signal degrading fast.

### Central-Bank Jackson Hole Jet Tracking
- **What it is**: Watch Gulfstream traffic into JAC airport late August; off-schedule landings by CB-affiliated jets can presage policy shifts.
- **Asset class**: USD, USTs, gold, SPX
- **Horizon**: 1-30 days
- **Entry/exit rule**: Count JAC arrivals of CB-registered jets week before Jackson Hole; if count > 5y avg + 1σ, position for dovish surprise.
- **Data source**: ADS-B Exchange, FAA registry (free)
- **Source**: ZeroHedge recurring annual posts; Bloomberg.
- **Reported edge**: Anecdotal only.
- **Originality**: 9
- **Backtest-feasibility**: 2
- **Notes**: N ~1 event/year. Sample size kills any statistical claim.

### Walmart/Target Parking-Lot Fill Rates
- **What it is**: Satellite car counts in big-box parking lots predict quarterly SSS beats/misses.
- **Asset class**: US retail (WMT, TGT, COST, HD, LOW)
- **Horizon**: Quarter-end to earnings (~30-60 days)
- **Entry/exit rule**: Aggregate satellite car counts vs prior-year quarter for top 200 stores; if YoY growth > consensus comp guidance + 200bps, buy at quarter-end through earnings.
- **Data source**: Orbital Insight, RS Metrics, SpaceKnow, Planet Labs (all $$$$)
- **Source**: Katona/Painter/Patatoukas/Zeng JFE 2024.
- **Reported edge**: ~3.5% per quarter pre-2018; halved post-disclosure.
- **Originality**: 5
- **Backtest-feasibility**: 1 (paid data)
- **Notes**: Free proxy: Placer.ai foot-traffic releases.

### Crude Oil Storage Tank Shadow Length
- **What it is**: Floating-roof tank shadows shorten as tanks fill; satellites estimate Cushing/Saudi/Chinese SPR inventories ahead of EIA.
- **Asset class**: WTI/Brent, USO, XLE
- **Horizon**: 1-7 days (around EIA Wed print)
- **Entry/exit rule**: If satellite-derived US inventory delta vs prior week > 1σ above EIA consensus, short crude into Wed 10:30 ET release; cover at print.
- **Data source**: Orbital Insight, Ursa Space, Kayrros (paid); free Sentinel-1 SAR via Copernicus
- **Source**: Bloomberg 2017; Kayrros methodology.
- **Reported edge**: Anecdotal.
- **Originality**: 7
- **Backtest-feasibility**: 2

### China Container-Port Throughput
- **What it is**: AIS + satellite of Shanghai/Ningbo/Shenzhen container yards predicts Chinese export PMI.
- **Asset class**: FXI, ASHR, BDRY, USDCNH, US importers
- **Horizon**: 1-3 months
- **Entry/exit rule**: Monthly z-score; z > +1.5 → long FXI vs short SPY; z < -1.5 → reverse.
- **Data source**: MarineTraffic (free tiers), IMF PortWatch (free, https://portwatch.imf.org/)
- **Source**: Cerdeiro/Komaromi/Liu/Saeed IMF WP 2020.
- **Reported edge**: ~0.8 correlation with monthly trade prints, 30+ days earlier.
- **Originality**: 6
- **Backtest-feasibility**: 3 (IMF PortWatch is now free/machine-readable)

### Glassdoor Review Trend
- **What it is**: Aggregate employee Glassdoor sentiment + recommend-CEO % predicts forward earnings surprises.
- **Asset class**: US large-cap
- **Horizon**: 6-12 months
- **Entry/exit rule**: Monthly 12m rolling change in Glassdoor rating per ticker; long top quintile, short bottom; rebalance monthly.
- **Data source**: Glassdoor scraped (TOS-restricted), Revelio Labs, Thinknum (paid)
- **Source**: Green/Huang/Wen/Zhou JFE 2019.
- **Reported edge**: 70-80 bps/month L/S alpha (~10% annualized).
- **Originality**: 6
- **Backtest-feasibility**: 2

### LinkedIn Hiring Velocity
- **What it is**: MoM growth in firm headcount / open job postings leads revenue growth by 1-2 quarters.
- **Asset class**: US equities
- **Horizon**: 3-9 months
- **Entry/exit rule**: Monthly QoQ change in firm-level LinkedIn headcount or active postings; long top decile, short bottom; rebalance monthly.
- **Data source**: Revelio Labs, LinkUp, Thinknum (paid); LinkedIn TOS blocks scraping
- **Source**: Gutiérrez/Lourie/Nekrasov/Shevlin Management Science 2020.
- **Reported edge**: ~6-8% annualized L/S alpha.
- **Originality**: 5
- **Backtest-feasibility**: 1 (no free historical panel)

### WARN Act Layoff Notices
- **What it is**: State-level Worker Adjustment and Retraining Notification filings list large planned layoffs 60+ days in advance. Free, public, often overlooked.
- **Asset class**: US equities
- **Horizon**: 1-6 months
- **Entry/exit rule**: Aggregate WARN per public ticker per month; if employees noticed > 2% of headcount in single month, short at filing + 5 days; cover 90 days.
- **Data source**: State labor dept WARN portals (free; CA, NY, TX, WA most useful); WARNtracker.com
- **Source**: Hallock AER 1998; Goldman "WARNing Signs" 2023; FT 2023 tech layoff coverage.
- **Reported edge**: Anecdotal — sign is ambiguous (large layoffs sometimes drive rallies).
- **Originality**: 7
- **Backtest-feasibility**: 4
- **Notes**: Conditioning on firm leverage helps.

### Wikipedia Pageview Spikes
- **What it is**: Spikes in pageviews for ticker/company articles precede returns (sign debated).
- **Asset class**: US equities, crypto
- **Horizon**: 1-4 weeks
- **Entry/exit rule**: For each DJIA/S&P 500 name, weekly pageview z-score vs trailing 52w; short z > +2; cover +20 days.
- **Data source**: Wikimedia Pageviews API (free)
- **Source**: Moat/Curme/Avakian/Kenett/Stanley/Preis Sci Rep 2013.
- **Reported edge**: Sharpe ~1.5 in-sample 2007-2012; OOS weak.
- **Originality**: 7
- **Backtest-feasibility**: 5

### Reddit r/wallstreetbets Mention Velocity
- **What it is**: WSB/Stocktwits mentions per ticker per day predict short-horizon retail momentum, then mean reversion.
- **Asset class**: US small/mid-cap, crypto
- **Horizon**: 1-15 trading days
- **Entry/exit rule**: Long top decile of 5-day mention growth; hold 5 days; (variant: short top decile after 10 days for reversal).
- **Data source**: Pushshift/Arctic Shift (free until 2023), Quiver Quant, Apewisdom.io (free)
- **Source**: Bradley/Hanousek/Jame/Xiao Management Science 2024; Pedersen JFE 2022.
- **Reported edge**: ~1.1% next-2-day return on WSB DD posts; reversal after.
- **Originality**: 5
- **Backtest-feasibility**: 4
- **Notes**: Post-GME 2021 more crowded; best in $1B-$10B mcap.

### App Store Ranking Velocity
- **What it is**: Daily change in App Store / Google Play rank predicts MAU/DAU surprises.
- **Asset class**: App-dependent firms (DASH, UBER, RBLX, SNAP, PINS, SPOT)
- **Horizon**: ~45 days to earnings
- **Entry/exit rule**: Quarterly avg overall-free rank for flagship app; long names whose rank improved >20% YoY; hold through earnings.
- **Data source**: data.ai, Sensor Tower, SimilarWeb (paid); free: Apple's RSS feeds for top charts
- **Source**: Bloomberg "App Annie SEC settlement" 2021.
- **Reported edge**: Anecdotal; SEC App Annie case suggests buyside finds it material.
- **Originality**: 6
- **Backtest-feasibility**: 2

### Magazine Cover Contrarian Indicator
- **What it is**: Hyperbolic Economist/BusinessWeek/Time covers — fade them.
- **Asset class**: Any
- **Horizon**: 6-24 months
- **Entry/exit rule**: Code covers as bull/bear for asset X; take opposite position at issue date; 5% weight; hold 12 months.
- **Data source**: Publisher cover archives (free)
- **Source**: Arnold/Earl/North FAJ 2007.
- **Reported edge**: 22-month CAR of -10% for bullish covers, +18% for bearish (n=44).
- **Originality**: 3
- **Backtest-feasibility**: 4
- **Notes**: Confirmation bias risk; only at extremes.

### Skyscraper / Construction Hubris Index
- **What it is**: World's tallest building completions historically coincide with macro tops.
- **Asset class**: Country equity indices, EM debt, REITs
- **Horizon**: 12-36 months
- **Entry/exit rule**: When top-3 globally building is announced, short host country index 25% weight on completion; hold 24 months.
- **Data source**: CTBUH (free, https://www.skyscrapercenter.com/)
- **Source**: Lawrence 1999 (original); Barr/Mizrach/Mundra Empirical Economics 2015.
- **Reported edge**: Barr et al. found NO predictive power after controlling for trend.
- **Originality**: 6
- **Backtest-feasibility**: 4
- **Notes**: N < 20 events globally. Sentiment overlay only.

### NDVI Crop Yield Front-Run
- **What it is**: MODIS/Sentinel-2 NDVI deviations predict USDA WASDE yield surprises.
- **Asset class**: Corn, soy, wheat futures; DBA; CTVA/NTR/MOS/ADM
- **Horizon**: 1-3 months (around USDA reports)
- **Entry/exit rule**: For US Corn Belt, Aug-15 NDVI z-score vs 10y same-pixel avg; if z < -1, long Dec corn into Sep WASDE.
- **Data source**: NASA MODIS MOD13Q1 (free), Sentinel-2 via Copernicus (free)
- **Source**: Johnson Remote Sensing of Environment 2014; USDA NASS uses MODIS internally.
- **Reported edge**: NDVI explains 60-80% of county-level yield variance in Aug.
- **Originality**: 7
- **Backtest-feasibility**: 3 (free but processing GeoTIFFs is real work)

### Heating/Cooling Degree-Day vs Natural Gas
- **What it is**: NOAA HDD/CDD forecasts vs market-implied weather expectations drive nat gas inventory surprises.
- **Asset class**: NG, UNG, utility equities
- **Horizon**: 1-2 weeks
- **Entry/exit rule**: Each Mon, compute next-14-day population-weighted HDD from NOAA CPC; if forecast > prior week's + 5%, long NG front-month at open; hold to Thu EIA.
- **Data source**: NOAA CPC (free, https://www.cpc.ncep.noaa.gov/), EIA Storage (free)
- **Source**: Mu Energy Economics 2007; Linn & Zhu JFM 2004.
- **Reported edge**: Weather shocks explain 30% of weekly NG volatility; Sharpe 0.5-0.8 net.
- **Originality**: 4
- **Backtest-feasibility**: 5
- **Notes**: Crowded; edge is in regional basis (Northeast, ERCOT) not Henry Hub.

### Big Mac PPP for FX Mean Reversion
- **What it is**: Economist Big Mac Index PPP deviations mean-revert over multi-year.
- **Asset class**: G10 and EM FX
- **Horizon**: 2-5 years
- **Entry/exit rule**: Twice-yearly Big Mac release: long top-quintile undervalued, short top-quintile overvalued; hold to next release.
- **Data source**: Economist Big Mac data (free CSV, https://github.com/TheEconomist/big-mac-data)
- **Source**: Cumby NBER 1996; Clements/Lan/Seah 2010.
- **Reported edge**: ~Sharpe 0.5 on Big Mac PPP carry overlay; long-horizon convergence stat-sig at 3-5y.
- **Originality**: 3
- **Backtest-feasibility**: 5
