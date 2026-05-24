# Phase 1O — Hunt Round 6: niche territories
# Returned by hunt agent. 12 signals targeting >10% CAGR, originality 8-9.

### O-1 SRF Usage Spike → Long ZT
- **Rule**: Long 2Y Treasury futures (ZT) / SHY next session when SRF daily take-up > $5B (vs typical zero).
- **Data**: newyorkfed.org/markets/desk-operations/repo (daily CSV)
- **Mechanism**: SRF is Fed emergency repo backstop priced above market; banks tap only when private repo breaks → flight-to-safety bid into front-end Treasuries.
- **CAGR**: 10-13%
- **Originality**: 9

### O-2 FICC Sponsored Repo Contraction → Short HYG
- **Rule**: Short HYG when FICC sponsored DVP repo daily volume drops > 15% WoW.
- **Data**: dtcc.com/charts/dtcc-gcf-repo-index + sponsored-repo XLSX (weekly)
- **Mechanism**: Sponsored repo funds basis trades; sudden contraction = hedge-fund deleveraging → spills into credit spreads in 2-3 weeks.
- **CAGR**: 11-14%
- **Originality**: 9

### O-3 Govt vs Prime MMF Rotation → Long XLU / Short XLF
- **Rule**: Long XLU / short XLF when ICI weekly MMF report shows Govt MMF assets growing faster than Prime for 3 consecutive weeks.
- **Data**: ici.org/research/stats/mmf (weekly XLS)
- **Mechanism**: Cash fleeing Prime (bank CP/CDs) → Govt = counterparty risk aversion = hurts banks, helps utilities via duration bid.
- **CAGR**: 10-12%
- **Originality**: 8

### O-4 Bitcoin Mempool Backlog → Long Miner Basket
- **Rule**: Long MARA+RIOT+CLSK when BTC mempool unconfirmed tx count exceeds 200,000 for 2 consecutive days.
- **Data**: mempool.space/api/mempool (free JSON)
- **Mechanism**: Mempool congestion → tx fees spike → miners get 5-15% revenue uplift → equity multiples re-rate within days.
- **CAGR**: 12-16%
- **Originality**: 8

### O-5 Bitcoin Difficulty Drop → Long Miners
- **Rule**: Long MARA+RIOT+CLSK basket when BTC difficulty adjustment is < -3% (every ~2016 blocks).
- **Data**: api.blockchain.info/charts/difficulty (free JSON)
- **Mechanism**: Negative difficulty = weaker miners capitulated → survivors have proportionally more hashrate + revenue per machine.
- **CAGR**: 11-15%
- **Originality**: 9

### O-6 NFIP Claims Surge → Short XHB
- **Rule**: Short XHB when FEMA NFIP weekly claims paid exceed $500M.
- **Data**: fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2 (OpenFEMA API)
- **Mechanism**: Large flood claims → regional builders flooded with insurance-funded demand initially, then collapse in new starts as labor diverted to repairs.
- **CAGR**: 10-13%
- **Originality**: 9

### O-7 Apartment List Rent Deflation → Long IEF
- **Rule**: Long IEF when Apartment List National Rent Index posts MoM decline > 0.4%.
- **Data**: apartmentlist.com/research/national-rent-data (monthly free CSV)
- **Mechanism**: Apartment List leads BLS shelter CPI by 6-12 months (captures new leases) → cooler shelter prints → rate cuts priced into belly.
- **CAGR**: 10-12%
- **Originality**: 8

### O-8 Built-for-Rent Share Inflection → Short SFR REITs
- **Rule**: Short INVH+AMH when Census built-for-rent share of single-family starts > 12% (quarterly release).
- **Data**: census.gov/construction/nrc/data/series.html (quarterly Survey of Construction)
- **Mechanism**: Rising BFR share floods SFR market → compresses rents → squeezes spread that listed SFR operators earn over cost of capital.
- **CAGR**: 10-13%
- **Originality**: 9

### O-9 Indeed Job Postings Breakdown → Short XLY
- **Rule**: Short XLY when Indeed Hiring Lab US job postings index falls > 8% YoY.
- **Data**: github.com/hiring-lab/data (free CSV weekly)
- **Mechanism**: Indeed postings lead BLS payrolls by 4-8 weeks (employer intent before formal hiring) → 8% YoY contraction precedes XLY earnings cuts.
- **CAGR**: 11-14%
- **Originality**: 8

### O-10 Business Formation High-Propensity → Long IWM
- **Rule**: Long IWM when Census Weekly Business Formation "High-Propensity" applications grow > 5% YoY for 4 consecutive weeks.
- **Data**: census.gov/econ/bfs (BFS weekly CSV; BA_BAHPC series)
- **Mechanism**: High-Propensity apps (likely to become employers) lead small-business hiring and capex by 2 quarters → small-caps track small-biz tighter than large-caps.
- **CAGR**: 10-12%
- **Originality**: 9

### O-11 3-2-1 Crack Spread Crash → Short XLE
- **Rule**: Short XLE when 3-2-1 crack spread (2×RBOB + HO - 3×WTI, front-month) falls below $10/bbl.
- **Data**: eia.gov/dnav/pet (daily spot prices for WTI, NY RBOB, NY ULSD)
- **Mechanism**: 3-2-1 crack < $10 = refiners can't cover variable costs → cut runs → XLE earnings compress (refiners drive Q/Q vol more than E&P).
- **CAGR**: 11-14%
- **Originality**: 8

### O-12 California Cap-and-Trade Auction → Long ICLN
- **Rule**: Long ICLN when California ARB quarterly allowance auction clears > 10% above prior auction price.
- **Data**: ww2.arb.ca.gov/our-work/programs/cap-and-trade-program/auction-information (quarterly PDF)
- **Mechanism**: Rising CCA allowance prices → higher implicit carbon cost for CA fossil generators → relative economics of clean alternatives → capital into renewable equity.
- **CAGR**: 10-13%
- **Originality**: 9
