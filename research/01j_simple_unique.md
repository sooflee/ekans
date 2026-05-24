# Phase 1J — Simple + Maximally Unique Signals
# Returned by hunt agent. 12 signals, all originality 8-10, one-line rules.

### J-1 TIC Foreign Custody Holdings Drop
- **Rule**: Short TLT for the month after TIC release when foreign official custody (FRED WCFOL) falls > 1% MoM.
- **Asset**: TLT short / ZB futures
- **Horizon**: 1 month
- **Data**: FRED WCFOL (weekly)
- **Mechanism**: Foreign CBs at NY Fed are the most price-insensitive Treasury buyer; sudden drawdowns (FX defense) remove that bid → term premium widens.
- **Originality**: 9
- **Simplicity**: 5
- **Notes**: Brad Setser flagged 2015 China devaluation and 2022 BoJ defense preceded 20-50bp 10Y backups.

### J-2 Fed H.8 C&I Loan Contraction
- **Rule**: Rotate SPY → SHV for the following week if BUSLOANS contracts two consecutive weeks.
- **Asset**: SPY vs SHV
- **Horizon**: 1-4 weeks
- **Data**: FRED BUSLOANS (weekly)
- **Mechanism**: C&I loans are sticky; 2 consecutive weekly contractions has historically only occurred at credit-cycle turning points (1990, 2001, 2008, 2020, 2023 SVB).
- **Originality**: 8
- **Simplicity**: 5

### J-3 SEC Form D Private-Placement Surge
- **Rule**: Short XBI for 60 days when weekly Form D filings by NAICS 5417/3254 issuers exceed trailing-52w 90th percentile.
- **Asset**: XBI short
- **Horizon**: 1-2 months
- **Data**: SEC Form D bulk dataset (free)
- **Mechanism**: Biotech PIPEs/Reg D raises cluster when public equity is funder-of-last-resort → dilution waves → sector underperformance.
- **Originality**: 9
- **Simplicity**: 4

### J-4 Mississippi River Stage at Memphis
- **Rule**: Long WEAT/CORN for 30 days when Memphis gauge < -5 feet for 5 consecutive days during Jul-Oct.
- **Asset**: CORN, WEAT
- **Horizon**: 1-3 months
- **Data**: NOAA AHPS (free JSON)
- **Mechanism**: ~60% of US grain export tonnage moves down Mississippi; low stages force barge load reductions → export basis widens.
- **Originality**: 9
- **Simplicity**: 4

### J-5 FAA NOTAM "GPS Interference" Spike
- **Rule**: Long ITA for 10 trading days when weekly FAA NOTAMs containing "GPS UNRELIABLE" exceed trailing-26w 90th percentile.
- **Asset**: ITA, XAR
- **Horizon**: 2 weeks
- **Data**: FAA NOTAM Search API (free)
- **Mechanism**: GPS-interference NOTAMs cluster around DoD EW exercises and active geopolitical jamming.
- **Originality**: 10
- **Simplicity**: 4

### J-6 CDC FluView Hospitalization Acceleration
- **Rule**: Long WMT+CVS / short JETS for 4 weeks when CDC FluSurv-NET rate rises >30% WoW during Oct-Mar.
- **Asset**: WMT+CVS pair vs JETS
- **Horizon**: 4 weeks
- **Data**: CDC FluView API (free CSV)
- **Mechanism**: Severe flu shifts spending to OTC/pharmacy + depresses short-haul leisure travel. ~2-3 week lag.
- **Originality**: 8
- **Simplicity**: 4

### J-7 Fed RRP Facility Drain
- **Rule**: Short TLT for 1 week when Fed ON-RRP usage (FRED RRPONTSYD) falls >$50B WoW.
- **Asset**: TLT short
- **Horizon**: 1-2 weeks
- **Data**: FRED RRPONTSYD (daily)
- **Mechanism**: RRP drains mean MMFs buying T-bills instead of parking at Fed → front-loads supply absorption short-end → long end clears at higher yields.
- **Originality**: 8
- **Simplicity**: 5

### J-8 USGS Copper Mine Production Disruption
- **Rule**: Long COPX for 30 days when USGS Mineral Industry Surveys monthly copper report shows ANY major mine (Escondida/Grasberg/Cobre Panamá/Las Bambas) reporting force-majeure.
- **Asset**: COPX, CPER, FCX
- **Horizon**: 1 month
- **Data**: USGS Mineral Industry Surveys (monthly PDF)
- **Mechanism**: Top 10 copper mines produce ~40% global supply; one offline tightens concentrate market measurably.
- **Originality**: 8
- **Simplicity**: 4
- **Notes**: Cobre Panamá closure Nov 2023 preceded ~25% copper rally Q1 2024.

### J-9 EIA Weekly Distillate Stocks
- **Rule**: Long XLE / short CL=F (crack spread proxy) for 2 weeks when distillate stocks fall >2σ below 5-year seasonal range.
- **Asset**: XLE vs CL pair
- **Horizon**: 2 weeks
- **Data**: EIA Weekly Petroleum Status (free CSV API)
- **Mechanism**: Distillate is tightest part of refined products; extreme draws force max-distillate yields → crack spreads widen → refiner equity outperforms crude.
- **Originality**: 8
- **Simplicity**: 4

### J-10 NASA FIRMS Brazilian Amazon Fire Count
- **Rule**: Long SOYB for 60 days when NASA FIRMS active fires in Mato Grosso + Pará exceed prior-3-year avg by 50% during Aug-Oct.
- **Asset**: SOYB
- **Horizon**: 2-3 months
- **Data**: NASA FIRMS API (free)
- **Mechanism**: Fire-count spikes correlate with drought lowering Brazil soy yield expectations; CONAB/USDA revise yields → global ending stocks tighten.
- **Originality**: 9
- **Simplicity**: 4

### J-11 TSA Throughput YoY Divergence
- **Rule**: Short JETS for 30 days when 7-day-avg TSA checkpoint throughput is >5% below same-day 2019 baseline on any non-holiday weekday.
- **Asset**: JETS, AAL, DAL
- **Horizon**: 1 month
- **Data**: TSA daily checkpoint counts (free CSV/scrape)
- **Mechanism**: TSA daily counts are highest-frequency RPM proxy; sustained underperformance precedes airline guidance cuts by 2-4 weeks.
- **Originality**: 8
- **Simplicity**: 4

### J-12 USDA Cattle on Feed Report Shock
- **Rule**: Long LE futures / COW for 30 days when monthly USDA Cattle on Feed placements are >5% below pre-release trade survey median.
- **Asset**: LE futures / COW
- **Horizon**: 1 month
- **Data**: USDA NASS monthly (PDF + CSV)
- **Mechanism**: Placements feed cattle that come to market 4-6 months out; placements shortfall tightens forward supply with long lag the futures curve underprices.
- **Originality**: 9
- **Simplicity**: 4
