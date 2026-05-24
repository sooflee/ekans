# Phase 1Q — Commodity-Specific Signals
# Returned by hunt agent. 15 signals, originality 8-10, one-line rules.

### Q-1 Saudi Aramco OSP Asia Shock → Long Brent
- **Rule**: Long BZ=F 30d when Aramco Arab Light OSP to Asia (vs Oman/Dubai avg) raised by > $1.50/bbl MoM.
- **Data**: aramco.com/en/investors monthly OSP table (released ~5th of month)
- **Mechanism**: Aramco hikes Asian OSP only when seeing inelastic demand from Asian refiners → forward-looking signal from world's lowest-cost producer.
- **CAGR**: 12-15%
- **Originality**: 9

### Q-2 Mexican Hannover Hedge Window → Short WTI
- **Rule**: Jun 1 - Sep 30, when WTI implied vol (OVX) < 30 AND WTI spot > $70, short CL=F 6 weeks.
- **Data**: cboe.com OVX daily; eia.gov WTI spot; Mexican Federal Budget annexes
- **Mechanism**: Mexico hedges 200-300M bbl annually via Asian puts Jun-Sep; counterparty banks delta-hedge by shorting WTI futures, pressing curve.
- **CAGR**: 10-12%
- **Originality**: 10

### Q-3 COMEX Gold Registered Inventory Crunch → Long Gold
- **Rule**: Long GC=F 60d when COMEX registered gold < 30% of total (reg + eligible) for 5 consecutive days.
- **Data**: cmegroup.com Metals Issues and Stops daily report
- **Mechanism**: Registered = pledged to delivery; falling share = market makers face higher cost-of-carry → EFP blowouts → spot rises.
- **CAGR**: 10-14%
- **Originality**: 9

### Q-4 Platinum/Palladium Ratio Mean Reversion
- **Rule**: Long cheaper metal / short richer when Pt/Pd > 1.5 OR < 0.4; hold 120d.
- **Data**: CME PL=F, PA=F daily settlement (or PPLT/PALL ETFs)
- **Mechanism**: Catalytic converter substitution with 12-18 month engineering lag forces ratio back toward 0.8-1.2 band.
- **CAGR**: 11-13%
- **Originality**: 8
- **Caveat**: EV adoption may permanently destroy historical range

### Q-5 Copper TC/RC Smelter Collapse → Long Copper
- **Rule**: Long HG=F 90d when spot copper concentrate TC/RC falls below $20/ton (vs $80+ benchmark).
- **Data**: metal.com SMM free headline weekly
- **Mechanism**: TC/RC collapse → smelters cut runs or close → refined copper removed from market.
- **CAGR**: 13-15%
- **Originality**: 9

### Q-6 Brent-Dubai EFS Asian Demand → Long Brent / Short WTI
- **Rule**: When Brent-Dubai EFS narrows below $0.50/bbl for 10 consecutive sessions, long BZ=F + short CL=F 45d.
- **Data**: ice.com Brent-Dubai EFS Swap Future daily
- **Mechanism**: Narrow EFS = Asian sour bid up to Atlantic light → Chinese/Indian refiners paying up = tightness shifting Atlantic→Pacific.
- **CAGR**: 10-12%
- **Originality**: 9

### Q-7 Indian Gold Festival Import Surge → Long Gold (Sep)
- **Rule**: Long GC=F entering 2nd week of September for 45d when India Aug monthly gold imports > 80 tons (DGCI&S).
- **Data**: tradestat.commerce.gov.in monthly trade data
- **Mechanism**: India ~25% global gold demand; pre-Diwali stocking Aug → strong physical premiums Sep-Oct → COMEX paper higher.
- **CAGR**: 10-12%
- **Originality**: 8

### Q-8 Brazil Sugarcane Mill Crush Shortfall → Long Sugar
- **Rule**: When UNICA bi-weekly Center-South sugarcane crush shows cumulative through Jul 31 > 5% below prior year, long SB=F 90d.
- **Data**: observatoriodacana.com.br bi-weekly UNICA reports
- **Mechanism**: Center-South Brazil = ~40% global exportable sugar; crush shortfalls compress global stocks-to-use within 60-90 days.
- **CAGR**: 12-15%
- **Originality**: 9

### Q-9 Indonesia Palm Oil Export Curb Reflex → Long CPO
- **Rule**: Within 3 trading days of any Indonesian palm-oil export levy hike / DMO increase / outright ban, long Bursa Malaysia CPO (FCPO) or soy oil (ZL=F) 30d.
- **Data**: kemendag.go.id press releases
- **Mechanism**: Indonesia ~58% global palm; export restriction reroutes buyers (India/China/EU) to Malaysian CPO = step-function price reset.
- **CAGR**: 12-14%
- **Originality**: 9

### Q-10 Gold Miner AISC Squeeze → Long Gold
- **Rule**: Long GC=F 120d when GDX-weighted avg AISC is within 10% of spot gold.
- **Data**: SEC EDGAR 10-Q/40-F of top 10 GDX constituents + World Gold Council AISC dataset
- **Mechanism**: Gold near marginal cost curve → miners cut mine plans → ETF investors interpret as floor signal → reflexive buying.
- **CAGR**: 10-12%
- **Originality**: 8

### Q-11 Cushing Operational Minimum Squeeze → WTI Calendar Spread
- **Rule**: When EIA weekly Cushing crude inventory < 25M bbl, long WTI front + short 1m deferred (CL1-CL2 spread) for 30d.
- **Data**: eia.gov Weekly Petroleum Status Report (Wed 10:30 ET)
- **Mechanism**: Cushing operational minimum ~20-22M bbl (tank bottoms); below 25M, refiners/traders bid up prompt barrels → steep backwardation.
- **CAGR**: 12-14%
- **Originality**: 8

### Q-12 OPEC+ Compliance Breakdown → Short Brent
- **Rule**: Short BZ=F 60d when IEA/Argus monthly OPEC+ compliance rate < 80% for 2 consecutive months.
- **Data**: iea.org Oil Market Report; opec.org MOMR self-reported production
- **Mechanism**: Compliance < 80% = cheating by UAE/Iraq/Kazakhstan/Russia under fiscal stress; cartel discipline myth dissolves.
- **CAGR**: 11-13%
- **Originality**: 8

### Q-13 Brazil Frost Belt Coffee Trigger → Long Arabica
- **Rule**: Between Jun 1 - Aug 15, when INMET minimum temperature at any Minas Gerais coffee-belt station (Varginha/Poços de Caldas/Patrocínio) < 2°C, long KC=F next session for 30d.
- **Data**: portal.inmet.gov.br hourly station data; NOAA GHCN as backup
- **Mechanism**: Arabica trees suffer permanent leaf burn < 2°C; Minas Gerais grows ~50% world arabica → speculators/roasters scramble for cover.
- **CAGR**: 13-15%
- **Originality**: 9

### Q-14 EU Gas Storage Above-Range Short → Short TTF
- **Rule**: Short TTF=F 45d when AGSI EU aggregate gas storage exceeds 5-year max for the week by > 5 percentage points.
- **Data**: agsi.gie.eu free daily dataset
- **Mechanism**: Storage above 5y max → injection demand collapses → pipeline/LNG suppliers compete for marginal slot → prompt TTF crushed.
- **CAGR**: 12-14%
- **Originality**: 8

### Q-15 JKM-TTF LNG Arbitrage Diversion
- **Rule**: When JKM front-month premium over TTF > $2.50/MMBtu for 5 consecutive sessions, long TTF=F + short JKM=F 45d.
- **Data**: spglobal.com Platts headline daily; ice.com settlements
- **Mechanism**: JKM-TTF > ~$2 covers tanker freight + boil-off; US Gulf/Qatari cargoes redirect Atlantic→Pacific → drains EU rebuild, loosens JKM.
- **CAGR**: 11-13%
- **Originality**: 9
