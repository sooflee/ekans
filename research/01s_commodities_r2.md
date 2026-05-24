# Phase 1S — Commodity Signals Round 2
# Returned by hunt agent. 15 signals, originality 8-9, one-line rules.

### S-1 Ghana Cocobod Farmgate Price Hike → Long Cocoa
- **Rule**: Long ICE Cocoa (CC) / NIB when Cocobod farmgate price hike > 20% vs prior season.
- **Data**: cocobod.gh/news (press releases)
- **CAGR**: 12-18%; Originality 9

### S-2 Cote d'Ivoire Port Arrivals Deficit → Long Cocoa
- **Rule**: Long CC when cumulative season-to-date Ivorian port arrivals run > 10% below prior season.
- **Data**: Reuters weekly Abidjan note
- **CAGR**: 15-25%; Originality 8

### S-3 Texas High Plains Cotton Drought → Long Cotton
- **Rule**: Long CT/BAL when US Drought Monitor classifies > 50% of Texas cotton-belt counties (Lubbock/Lamesa) at D2+ during May-July.
- **Data**: droughtmonitor.unl.edu shapefiles + USDA-NASS county map
- **CAGR**: 10-15%; Originality 8

### S-4 Florida Citrus Greening Production Step-Down → Long OJ
- **Rule**: Long OJ when USDA October Florida Citrus Forecast cuts orange production > 15% vs prior season final.
- **Data**: nass.usda.gov Florida Citrus Forecast (monthly Oct PDF)
- **CAGR**: 15-30%; Originality 9

### S-5 Kazatomprom Production Guidance Cut → Long Uranium
- **Rule**: Long URNM / U.UN / CCJ when Kazatomprom quarterly results lower full-year U3O8 guidance > 5%.
- **Data**: kazatomprom.kz/en/category/press_releases
- **CAGR**: 20-40%; Originality 9

### S-6 LME Aluminum Stock Drawdown → Long Aluminum
- **Rule**: Long AA/CENX when LME on-warrant aluminum stocks fall > 25% in rolling 8-week window.
- **Data**: lme.com Market-Data Reports stocks daily CSV
- **CAGR**: 10-15%; Originality 8

### S-7 China Iron Ore Port Stock Glut → Short Iron Ore
- **Rule**: Short SGX iron ore (FEF) / VALE when total Chinese port stockpiles > 145Mt per Mysteel weekly.
- **Data**: mysteel.net 45-ports weekly inventory
- **CAGR**: 10-15%; Originality 8

### S-8 SHFE Rebar Inventory Seasonal Squeeze → Long Rebar
- **Rule**: Long SHFE rebar (RB) when total social+mill inventory falls > 20% below 5y avg for the same calendar week during March-May.
- **Data**: mysteel.net weekly rebar social inventory
- **CAGR**: 12-18%; Originality 8

### S-9 WTI Calendar Spread Backwardation Extreme → Long Crude
- **Rule**: Long CL1 / short CL2 (or USO outright) when M1-M2 spread > +$2.00/bbl backwardation.
- **Data**: cmegroup.com free CL settlements
- **CAGR**: 10-20%; Originality 8

### S-10 Crude OVX vs Equity VIX Decoupling → Short Crude
- **Rule**: Short CL/USO when OVX/VIX > 2.0x for 5 consecutive sessions.
- **Data**: cboe.com free daily OVX + VIX
- **CAGR**: 12-18%; Originality 9

### S-11 US Gasoline Demand Collapse → Short RBOB
- **Rule**: Short RBOB / UGA when EIA weekly Product Supplied of Finished Motor Gasoline falls > 5% below 4-week avg AND below prior year for 2 consecutive weeks during May-August.
- **Data**: eia.gov/petroleum/weekly Product Supplied series
- **CAGR**: 10-15%; Originality 8

### S-12 Egypt GASC Tender Bid Spread → Long Wheat
- **Rule**: Long CBOT wheat (ZW) when most recent GASC tender clears at CFR price > $25/tonne above prior tender.
- **Data**: agricensus.com / Reuters GASC tender results
- **CAGR**: 10-15%; Originality 9

### S-13 California NRCS Snowpack Deficit → Long Milk/Cheese
- **Rule**: Long CME Class III Milk (DC) / Cheese (CSC) when April 1 California statewide SWE < 70% of 30y normal per USDA NRCS.
- **Data**: nrcs.usda.gov SNOTEL/snow course basin reports
- **CAGR**: 10-14%; Originality 9

### S-14 Pilbara Spodumene Auction Step-Down → Short Lithium
- **Rule**: Short LIT/ALB when Pilbara Minerals BMX auction clears > 15% below prior auction.
- **Data**: pilbaraminerals.com.au/investors ASX releases
- **CAGR**: 15-25%; Originality 9

### S-15 Russian Urals Discount Blowout → Long Brent / Short WTI
- **Rule**: Long Brent (BZ) / short WTI (CL) when Urals-to-Brent discount widens beyond -$20/bbl.
- **Data**: Russian MinFin monthly + Argus weekly
- **CAGR**: 10-15%; Originality 9
