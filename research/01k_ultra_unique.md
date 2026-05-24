# Phase 1K — Ultra-Unique + Simple Signals (Round 2)
# Returned by hunt agent. 15 signals, all originality 8-10, one-line rules.
# Theme: subindex-rather-than-headline, obscure-feed, calendar-fixed catalysts.

### K-1 Architecture Billings Inquiries Subindex Crossover
- **Rule**: Long XHB for next month when AIA "Project Inquiries" subindex > 55.
- **Asset**: XHB / ITB
- **Horizon**: 20-day hold
- **Data**: aia.org ABI monthly PDF
- **Mechanism**: Inquiries leads billings by 3-6 months; headline ABI is widely tracked but the subindex isn't.
- **Originality**: 8

### K-2 FDA Drug Shortage Net Additions
- **Rule**: Long IHE day after FDA weekly drug-shortage list shows net additions > 5.
- **Asset**: IHE
- **Horizon**: 10 days
- **Data**: accessdata.fda.gov/scripts/drugshortages (free)
- **Mechanism**: Net additions signal generic-supply stress → pricing power for branded incumbents.
- **Originality**: 9

### K-3 Beige Book "Uncertainty" Word Frequency
- **Rule**: Short SPY 30 days when "uncertain/uncertainty" count in latest Beige Book exceeds 25.
- **Asset**: SPY
- **Horizon**: 30 days, ~8 events/year
- **Data**: federalreserve.gov Beige Book PDFs back to 1996
- **Mechanism**: Regional Fed contacts use "uncertainty" when capex decisions are being delayed.
- **Originality**: 8

### K-4 SLOOS "Willingness to Lend to Consumers" Reversal
- **Rule**: Long XLY next quarter when SLOOS DRIWCIL net % rises 10+ points QoQ.
- **Asset**: XLY
- **Horizon**: 60 days quarterly
- **Data**: FRED DRIWCIL
- **Mechanism**: Bank willingness leads consumer credit origination by 1 quarter → discretionary spending.
- **Originality**: 8

### K-5 ClinicalTrials.gov Phase-3 Completion-Date Slip
- **Rule**: Short sponsor 20 days when a Phase-3 trial's Primary Completion Date slips >90 days in single update.
- **Asset**: single-name biotech sponsor
- **Horizon**: 20 days
- **Data**: clinicaltrials.gov/api/v2 StudyVersions endpoint
- **Mechanism**: 90+ day slips reflect enrollment/efficacy issues; registry updates ahead of 8-K.
- **Originality**: 9

### K-6 USDA Grain Barge Rate Surge
- **Rule**: Short MOO 15 days when St Louis-NOLA grain barge rate rises >30% WoW.
- **Asset**: MOO, ADM short
- **Horizon**: 15 days
- **Data**: ams.usda.gov/services/transportation-analysis/gtr weekly PDF
- **Mechanism**: Barge-rate spikes (drought, lock closure) compress grain-handler margins faster than futures.
- **Originality**: 9

### K-7 EPA SO2 Allowance Auction Clearing Price Collapse
- **Rule**: Long BTU/coal basket 60 days when EPA annual SO2 Acid Rain auction clears <$1.
- **Asset**: BTU, KOL
- **Horizon**: 60 days (annual auction in March)
- **Data**: epa.gov/power-sector/acid-rain-program-auctions
- **Mechanism**: Near-zero clearing = perceived negligible future coal-burning = peak pessimism = contrarian.
- **Originality**: 10

### K-8 NAHB Traffic-of-Prospective-Buyers Z-Score
- **Rule**: Long ITB 30 days when NAHB Traffic subindex prints 1σ above 12m mean.
- **Asset**: ITB
- **Horizon**: 30 days monthly
- **Data**: nahb.org Housing Market Index monthly
- **Mechanism**: Buyer foot-traffic leads contract signings by 30-60 days; subindex more volatile than smoothed HMI.
- **Originality**: 8

### K-9 Treasury Daily Statement DoD Outlays Surge
- **Rule**: Long ITA 10 trading days when DTS DoD outlays exceed $5B single day.
- **Asset**: ITA
- **Horizon**: 10 days
- **Data**: fiscaldata.treasury.gov daily-treasury-statement Table III-A (daily JSON API)
- **Mechanism**: Large single-day DoD outlays = milestone payments on weapons contracts not yet in earnings.
- **Originality**: 9

### K-10 China Customs Rare Earth Export Volume Drop
- **Rule**: Long REMX 30 days when China monthly rare earth export volume falls >20% YoY.
- **Asset**: REMX
- **Horizon**: 30 days
- **Data**: english.customs.gov.cn monthly export tables (codes 28053000 + 280530)
- **Mechanism**: China supplies ~85% rare-earth processing globally; export drops tighten ex-China supply.
- **Originality**: 8

### K-11 BPA Wind Generation Forecast-Bust Days
- **Rule**: Long FAN 5 days when BPA day-ahead wind forecast underperforms realized by >25% for 3 consecutive days.
- **Asset**: FAN
- **Horizon**: 5 days
- **Data**: transmission.bpa.gov 5-minute CSV (free)
- **Mechanism**: Sustained forecast-busts upward = Pacific NW weather regime lifting wind capacity factors nationally; contrarian buy when sentiment depressed.
- **Originality**: 10

### K-12 FEMA Major Disaster Declaration Cluster
- **Rule**: Short P&C insurer basket (TRV+ALL+CB) 20 days when FEMA issues >5 Major Disaster Declarations in rolling 30-day window.
- **Asset**: P&C insurers
- **Horizon**: 20 days
- **Data**: fema.gov OpenFEMA API real-time JSON
- **Mechanism**: Declaration clusters correlate with cat-loss surprises feeding P&C earnings revisions; equity lags FEMA bulletin.
- **Originality**: 8

### K-13 SBA 7(a) Weekly Loan Approval Volume Drop
- **Rule**: Short IWM 20 days when SBA 7(a) weekly approved loan $ volume falls >25% WoW (excluding holiday weeks).
- **Asset**: IWM
- **Horizon**: 20 days
- **Data**: sba.gov weekly lending report (CSV)
- **Mechanism**: SBA 7(a) is main credit channel for sub-$10M-revenue businesses; collapses precede small-business hiring/capex deceleration affecting Russell 2000 earnings.
- **Originality**: 9

### K-14 Dallas Fed Manufacturing Capex-Future Subindex
- **Rule**: Short XLI 30 days when Dallas Fed Manufacturing Survey "Capex 6 Months Ahead" sub falls >15 points MoM.
- **Asset**: XLI
- **Horizon**: 30 days monthly
- **Data**: dallasfed.org Manufacturing Survey
- **Mechanism**: Texas manufacturers (energy-services tilt) flag capex pullbacks faster than national surveys.
- **Originality**: 8

### K-15 CMS Hospital Cost Report Bad-Debt Ratio Spike
- **Rule**: Short hospital operators (HCA+UHS+THC) 90 days when latest CMS quarterly bad-debt-to-revenue rises >50 bps QoQ.
- **Asset**: hospital basket
- **Horizon**: 90 days
- **Data**: cms.gov Healthcare Cost Report Information System (quarterly aggregate)
- **Mechanism**: Bad-debt is leading margin indicator; CMS aggregates capture industry-wide stress before any single 10-Q.
- **Originality**: 9
