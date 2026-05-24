# Phase 1T — Commodity Signals Round 3 (Fresh Territory)
# Returned by hunt agent. 12 signals, originality 8-10.

### T-1 Class IV - Class III Milk Spread Reversion
- **Rule**: Long DK (Class IV) / short DC (Class III) 1:1 when front-month spread < -$3.50/cwt; exit at zero.
- **Asset**: CME DK vs DC
- **Horizon**: 2-6 months
- **Data**: USDA AMS Dairy Market News weekly + CME settlements
- **Originality**: 9 — dairy spreads almost never traded systematically
- **CAGR**: 12-15%

### T-2 Boxed Beef Cutout vs Live Cattle Pack Margin
- **Rule**: Short LE front-month when USDA Choice cutout / live cattle ratio < 1.55; hold 30 days.
- **Asset**: CME LE
- **Horizon**: 1-2 months
- **Data**: USDA AMS LM_XB459 daily cutout / LM_CT150 weekly steer
- **Originality**: 9 — industry insider trade, never systematized
- **CAGR**: 10-13%

### T-3 Indian Monsoon Onset Delay → Long Sugar
- **Rule**: Long SB when IMD June 1-30 cumulative all-India rainfall < 88% of LPA; exit Oct 31.
- **Asset**: ICE SB #11
- **Horizon**: 4 months
- **Data**: mausam.imd.gov.in daily rainfall
- **Originality**: 8
- **CAGR**: 14-18%

### T-4 Capesize Spot Rate Spike → Long Iron Ore
- **Rule**: Long SGX FEF when Baltic Capesize 5TC daily spot rate jumps > 35% WoW.
- **Asset**: SGX FEF
- **Horizon**: 3-8 weeks
- **Data**: Baltic Exchange via Hellenic Shipping News free weekly
- **Originality**: 8
- **CAGR**: 15-20%

### T-5 Henry Hub vs Waha Basis Blow-Out → Long NG
- **Rule**: Long NG front-month when Waha next-day cash basis to Henry Hub < -$3.00/MMBtu for 5 consecutive days.
- **Asset**: NYMEX NG
- **Horizon**: 2-6 weeks
- **Data**: EIA daily spot natural gas prices
- **Originality**: 9 — Waha basis followed by midstream specialists, never traded as HH long
- **CAGR**: 12-16%

### T-6 Soybean Board Crush Compression → Long Beans
- **Rule**: Long ZS front-month when board crush (ZL*11 + ZM*2.2 - ZS) falls below $0.40/bu for 3 consecutive sessions.
- **Asset**: CBOT ZS
- **Horizon**: 4-10 weeks
- **Data**: CME daily settlements for ZS, ZM, ZL
- **Originality**: 8 (long-beans-at-compressed-margin non-obvious)
- **CAGR**: 11-14%

### T-7 LCFS Credit Price Floor Reversion → Long KRBN
- **Rule**: Long KRBN when CARB monthly LCFS credit weighted-avg price drops < $60/MT.
- **Asset**: KRBN (KraneShares Global Carbon)
- **Horizon**: 6-18 months
- **Data**: CARB monthly LCFS credit transfer activity PDF
- **Originality**: 9 — LCFS rarely traded by macro/commodity systematic
- **CAGR**: 15-25%

### T-8 South African Load-Shedding → Long Platinum
- **Rule**: Long PL front-month when Eskom announces Stage 4+ load-shedding for 7+ consecutive days.
- **Asset**: NYMEX PL
- **Horizon**: 1-4 months
- **Data**: eskom.co.za official + EskomSePush archive
- **Originality**: 9 — SA-news territory, almost no systematic strategy uses it
- **CAGR**: 14-20%

### T-9 AUDUSD vs Iron Ore Divergence Reversion → Long Iron Ore
- **Rule**: Long SGX FEF when 20-day rolling corr AUDUSD-FEF < 0.20 AND AUDUSD outperforms FEF by > 5% in prior 30 days.
- **Asset**: SGX FEF (or yfinance proxy)
- **Horizon**: 4-8 weeks
- **Data**: FRED DEXUSAL + SGX FEF
- **Originality**: 8
- **CAGR**: 11-15%

### T-10 Tampa Urea Premium → Long Corn
- **Rule**: Long ZC when Tampa urea barge minus Black Sea FOB urea > $80/MT for 3 consecutive weeks.
- **Asset**: CBOT ZC (new-crop Dec during planting)
- **Horizon**: 2-6 months
- **Data**: World Bank Pink Sheet monthly + DTN weekly
- **Originality**: 9 — fertilizer-to-grain throughput trade
- **CAGR**: 10-13%

### T-11 US Ethanol Crush Margin Counter-Intuitive → Long Corn
- **Rule**: Long ZC when EIA weekly ethanol production drops > 5% WoW AND ethanol-corn crush margin negative for 2 consecutive weeks.
- **Asset**: CBOT ZC
- **Horizon**: 4-10 weeks
- **Data**: EIA Weekly Petroleum Status + CBOT ethanol futures + USDA DDG
- **Originality**: 9 — DDG-substitution counter-mechanism widely overlooked
- **CAGR**: 10-14%

### T-12 LBMA Silver Lease Rate Spike → Long Silver ⭐
- **Rule**: Long SI front-month when LBMA 1-month silver lease rate > 4.0% annualized for 3 consecutive days.
- **Asset**: COMEX SI
- **Horizon**: 1-3 months
- **Data**: Kitco / Bullionvault daily lease rate
- **Originality**: 10 — silver lease rates absent from systematic literature
- **CAGR**: 14-22%
