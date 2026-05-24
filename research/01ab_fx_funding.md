# Phase 1AB — FX + Cross-Asset Funding/Vol Signals
# 10 signals. Renamed F→AB to avoid collision with policy 1F batch. Dedup: all NEW (2 related-variants).

### AB-1 CAD/WTI Residual Snap-back → USDCAD
- **Rule**: Long USDCAD when 5-day rolling residual of CAD on WTI < -2σ (60d window); exit at mean-reversion to 0.
- **Data**: FRED DEXCAUS + DCOILWTICO
- **CAGR**: 10-14%; Originality 8

### AB-2 NOK-Brent Sovereign-Wealth Lag → USDNOK
- **Rule**: Long NOK when 20d Brent return > +8% AND 20d NOK appreciation < 50% of Brent-implied move.
- **Data**: FRED DEXNOUS + DCOILBRENTEU
- **Mechanism**: Norway's GPFG mechanically rebalances NOK conversions with 3-6 week lag
- **CAGR**: 12-16%; Originality 9

### AB-3 MXN-Oil Decoupling + Banxico Carry → Short USDMXN
- **Rule**: Short USDMXN when 90d corr(MXN, WTI) < +0.15 AND Mexico–US 1Y rate diff > +500bps.
- **Data**: FRED DEXMXUS + WTI + Mexico short rate + DGS1
- **Mechanism**: Petro-currency → carry-currency regime switch
- **CAGR**: 14-18%; Originality 9

### AB-4 BRL Soft-Commodity Composite Z-Divergence
- **Rule**: Long BRL when equal-weighted z-score of {iron ore, sugar, coffee, soy} 60d returns > +1.5σ AND BRL 60d z < 0.
- **Data**: FRED DEXBZUS + yfinance commodity futures
- **CAGR**: 12-15%; Originality 9

### AB-5 CLP Leads Copper → Long Copper
- **Rule**: Long HG=F when CLP 10d return outperforms copper 10d return by > 2σ.
- **Data**: yfinance CLP=X + HG=F
- **Mechanism**: Chilean miner hedging makes CLP front-run physical copper by 3-10 days
- **CAGR**: 11-14%; Originality 9

### AB-6 MOVE / 3M T-Bill Funding Stress Ratio → Short SPY
- **Rule**: Short SPY when MOVE/(DTB3*100) > 1y 90th pct for 3 consec sessions.
- **Data**: yfinance ^MOVE + FRED DTB3
- **CAGR**: 11-15%; Originality 8
- **Dedup**: Variant of C11 (MOVE/VIX) — uses DTB3 denominator for absolute funding-cost stress

### AB-7 VIX9D/VIX Acute Stress Regime → Long SPY (Mean Reversion)
- **Rule**: Long SPY on open after VIX9D/VIX > 1.20 close AND VIX itself < 1y 70th pct; hold 5d.
- **Data**: yfinance ^VIX9D + ^VIX
- **CAGR**: 12-16%; Originality 8
- **Dedup**: Variant of B01 (VIX/VIX3M) — ultra-short term + level filter

### AB-8 US-JP 10Y Real-Rate Differential → USDJPY
- **Rule**: Long USDJPY when (DFII10 − JP 10Y real proxy) rises > +25bps over 20d AND USDJPY 20d return is < half the expected move.
- **Data**: FRED DFII10 + Japan 10Y nominal + JP CPI
- **CAGR**: 10-13%; Originality 8

### AB-9 CNH PBoC Fix Drift Devaluation Signal → Long USDCNH
- **Rule**: Long USDCNH (or short FXI + long DXY) when 5d cumulative PBoC mid-fix change > +0.5% AND spot within 0.3% of upper 2% band.
- **Data**: FRED DEXCHUS + yfinance USDCNH=X
- **CAGR**: 10-14%; Originality 9

### AB-10 STLFSI Acceleration → Reduce Equity Exposure
- **Rule**: Reduce equity (short SPY / long TLT) when 4-week change in STLFSI4 > +0.30 σ units AND STLFSI4 level still < +1.
- **Data**: FRED STLFSI4 (weekly)
- **CAGR**: 10-13%; Originality 8
