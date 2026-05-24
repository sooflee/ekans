# Phase 1V — YouTube / Podcast Round 2 (different creators)
# Returned by hunt agent. 12 signals.
# Dedup note: see annotations on each signal vs existing catalog.

### V-1 Cole Long-Vol "5% Move" Allocation Pulse ✓ NEW
- **Creator**: Chris Cole (Artemis) — RCM "The Derivative" #100
- **Rule**: After SPX makes ±5% one-month return, scale up 1m 5%-OTM SPX put-spread sleeve to 2x normal weight, holding 21 trading days.
- **Asset**: SPX options (proxy VIXM)
- **Mechanism**: 5% monthly move = empirical threshold where vol-of-vol + dealer hedging flip from short-gamma to crisis-trending
- **CAGR**: 6-9% standalone; +1-2pp Sharpe boost as overlay
- **Originality**: 9

### V-2 Verdad HY OAS > 600bps Crisis Switch
- **Creator**: Dan Rasmussen (Verdad)
- **Rule**: When HY OAS > 600bps closes, rotate cash → 50/50 HYG+IJS basket; exit when OAS < 400bps.
- **Asset**: HYG, IJS
- **CAGR**: 30-40% on triggered windows; ~10-12% blended
- **Originality**: 8
- **Dedup**: Related to C10 (HY OAS regime, de-risk side); Verdad's is the BUY side with hysteresis + small-cap value tilt. Different rule, different assets. Keep as distinct.

### V-3 Felder Margin-Debt/GDP Reversal
- **Creator**: Jesse Felder
- **Rule**: Cut equity beta to 50% when 12m change in (FINRA Margin Debt / Nominal GDP) rolls negative from above 2.5% of GDP; restore to 100% only after fresh 12m positive change.
- **Asset**: SPY
- **Originality**: 8
- **Dedup**: Overlaps F03 (Margin Debt YoY) — Felder normalizes by GDP and uses 12m rolling diff. Tests differently; potentially still distinct enough.

### V-4 Doomberg LNG-Brent Btu Parity Pair ✓ NEW
- **Creator**: Doomberg
- **Rule**: Long UNG / short BNO when ratio of landed JKM LNG ($/MMBtu) to Brent ($/MMBtu) > 2σ above 3y mean.
- **Asset**: UNG / BNO pair
- **Mechanism**: When LNG runs far above Brent's Btu-equivalent, hot-spot demand pulls US gas higher
- **CAGR**: 8-15%; Originality 9

### V-5 Apollo LA Port TEU Recession Switch
- **Creator**: Torsten Slok (Apollo Academy)
- **Rule**: When 4w MA of inbound LA TEUs < -15% YoY for 3 consec weeks, rotate cyclicals (XLI/XLY/XTN) → defensives (XLP/XLU).
- **Originality**: 8
- **Dedup**: Different port from G10 (IMF PortWatch China). LA port is US-specific; can be distinct.

### V-6 Bob Elliott HFRX Macro Replication ✓ NEW
- **Creator**: Bob Elliott (Unlimited)
- **Rule**: Monthly, regress 24m rolling HFRX Macro/CTA returns on 6 Fung-Hsieh trend factors; replicate top-3 positively-loaded futures betas with 30% notional via ETF basket.
- **Asset**: UUP/GLD/DBC/TLT/SHY rotation
- **CAGR**: 5-7% Sharpe ~0.7, low SPX correlation
- **Originality**: 9

### V-7 Papic GDELT Geopolitical Crisis-Dip Buy
- **Creator**: Marko Papic (Clocktower)
- **Rule**: When SPX falls > 5% within 5 trading days AND GDELT Global Conflict Index jumps > 2σ, buy 100% SPX hold 12 months.
- **Originality**: 9
- **Dedup**: Related to G-23 (GPR tail-spike BUY SPY, which underperformed). Papic uses GDELT + price drop combo — different trigger. Worth testing.

### V-8 Crittenden 12-Asset 200d Multi-Asset Trend
- **Creator**: Eric Crittenden (Standpoint / BLNDX)
- **Rule**: Hold equal-weight long across 12 futures assets only when current price > 200d SMA; reweight monthly.
- **Asset**: SPY/EFA/EEM/GLD/USO/CPER/DBA/TLT/UUP/UNG/SOYB/WEAT
- **Originality**: 8
- **Dedup**: Related to C05 Faber GTAA (5-asset 10-mo MA). Crittenden's is wider asset set + 200d. Worth testing as a separate variant.

### V-9 Hoffstein Trend-Voter Ensemble ✓ NEW
- **Creator**: Corey Hoffstein (Newfound)
- **Rule**: For each of 8 trend models on SPY (50/200 SMA, 12-1 momentum, ROC, Donchian-50, MACD, regression slope, KAMA, BB-breakout), tally positive votes; size SPY exposure as (positive votes / 8).
- **Asset**: SPY
- **CAGR**: 9-10% with lower DD; +1.5-2% risk-adjusted alpha
- **Originality**: 8

### V-10 Alden Global M2 13-week BTC ❌ DUPE
- **DROP**: This is fundamentally the same as **P6 Howell global liquidity 13-week → BTC** (43% CAGR / Sharpe 1.04 / t=4.29 already in sidebar). Same target (BTC), same lag (13w), same mechanism (global liquidity). Not testing.

### V-11 Sidial COR3M + VVIX/VIX Dispersion Gate
- **Creator**: Kris Sidial (Ambrus)
- **Rule**: Enter long single-stock straddle vs short SPX straddle (dispersion) when implied-correlation (CBOE COR3M) < 1y 20th pct AND VVIX/VIX > 6.5.
- **Originality**: 9
- **Dedup**: B03 (VVIX/VIX alone) underperformed in our backtest. Sidial adds COR3M gate which is genuinely a different filter. Worth testing if COR3M data accessible.

### V-12 Kuppy URNM-SRUUF Pair
- **Creator**: Harris "Kuppy" Kupperman (Praetorian)
- **Rule**: Long URNM / short SRUUF when URNM has lagged SRUUF by > 20% over trailing 6 months; close when gap < 5%.
- **Originality**: 9
- **Dedup**: Distinct from X5 (Sprott NAV → CCJ). Kuppy's is equity-vs-physical pair trade.
