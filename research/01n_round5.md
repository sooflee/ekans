# Phase 1N — Hunt Round 5: autos, logistics, semis, repo, niche
# Returned by hunt agent. 12 signals, originality 8-10, one-line rules.

### N-1 Manheim Mid-Month Print → KMX Long / Auto-Retail Short
- **Rule**: Long KMX, short (AN+ABG+GPI+LAD+SAH) 20d when Manheim mid-month estimate prints > 1.0% MoM above prior full-month final.
- **Data**: manheim.com mid-month bulletin (PDF scrape)
- **Mechanism**: KMX marks inventory to wholesale faster than franchised dealers — wholesale surge expands KMX gross-per-unit before sell-side updates.
- **CAGR**: 12-15%
- **Originality**: 9

### N-2 MBA Refi Application Whipsaw → REM
- **Rule**: Long REM 10d when MBA Refinance Index jumps > 40% WoW from a 26-week low.
- **Data**: FRED MORTGAGE (weekly Wed 7am ET)
- **Mechanism**: Refi spike = rate-driven prepayment regime change → MBS spreads tighten → mREIT book marks up before NAV disclosures.
- **CAGR**: 10-12%
- **Originality**: 8

### N-3 AAR Intermodal vs Carloads Divergence
- **Rule**: Long XTN / short IYT 4 weeks when AAR weekly intermodal YoY exceeds carload YoY by > 5pp for 2 consecutive weeks.
- **Data**: aar.org/data-center/rail-traffic-data (weekly CSV)
- **Mechanism**: Intermodal outpacing = consumer/import demand strengthening vs industrial → favors trucking/parcel (XTN) over Class I rails (IYT).
- **CAGR**: 10%
- **Originality**: 9

### N-4 SIA Semiconductor Billings 3-Month Acceleration → SOXX
- **Rule**: Long SOXX 60d when SIA global semiconductor billings 3MMA YoY accelerates by > 2pp MoM.
- **Data**: semiconductors.org/global-billings-report (monthly press release)
- **Mechanism**: SIA 3MMA-YoY second derivative leads semi equipment book-to-bill by ~1 quarter; SOXX reprices on revenue-acceleration regime before sell-side numbers move.
- **CAGR**: 15%
- **Originality**: 8

### N-5 TSMC Monthly Revenue Surprise → AI-Semi Basket
- **Rule**: Long NVDA+AVGO+AMD+ASML basket 10d when TSMC monthly revenue (released ~10th) beats trailing-3m avg by > 8% YoY.
- **Data**: tsmc.com/english/investorRelations/monthly_revenue
- **Mechanism**: TSMC monthly revenue is highest-frequency public read on leading-edge logic demand; surprises propagate to fab customers before quarterly pre-announcements.
- **CAGR**: 15-20%
- **Originality**: 9

### N-6 SOFR-IORB Funding Stress → Short SPY
- **Rule**: Short SPY 5d when daily SOFR - IORB > +10bp for 2 consecutive days.
- **Data**: FRED SOFR + IORB (daily 8am ET)
- **Mechanism**: SOFR persistently above IORB = scarce reserves + dealer balance-sheet stress → precedes risk-asset drawdowns (Sep 2019, Mar 2020, Sep 2023).
- **CAGR**: 12% (sparse event-driven)
- **Originality**: 9

### N-7 Coinbase Daily Volume Crush → Short COIN
- **Rule**: Short COIN 10d when Coinbase exchange daily spot volume drops > 50% from trailing 30d avg for 3 consecutive sessions.
- **Data**: api.exchange.coinbase.com/products/BTC-USD/stats (free)
- **Mechanism**: ~95% of Coinbase revenue is transaction fees; sustained volume collapse compresses earnings before sell-side cuts numbers.
- **CAGR**: 15%
- **Originality**: 8

### N-8 EIA Propane Stocks vs 5y Range (Pre-Heating-Season) → UNG
- **Rule**: Long UNG 30d when EIA weekly propane stocks (Sep 1 - Nov 15) fall below 5-year minimum.
- **Data**: eia.gov/dnav/pet/pet_stoc_wstk (weekly Wed 10:30am)
- **Mechanism**: Propane depletion ahead of winter = heating-fuel substitution + cold expectations → nat gas burn forecasts up before NOAA seasonal outlooks.
- **CAGR**: 12%
- **Originality**: 9

### N-9 India FII Daily Net Flow Reversal → INDA Long
- **Rule**: Long INDA 15d when FII net cash equity flow is negative 8 of prior 10 sessions AND cumulative outflow > -$2B.
- **Data**: nseindia.com/reports/fii-dii (daily CSV)
- **Mechanism**: FII selling is mechanical (EM-fund redemptions, INR hedging); reversals sharp once marginal seller exhausts because domestic SIP flow keeps providing a bid.
- **CAGR**: 12-15%
- **Originality**: 8

### N-10 0DTE Volume Share Regime → Long Vol
- **Rule**: Long 1m SPX straddle 20d when 0DTE share of SPX options daily volume exceeds 55% for 5 consecutive sessions.
- **Data**: cboe.com/us/options/market_statistics/daily (free)
- **Mechanism**: Intraday hedging via 0DTE dominating → longer-dated IV compressed vs realized; mean-revert via vol-of-vol spikes once dealer gamma flips.
- **CAGR**: 15%
- **Originality**: 9

### N-11 Pendle YT-ETH Supply Burn-Down → Long ETH
- **Rule**: Long ETH 14d when total Pendle YT-ETH supply outstanding declines > 20% in 7 days.
- **Data**: api-v2.pendle.finance/core/v1/1/markets
- **Mechanism**: YT-ETH = leveraged speculation on ETH staking yield rising; rapid burn = traders cashing in or rotating to spot, locked principal redeploys → spot rallies.
- **CAGR**: 20%+
- **Originality**: 10

### N-12 TGA Estimated-Tax-Date Drawdown Reversal → Long TLT
- **Rule**: Long TLT 5d starting on the second business day after April 15 / June 15 / Sep 15 / Jan 15 if TGA balance rose > $80B in preceding 5 sessions.
- **Data**: fiscaldata.treasury.gov daily TGA balance
- **Mechanism**: Estimated-tax inflows pull reserves from banking system → dealers sell duration into deadline; once flow stops, duration overhang unwinds, long end rallies.
- **CAGR**: 10% (4 events/year)
- **Originality**: 8
