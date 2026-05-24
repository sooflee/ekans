# Phase 1B — Academic / Quant Literature Signals
# Returned by research agent. Citations to be verified before final report.

### Value (HML)
- **What it is**: Long high book-to-market stocks, short low book-to-market. Captures the cheapness premium.
- **Asset class**: equities (extended to all asset classes by AQR)
- **Horizon**: months to years
- **Source**: Fama & French (1992, 1993), "Common risk factors in the returns on stocks and bonds," Journal of Financial Economics. https://doi.org/10.1016/0304-405X(93)90023-5
- **Reported edge**: ~4-5% annual premium 1963-1990 in original sample
- **Backtest-feasibility with free data (1-5)**: 5 (Ken French data library)
- **Originality (1-10)**: 1
- **Decay risk**: High. Severe drawdown 2010-2020; debated whether premium still exists post-publication. McLean/Pontiff (2016) finds ~32% post-publication decay across anomalies generally.

### Cross-sectional Momentum (UMD)
- **What it is**: Long past 12-month winners (skip last month), short losers.
- **Asset class**: equities, but documented across all asset classes
- **Horizon**: 1-12 months
- **Source**: Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers," Journal of Finance. https://doi.org/10.1111/j.1540-6261.1993.tb04702.x
- **Reported edge**: ~1% per month in original 1965-1989 sample; ~8% annual long-run
- **Backtest-feasibility with free data (1-5)**: 5
- **Originality (1-10)**: 1
- **Decay risk**: Moderate. Survived OOS but suffers severe crashes (2009, 2020 Q2). Daniel & Moskowitz (2016) "Momentum Crashes."

### Quality / Profitability
- **What it is**: Long high gross-profits-to-assets, short low. Novy-Marx's "other side of value."
- **Asset class**: equities
- **Horizon**: months to years
- **Source**: Novy-Marx (2013), "The other side of value: The gross profitability premium," JFE. https://doi.org/10.1016/j.jfineco.2013.01.003
- **Reported edge**: ~0.31% per month abnormal return
- **Backtest-feasibility with free data (1-5)**: 2 (needs Compustat)
- **Originality (1-10)**: 2
- **Decay risk**: Low; survived as part of FF5 (Fama-French 2015).

### Betting Against Beta (BAB)
- **What it is**: Long leveraged low-beta stocks, short high-beta. Funding-constraint-driven premium.
- **Asset class**: equities, bonds, FX, commodities
- **Horizon**: months
- **Source**: Frazzini & Pedersen (2014), "Betting against beta," JFE. https://doi.org/10.1016/j.jfineco.2013.10.005
- **Reported edge**: US equity BAB Sharpe ~0.78
- **Backtest-feasibility with free data (1-5)**: 4 (yfinance + market index)
- **Originality (1-10)**: 2
- **Decay risk**: Novy-Marx & Velikov (2022) argue construction-specific; some decay debated.

### Quality Minus Junk (QMJ)
- **What it is**: Long stocks that are profitable, growing, safe, well-paying-out; short the opposite. Multi-dimensional quality.
- **Asset class**: equities (24 countries in original)
- **Horizon**: months
- **Source**: Asness, Frazzini & Pedersen (2019), "Quality Minus Junk," Review of Accounting Studies. https://doi.org/10.1007/s11142-018-9470-2 (working paper 2013)
- **Reported edge**: US Sharpe ~0.75; global ~1.0 after risk adjustments
- **Backtest-feasibility with free data (1-5)**: 2 (Compustat needed for full construction; AQR posts factor returns)
- **Originality (1-10)**: 3
- **Decay risk**: Low; out-of-sample evidence in global markets supportive.

### Post-Earnings Announcement Drift (PEAD)
- **What it is**: Stocks with positive earnings surprises continue to drift up for 60+ days post-announcement; negative SUE drifts down.
- **Asset class**: equities
- **Horizon**: 1-3 months post-earnings
- **Source**: Bernard & Thomas (1989), "Post-earnings-announcement drift: Delayed price response or risk premium?" Journal of Accounting Research. https://doi.org/10.2307/2491062
- **Reported edge**: ~18% annualized for top-bottom SUE decile in original sample
- **Backtest-feasibility with free data (1-5)**: 3 (earnings dates via yfinance/EDGAR; SUE construction is non-trivial without IBES)
- **Originality (1-10)**: 3
- **Decay risk**: Mixed. Chordia/Goyal/Sadka/Shivakumar (2009) show decay in liquid stocks; persists in illiquid names.

### Idiosyncratic Volatility Puzzle
- **What it is**: Stocks with high idiosyncratic vol (residual from FF3) earn LOWER subsequent returns — opposite of risk-reward intuition.
- **Asset class**: equities
- **Horizon**: 1 month
- **Source**: Ang, Hodrick, Xing, Zhang (2006), "The cross-section of volatility and expected returns," Journal of Finance. https://doi.org/10.1111/j.1540-6261.2006.00836.x
- **Reported edge**: -1.06% per month, quintile spread
- **Backtest-feasibility with free data (1-5)**: 4 (daily returns from yfinance)
- **Originality (1-10)**: 4
- **Decay risk**: Partially attributable to MAX effect (Bali et al.); Stambaugh/Yu/Yuan (2015) tie it to mispricing.

### 52-Week High Momentum
- **What it is**: Stocks near their 52-week high outperform — proximity to 52-week high (not past return) is the signal.
- **Asset class**: equities
- **Horizon**: months
- **Source**: George & Hwang (2004), "The 52-week high and momentum investing," Journal of Finance. https://doi.org/10.1111/j.1540-6261.2004.00695.x
- **Reported edge**: 0.45% per month after controlling for J-T momentum
- **Backtest-feasibility with free data (1-5)**: 5 (just need daily price)
- **Originality (1-10)**: 5
- **Decay risk**: Liu/Liu/Ma (2011) show it works globally OOS; survived.

### Industry Momentum
- **What it is**: Past industry winner industries outperform past loser industries; explains much of stock-level momentum.
- **Asset class**: equities
- **Horizon**: 1-12 months
- **Source**: Moskowitz & Grinblatt (1999), "Do industries explain momentum?" Journal of Finance. https://doi.org/10.1111/0022-1082.00146
- **Reported edge**: ~0.43% per month for top-bottom industry portfolios at 6-month horizon
- **Backtest-feasibility with free data (1-5)**: 5 (Ken French 49-industry portfolios)
- **Originality (1-10)**: 4
- **Decay risk**: Grundy/Martin and others have replicated; remains.

### Time-Series Momentum (TSMOM)
- **What it is**: Each asset's own past 12-month return predicts its next-month return; long winners, short losers in same instrument.
- **Asset class**: any (equities, bonds, FX, commodities — 58 instruments in original)
- **Horizon**: 1-12 months
- **Source**: Moskowitz, Ooi & Pedersen (2012), "Time series momentum," JFE. https://doi.org/10.1016/j.jfineco.2011.11.003
- **Reported edge**: Diversified TSMOM Sharpe ~1.4 in original 1985-2009 sample
- **Backtest-feasibility with free data (1-5)**: 4 (futures data harder; ETFs work for proxy)
- **Originality (1-10)**: 3
- **Decay risk**: Huang/Li/Wang/Zhou (2020) "Time-series momentum: Is it there?" claim weak OOS; AQR rebuts.

### MAX / Lottery Effect
- **What it is**: Stocks with highest single-day returns over prior month underperform — investors overpay for lottery-like upside.
- **Asset class**: equities
- **Horizon**: 1 month
- **Source**: Bali, Cakici & Whitelaw (2011), "Maxing out: Stocks as lotteries and the cross-section of expected returns," JFE. https://doi.org/10.1016/j.jfineco.2010.08.014
- **Reported edge**: -1.03% per month (raw); -0.65% FF4-alpha for MAX(5) decile spread
- **Backtest-feasibility with free data (1-5)**: 5
- **Originality (1-10)**: 5
- **Decay risk**: Replicated internationally (Annaert et al. 2013). Has held up.

### Short Interest
- **What it is**: High short interest predicts negative returns; arbitrageurs are informed.
- **Asset class**: equities
- **Horizon**: 1 month
- **Source**: Boehmer, Jones & Zhang (2008), "Which shorts are informed?" Journal of Finance. https://doi.org/10.1111/j.1540-6261.2008.01324.x; also Asquith/Pathak/Ritter (2005) JFE.
- **Reported edge**: ~1.6% per month for top-decile short interest (value-weighted)
- **Backtest-feasibility with free data (1-5)**: 3 (FINRA short interest is bi-monthly free; daily SI requires paid feed)
- **Originality (1-10)**: 4
- **Decay risk**: Persists but more arbitraged in large caps.

### Analyst Dispersion
- **What it is**: Stocks with high dispersion in analyst forecasts earn LOWER returns — proxy for opinion divergence + short-sale constraints.
- **Asset class**: equities
- **Horizon**: months
- **Source**: Diether, Malloy & Scherbina (2002), "Differences of opinion and the cross-section of stock returns," Journal of Finance. https://doi.org/10.1111/1540-6261.00490
- **Reported edge**: -9.5% annualized for high-vs-low dispersion quintile
- **Backtest-feasibility with free data (1-5)**: 2 (needs IBES; some free proxies via Yahoo Finance analyst tab — noisy)
- **Originality (1-10)**: 5
- **Decay risk**: Mixed; Avramov/Chordia/Jostova/Philipov (2009) tie it to credit risk.

### Accruals Anomaly
- **What it is**: High-accruals firms (earnings driven by non-cash items) underperform low-accruals firms — market doesn't fully discount earnings quality.
- **Asset class**: equities
- **Horizon**: 1 year
- **Source**: Sloan (1996), "Do stock prices fully reflect information in accruals and cash flows about future earnings?" The Accounting Review. https://www.jstor.org/stable/248290
- **Reported edge**: 10.4% annual hedge return (extreme deciles)
- **Backtest-feasibility with free data (1-5)**: 2 (Compustat)
- **Originality (1-10)**: 4
- **Decay risk**: Green/Hand/Soliman (2011) "Going, going, gone? The apparent demise of the accruals anomaly." Substantial decay post-2003 once known.

### Net Stock Issuance
- **What it is**: Firms that issue net new shares underperform; net repurchasers outperform.
- **Asset class**: equities
- **Horizon**: 1-3 years
- **Source**: Pontiff & Woodgate (2008), "Share issuance and cross-sectional returns," Journal of Finance. https://doi.org/10.1111/j.1540-6261.2008.01334.x; Daniel & Titman (2006).
- **Reported edge**: ~10% annual spread (decile)
- **Backtest-feasibility with free data (1-5)**: 4 (shares-outstanding deltas from yfinance/EDGAR)
- **Originality (1-10)**: 4
- **Decay risk**: Survives; one of the more robust anomalies in McLean/Pontiff (2016).

### Asset Growth
- **What it is**: Firms growing total assets fastest underperform; "investment factor" intuition.
- **Asset class**: equities
- **Horizon**: 1 year
- **Source**: Cooper, Gulen & Schill (2008), "Asset growth and the cross-section of stock returns," Journal of Finance. https://doi.org/10.1111/j.1540-6261.2008.01370.x
- **Reported edge**: 20% annual value-weighted hedge return (decile spread)
- **Backtest-feasibility with free data (1-5)**: 3 (total assets via EDGAR XBRL or financial APIs)
- **Originality (1-10)**: 3
- **Decay risk**: Folded into FF5 / Hou-Xue-Zhang q-factor; survives.

### R&D Intensity
- **What it is**: High R&D-to-market firms outperform — market underprices intangibles.
- **Asset class**: equities
- **Horizon**: years
- **Source**: Chan, Lakonishok & Sougiannis (2001), "The stock market valuation of research and development expenditures," Journal of Finance. https://doi.org/10.1111/0022-1082.00411; Eberhart/Maxwell/Siddique (2004) for R&D increases.
- **Reported edge**: ~6.1% annualized abnormal for high R&D/MV
- **Backtest-feasibility with free data (1-5)**: 3 (XBRL R&D line item)
- **Originality (1-10)**: 5
- **Decay risk**: Persistent; tied to intangibles missing from book value.

### Lazy Prices (10-K Textual Similarity)
- **What it is**: Firms that change the language of their 10-K/10-Q year-over-year underperform — changes signal hidden bad news.
- **Asset class**: equities
- **Horizon**: months
- **Source**: Cohen, Malloy & Nguyen (2020), "Lazy prices," Journal of Finance. https://doi.org/10.1111/jofi.12885
- **Reported edge**: ~30-60 bp/month long-short (depending on similarity metric), Sharpe ~1.3 reported in working paper
- **Backtest-feasibility with free data (1-5)**: 4 (EDGAR full-text + cosine similarity; computationally intensive)
- **Originality (1-10)**: 7
- **Decay risk**: Recent enough that decay limited; signal is hard to arbitrage at scale.

### Loughran-McDonald 10-K Sentiment
- **What it is**: Counts of negative-tone words in 10-Ks predict lower future returns and higher volatility.
- **Asset class**: equities
- **Horizon**: weeks to months
- **Source**: Loughran & McDonald (2011), "When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks," Journal of Finance. https://doi.org/10.1111/j.1540-6261.2010.01625.x; updated dictionaries at https://sraf.nd.edu/
- **Reported edge**: ~-4% three-day filing-window CAR for high-negative tertile vs low
- **Backtest-feasibility with free data (1-5)**: 5 (EDGAR + L-M dictionary free)
- **Originality (1-10)**: 5
- **Decay risk**: Mixed; effect remains but well-known.

### Insider Buying Clusters
- **What it is**: Routine vs opportunistic insider trades — opportunistic insider buys predict ~10% annual abnormal returns.
- **Asset class**: equities
- **Horizon**: 6 months
- **Source**: Cohen, Malloy & Pomorski (2012), "Decoding inside information," Journal of Finance. https://doi.org/10.1111/j.1540-6261.2012.01740.x
- **Reported edge**: 82 bp/month abnormal for opportunistic insider buys (10% annualized)
- **Backtest-feasibility with free data (1-5)**: 4 (SEC Form 4 filings free via EDGAR; OpenInsider scrapes)
- **Originality (1-10)**: 5
- **Decay risk**: Persists; constrained by capacity since insider buys are infrequent.

### Pre-FOMC Announcement Drift
- **What it is**: Equity returns are abnormally large in the 24 hours BEFORE scheduled FOMC announcements — most of the equity premium since 1994 accrues then.
- **Asset class**: equities (S&P 500 most cleanly)
- **Horizon**: ~24 hours, ~8 events/year
- **Source**: Lucca & Moench (2015), "The pre-FOMC announcement drift," Journal of Finance. https://doi.org/10.1111/jofi.12196
- **Reported edge**: ~49 bp average over 24-hour pre-announcement window; ~3.9% annualized just from these 8 days
- **Backtest-feasibility with free data (1-5)**: 5 (FOMC calendar + SPY intraday)
- **Originality (1-10)**: 6
- **Decay risk**: Cieslak/Vissing-Jorgensen and others suggest weakening post-2011; debated.

### FOMC Cycle (Even-Week Effect)
- **What it is**: Stock returns are concentrated in even weeks of the FOMC cycle (weeks 0, 2, 4, 6 after FOMC); essentially zero in odd weeks.
- **Asset class**: equities
- **Horizon**: biweekly cycle
- **Source**: Cieslak, Morse & Vissing-Jorgensen (2019), "Stock returns over the FOMC cycle," Journal of Finance. https://doi.org/10.1111/jofi.12818
- **Reported edge**: Entire US equity premium 1994-2016 earned in even weeks; ~0 in odd weeks
- **Backtest-feasibility with free data (1-5)**: 5 (FOMC dates + SPY)
- **Originality (1-10)**: 7
- **Decay risk**: Hillenbrand (2021 working paper) shows pattern survives post-publication but with reduced magnitude.

### Halloween / Sell-in-May Effect
- **What it is**: November-April returns dominate May-October returns globally.
- **Asset class**: equities (and now extended to industries/factors)
- **Horizon**: 6-month seasonal
- **Source**: Bouman & Jacobsen (2002), "The Halloween indicator, 'Sell in May and go away,'" AER. https://www.aeaweb.org/articles?id=10.1257/000282802762024683; updated by Zhang & Jacobsen (2021), "The Halloween indicator: Everywhere and all the time," International Review of Finance.
- **Reported edge**: ~10% annualized spread (Nov-Apr vs May-Oct) across 65 countries in 2021 update
- **Backtest-feasibility with free data (1-5)**: 5
- **Originality (1-10)**: 4
- **Decay risk**: Survived OOS — Zhang/Jacobsen 2021 update confirms strengthening post-publication.

### Sunshine / Weather Effect
- **What it is**: Morning sunshine at the city of a stock exchange correlates with positive same-day index returns.
- **Asset class**: equities (index)
- **Horizon**: daily
- **Source**: Hirshleifer & Shumway (2003), "Good day sunshine: Stock returns and the weather," Journal of Finance. https://doi.org/10.1111/1540-6261.00556; Saunders (1993) AER for NYC original.
- **Reported edge**: NYC: 24.8% annualized on perfectly cloudy vs clear days in Saunders; 26 cities ~ economically modest but statistically significant
- **Backtest-feasibility with free data (1-5)**: 4 (NOAA weather + index data free)
- **Originality (1-10)**: 7
- **Decay risk**: Likely attenuated by electronic trading; Goetzmann/Zhu (2005) find no effect on individual investor trading.

### Lunar Cycles
- **What it is**: Stock returns are ~3-5% per year lower around full moon days vs new moon days, across 48 countries.
- **Asset class**: equities (index level)
- **Horizon**: ~15-day half-cycle
- **Source**: Yuan, Zheng & Zhu (2006), "Are investors moonstruck? Lunar phases and stock returns," Journal of Empirical Finance. https://doi.org/10.1016/j.jempfin.2005.06.001
- **Reported edge**: 3-5% annualized spread between new-moon-window and full-moon-window returns
- **Backtest-feasibility with free data (1-5)**: 5 (lunar phase calendars + index)
- **Originality (1-10)**: 9
- **Decay risk**: Some replication, but small sample of "events"; OOS questionable.

### Daylight Saving Anomaly
- **What it is**: Stock returns on the Monday after DST changes are abnormally negative — sleep desynchronosis effect.
- **Asset class**: equities (index)
- **Horizon**: 1 trading day, ~2x/year
- **Source**: Kamstra, Kramer & Levi (2000), "Losing sleep at the market: The daylight saving anomaly," AER. https://www.aeaweb.org/articles?id=10.1257/aer.90.4.1005
- **Reported edge**: ~-27 bp on DST-change Mondays in US/UK/Germany/Canada; 200-500% larger negative return than regular Mondays
- **Backtest-feasibility with free data (1-5)**: 5
- **Originality (1-10)**: 8
- **Decay risk**: Pinegar (2002) and others have challenged statistical robustness; effect contested.

### Turn-of-Month Effect
- **What it is**: Equity returns concentrated in last trading day + first 3 trading days of each month; rest of month near zero.
- **Asset class**: equities (and bonds, McConnell-Xu 2008)
- **Horizon**: ~4 days/month
- **Source**: Ariel (1987), "A monthly effect in stock returns," JFE. https://doi.org/10.1016/0304-405X(87)90066-3; McConnell & Xu (2008), "Equity returns at the turn of the month," FAJ.
- **Reported edge**: Entire monthly return earned in 4-day TOM window; ~0.13% per TOM day vs ~0% otherwise
- **Backtest-feasibility with free data (1-5)**: 5
- **Originality (1-10)**: 5
- **Decay risk**: McConnell/Xu (2008) confirm OOS through 2005; later evidence weaker in large caps.

### Treasury Auction Cycle
- **What it is**: Treasury yields rise into auction announcements and fall after — primary dealers' inventory management creates predictable supply-driven price pattern.
- **Asset class**: bonds (Treasuries)
- **Horizon**: ~5 days around each auction
- **Source**: Lou, Yan & Zhang (2013), "Anticipated and repeated shocks in liquid markets," Review of Financial Studies. https://doi.org/10.1093/rfs/hht034
- **Reported edge**: ~9-12 bp predictable yield round-trip per auction; cumulatively very economically significant given auction frequency
- **Backtest-feasibility with free data (1-5)**: 4 (TreasuryDirect auction schedule + FRED yields)
- **Originality (1-10)**: 8
- **Decay risk**: Fleming/Liu/Peterson/Sarkar (2022 Fed staff report) find continued evidence; minimal decay reported.

### Variance Risk Premium (VRP)
- **What it is**: Implied variance (VIX^2) systematically exceeds realized variance; sellers of volatility earn a premium.
- **Asset class**: equities (also extended to commodities, FX)
- **Horizon**: weeks to 1 month
- **Source**: Carr & Wu (2009), "Variance risk premiums," Review of Financial Studies. https://doi.org/10.1093/rfs/hhn038; Bollerslev/Tauchen/Zhou (2009) on VRP as equity-return predictor.
- **Reported edge**: Average VRP ~+3 to +5 vol points; SPX vol-selling Sharpe ~1.0 historically (with severe LTCM/2008/2020 drawdowns)
- **Backtest-feasibility with free data (1-5)**: 5 (VIX + SPX realized vol from yfinance/CBOE)
- **Originality (1-10)**: 4
- **Decay risk**: Persists structurally but compressed since 2014; February 2018 / March 2020 showed crash risk.

### Currency Carry
- **What it is**: Long high-interest-rate currencies, short low — earns positive returns on average despite UIP prediction.
- **Asset class**: FX
- **Horizon**: months
- **Source**: Lustig & Verdelhan (2007) AER, "The cross-section of foreign currency risk premia and consumption growth risk." https://www.aeaweb.org/articles?id=10.1257/aer.97.1.89; also Lustig/Roussanov/Verdelhan (2011 RFS) for HML-FX factor.
- **Reported edge**: G10 carry Sharpe ~0.5-0.7 pre-2008; severely impaired post-GFC
- **Backtest-feasibility with free data (1-5)**: 4 (FX spots + short rates from FRED; futures via CFTC/Quandl)
- **Originality (1-10)**: 3
- **Decay risk**: Substantial. 2008 crash, post-2014 flat. Conventional carry largely capacity-constrained.

### Commodity Backwardation/Roll Yield
- **What it is**: Long backwardated commodities (front < back), short contangoed; captures convenience-yield-driven roll return.
- **Asset class**: commodities (futures)
- **Horizon**: monthly rebalance
- **Source**: Erb & Harvey (2006), "The strategic and tactical value of commodity futures," FAJ. https://doi.org/10.2469/faj.v62.n2.4084; Gorton/Rouwenhorst (2006).
- **Reported edge**: ~10% annualized spread between backwardated and contangoed commodity baskets
- **Backtest-feasibility with free data (1-5)**: 3 (futures curves harder; some via Quandl/CFTC continuous contracts)
- **Originality (1-10)**: 4
- **Decay risk**: Roll yield compressed in 2010s as long-only commodity index investing grew; partial recovery 2021-2022.

### Glassdoor Employee Satisfaction
- **What it is**: Firms with high Glassdoor employee ratings earn ~2.3-3.8% annual abnormal returns; intangible human-capital signal.
- **Asset class**: equities
- **Horizon**: months to year
- **Source**: Green, Huang, Wen & Zhou (2019), "Crowdsourced employer reviews and stock returns," JFE. https://doi.org/10.1016/j.jfineco.2019.03.012 (also Edmans 2011 JFE "Does the stock market fully value intangibles? Employee satisfaction and equity prices" using 100 Best Companies)
- **Reported edge**: 0.31% monthly abnormal (4-factor) on long-short Glassdoor rating-change portfolio
- **Backtest-feasibility with free data (1-5)**: 2 (Glassdoor data not free at scale; scraping faces ToS issues)
- **Originality (1-10)**: 8
- **Decay risk**: Recent, limited OOS — capacity likely small but signal seems robust where reproduced.

### EDGAR Search Traffic
- **What it is**: Abnormal SEC EDGAR downloads of a firm's filings by institutional IPs predicts upcoming returns and earnings news.
- **Asset class**: equities
- **Horizon**: days to weeks
- **Source**: Drake, Roulstone & Thornock (2015), "The determinants and consequences of information acquisition via EDGAR," Contemporary Accounting Research; Lee, Ma & Wang (2015) "Search-based peer firms." Also Ryans (2017 working paper) "Using the EDGAR log file dataset." https://www.sec.gov/dera/data/edgar-log-file-data-set
- **Reported edge**: Bid-ask spreads narrow ~5%, returns drift ~0.5% over 3 days after abnormal institutional download spikes (Drake et al.)
- **Backtest-feasibility with free data (1-5)**: 4 (EDGAR log files free but huge — TB-scale)
- **Originality (1-10)**: 9
- **Decay risk**: SEC anonymized post-2017, complicating identification; signal hardened.

### Seasonality (SEAS) — Same-Calendar-Month Momentum
- **What it is**: Stocks that historically outperform in calendar month M tend to outperform in month M of future years too — return seasonalities persist.
- **Asset class**: equities (and extended to other assets)
- **Horizon**: monthly, with annual recurrence
- **Source**: Heston & Sadka (2008), "Seasonality in the cross-section of stock returns," JFE. https://doi.org/10.1016/j.jfineco.2007.09.005; Keloharju/Linnainmaa/Nyberg (2016 JF) "Return seasonalities."
- **Reported edge**: 0.7% per month long-short for sorts on same-month historical performance
- **Backtest-feasibility with free data (1-5)**: 5
- **Originality (1-10)**: 7
- **Decay risk**: Keloharju et al. (2016) document persistence across countries and asset classes; held up OOS.

### Index Inclusion / Addition Effect
- **What it is**: Stocks added to S&P 500 experience permanent price bump on announcement; classic demand-curve-slopes-down evidence.
- **Asset class**: equities
- **Horizon**: days to weeks around announcement
- **Source**: Shleifer (1986), "Do demand curves for stocks slope down?" Journal of Finance. https://doi.org/10.1111/j.1540-6261.1986.tb04518.x; updated Chen/Noronha/Singal (2004) JF; Greenwood/Sammon (2024 working paper, "The disappearing index effect") shows recent decline.
- **Reported edge**: ~3.5% addition CAR in 1990s; declined to near-zero by 2010s per Greenwood/Sammon 2024
- **Backtest-feasibility with free data (1-5)**: 4 (S&P announcement dates + prices)
- **Originality (1-10)**: 5
- **Decay risk**: HIGH — Greenwood/Sammon NBER w31971 (2024) "The disappearing index effect." https://www.nber.org/papers/w31971 documents near-complete decay post-2010.

### Trend-Following (Century of Evidence)
- **What it is**: Same-instrument trend on 1/3/12-month horizons applied across 67 markets globally — robust through 137 years of OOS.
- **Asset class**: equities indices, bonds, FX, commodities
- **Horizon**: 1-12 months
- **Source**: Hurst, Ooi & Pedersen (2017), "A century of evidence on trend-following investing," Journal of Portfolio Management. https://doi.org/10.3905/jpm.2017.44.1.015; AQR working paper version: https://www.aqr.com/Insights/Research/Working-Paper/A-Century-of-Evidence-on-Trend-Following-Investing
- **Reported edge**: Sharpe ~1.0 with consistent positive returns in every decade 1880-2016
- **Backtest-feasibility with free data (1-5)**: 4 (futures hard; ETF basket workable)
- **Originality (1-10)**: 3
- **Decay risk**: 2010s were weakest decade; 2022 strong revival. Capacity-aware but structurally intact.

### Crash Risk / Negative Coskewness Premium
- **What it is**: Stocks with high negative coskewness (large losses when market falls) earn higher returns — investors demand premium for crash exposure.
- **Asset class**: equities
- **Horizon**: months
- **Source**: Harvey & Siddique (2000), "Conditional skewness in asset pricing tests," Journal of Finance. https://doi.org/10.1111/0022-1082.00247; Chang/Christoffersen/Jacobs (2013) on option-implied skew.
- **Reported edge**: 3.6% annual premium for top-bottom coskewness deciles
- **Backtest-feasibility with free data (1-5)**: 4 (daily returns; option-implied versions need OptionMetrics)
- **Originality (1-10)**: 7
- **Decay risk**: Mixed; Conrad/Dittmar/Ghysels (2013) document option-implied skew as a robust predictor.
