# Master Signal Catalog (Phase 2 consolidation)

Consolidating Phase 1A-E outputs (and to be extended with 1F-I as those agents complete).
Duplicates collapsed to a single canonical entry.
Each row: ID | Name | Category | Asset | Horizon | Orig (1-10) | BT-feasibility (1-5) | Status
Status = `todo` (not yet attempted), `bt-done` (backtest complete), `bt-fail` (data not available or signal not implementable with free data), `cited-only` (well-documented in literature; not re-tested).

## Categories
- A. Calendar / seasonal (time-of-year, time-of-month, day-of-week)
- B. Volatility / options-implied / dealer-positioning
- C. Cross-asset macro / rotation (ratios, regimes)
- D. Equity factor / cross-sectional (value, momentum, quality, etc.)
- E. Event-driven / corporate actions (insider, 13F, 13D, 8-K, spin-off, lockup)
- F. Sentiment / behavioral / breadth
- G. Alt-data (satellite, web, employment, app store)
- H. Crypto on-chain / structural
- I. Pattern / technical (retail TA: ICT, Wyckoff, harmonic, Elliott)
- J. Behavioral / pop-culture / curiosity
- K. Policy / geopolitical / event calendar

---

## Master Table

| ID | Signal | Cat | Asset | Horiz | Orig | BT | Notes / Status |
|---|---|---|---|---|---|---|---|
| A01 | Halloween (Sell-in-May) | A | equities | 6mo | 3 | 5 | Bouman-Jacobsen 2002, Zhang-Jacobsen 2021. `todo` |
| A02 | Turn-of-Month | A | equities | 4 days/mo | 5 | 5 | Ariel 1987; McConnell-Xu 2008. `todo` |
| A03 | Santa Claus Rally | A | equities | 7 days | 2 | 5 | Hirsch (1972). `todo` |
| A04 | First Five Days of January | A | equities | 12mo | 3 | 5 | Hirsch. Sample of 5 days is statistically weak. `todo` |
| A05 | January Effect (small caps) | A | equities (small) | 1mo | 3 | 5 | Classic; tax-loss harvesting. `todo` |
| A06 | FOMC even-week (Cieslak-Morse-Vissing-Jorgensen) | A | equities | biweekly | 7 | 5 | Cieslak et al. JF 2019. `todo` |
| A07 | Pre-FOMC drift | A | equities | 24h | 6 | 5 | Lucca-Moench JF 2015. Drift weakened post-2015. `todo` |
| A08 | Monthly OPEX week | A | equities | 1wk/mo | 6 | 5 | Quantpedia; delta-hedge unwind. `todo` |
| A09 | Sunshine effect (NYC weather) | A | equities | daily | 7 | 4 | Saunders 1993, Hirshleifer-Shumway 2003. `todo` |
| A10 | Lunar phase (full moon negative) | A | equities | 15d half-cycle | 9 | 5 | Yuan-Zheng-Zhu 2006. Low statistical power. `todo` |
| A11 | DST anomaly (Monday after) | A | equities | 1 day 2x/yr | 8 | 5 | Kamstra-Kramer-Levi 2000. Contested. `todo` |
| A12 | Treasury auction cycle | A | bonds | 5d per auction | 8 | 4 | Lou-Yan-Zhang RFS 2013. `todo` |
| A13 | Presidential cycle year 3 | A | equities | 1yr | 4 | 5 | Hirsch; Beyer JPM. `todo` |
| A14 | Halving cycle (BTC) | A | crypto | multi-year | 2 | 5 | N=3, weak. `todo` |
| A15 | Halloween + Turn-of-Month combo | A | equities | various | 5 | 5 | Stack signals; check additivity. `todo` |
| A16 | Seasonality (Heston-Sadka same-month) | A | equities | monthly | 7 | 5 | Heston-Sadka JFE 2008; Keloharju et al 2016. `todo` |
| B01 | VIX term structure (VIX/VXV) | B | equities | days-wks | 4 | 5 | VIX and More; signal filter. `todo` |
| B02 | VIX > 30 contrarian buy | B | equities | 1-3mo | 3 | 5 | Whaley. `todo` |
| B03 | VVIX/VIX ratio | B | equities/vol | 1-4wk | 5 | 5 | `todo` |
| B04 | SKEW index extremes | B | equities | 1-3mo | 4 | 5 | `todo` |
| B05 | Put/call ratio extremes | B | equities | 1-4wk | 2 | 5 | Sentimentrader. 2020-21 distortion. `todo` |
| B06 | Variance Risk Premium (VRP) | B | equities | 1mo | 4 | 5 | Carr-Wu RFS 2009; Bollerslev et al 2009. `todo` |
| B07 | Dealer gamma (GEX) regime filter | B | equities | days-wks | 8 | 1 | Paid data. `bt-fail` (data) |
| B08 | JPM Collar quarterly pin | B | equities/vol | 1-2wk/qtr | 7 | 2 | Need historical strikes. `bt-fail` (data) |
| C01 | Lumber/Gold (Gayed) | C | equities/bonds | 13wk | 4 | 4 | LBR contract redesigned 2022. `todo` |
| C02 | Utilities/SPY ratio (Gayed) | C | equities | 1mo | 3 | 5 | `todo` |
| C03 | Copper/Gold → 10Y yield | C | bonds | 1-6mo | 5 | 5 | Hua-Wang 2023. `todo` |
| C04 | Gold/Silver ratio mean reversion | C | commodities | 6-24mo | 4 | 5 | `todo` |
| C05 | Faber GTAA 10-mo MA | C | multi-asset | monthly | 2 | 5 | Faber 2007. `todo` |
| C06 | Dual Momentum (Antonacci GEM) | C | equities/bonds | monthly | 3 | 5 | Antonacci 2013. `todo` |
| C07 | Accelerating Dual Momentum | C | equities/bonds | monthly | 5 | 5 | EngineeredPortfolio 2018. 2022 brutal. `todo` |
| C08 | Yield curve inversion (10Y-2Y) | C | equities/bonds | months-yrs | 3 | 5 | `todo` |
| C09 | 3m10y inversion + un-inversion | C | equities/bonds | 6-18mo | 4 | 5 | Estrella-Mishkin 1998. `todo` |
| C10 | HY OAS credit spread regime | C | equities/credit | 1-12mo | 3 | 5 | Verdad Capital. `todo` |
| C11 | MOVE/VIX ratio | C | cross-asset | 2-8wk | 6 | 4 | `todo` |
| C12 | ACM 10Y term premium | C | bonds/equities | months-yrs | 7 | 5 | NY Fed. `todo` |
| C13 | TIPS breakeven (commodities trigger) | C | commodities | 1-3mo | 5 | 5 | `todo` |
| C14 | Time-Series Momentum (TSMOM) | C | multi-asset | 1-12mo | 3 | 4 | Moskowitz-Ooi-Pedersen 2012; AQR Century of Evidence. `todo` |
| C15 | Trend-following (Hurst-Ooi-Pedersen 100y) | C | multi-asset | 1-12mo | 3 | 4 | AQR 2017. `todo` |
| C16 | Currency carry (G10) | C | FX | months | 3 | 4 | Lustig-Verdelhan 2007. Decayed post-2008. `todo` |
| C17 | Commodity backwardation/roll | C | commodities | monthly | 4 | 3 | Erb-Harvey FAJ 2006. `todo` |
| C18 | Big Mac PPP for FX | C | FX | 2-5yr | 3 | 5 | Cumby NBER 1996. `todo` |
| D01 | Value (HML, book-to-market) | D | equities | months-yrs | 1 | 5 | Fama-French. Decayed 2010s. `todo` |
| D02 | Cross-sectional momentum (UMD) | D | equities | 1-12mo | 1 | 5 | Jegadeesh-Titman 1993. `todo` |
| D03 | 52-week high momentum | D | equities | months | 5 | 5 | George-Hwang JF 2004. `todo` |
| D04 | Industry momentum | D | equities | 1-12mo | 4 | 5 | Moskowitz-Grinblatt 1999. `todo` |
| D05 | MAX / lottery effect | D | equities | 1mo | 5 | 5 | Bali-Cakici-Whitelaw JFE 2011. `todo` |
| D06 | Idiosyncratic vol puzzle | D | equities | 1mo | 4 | 4 | Ang-Hodrick-Xing-Zhang JF 2006. `todo` |
| D07 | Quality (gross profitability) | D | equities | months-yrs | 2 | 2 | Novy-Marx 2013. Needs Compustat. `cited-only` |
| D08 | Quality Minus Junk (QMJ) | D | equities | months | 3 | 2 | Asness-Frazzini-Pedersen. AQR posts factor returns. `cited-only` |
| D09 | Betting Against Beta (BAB) | D | equities | months | 2 | 4 | Frazzini-Pedersen 2014. `todo` |
| D10 | Post-Earnings Announcement Drift (PEAD) | D | equities | 1-3mo | 3 | 3 | Bernard-Thomas 1989. SUE needs estimates. `todo` (rough version) |
| D11 | Short interest | D | equities | 1mo | 4 | 3 | FINRA bi-monthly. `todo` |
| D12 | Analyst dispersion | D | equities | months | 5 | 2 | Needs IBES. `cited-only` |
| D13 | Accruals anomaly | D | equities | 1yr | 4 | 2 | Sloan 1996. Needs Compustat. `cited-only` |
| D14 | Net stock issuance | D | equities | 1-3yr | 4 | 4 | `todo` |
| D15 | Asset growth | D | equities | 1yr | 3 | 3 | Cooper-Gulen-Schill 2008. `todo` |
| D16 | R&D intensity | D | equities | years | 5 | 3 | Chan-Lakonishok-Sougiannis 2001. `cited-only` |
| D17 | Crash risk (negative coskewness) | D | equities | months | 7 | 4 | Harvey-Siddique JF 2000. `todo` |
| D18 | Lazy Prices (10-K text similarity) | D | equities | 12mo | 7 | 4 | Cohen-Malloy-Nguyen JF 2020. `cited-only` (text munging too heavy for this pass) |
| D19 | Loughran-McDonald 10-K sentiment | D | equities | wks-mo | 5 | 5 | `cited-only` (NLP heavy) |
| E01 | Form 4 insider cluster buying | E | equities | 6-12mo | 3 | 5 | EDGAR free. `todo` |
| E02 | Opportunistic vs routine insiders | E | equities | 6mo | 5 | 4 | Cohen-Malloy-Pomorski JF 2012. `todo` |
| E03 | Form 144 pre-sale drift | E | equities | 1-3mo | 6 | 3 | Only since Apr 2023. `cited-only` |
| E04 | 10b5-1 plan adoption (red flag) | E | equities | 3-12mo | 7 | 3 | NLP on 10-Qs. `cited-only` |
| E05 | Friday-after-close 8-K drift | E | equities | 1-2mo | 6 | 5 | DellaVigna-Pollet JF 2009. `todo` |
| E06 | Item 4.02 restatement short | E | equities | 6-12mo | 4 | 5 | Hennes-Leone-Miller. `todo` |
| E07 | Schedule 13D activist drift | E | equities | 3-18mo | 3 | 5 | Brav et al JF 2008. `todo` |
| E08 | 13F famous-fund mirror | E | equities | 3-12mo | 2 | 5 | 45-day lag is the killer. `todo` |
| E09 | IPO lockup expiry short | E | equities | 5d/expiry | 3 | 4 | Field-Hanka 2001. Edge shrunk to ~50bp. `todo` |
| E10 | Spin-off drift (Greenblatt) | E | equities | 12-36mo | 4 | 4 | Cusatis-Miles-Woolridge JFE 1993. `todo` |
| E11 | Pelosi disclosed-trade mirror | E | equities | 3-12mo | 2 | 4 | Unusual Whales tracker; NANC ETF. `todo` |
| E12 | House Fin Services Committee subset | E | equities | 1-6mo | 4 | 4 | `todo` |
| E13 | Index inclusion drift (S&P 500 add) | E | equities | days-wks | 5 | 4 | Greenwood-Sammon NBER 2024: largely decayed. `todo` |
| F01 | AAII bull-bear extremes | F | equities | 4-12wk | 2 | 5 | `todo` |
| F02 | NAAIM exposure extremes | F | equities | 4-12wk | 4 | 5 | `todo` |
| F03 | Margin debt YoY change | F | equities | 3-12mo | 5 | 5 | FINRA monthly. `todo` |
| F04 | Hindenburg Omen | F | equities | 1-3mo | 6 | 4 | NYSE breadth. `todo` |
| F05 | Coppock Curve (long-term bottoms) | F | equities | multi-yr | 6 | 5 | N~8/century. `todo` (bottom-only) |
| F06 | DeMark TD Sequential 9-13 | F | any | days-wks | 5 | 5 | Reversal trigger. `todo` |
| F07 | Bullish Percent Index (BPI) | F | equities | 1-6mo | 5 | 4 | Stockcharts $BPSPX. `todo` |
| F08 | RSI(2) mean reversion (Connors) | F | equities | days | 5 | 5 | Connors 2009. `todo` |
| F09 | Golden Cross / Death Cross | F | equities | months-yrs | 1 | 5 | `todo` |
| F10 | MACD bullish divergence | F | equities | days-wks | 3 | 5 | `todo` |
| F11 | Bollinger Band squeeze breakout | F | equities | days-wks | 3 | 5 | `todo` |
| F12 | Magazine cover contrarian | F | any | 6-24mo | 3 | 4 | Arnold-Earl-North FAJ 2007. Manual coding. `todo` (small basket only) |
| F13 | "Buy the dip" anti-signal | F | equities | years | 6 | 5 | AQR (2025) — null hypothesis test. `todo` |
| G01 | Google Trends "recession" spike | G | equities | 1-6mo | 6 | 4 | Preis-Moat-Stanley Nature 2013. Rebasing tricky. `todo` |
| G02 | Wikipedia pageview anomalies | G | equities | 1-4wk | 7-8 | 4 | Moat et al. Nature 2013. Wikimedia API free. `todo` |
| G03 | WSB / Reddit mention velocity | G | small/mid-cap | 1-15d | 5 | 4 | Apewisdom free. Best in $1-10B mcap. `todo` (limited history) |
| G04 | Glassdoor employee reviews | G | equities | 6-12mo | 6-8 | 2 | Green-Huang-Wen-Zhou JFE 2019. Glassdoor TOS blocks. `cited-only` |
| G05 | LinkedIn hiring velocity | G | equities | 3-9mo | 5 | 1 | Paid. `bt-fail` (data) |
| G06 | App Store rank velocity | G | tech | 45d | 6 | 2 | Paid. `bt-fail` (data) |
| G07 | WARN Act layoff notices | G | equities | 1-6mo | 7 | 4 | State portals free. Sign ambiguous. `todo` |
| G08 | Walmart/Target satellite parking | G | retail | qtr | 5 | 1 | Paid. `bt-fail` (data) |
| G09 | Crude oil tank shadows | G | oil | 1-7d | 7 | 2 | Paid (Sentinel-1 free but heavy processing). `bt-fail` (effort) |
| G10 | China container-port throughput | G | China/shipping | 1-3mo | 6 | 3 | IMF PortWatch free. `todo` |
| G11 | NDVI crop yield front-run | G | grains | 1-3mo | 7 | 3 | MODIS free; heavy GeoTIFF processing. `bt-fail` (effort) |
| G12 | HDD/CDD vs nat gas | G | nat gas | 1-2wk | 4 | 5 | NOAA CPC free, EIA free. `todo` |
| G13 | Corporate jet M&A | G | equities | 1-90d | 8 | 2 | ADS-B free but registration matching laborious. FAA Privacy program degrading signal. `bt-fail` (data + signal decay) |
| G14 | Central-bank Jackson Hole jets | G | rates/equities | 1-30d | 9 | 2 | N~1/yr. `bt-fail` (data, sample) |
| G15 | EDGAR search traffic | G | equities | days-wks | 9 | 4 | EDGAR log files free but TB-scale. SEC anonymized post-2017. `cited-only` |
| H01 | Perp funding rate extremes (BTC/ETH) | H | crypto | 1-30d | 4 | 5 | Coinglass / Binance API free. `todo` |
| H02 | Stablecoin Supply Ratio (SSR) | H | crypto | wks-mo | 5 | 4 | Glassnode/DefiLlama. `todo` |
| H03 | Coinbase Premium Index | H | crypto | 1-30d | 5 | 4 | CryptoQuant/Coinglass. `todo` |
| H04 | Exchange netflow | H | crypto | 3-30d | 4 | 3 | CryptoQuant free chart only. `todo` (best-effort) |
| H05 | Hash Ribbons (Edwards) | H | crypto | 3-12mo | 6 | 5 | mempool.space free. `todo` |
| H06 | MVRV Z-score | H | crypto | 6-24mo | 3 | 5 | BitcoinMagazinePro free. `todo` |
| H07 | Puell Multiple | H | crypto | 6-24mo | 4 | 5 | `todo` |
| H08 | NUPL | H | crypto | 6-24mo | 4 | 4 | `todo` |
| H09 | Spot BTC ETF net flows | H | crypto | 1-30d | 6 | 5 | Farside free. Short history. `todo` |
| H10 | Coin Days Destroyed | H | crypto | 1-12mo | 6 | 4 | `todo` |
| H11 | BTC-NDX / BTC-Gold correlation regime | H | crypto/equity | months | 5 | 5 | Yahoo + FRED. `todo` |
| H12 | COT commercial-hedger extreme | H | commodities/FX | 4-12wk | 5 | 5 | CFTC free. `todo` |
| I01 | ICT Fair Value Gap fill | I | any | days-wks | 3 | 4 | `todo` (single-name, ES proxy) |
| I02 | ICT Order Block retest | I | any | days-wks | 3 | 4 | `todo` |
| I03 | Liquidity sweep reversal | I | FX/crypto/futures | days-wks | 4 | 5 | Failed-breakout rebrand. `todo` |
| I04 | Premium/discount bias (ICT) | I | any | wks | 4 | 5 | Mean reversion within range. `todo` |
| I05 | Wyckoff Spring (daily) | I | equities/crypto | wks-mo | 4 | 5 | Volume-confirmed sweep. `todo` |
| I06 | Sam Seiden supply/demand zone | I | any | days-wks | 5 | 4 | Parameter-sensitive. `todo` |
| I07 | 0.618 Fib pullback | I | any | wks | 2 | 4 | Swing detection. `todo` |
| I08 | Bullish butterfly harmonic | I | equities/FX | days-wks | 6 | 3 | Pattern detection nontrivial. `bt-fail` (effort) |
| I09 | Elliott Wave 3 entry | I | any | wks-mo | 4 | 2 | Subjective labeling. `bt-fail` (subjective) |
| J01 | Skyscraper Indicator | J | global equities | 12-36mo | 6 | 4 | N~6-7/century. `cited-only` |
| J02 | Magazine cover (separate) | J | any | 6-24mo | 3 | 4 | Same as F12. (collapse) |
| J03 | Super Bowl Indicator | J | equities | 12mo | 2 | 5 | Debunked post-1989. `todo` (as null test) |
| K01 | Treasury auction cycle | K | bonds | 5d | 8 | 4 | Same as A12. (collapse) |
| K02 | FOMC blackout drift | K | equities | days | 6 | 5 | `todo` |
| K03 | OPEC meeting pre-drift (oil) | K | oil | 5d | 6 | 5 | Awaiting Phase 1G agent. |

(More K-category signals will be added as Phase 1F, 1G policy/geopolitical agents complete.)

---

## Counts so far
- A (calendar): 16
- B (vol/options): 8
- C (cross-asset macro): 18
- D (equity factor): 19
- E (event-driven): 13
- F (sentiment/breadth/TA-classic): 13
- G (alt-data): 15
- H (crypto): 12
- I (retail pattern TA): 9
- J (curiosity): 3
- K (policy/geopol): 3 (more pending)

**Total so far: 129** unique signals catalogued. Once 1F/G/H/I agents finish we'll add ~50 more, target ~170-180 unique signals. Backtest-feasible-with-free-data subset: ~85.

## Status tally
- `todo`: ~85 signals to attempt backtesting on
- `cited-only`: ~12 (data not free or NLP too heavy for this pass)
- `bt-fail (data)`: ~10 (paid feed required; mark as tried)
- `bt-fail (effort/subjective)`: ~5 (Elliott Wave labeling, satellite GeoTIFFs at scale, harmonic pattern detection at scale)
