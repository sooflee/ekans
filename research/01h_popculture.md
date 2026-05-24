# Phase 1H — Behavioral / Pop-Culture / Weird-Tail Signals
# Returned by research agent. Spread from "real edge candidates" to "pure curiosities."

### 1. Lipstick Index
- **What it is**: Leonard Lauder claim (~2001) that lipstick sales rise during recessions.
- **Asset class**: cosmetics (EL, ULTA, COTY); macro
- **Horizon**: 6-18 months
- **Entry/exit rule**: YoY lipstick unit sales (Circana/NPD) accelerate >10% while discretionary spending decelerates → overweight defensive consumer + short discretionary luxury.
- **Source**: NYT 2008. Lauder later walked it back (broke in 2008).
- **Reported edge**: Anecdotal/dead.
- **Originality**: 3
- **Backtest-feasibility**: 2
- **Notes**: Curiosity-only.

### 2. Men's Underwear Index
- **What it is**: Greenspan claim — men's underwear is so stable that dips signal genuine distress.
- **Asset class**: discretionary; macro recession signal
- **Horizon**: 6-12 months leading
- **Entry/exit rule**: Circana unit sales decline YoY >3% → reduce equity beta. Reverse on first positive YoY print.
- **Source**: NPR Planet Money 2008.
- **Originality**: 5
- **Backtest-feasibility**: 2
- **Notes**: N~2 recessions.

### 3. Cardboard Box / Containerboard Index
- **What it is**: Containerboard production as physical-goods movement proxy.
- **Asset class**: industrials, transports (XLI, IYT), packaging (PKG, IP, WRK)
- **Horizon**: 1-6 months leading
- **Entry/exit rule**: Fibre Box Association quarterly shipments negative YoY for 2 consecutive quarters → underweight industrials.
- **Source**: FBA reports. Bloomberg 2023.
- **Reported edge**: Correlation ~0.6+ with ISM PMI; led 2008 and 2020 by 1-2 quarters.
- **Originality**: 6
- **Backtest-feasibility**: 4
- **Notes**: One of the more legitimate signals.

### 4. Magazine Cover Contrarian
- **What it is**: Mainstream non-financial magazine extreme directional cover → fade.
- **Asset class**: any
- **Horizon**: 3-24 months
- **Entry/exit rule**: Non-specialist news magazine cover with extreme sentiment → opposite position 30 days after, hold 12 months.
- **Source**: Marks & Donnelly (Citi); Paul Macrae Montgomery research.
- **Reported edge**: Citi: ~70% hit rate on currency covers; ~14% average reversal over 12 months (n=44).
- **Originality**: 4
- **Backtest-feasibility**: 3

### 5. Sports Illustrated Swimsuit Cover Country Index
- **What it is**: Country of origin of SI cover model predicts country equity outperformance over following year.
- **Asset class**: country ETFs
- **Horizon**: 12 months
- **Entry/exit rule**: Long cover model's MSCI country ETF vs short MSCI World for 12 months.
- **Source**: MarketWatch 2015.
- **Reported edge**: ~70% hit rate (n tiny).
- **Originality**: 8
- **Backtest-feasibility**: 4
- **Notes**: Almost certainly p-hacked.

### 6. Skyscraper Index
- **What it is**: Record-breaking skyscraper completions cluster at market tops.
- **Asset class**: host-country equities
- **Horizon**: 12-36 months
- **Source**: Lawrence (Dresdner Kleinwort, 1999); Barclays update 2012; CTBUH database (free).
- **Reported edge**: ~80% hit rate per Barclays (n~13 over century).
- **Originality**: 7
- **Backtest-feasibility**: 4
- **Notes**: N hopeless. China boom since 2010 weakened signal.

### 7. Super Bowl Indicator
- **What it is**: NFC win → bull; AFC → bear.
- **Asset class**: SPX
- **Horizon**: 1 calendar year
- **Entry/exit rule**: NFC wins → long SPY Feb-Dec.
- **Source**: Stovall 1978; Krueger-Kennedy JF 1990.
- **Reported edge**: Original ~80% (1967-89); near-random since 1990. Confounded by 70% base rate.
- **Originality**: 2
- **Backtest-feasibility**: 5
- **Notes**: Textbook data mining. Include as null test.

### 8. Mercury Retrograde
- **What it is**: 3-4x yearly retrograde periods correlate with market choppiness.
- **Asset class**: equities (XLC, XLK)
- **Horizon**: 3-week windows
- **Entry/exit rule**: Reduce gross during retrograde windows OR long VIX calls into retrograde start.
- **Source**: Prechter socionomics; Bill Meridian "Planetary Stock Trading."
- **Reported edge**: No rigorous study finds significance.
- **Originality**: 9
- **Backtest-feasibility**: 5
- **Notes**: Zero mechanism. Pure curiosity.

### 9. GLP-1 / Ozempic Prescription Pair Trade
- **What it is**: Rising GLP-1 prescriptions reduce caloric intake 20-30%; affects snack food, soda, alcohol, fast food.
- **Asset class**: long NVO/LLY, short PEP/MDLZ/KHC/MCD/KO/BUD
- **Horizon**: 12-36 months
- **Entry/exit rule**: Monthly GLP-1 Rx count (IQVIA, Symphony) — 3mo growth >25% YoY → increase short processed-food basket.
- **Source**: WMT CEO Oct 2023; Morgan Stanley Kaufman; GS GLP-1 basket.
- **Reported edge**: GS GLP-1 short basket -8% late 2023 vs SPX +5%; NVO+LLY +60% in 2023. Mean-reverted 2024.
- **Originality**: 7
- **Backtest-feasibility**: 3
- **Notes**: Real mechanism. Crowded by 2024.

### 10. WSB / Retail Mania Heat Index
- **What it is**: WSB DAU + ticker mentions + TikTok #stocktok views as contrarian top indicator.
- **Asset class**: single names + broad
- **Horizon**: 1-12 weeks contrarian
- **Entry/exit rule**: Daily WSB mention count > 3σ above 90-day baseline AND price up >50% MTD → fade via puts within 2 weeks.
- **Source**: Bradley/Hanousek/Jame/Xiao SSRN 2021; ApeWisdom.io free.
- **Reported edge**: +5d return after mention surge but -20d+ returns. ~3-5% mean reversion on top names.
- **Originality**: 6
- **Backtest-feasibility**: 4

### 11. SPAC Issuance Froth Gauge
- **What it is**: Monthly SPAC IPO count as market-cycle peak indicator.
- **Asset class**: ARKK-style growth, IPO, IWO
- **Horizon**: 3-12 months
- **Entry/exit rule**: 3mo SPAC IPO count > 75th percentile of 5-year window → reduce high-multiple growth.
- **Source**: SPACInsider; Klausner/Ohlrogge/Ruan Yale 2022.
- **Reported edge**: Post-merger SPACs -65% vs SPY in 12 months post-de-SPAC (2019-2020 cohort).
- **Originality**: 5
- **Backtest-feasibility**: 4

### 12. Memecoin Launches Per Week
- **What it is**: Weekly count of new memecoins on Solana (pump.fun) + ETH as euphoria gauge.
- **Asset class**: BTC, ETH, broad crypto
- **Horizon**: 2-12 weeks
- **Entry/exit rule**: pump.fun daily launches > 40k for 7+ days → reduce crypto exposure 50%.
- **Source**: Dune Analytics, pump.fun trackers, The Block, CoinDesk.
- **Reported edge**: Memecoin breadth peaked late March 2024, BTC topped same week before -25%.
- **Originality**: 8
- **Backtest-feasibility**: 4
- **Notes**: One-cycle sample. Plausible mechanism.

### 13. Google Trends "How to Buy Stocks" Top Indicator
- **What it is**: Retail-entry query spikes near tops.
- **Asset class**: SPY, QQQ, BTC
- **Horizon**: 1-6 months contrarian
- **Entry/exit rule**: Google Trends "how to buy stocks" > 90 in US → reduce equity gross 25%.
- **Source**: Da/Engelberg/Gao JF 2011 ("In Search of Attention"); SVI literature.
- **Reported edge**: Da et al.: high SVI predicts +2% in 2 weeks, -5% over 12 months on Russell 3000. "How to buy bitcoin" peaked Dec 2017 and Nov 2021.
- **Originality**: 5
- **Backtest-feasibility**: 5
- **Notes**: Trends data is rebased; use relative z-scores.

### 14. Box Office Recession Proxy
- **What it is**: Movie attendance / consumer escapism mood gauge.
- **Asset class**: cinema (CNK, AMC, IMAX), XLY
- **Horizon**: 3-12 months
- **Entry/exit rule**: YoY US box office negative 2 consecutive months + falling consumer confidence → short XLY / long XLP.
- **Source**: Pautz 2002; Box Office Mojo (free).
- **Reported edge**: Inconsistent — 2008 +10% (escapism); 2020 destroyed by COVID; 2023 muddled by streaming.
- **Originality**: 4
- **Backtest-feasibility**: 5
- **Notes**: Structural break post-streaming.

### 15. Champagne / Cognac Shipments
- **What it is**: Global champagne/cognac shipments are cyclical with luxury demand.
- **Asset class**: European luxury (LVMUY, MC.PA, RMS.PA, CFR.SW)
- **Horizon**: 3-9 months
- **Entry/exit rule**: BNIC cognac shipment YoY < -10% → underweight luxury. Re-enter on positive YoY.
- **Source**: Comité Champagne, BNIC (free monthly); Bernstein Solca; Reuters July 2024 (cognac to China -22%).
- **Reported edge**: BNIC cognac led LVMH wines-and-spirits revenue by ~1 quarter through 2023-24 downturn.
- **Originality**: 7
- **Backtest-feasibility**: 4
- **Notes**: Genuinely useful sector signal.

### 16. First-Class / Premium Cabin Bookings
- **What it is**: Corporate first-class collapses early in recessions.
- **Asset class**: airlines (DAL, UAL, AAL), hotels
- **Horizon**: 3-9 months leading
- **Entry/exit rule**: Premium-cabin YoY decline >15% for 2 months while economy positive → short XLY airlines basket.
- **Source**: ARC monthly reports; KAYAK; Skift.
- **Reported edge**: Premium-cabin yields fell -30% YoY in late 2008 ahead of broad airline cap-ex cuts.
- **Originality**: 6
- **Backtest-feasibility**: 3

### 17. Spotify Top-50 Mood / Valence Index
- **What it is**: Average "valence" of weekly Spotify Top 50 Global as consumer-mood proxy.
- **Asset class**: XLY, broad sentiment
- **Horizon**: 1-6 months
- **Entry/exit rule**: 8-week MA of Top 50 valence < 0.35 → reduce XLY beta. > 0.55 → increase.
- **Source**: Spotify Web API audio-features (free); Glasgow Univ. chart valence study.
- **Reported edge**: Glasgow: r~0.25 with contemporaneous consumer confidence; not predictive at longer horizons.
- **Originality**: 8
- **Backtest-feasibility**: 5
- **Notes**: Highly underexploited. Coding project potential.

### 18. Olympic / World Cup Host-Country Drift
- **What it is**: Pre-event infrastructure outperformance, post-event hangover.
- **Asset class**: country ETFs
- **Horizon**: 24mo pre long; 12mo post short
- **Source**: Veraros/Kasimati/Dawson 2004; Maennig/Zimbalist.
- **Reported edge**: Sydney 2000, Athens 2004 pre-drift; Beijing 2008, Rio 2016 negative throughout.
- **Originality**: 6
- **Backtest-feasibility**: 4
- **Notes**: N too small + macro-confounded.

### 19. AI/Hype-Term News Density
- **What it is**: Count of "AI," "GenAI," "LLM" mentions in financial news + earnings calls as saturation indicator.
- **Asset class**: theme baskets (BOTZ, AIQ, NVDA, PLTR)
- **Horizon**: 3-12 months
- **Entry/exit rule**: Earnings-call "AI" mentions > 40% of S&P 500 in a quarter AND theme up >30% YTD → reduce theme exposure 50%.
- **Source**: FactSet quarterly Earnings Insight.
- **Reported edge**: Theme-rotation alpha is well-documented (Greenwood-Hanson "Hot Issues"; Barberis et al on bubbles); generic ~5-10% over 6 months mean reversion after saturation.
- **Originality**: 6
- **Backtest-feasibility**: 3

### 20. "Quiet Quitting" / "Soft Life" Google Trends
- **What it is**: Workforce disengagement search terms as labor-market proxy.
- **Asset class**: XLY vs XLP; staffing (RHI, MAN)
- **Horizon**: 6-18 months
- **Entry/exit rule**: 3mo MA of "quiet quitting"+"soft life"+"lying flat" Trends index up >50% YoY → underweight discretionary.
- **Source**: Kyla Scanlon "vibecession"; Google Trends free.
- **Reported edge**: "Quiet quitting" peaked Sep 2022 coincident with U Mich sentiment trough.
- **Originality**: 9
- **Backtest-feasibility**: 5
- **Notes**: Speculative but novel and codifiable.

## Agent's honest closing notes
- Real edge candidates: #3 (containerboard), #9 (GLP-1), #10 (WSB), #11 (SPAC), #15 (cognac), #17 (Spotify), #19 (AI density).
- Pure curiosities: #5, #7, #8.
- Sample-size traps: #6, #4, #18.
- Common failure mode: post-hoc selection / survivorship bias.
- Best general data source: Google Trends (free, 20-year history, pytrends API).
