# Phase 1I — Creative Wild Card / Unique Signals
# Returned by research agent. 15 signals, mostly originality 7+.

### I-1 CEO Mansion / Trophy-Home Purchase short signal
- **What**: CEO buys >10k sq ft mansion or extravagant estate (especially financed by stock liquidation) → firm underperforms.
- **Asset**: single-name short
- **Horizon**: 6-36mo
- **Rule**: Short stocks where CEOs purchase homes >10k sq ft AND within 30 days of insider stock sales. Hold 12-24 months.
- **Data**: County recorder deeds (free, decentralized) + SEC Form 4 + Zillow/Redfin
- **Source**: Liu-Yermack "Where are the Shareholders' Mansions?" SSRN 970413
- **Reported edge**: Top decile mansion size underperforms market by ~25% over 3 years
- **Originality**: 9, **BT**: 2

### I-2 Manager humor/laughter on earnings calls
- **What**: Detected laughter or humor in calls predicts positive returns.
- **Asset**: single-name event trades
- **Horizon**: 1-90 days post-call
- **Data**: Call audio (Seeking Alpha, IR sites) + wav2vec/openSMILE laughter detector
- **Source**: Call/Flam/Lee/Sharp SSRN 4372000 (2023)
- **Originality**: 8, **BT**: 3

### I-3 Director-network interlock trades
- **What**: Director at company A buys stock of company B where someone at B is on A's board.
- **Asset**: US equities
- **Horizon**: 3-12mo
- **Data**: SEC DEF14A + Form 4 (both EDGAR free)
- **Source**: JBF 2020 (Director Network paper)
- **Originality**: 8, **BT**: 4

### I-4 USPTO trademark filing intensity L/S
- **What**: Top-tercile trademark filers (scaled by total assets) outperform bottom tercile.
- **Asset**: US equities cross-sectional
- **Horizon**: 12 months
- **Data**: USPTO Trademark Case Files Dataset (free bulk download)
- **Source**: UCLA Anderson Review summary; underlying Management Science paper
- **Reported edge**: ~5.2% annualized hedged return
- **Originality**: 7, **BT**: 5

### I-5 Private-jet tail-number M&A lurker
- **What**: Public-company jets repeatedly visit airports near target/rival HQs before M&A.
- **Asset**: single-name M&A target longs
- **Horizon**: days-weeks
- **Data**: ADS-B Exchange (free historical), OpenSky Network (free), FAA registry
- **Source**: Oxford study (Matthew Smith); Bloomberg 2019
- **Originality**: 9, **BT**: 3

### I-6 Lunar January effect in Chinese A-shares
- **What**: Strong "January effect" in Chinese lunar calendar (not Gregorian); driven by domestic retail.
- **Asset**: Chinese A-shares (vs B-shares)
- **Horizon**: 1 lunar month per year
- **Data**: Lunar calendar (free) + AkShare for A/B-share data
- **Source**: Liang-Liu-Zebedee SSRN 4209010 "One Country, Two Calendars"
- **Originality**: 8, **BT**: 5

### I-7 Geomagnetic storm mood trade
- **What**: High Kp/Dst index periods → US equities underperform; attributed to mood-mediation in retail.
- **Asset**: SPY, especially small caps
- **Horizon**: 1-10 days
- **Rule**: 5-day rolling Kp in top decile → lighten equity beta; mean-revert long after storm subsides.
- **Data**: NOAA SWPC Kp/Dst daily 1932+ (free)
- **Source**: Krivelyova-Robotti (Atlanta Fed); Hindawi 2019 replication
- **Originality**: 9, **BT**: 5

### I-8 Patent-citation surprise residual
- **What**: Citations a firm receives ABOVE statistical expectation predicts 2-year positive returns.
- **Asset**: tech/biotech/industrial equities
- **Horizon**: 12-24 months
- **Data**: USPTO PatentsView (free); Kogan-Papanikolaou-Seru-Stoffman patent value data (free academic)
- **Source**: "Taking the Road Less Traveled" AFA paper; Hirshleifer-Hsu-Li
- **Originality**: 7, **BT**: 4

### I-9 CEO divorce filing as risk-reduction signal
- **What**: In year of CEO divorce, firm risk falls (R&D, leverage, vol).
- **Asset**: single-name equities; vol surfaces
- **Horizon**: 6-24 months
- **Data**: PACER (cheap per query) + state court records (free, decentralized)
- **Source**: Neyland "Love or Money"; Nicolosi-Yore "I Do"
- **Originality**: 9, **BT**: 2

### I-10 Containerboard / Cardboard box recession pulse
- **What**: AF&PA monthly containerboard production as physical-goods leading indicator.
- **Asset**: macro overlay; US small caps, industrials, freight; HY credit
- **Horizon**: 3-12 months
- **Rule**: 6m YoY containerboard production drops > 1σ below trend → reduce small-cap + HY exposure.
- **Data**: AF&PA monthly press releases (free), Fibre Box Association
- **Originality**: 7, **BT**: 4

### I-11 Wikipedia pageview momentum L/S
- **What**: Cross-section of firms by MoM change in Wikipedia views; long top, short bottom.
- **Asset**: US equities
- **Horizon**: 1 month
- **Rule**: Monthly long top-decile views Δ, short bottom-decile, sector-neutral.
- **Data**: Wikimedia Pageviews API (free, hourly) + ticker→article mapping
- **Source**: Pyun "Wikipedia Effect" Econ Letters 2024; Behrendt-Zimmermann SSRN 3220053
- **Originality**: 7, **BT**: 5
- **Notes**: Cleaner than Google Trends (no search-term ambiguity).

### I-12 ENSO (El Niño / La Niña) ag commodity trade
- **What**: ENSO predicts ag commodity variance + price 6-12 months ahead.
- **Asset**: ZS (soy), ZC (corn), ZW (wheat), KC (coffee), SB (sugar); ADM, BG, MOS
- **Horizon**: 6 months to multi-year
- **Rule**: NOAA El Niño declared → long soy/palm oil vol straddles + short Brazil/Indonesia ag exporters. Reverse for La Niña on corn/wheat.
- **Data**: NOAA ONI / Nino 3.4 SST (free monthly)
- **Originality**: 7, **BT**: 5

### I-13 ClinicalTrials.gov enrollment velocity acceleration
- **What**: 2nd derivative of trial enrollment leaks info about readout probability + timing.
- **Asset**: small/mid biotech
- **Horizon**: 3-18 months
- **Data**: ClinicalTrials.gov API (free) + AACT database
- **Originality**: 8, **BT**: 4

### I-14 Lunar New Year pre-holiday drift in Asian markets
- **What**: Across HK, TW, JP, KR, SG, MY: positive abnormal returns in days BEFORE Lunar New Year; partial reversal after.
- **Asset**: EWH, EWS, EWT; index futures
- **Horizon**: 5-15 trading days around LNY
- **Rule**: Long HK + Singapore + Vietnam ETFs for 10 sessions ending on LNY-eve; flat or hedge post-holiday.
- **Source**: Yuan-Gupta (J Int Financial Markets)
- **Originality**: 7, **BT**: 5

### I-15 Conference-call vocal dissonance → misreporting short
- **What**: Pitch tremor / jitter changes in CEO answers predict restatements.
- **Asset**: short basket
- **Horizon**: 6-24 months
- **Data**: Call audio + openSMILE/Praat (free)
- **Source**: Hobson-Mayew-Venkatachalam SSRN 1531871; Mayew-Venkatachalam "Power of Voice"
- **Originality**: 8, **BT**: 3
