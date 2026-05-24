# Phase 1X — Commodity Equity Event-Driven Plays
# Returned by hunt agent. 12 signals on single-name commodity equities (not futures).

### X-1 Reserve Replacement Ratio < 80% → Short Oil Majors
- **Rule**: Short XOM/CVX/SHEL/BP/TTE next day after 10-K if RRR < 80% on 3-yr rolling. Cover 90d.
- **Data**: SEC EDGAR 10-K text search for "reserve replacement ratio"
- **CAGR**: 12-18%; Originality 9

### X-2 Royalty Streamer P/NAV Premium Compression
- **Rule**: Long FNV+WPM+RGLD basket when 90d P/NAV premium vs NEM/GOLD/AEM compresses to 2-yr low decile.
- **Data**: 10-Q/40-F MD&A NAV; SEDAR+ + Stooq prices
- **CAGR**: 14-22%; Originality 9

### X-3 USGC vs NYH Refiner Crack Gap → Long VLO / Short PBF
- **Rule**: Long VLO + short PBF when 4-week USGC 3-2-1 minus NYH 3-2-1 > $8/bbl; close < $3.
- **Data**: EIA daily wholesale prices
- **CAGR**: 11-15%; Originality 8

### X-4 VALE Jurisdictional Discount Snap-Back
- **Rule**: Long VALE when forward EV/EBITDA discount to BHP+RIO avg > 3.5 turns AND no ANM new licensing penalty for 30 days.
- **Data**: Yahoo EBITDA estimates; gov.br/anm news scrape
- **CAGR**: 13-19%; Originality 9

### X-5 Sprott Physical Uranium Trust NAV Premium Flip → Long CCJ
- **Rule**: Long CCJ when U.UN flips from > -5% NAV discount to NAV-flat for 2 consecutive weekly NAVs.
- **Data**: sprott.com weekly NAV + Stooq U.UN TSX price
- **CAGR**: 18-28%; Originality 9

### X-6 Polysilicon Crash → Solar Module-Maker Margin Lag
- **Rule**: Long FSLR+JKS 30d after monthly China polysilicon spot drops > 25% from 90d high. Hold 90d.
- **Data**: pv-magazine.com module price index + BNEF
- **CAGR**: 14-20%; Originality 8

### X-7 Royalty Streaming Deal Announcement Drift
- **Rule**: Long WPM or FNV the morning of any announced new precious-metal stream/royalty acquisition > $200M. Hold 45 trading days.
- **Data**: SEDAR+ + company press releases
- **CAGR**: 12-16%; Originality 9

### X-8 Coal-Gas Henry Hub Inversion → Long Coal Equities
- **Rule**: Long BTU+ARCH when Henry Hub minus PRB-coal-equivalent ($/MMBtu) negative for 10 consec days. Exit when spread > $0.50/MMBtu.
- **Data**: EIA daily Henry Hub + weekly PRB coal
- **CAGR**: 15-22%; Originality 8

### X-9 European Gas Crush → Long CF Industries
- **Rule**: Long CF when TTF Dutch front-month gas drops > 30% in 60d while Henry Hub flat ±10%. Hold 90d.
- **Data**: ICE TTF / EEX free chart + EIA Henry Hub
- **CAGR**: 12-18%; Originality 9

### X-10 Steel HRC Import Parity Widening → Long NUE/STLD
- **Rule**: Long NUE+STLD when CME HRC front-month minus US Midwest import-parity > $150/ton for 4 consec weeks.
- **Data**: CME HRC settlements + Trade Map + SteelOrbis
- **CAGR**: 13-19%; Originality 8

### X-11 US E&P Capex Discipline Beat
- **Rule**: Long any large-cap US E&P (FANG/EOG/DVN/PXD) morning after 8-K if announced full-year capex cut > 10% with production held flat or raised.
- **Data**: SEC EDGAR 8-K full-text search
- **CAGR**: 14-20%; Originality 9

### X-12 Junior Explorer Basket on Senior RRR Cliff ⭐
- **Rule**: Long basket of TSX-V exploration-stage juniors with NI 43-101 indicated resources > 2 Moz Au-eq or > 1 Mt Cu, opened quarter after any senior (NEM/GOLD/AEM/FCX) reports RRR < 80% in 10-K.
- **Data**: SEC 10-K RRR (X-1 trigger) + SEDAR+ NI 43-101 reports
- **CAGR**: 18-30% on basket; Originality 10
- **Notes**: Pairs senior accounting trigger with junior takeout target screen
