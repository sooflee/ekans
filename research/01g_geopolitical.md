# Phase 1G — Geopolitical / War / Sanctions / Defense / Election Signals
# Returned by research agent. 22 signals.

### G-1 Pre-war drift in defense stocks (GPR Threats index)
- **What**: When tension is telegraphed (vs surprise attack), US defense primes accrue +CAR before hostilities. Federman-Pace 2025: ~10pp CAR after Russia invasion vs ~0 after Hamas surprise.
- **Asset**: LMT, RTX, NOC, GD, LHX; ITA/XAR
- **Horizon**: 30-90d
- **Rule**: When Caldara-Iacoviello GPR Threats (GPRT) closes > 1.5σ above 12m mean for 5 consec days AND involves NATO/major-power flashpoint → long equal-weight basket. Exit day 60 or GPRT mean-reverts.
- **Data**: matteoiacoviello.com GPR daily CSV (free), yfinance
- **Originality**: 5, **BT**: 5

### G-2 Israel/Iran flashpoint oil spike fade
- **Asset**: WTI (CL=F), Brent (BZ=F), USO
- **Horizon**: 5-15d
- **Rule**: On day Brent closes up >4% with confirmed Israel-Iran direct strike (GDELT 190-204 between ISR/IRN), short Brent next open. Cover at 50% retrace or day +10. Skip if Hormuz tanker actually struck.
- **Data**: GDELT 2.0 free, yfinance
- **Originality**: 6, **BT**: 4
- **Notes**: Asymmetric tail risk if Hormuz closes.

### G-3 European gas / wheat drift on Russia-tail risk
- **Asset**: ZW=F (wheat), UNG/UNL (gas), WEAT ETF
- **Horizon**: 10-60d
- **Rule**: GPR Russia > 2σ above 6m mean AND troop-mobilization keyword count (E.Europe) > 90th percentile → long equal-weight (ZW=F, UNG). Exit day 45 or GPR Russia mean-reverts 50%.
- **Originality**: 6, **BT**: 3

### G-4 Houthi/Red Sea shipping attack tanker trade
- **Asset**: FRO, DHT, STNG, INSW, GOGL, ZIM; BDI confirming
- **Horizon**: 5-30d
- **Rule**: Attacks per 7d ≥ 3 AND major liner announces Cape reroute → long equal-weight FRO+STNG+INSW. Exit day 20.
- **Data**: UKMTO RSS, ACLED, yfinance
- **Originality**: 7, **BT**: 4

### G-5 North Korea missile test KOSPI/KRW fade
- **Asset**: EWY, ^KS11, KRW=X
- **Horizon**: 3-5d
- **Rule**: On day EWY closes down >1.5% AND DPRK ballistic launch reported → buy EWY at next open, short USDKRW. Hold 4 days. Skip nuclear (ground detonation) tests.
- **Data**: Wikipedia "List of NK missile tests" (free dated table), yfinance
- **Originality**: 8, **BT**: 4

### G-6 OPEC pre-meeting Brent drift
- **Asset**: BZ=F, USO
- **Horizon**: 5-15d
- **Rule**: 10 trading days before scheduled OPEC ministerial: if OECD inventory z < -1σ (tight), long Brent into meeting + 3 days; if z > +1σ (loose), short.
- **Data**: opec.org press release dates, FRED OECD crude stocks, yfinance
- **Originality**: 5, **BT**: 5

### G-7 Saudi off-cycle surprise voluntary cut
- **Asset**: BZ=F, USO
- **Horizon**: 5-10d
- **Rule**: Sunday/weekend Saudi voluntary cut announcement → buy Brent Mon open, exit day 7 or first close below Mon high.
- **Data**: SPA RSS, OPEC site, yfinance
- **Originality**: 6, **BT**: 4

### G-8 OFAC Russia/Iran tanker SDN addition drift
- **Asset**: FRO, STNG, INSW, DHT
- **Horizon**: 5-20d
- **Rule**: OFAC press release adding ≥5 vessels OR ≥3 shipping entities to SDN under EO 14024 (Russia) or Iran-oil EOs → buy equal-weight FRO+STNG+INSW next open. Exit day 15.
- **Data**: ofac.treasury.gov/recent-actions RSS (free), OpenSanctions delta feed (free)
- **Originality**: 8, **BT**: 4

### G-9 Russia Brent-Urals price-cap spread persistence
- **Asset**: Brent (leg); Russian-exposed equities short leg
- **Horizon**: 1-12 months
- **Rule**: When Urals-Brent discount narrows to < $10 (vs post-cap mean ~$15-20), short Russian-exposed proxies until discount re-widens > $15.
- **Originality**: 7, **BT**: 2 (Urals series gappy)

### G-10 NDAA-passage defense sector drift (mid-Dec)
- **Asset**: ITA ETF
- **Horizon**: 10-30d
- **Rule**: 10 trading days before Senate conference vote on NDAA → long ITA. Exit 3 days after presidential signature.
- **Data**: Congress.gov NDAA tracker (free)
- **Originality**: 7, **BT**: 5

### G-11 Continuing-resolution defense underperformance
- **Asset**: ITA vs SPY; or large defense vs small defense
- **Horizon**: CR duration (30-180d)
- **Rule**: From start of any CR > 30d, long large-cap defense (LMT, RTX, NOC, GD) / short small-cap defense (KTOS, AVAV, RKLB). Close at full appropriations passage.
- **Originality**: 8, **BT**: 4

### G-12 DoD daily contract-award single-stock drift
- **Asset**: LHX, KTOS, AVAV, MRCY, BAH, CACI, LDOS
- **Horizon**: 1-5d
- **Rule**: Each evening scrape defense.gov/News/Contracts/. If award > 1% of TTM revenue for mapped public contractor → buy next open. Exit close of day +3.
- **Originality**: 9, **BT**: 4

### G-13 NATO summit European defense run-up
- **Asset**: RHM.DE (Rheinmetall), BAE.L, LDO.MI, HO.PA; EUAD ETF
- **Horizon**: 15-45d
- **Rule**: 15 trading days before scheduled NATO summit communiqué → long equal-weight RHM.DE + BAE.L + LDO.MI. Exit 5d after summit close OR earlier if communiqué disappoints.
- **Originality**: 6, **BT**: 5
- **Notes**: Regime — only works post-2022.

### G-14 Pre-FOMC drift, regime-conditional
- **Asset**: SPY
- **Horizon**: 1-2d
- **Rule**: At close of T-1 before FOMC, long SPY. Sell at close of FOMC day. CONDITIONAL: only trade when 1-month change in 2y Treasury yield is below median (dovish regime).
- **Originality**: 4 (base) / 6 (regime), **BT**: 5

### G-15 US presidential election pre-vote VIX ramp
- **Asset**: SPY straddles; VIXY/SVXY
- **Horizon**: 30-180d
- **Rule**: 4 months before US presidential election (T-120 days), buy 3m ATM SPY straddle, roll monthly until T-5. Close on T+10.
- **Originality**: 4, **BT**: 5
- **Notes**: 8 cycle sample. Roll-cost drag.

### G-16 Year-3 presidential cycle + year-4 small-cap tilt
- **Asset**: SPY / IWM rotation
- **Horizon**: 1-2 years
- **Rule**: Year 3 of presidential term → 100% SPY. Year 4 → switch to IWM. Year 1 → cash for first 9 months. Year 2 → re-enter at mid-term-election trough (~Oct).
- **Originality**: 3, **BT**: 5
- **Notes**: N~22 cycles since 1933.

### G-17 BoJ surprise YCC tweak reaction
- **Asset**: USDJPY (JPY=X), Nikkei (^N225), EWJ
- **Horizon**: 1-5d
- **Rule**: On BoJ meeting with hawkish surprise (10y ceiling raised OR unexpected rate hike) → short USDJPY at next NY open, exit at 5-day close.
- **Originality**: 7, **BT**: 4

### G-18 PBOC RRR cut Hang Seng momentum
- **Asset**: ^HSI, HSCEI, FXI, KWEB
- **Horizon**: 1-10d
- **Rule**: Any PBOC RRR cut → buy FXI next session open. Exit day 10. Boost size if paired with rate cut within 30 days.
- **Originality**: 6, **BT**: 5

### G-19 China NPC two-sessions post-meeting drift
- **Asset**: FXI, MCHI, CSI 300
- **Horizon**: 10-15d
- **Rule**: Short FXI 5 days before NPC convening. Cover at NPC close. Flip to long FXI for 10 days post-NPC close.
- **Originality**: 8, **BT**: 5
- **Notes**: -0.9% during, +1.2% after; post-meeting positive 13/24 years.

### G-20 Section 232/301 tariff announcement sector short
- **Asset**: SOXX (chip), MOO (ag), XLB/XLI (steel)
- **Horizon**: 5-30d
- **Rule**: On USTR press release / White House EO with tariff lists → short affected sector ETF next open. Cover day 15 or on first carve-out.
- **Originality**: 5, **BT**: 4

### G-21 Brexit-style binary referendum vol trade
- **Asset**: local FX, local equity index, straddles
- **Horizon**: 7-30d
- **Rule**: Sovereign-level binary referendum: buy 1m straddle on local-currency vs USD 2 weeks pre-vote; close on vote close.
- **Originality**: 5, **BT**: 3 (option vol not free)

### G-22 EM election surprise FX fade
- **Asset**: USDMXN, USDBRL, USDTRY, USDINR; EWZ, EWW, TUR, INDA
- **Horizon**: 10-30d
- **Rule**: On day USD/EM pair gaps >2% on election surprise (T0 vs T+1 close) → fade by going opposite direction at T+2 open. Exit T+20 close.
- **Originality**: 7, **BT**: 4

### G-23 GPR tail-spike equity reversal (CONTRARIAN BUY)
- **Asset**: SPY; ITA co-trade
- **Horizon**: 30-90d
- **Rule**: When daily GPR closes > 3σ above 36m mean AND NBER recession indicator OFF → buy SPY next close. Exit day 60 OR GPR closes within 1σ.
- **Data**: matteoiacoviello.com GPR daily, FRED USREC, yfinance
- **Originality**: 6, **BT**: 5
- **Notes**: Counter-intuitive — fear overpriced when economy OK.
