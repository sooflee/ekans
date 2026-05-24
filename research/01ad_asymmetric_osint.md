# Phase 1AD — Asymmetric / OSINT Signals
# 22 signals across 6 territories. All NEW vs existing catalog. Dedup exclusion list applied: E11, G-10, G-13, G-14, F1-F15, AC-1, AC-6, Z6.

## Tractability legend
- **★ T**: Tractable on free data — backtest in BT-AD round
- **△ T**: Partially tractable (small N or hand-coded events) — include in BT-AD with caveats
- **✗ H**: Hard / infeasible on free data — log only

---

## T-batch — Trump Truth Social / X

### AD-T1 Truth-Social → X cross-post latency arb  ✗ H
- **Rule**: When a market-moving Truth Social post hits RSS but does NOT yet appear on X, take directional trade in implied asset; exit when X cross-post appears OR T+15min.
- **Asset**: ES/NQ futures, single names by keyword
- **Mechanism**: Truth-first contractual obligation (WaPo 2025-06-03) creates 30s–6min lag before X mirrors
- **CAGR**: 15-35% (dying-edge — front-load 2026)
- **Failure**: Needs real-time parallel feeds; no historical replay.

### AD-T2 Burst-rate intensity → VIX long  △ T
- **Rule**: Trump posts ≥25 times in rolling 60-min window between 6pm-7am ET → buy VIX 1-2wk ~10% OTM calls at next 9:30 ET open; exit T+3 close or +50%.
- **Asset**: VIX calls / VXX / short SPY tail
- **Mechanism**: Manic posting episodes empirically precede policy reversals, firings, escalations
- **CAGR**: 10-20% (low base rate ~8-15 events/yr)

### AD-T3 Named-foreign-leader caps → defense/EM pair  △ T
- **Rule**: Truth post mentions {Xi/Putin/Zelensky/Netanyahu/Kim/Khamenei/Maduro} AND post is ≥40% ALL CAPS or ≥3 exclamations → long ITA + short matched EM ETF (EWZ/MCHI/EIDO); hold 3d. Add "deal/agreement" exclusion.
- **CAGR**: 8-15%

### AD-T4 ★ Verbatim "GREAT TIME TO BUY" → SPY long  ★ T
- **Rule**: Trump post during US hours or 12h pre-open contains exact phrase ("THIS IS A GREAT TIME TO BUY" / "great time to buy" / "DJT" + "STOCK MARKET" + "GREAT" + "!") → buy SPY/ES at next print; exit T+1 15:55 ET or -1% stop.
- **Anchor**: Apr 9 2025 9:33 ET "THIS IS A GREAT TIME TO BUY!!! DJT" preceded tariff-pause + 9.5% SPY day
- **CAGR**: 12-25% (~4-10 events/yr)
- **Failure**: SEC investigation chills pattern.

---

## D-batch — DoD Daily Contracts Wire

### AD-D1 Sub-prime supplier NLP drift  △ T (needs scrape infra)
- **Rule**: DoD contract announcement names sub-tier supplier (MRCY/KTOS/AVAV/CW/HEI/MOG.A/TDG/DCO/RKLB) in description text AND program is identified → long named supplier next open; exit T+5.
- **Mechanism**: Headline bots match only prime field; sub-tier mentions diffuse over 1-5 days
- **CAGR**: 14-20%; **Originality 9**

### AD-D2 Contract-type regime pair  ✗ H
- **Rule**: Rolling 20-day cost-plus vs FFP share for mid-cap prime; when CPFF share +2σ vs 12mo mean → long that prime T+0 to T+20.
- **Asset**: LHX, BAH, CACI, LDOS, SAIC, KBR, PSN, VVX, AIT
- **Failure**: Needs full historical DoD contract type parsing

### AD-D3 Modification-vs-new-award polarity  ✗ H
- **Rule**: Classify each posting {NEW / MOD-CEILING-UP / MOD-OPTION / MOD-DESCOPE}; long top decile primes by ceiling-up; short bottom decile by descope; rebalance weekly hold 10d.
- **Originality 10**, but full text classification heavy

### AD-D4 Late-Friday end-of-quarter postings  ✗ H
- **Rule**: Contracts posted last Friday of FY quarter for small/mid-cap primes (MCAP <$5B) underperform Mon open → short basket Mon, cover Tue close. Cap $200M threshold.
- **CAGR**: 8-12%; needs timestamp scrape

---

## A-batch — Military / Executive ADS-B

### AD-A1 E-4B/E-6B simultaneous airborne cluster  ✗ H
- **Rule**: 2+ Nightwatch/Mercury hexes simultaneously airborne >2h outside scheduled GLOBAL THUNDER → long XAR + XLE + VIX calls, fade T+24-72h
- **Mechanism**: Baseline = 1 Mercury orbit max; clusters historic with crisis windows (Oct 2023 Israel, Feb 2022 Ukraine, Jan 2020 Soleimani)
- **Failure**: ADS-B history not freely available at scale

### AD-A2 SecState C-32 pre-summit country ETF drift  ✗ H
- **Rule**: C-32 (ADFEAD-ADFEB0) lands in non-US country with no prior 7-day State Dept readout → long that country's MSCI ETF + sell ATM USD/local 1wk straddle; hold 5-10d.
- **Asset**: EWZ, EWW, EWY, EZA, TUR, EWT, GREK, ARGT, INDA by destination

### AD-A3 KC-135/KC-46 tanker surge to CENTCOM/INDOPACOM  ✗ H
- **Rule**: ≥5 USAF tanker hexes cross 30E (CENTCOM) or 140E (INDOPACOM) in 72h vs trailing 30d baseline → long BNO + XLE + ITA + VIX 1M
- **Originality 8**; tanker hex registry maintenance heavy

### AD-A4 Doomsday cap-region loiter anomaly  ✗ H
- **Rule**: E-4B executes >4h racetrack orbit within 200nm of DC on non-Tuesday/non-exercise day → long VIX 1W 20-25 strike calls + GLD; flatten on landing.
- **Originality 10**

---

## O-batch — OFAC Sanctions

### AD-O1 Crypto-mixer SDN add → compliance vs privacy  △ T (hand-code)
- **Rule**: OFAC adds mixer/privacy-tool/smart-contract to SDN (Tornado Cash Aug 2022, Blender May 2022, Sinbad Nov 2023, Samourai Apr 2024, etc.) → long COIN+RIOT+CLSK, short XMR/USD; hold 5-15d
- **N ≈ 6** designations 2022-2025
- **Originality 9**

### AD-O2 ★ SDN delisting → listed-parent positive surprise  △ T (hand-code)
- **Rule**: OFAC removes name from SDN that maps to listed parent's subsidiary/supplier → buy parent next open; exit T+20.
- **Anchors**: Rusal/En+ Jan 2019 → GLEN.L +6%/10d; Karpowership KAR.TR; MTS family removals
- **Originality 9** — entire sanctions literature trades adds; removals untouched
- **N ≈ 6-10/yr globally**

### AD-O3 Venezuela General License → CVX drift  △ T (hand-code)
- **Rule**: Treasury issues new GL authorizing oil-sector activity (GL 41 Oct-2022, GL 44 Oct-2023, renewals) → long CVX 50% + REP.MC 25% + ENI.MI 25% next open; exit T+15 or on GL revocation.
- **CAGR**: 6-10%
- **Originality 8**

### AD-O4 EO 14114 secondary-sanctions warning → HK bank fade  △ T (hand-code)
- **Rule**: Treasury press release with {"secondary sanctions" / "EO 14114" / "correspondent account" / "FFI"} naming Chinese banks → short FXI 40% + EWH 40% + KWEB 20%; exit T+10
- **Events**: Jan/Mar/May 2024 cycles
- **CAGR**: 5-9%

---

## S-batch — STOCK Act Cluster (distinct from E11 Pelosi-mirror)

### AD-S1 Same-committee defense cluster pre-NDAA markup  △ T
- **Rule**: 3+ HASC/SASC members file PTRs buying same defense prime (LMT/RTX/NOC/GD/BA/LHX/HII) within 14-day window AND NDAA subcommittee markup in 5-30d → long that ticker 30-60d.
- **Mechanism**: Subcommittee program briefings 4-8 weeks pre-markup
- **CAGR**: 12-25% on cluster basis; ~4-8 firings/yr
- **Failure**: 30-day filing window means visibility lag

### AD-S2 ★ Bipartisan convergence cluster  △ T (CapitolTrades data)
- **Rule**: ≥2 Democrats AND ≥2 Republicans buy same ticker within 21-day window (zero offsetting sales), exclude mega-cap noise (AAPL/MSFT/NVDA/GOOGL/AMZN) → long 60-90d.
- **Mechanism**: Partisan trading bias removed; implies non-public catalyst crossing party lines
- **CAGR**: 15-30% per event; 6-12% portfolio alpha on ~5-15 events/yr
- **Originality 9**

### AD-S3 Late-filer bunch fade  △ T
- **Rule**: Chronic late-filer (>60-day avg lag — Tuberville, Fallon, Crenshaw, Blake Moore) dumps ≥10 backlog transactions on single PTR, ≥3 sector-concentrated → SHORT/fade sector ETF for 30-45d.
- **Mechanism**: Trades are stale; disclosure event triggers ethics-complaint news cycle adding fade pressure
- **N ≈ 6-12/yr**; portfolio CAGR 8-15%
- **Originality 9** (contrarian inversion of mirror paradigm)

---

## F-batch — Federal Register Friday-Dumps

### AD-F1 ★ Friday-evening EPA final-rule → energy/utility pair  ★ T
- **Rule**: EPA Final Rule (CFR title 40) published Friday AND economically-significant AND OIRA-cleared in prior 7d → short XLU + long XLE pair Fri close, exit T+5.
- **Data**: federalregister.gov/api/v1/articles.json (free, queryable); reginfo.gov OIRA dashboard for clearance cross-check
- **CAGR**: 7-12%
- **Originality 9**

### AD-F2 OIRA "midnight" rule surge → next-admin reversal pair  ✗ H (N too small)
- **Rule**: 90-day pre-inauguration after party-flip election: when OIRA concluded-review count > trailing 2yr p95 for agency → pair-trade AGAINST lame-duck tilt; hold 40-90d post-inauguration.
- **Anchors**: 2008→09, 2016→17, 2020→21, 2024→25 (N=4)
- **CAGR**: 15-30% during active windows
- **Originality 10**

### AD-F3 ★ CMS Friday/pre-holiday payment rule timing  ★ T
- **Rule**: CMS publishes payment-rate Final Rule (IPPS/OPPS/MPFS/MA Final Rule) on Friday OR pre-holiday OR after 4:30pm ET → take OPPOSITE of typical insurer/provider reaction T+0 to T+3.
- **Asset**: MA insurer basket {UNH, HUM, CVS, ELV, CI, MOH} long/short matched against hospital REITs {HCA, THC, UHS, CYH}
- **Data**: federalregister.gov API; gate on |surprise vs Advance Notice| < threshold
- **CAGR**: 9-15% (~4-8 firings/yr)

---

## Tractability summary

**★ Tractable, top priority for BT-AD**:
- AD-T4 verbatim Trump phrase (anchor trade Apr 9 2025)
- AD-F1 EPA Friday dump (clean FR API)
- AD-F3 CMS Friday payment-rule (clean FR API)

**△ Tractable with hand-coded event list**:
- AD-T2 burst rate (Trump archive)
- AD-T3 named-leader caps (Trump archive)
- AD-O1 crypto-mixer SDN (N≈6 events, hand-code)
- AD-O2 SDN delisting (N≈10-15 events, hand-code)
- AD-O3 Venezuela GL (hand-code GL dates)
- AD-O4 EO 14114 (hand-code 3-5 events)
- AD-D1 sub-prime supplier (scrape feasible)
- AD-S1 HASC/SASC cluster (Quiver/CapitolTrades data)
- AD-S2 bipartisan cluster (CapitolTrades data)
- AD-S3 late-filer fade

**✗ Hard/infeasible on free data**:
- AD-T1 latency arb (real-time only, no replay)
- AD-D2/D3/D4 (heavy DoD wire scrape)
- AD-A1/A2/A3/A4 (ADS-B history not free at scale)
- AD-F2 midnight rule (N=4 too small)

Backtest plan: AD-T4, AD-F1, AD-F3 as ★ priority; AD-T2, AD-T3, AD-O1, AD-O2, AD-O3, AD-O4, AD-S2, AD-S3 as △ priority on hand-coded event tables.
