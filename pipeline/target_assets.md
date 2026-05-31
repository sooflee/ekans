# Target Assets — Preferred Trading Universe

This file lists the assets the project actually wants to predict and trade. The
idea_scout and strategy_developer specs reference it to bias signals away from
the lazy SPY/QQQ/BTC default and toward instruments with concrete economic
exposure.

**How to use this list**:
- Idea scouts: when designing a causal chain, prefer terminal assets from this
  list when the mechanism plausibly touches them. A signal that targets HCA on
  a hospital-margin shock is more useful than the same mechanism reformulated
  as "long XLV."
- Strategy developers: same — when reformulating an idea, prefer specific names
  here over broad baskets.
- Backtester: no special rule — backtest whatever the strategy specifies.

**This is a project preference, not a hard constraint.** If a mechanism's most
direct exposure is a name not on this list (e.g., a new IPO, an industry
specialist), use that name. The list is a default, not a fence.

---

## Single-name equities

### Mega-cap / liquid US large-cap
Use when an idea has direct exposure to one of these names. Not as a general
"long mega-cap" basket.

- AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA — the magnificent few
- BRK.B (Buffett-overlap; cleanest financial conglomerate)

### Sector specialists with clean mechanism exposure

**Health care**: HCA, THC, UHS (hospitals); CPRT, LKQ (salvage/parts);
SGRY, USPH, ENSG, DOC (ambulatory / ASC); DHR, TMO, BIO, A (life-science tools);
GILD, VRTX, REGN (large biotech). Avoid generic "long XLV" baskets.

**Industrials / capex**: UNP, CSX, NSC, CP, CNI (Class I rails);
WMS, AOS, MMM, PH (water / industrials); EME, PRIM, FIX, BLD (construction
services / CHIPS-act capex beneficiaries); GE, RTX, BA, LMT (aero/defense).

**Insurance / financials**: PGR, ALL, TRV, AIG, HIG, CINF, EG, RNR, AXS, RLI
(P&C); MET, PRU, AFL (life); BAC, JPM, USB, KEY (large banks); KRE constituents
for regional bank pair trades. Pair-trade structure preferred (e.g., long KBE /
short KRE) for credit-stress mechanisms.

**Energy / commodities equity**: XOM, CVX, COP, OXY, FANG (majors); EQT, AR,
CTRA, RRC (Appalachian gas); VLO, MPC, PSX (refiners); LNG, CQP, NEXT, TLN
(LNG/IPP); VST, NRG, CEG (IPP); HCC, BTU (met coal); FCX, SCCO, TECK (copper);
NEM, GOLD, AEM (gold majors).

**Consumer**: HD, LOW (home improvement); WMT, COST, BJ, TGT (broad retail);
DG, DLTR, FIVE, OLLI (dollar stores — for SNAP / wage signals);
SBUX, MCD, QSR, JACK, YUM (QSR); DRI, BLMN (casual dining);
CMG, WING (premium QSR).

**Tech subindustries**: COHR, GLW, WOLF, MTSI (compound semi);
ASML, AMAT, LRCX, KLAC (semicap); MU (memory); TSM, INTC, GFS (foundry);
ANET, CIEN, JNPR (networking); CRWD, ZS, S, NET (security).

**Real estate / REITs**: CPT, AVB, EQR (multifamily);
EQIX, DLR (data center); PLD, REXR (industrial);
WY, RYN (timber); IRM (storage).

### Single-country / international names
Use these for non-US mechanism exposure, not as a generic "international" play.

- EWZ, EWY, EWJ, EWG, EWU (country ETFs as proxies for non-yfinance equity)
- VALE, PBR, ABEV, BRFS (Brazil specifics)
- TSM, UMC (Taiwan semi)
- MOWI.OL, BAKKA.OL, SALM.OL (Norwegian salmon)
- ABUK.CA, MFPC.CA (Egyptian fertilizer)
- INFY, WIT (India IT)

## Commodity futures / commodity ETFs

Prefer direct commodity exposure for supply-shock mechanisms over commodity
equity proxies (the equity adds idiosyncratic noise).

- **Metals**: HG=F (copper), GLD/SLV (gold/silver via ETF), PA=F (palladium),
  PL=F (platinum)
- **Energy**: USO/BNO (oil ETFs), UNG (natgas), CL=F/BZ=F (crude futures),
  NG=F (natgas futures)
- **Grains/softs**: CORN, SOYB, WEAT (grain ETFs); ZC=F, ZS=F, ZW=F (futures);
  SUGAR/SGG, JO (coffee), BAL (cotton), CANE (sugar)
- **Livestock**: LE=F (live cattle), GF=F (feeder cattle), HE=F (lean hogs)

## Rates / credit / bonds

- **Treasuries**: TLT (20y+), IEF (7-10y), SHY (1-3y), BIL (T-bills);
  DGS10/DGS2/T10Y2Y on FRED for direct curve
- **Credit**: HYG, JNK (HY); LQD (IG); HYS, FALN (rising stars / fallen angels);
  EMB, EMLC, LEMB (EM USD / local)
- **Inflation**: TIP, STIP, SCHP; T10YIE / T5YIE on FRED

## FX (via single-country ETFs or futures)

- USD/EUR, USD/JPY, USD/CNH via futures or DXY (UUP ETF)
- Single-country ETFs (EWZ, EWY, etc.) carry FX exposure baked in — use them
  when the mechanism is country-specific rather than pure-FX

## Crypto / crypto-proxy equities

- **Direct**: BTC-USD, ETH-USD
- **ETFs**: IBIT, FBTC (BTC); ETHA, FETH (ETH)
- **Crypto-equity proxies**: COIN, MSTR, MARA, RIOT, CLSK, CIFR (miners),
  MUFG/SBNY-type crypto-exposed banks

## Anti-targets — avoid when possible

These instruments are the default "lazy" targets that bloat the catalog. Use
them only when:
- They're an explicit counter-signal (counter_signal: true)
- The mechanism is genuinely index-wide (rare)
- The benchmark slot, not the primary trade

Anti-targets:
- **SPY, QQQ, IWM** — the broad-equity defaults
- **VTI, ITOT** — total-market ETFs (same problem at larger scope)
- **SOXX, XLK** — broad tech (prefer specific semi sub-industry names)
- **XLF** — broad financials (prefer KRE/KBE pair, or specific insurer/bank)
- **XLE** — broad energy (prefer specific subsector: XOP refiners, etc.)

Sector ETFs (XLY, XLP, XLU, XLV, XLI, XLB, XLE, XLF, XLK, XRE) are **acceptable**
when the mechanism is genuinely sector-wide. They are NOT acceptable as a
disguised SPY proxy — i.e., "long XLY+XLP" is a thinly-veiled SPY trade.
