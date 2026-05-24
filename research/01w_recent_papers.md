# Phase 1W — Recent (2023-2026) Research Papers + Practitioner Notes
# 13 signals; dedup audit complete (no strong outright dupes vs prior catalog).

### W-1 FinGPT / LLaMA-2 MD&A Sentiment L/S ✓ NEW
- **Source**: Kim/Muhn/Nikolaev (2024-25) J Corp Finance — finance-tuned LLM sentiment
- **Rule**: Quarterly, rank Russell 3000 by FinGPT v3.3 sentiment on latest MD&A; long top decile / short bottom decile; hold 1 quarter.
- **Data**: SEC EDGAR 10-K/10-Q text + HuggingFace FinGPT model
- **Reported Sharpe**: ~3.05 in paper; 74% sign-of-return accuracy OOS
- **Originality**: 9
- **Dedup**: Distinct from D19 LM 10-K sentiment (lexicon vs LLM)

### W-2 CBOE DSPX Dispersion Regime → SPY vs RSP ✓ NEW
- **Source**: CBOE DSPX methodology (2023) + Ji (2025), Resonanz (2025)
- **Rule**: When DSPX in bottom quartile of trailing 252d, overweight RSP vs SPY; flip when DSPX > top quartile.
- **Data**: cboe.com/us/indices/dashboard/DSPX
- **Originality**: 9 (DSPX is brand new)

### W-3 Mag-7 Hedge-Fund De-Crowding ✓ NEW
- **Source**: CFM (2025), Resonanz (2025)
- **Rule**: When GS Prime weekly hedge-fund net Mag-7 exposure drops > 2σ WoW, long equal-weight Mag-7 for 10 trading days.
- **Data**: WhaleWisdom 13F proxy or GS Prime via Twitter/ZeroHedge reposts
- **Originality**: 8

### W-4 Treasury Auction Tail → Short Long-Duration Equities
- **Source**: Lou-Sadka-Yan (2024) J Banking Finance; HBS WP 26-033
- **Rule**: On day a 10/30-yr auction tails > 1.5bp (yield - WI), short XLU+XLRE+ARKK at 1pm close, cover next-day close.
- **Data**: TreasuryDirect.gov auction query at 1:01pm ET + CME WI
- **Reported edge**: 25-40bp/event × 10 events/year
- **Originality**: 9
- **Dedup**: Variant of L3-inv (TLT long after weak directs); this shorts EQUITIES instead

### W-5 RN Skewness Sign Flip → Long High-Skew / Short Low-Skew
- **Source**: Sai Ke (2024) SSRN 4777204
- **Rule**: Monthly, sort optionable US stocks by 30d option-implied skewness; long top decile / short bottom decile.
- **Data**: CBOE EOD chains (Bakshi-Kapadia-Madan formula)
- **Reported edge**: 80bp/month, t-stat 3.5
- **Originality**: 9 (opposite-sign vs classical Bali-Hu-Murray)
- **Dedup**: Variant of D17 — option-implied vs historical

### W-6 Insider + Anomaly Confirmation
- **Source**: Tian-Xiang-Xu (2025) SSRN 5237160
- **Rule**: Monthly, restrict 11-anomaly composite L/S universe to stocks where insider net-buying (Form 4, 90d) is on the same side as anomaly prediction; hold 1 month.
- **Data**: EDGAR Form 4 + Open Asset Pricing for anomaly signals
- **Reported edge**: 2x baseline anomaly alpha (1.3% vs 0.6%/mo)
- **Originality**: 8
- **Dedup**: Layered variant of E01/E02 + D-factors; novel via interaction

### W-7 RIVS + Abnormal Turnover ✓ NEW
- **Source**: Eksi-Roy (2025) SSRN 5234112
- **Rule**: Long bottom decile of (RV30 - IV30) conditional on bottom decile of (turnover / 12m avg); short symmetric top deciles; monthly.
- **Data**: Yahoo RV + AlphaQuery IV + free CRSP samples
- **Reported edge**: 1.1%/month Sharpe ~1.4 net of t-costs
- **Originality**: 8

### W-8 Post-GME WSB Reversal (Short Top Mentions)
- **Source**: Bradley/Hanousek/Jame/Xiao (2024) RFS
- **Rule**: Each day, short top decile of stocks by WSB daily mention count, beta-hedge with SPY; hold 5 trading days.
- **Data**: SwaggyStocks / Pushshift / ApeWisdom
- **Reported edge**: -1.6%/5d alpha on short leg post-GME
- **Originality**: 9 (flipped-sign trade vs older WSB papers)
- **Dedup**: We had G03 WSB long with small sample; this is the SHORT side which we didn't fully test

### W-9 Managed-Futures Replication Slippage Premium ✓ NEW
- **Source**: Braun-Hoffstein-Jablecki (2024) SSRN 4990063
- **Rule**: Monthly, long SocGen Trend Index basket (KMLM+DBMF+CTA) and short an in-house 12m TSMOM across 24 liquid futures.
- **Data**: SocGen Trend Index daily + CME futures
- **Reported edge**: 3-4% annualized Sharpe ~0.8
- **Originality**: 9 (specific to 2024 replication ETF wave)

### W-10 Bond Short Interest → Equity Beta Tilt ✓ NEW
- **Source**: Duong/Gorbenko/Kalev/Tian (2025) SSRN 5099127
- **Rule**: When aggregate corporate-bond short interest (% of LQD+HYG+JNK float) > 1σ above 24m mean, reduce equity beta by 50% next month.
- **Data**: FINRA biweekly ETF short interest
- **Reported edge**: 1.4%/month alpha
- **Originality**: 9 (cross-asset bond → equity novel)

### W-11 Anomaly Tail Loss (Option IV Slope)
- **Source**: Vilkov-Xiao (2024 updated) + Alexiou-Rompolis (2024)
- **Rule**: Monthly, compute stock TLM from linear slope of OTM put IVs (90% to 60% moneyness, 30d); short top decile, long bottom decile.
- **Data**: CBOE chains delayed
- **Reported edge**: 90bp/month Sharpe 1.3
- **Originality**: 8
- **Dedup**: Variant of D17 — option-IV slope cross-sectional

### W-12 ML Anomaly-Combination Survivor Set ✓ NEW
- **Source**: Azevedo-Hoegner-Velikov (2024 rev 2025) SSRN 4702406
- **Rule**: LightGBM/XGBoost on 17 OpenAssetPricing anomalies that survived 2018-22 quant winter; trade top-bottom decile of predicted next-month return.
- **Data**: openassetpricing.com free signals
- **Reported edge**: Sharpe ~1.8, ~12% net CAGR
- **Originality**: 8

### W-13 IG Bond Issuance Surge → LQD/HYG ✓ NEW
- **Source**: Baslandze-Fuchs Atlanta Fed WP 2025
- **Rule**: When monthly IG corporate bond gross issuance > 1.5σ above 12m mean, short HYG + long LQD for 60 days.
- **Data**: SIFMA monthly issuance free
- **Reported edge**: ~6 trades/year, 1.8% spread, Sharpe 1.1
- **Originality**: 8
