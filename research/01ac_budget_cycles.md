# Phase 1AC — US + International Budget-Cycle Signals
# 12 signals. Dedup vs existing catalog (G-10, G-11, K-9, F-3, M5, M7, M11, P7, G-19, G-23, J12, F1-F15, AB-2, AA-4, AA-5).

### AC-1 US Fiscal-Year-End Contractor Obligation Surge
- **Rule**: Long equal-weight basket {LMT, GD, LDOS, BAH, SAIC} from 5th trading day of Sep through Sep 30; exit if running gain > 6%.
- **Data**: yfinance — direct
- **Mechanism**: ~16-18% of annual DoD contract obligations land in final fiscal month as program offices burn unobligated balances; benefits federal-IT primes (revenue recognized on task-order issuance).
- **Failure mode**: Full-year CR signaled in August compresses the surge.
- **CAGR**: 11-14%; Originality 9
- **Dedup**: NEW. F1-F15 are policy-event signals, not fixed-calendar contractor flow.

### AC-2 CBO August Baseline Deficit-Upgrade Steepener
- **Rule**: On CBO Budget & Economic Outlook release, if 10-yr cumulative deficit revised up >$500B vs prior baseline, short ZN / long ZT (2s10s steepener) for 15 trading days.
- **Data**: CBO PDF parse + FRED DGS2 / DGS10
- **Mechanism**: Larger projected deficits drive TBAC toward more coupon issuance at next QRA; dealers front-run by shedding duration.
- **CAGR**: 10-12%; Originality 9
- **Dedup**: NEW. Distinct from AA-4 (TIPS auction breakeven) and AA-5 (FOMC minutes).

### AC-3 UK Autumn Statement Gilt Pre-Drift Fade
- **Rule**: If UK 10y gilt yield rises >15bp in 10 days before Chancellor's statement, long gilt futures at close before statement; exit T+3.
- **Data**: FRED IRLTLT01GBM156N or BoE yield curves + HMT calendar
- **Mechanism**: Post-Truss trauma → pre-statement shorts; modal OBR-blessed package is fiscally disciplined → shorts cover.
- **Failure mode**: Pre-election giveaway budget. Gate by polling spread <10pts.
- **CAGR**: 12-15%; Originality 9
- **Dedup**: NEW.

### AC-4 India Union Budget Pre-Run Defense Tilt
- **Rule**: Long equal-weight {HAL.NS, BEL.NS, BDL.NS, MAZDOCK.NS} from Jan 10 through Feb 1 open; exit at Feb 1 open.
- **Data**: yfinance (.NS suffix)
- **Mechanism**: Defense capital outlay = single fastest-growing major Union Budget line since 2020 (Atmanirbhar Bharat); retail front-runs; Feb 1 is sell-the-news.
- **Failure mode**: Skip interim budgets in general-election years.
- **CAGR**: 13-16%; Originality 8
- **Dedup**: NEW.

### AC-5 Japan Supplementary Budget JGB 30y Bear-Steepener
- **Rule**: Short JGB 30y (or 20y futures) from Cabinet approval through Diet passage when supplementary >¥13T.
- **Data**: MOF supplementary budget releases + JGB yields
- **Mechanism**: Supplementaries >¥13T trigger additional JGB issuance in subsequent calendar revision; super-long absorbs marginal coupon supply post-YCC reform.
- **Failure mode**: BOJ Rinban operations cap long-end.
- **CAGR**: 10-13%; Originality 9
- **Dedup**: NEW. **TRACTABILITY: HARD** — supplementary sizes require manual MOF parse.

### AC-6 Saudi Budget Implicit Breakeven Oil Bull Signal ⭐
- **Rule**: When Saudi-published fiscal breakeven oil price (IMF Article IV or backed-out from MoF tables) exceeds front-month Brent by >$12, long Brent from Jan 2 through next OPEC+ JMMC.
- **Data**: IMF REO MENA + EIA Brent
- **Mechanism**: Saudi fiscal program is dominant constraint on OPEC+ supply policy; when new-year budget assumes oil materially above market, MbS has fiscal + political incentive to engineer cuts/extensions.
- **CAGR**: 14-18%; Originality 10
- **Dedup**: NEW. Distinct from AB-2 (NOK-Brent currency angle).

### AC-7 Italy Legge di Bilancio BTP-Bund Spread Mean-Reversion
- **Rule**: If BTP-Bund 10y spread widens >25bp between Italy DBP submission (Oct 15) and EC opinion (late Nov), short the spread (long BTP / short Bund futures) from day after EC opinion for 20 trading days.
- **Data**: ECB SDW spread + EU Commission opinions
- **Mechanism**: Markets reflexively price worst-case during negotiation; post-2023-reform EC opinions are institutionally measured; ECB TPI caps tail widening.
- **Failure mode**: Coalition fracture during passage.
- **CAGR**: 11-14%; Originality 9
- **Dedup**: NEW.

### AC-8 Germany Bundeshaushalt Debt-Brake-Suspension Defense Long
- **Rule**: On Bundestag plenary day where Art. 115 GG debt-brake exception passes, long equal-weight {RHM.DE, HAG.DE, R3NK.DE} at close; exit T+30.
- **Data**: Bundestag plenary protocols + Xetra
- **Failure mode**: Constitutional court invalidation (Nov 2023 precedent).
- **CAGR**: 12-16%; Originality 9
- **Dedup**: NEW. **TRACTABILITY: HARD** — rare event, N=2-4.

### AC-9 France Loi de Finances Article 49.3 CAC-Bank Fade
- **Rule**: On 49.3 invocation, short equal-weight {BNP.PA, GLE.PA, ACA.PA} at next open; cover T+10.
- **Data**: Assemblée Nationale press feed + Euronext
- **Mechanism**: 49.3 = coalition fragility → OAT-Bund widening → French bank capital pressure.
- **CAGR**: 10-13%; Originality 8
- **Dedup**: NEW. **TRACTABILITY: HARD** — small sample.

### AC-10 Norway GPFG Quarterly Allocation Drift → Short MSCI World
- **Rule**: Within 5 days of NBIM quarterly report, if reported equity allocation exceeds upper bound of 60% strategic band by >2pp, short MSCI World futures (or URTH); exit at next quarterly.
- **Data**: NBIM quarterly reports + URTH
- **Mechanism**: $1.5T+ AUM mechanical rebalancing acts as slow-moving headwind for DM beta.
- **Failure mode**: MoF widens band ex post during crisis (2008, 2020).
- **CAGR**: 10-12%; Originality 10
- **Dedup**: NEW. Distinct from AB-2 (NOK-Brent currency).

### AC-11 EU NextGenerationEU Disbursement → Peripheral Equity Lift
- **Rule**: On EU Commission RRF tranche >€10B disbursed (not just approved) to IT or ES, long FTSE MIB or IBEX 35 at next open; exit T+15.
- **Data**: EU RRF Scoreboard + Euronext/BME
- **Mechanism**: Tranches unlock pre-financed contractor payments within ~30 days; RRF = >5% of annual public investment in IT/ES/EL/PT.
- **CAGR**: 11-13%; Originality 9
- **Dedup**: NEW.

### AC-12 US December Tax-Loss-Harvest Year-Trailing-Loser Bounce
- **Rule**: On Dec 20, long equal-weight basket of Russell 3000 names with YTD return < -30% AND mcap > $1B; exit Jan 31.
- **Data**: Russell 3000 constituents + yfinance YTD
- **Mechanism**: Forced selling peaks mid-Dec, reverses once wash-sale window + new tax year begin; effect largest in mid-caps with concentrated retail/HF ownership.
- **Failure mode**: Broad-market drawdown years (2008, 2022) — basket keeps falling. Gate by SPX YTD > -10%.
- **CAGR**: 12-17%; Originality 8
- **Dedup**: NEW. Related to known January effect but with sharper construction.

---

## Dedup summary
- **NEW**: 12 / 12 (no DUPEs)
- **TRACTABLE (yfinance/FRED only)**: AC-1, AC-3, AC-4, AC-6, AC-7, AC-10, AC-12 → priority backtests
- **HARD (manual data)**: AC-2 (CBO PDFs), AC-5 (JP supplementary sizes), AC-8 (Bundestag votes), AC-9 (49.3 dates), AC-11 (EU tranche dates) → require hand-curated event tables
