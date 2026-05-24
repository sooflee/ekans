# Trading Signals Research

A catalog of ~350 trading signals mined from academic literature, retail social media, FinTwit, alt-data, crypto on-chain, geopolitical events, podcast transcripts, and research papers. Each one is honestly backtested using free public data only (yfinance, FRED, EDGAR, public APIs).

## Live site

**https://sooflee.github.io/ekans/**

- Sidebar groups strategies into **Non-crypto** vs **Crypto** tabs
- Adjustable **CAGR threshold slider** (default 10% = S&P long-run baseline)
- Sort by CAGR or Sharpe
- Click any strategy to read its mechanism, rule, results, caveats, and source

Full backtest table (all ~350 attempts, including duds): **https://sooflee.github.io/ekans/full_catalog.html**

## What's in this repo

| Path | Contents |
|---|---|
| `index.html` | Curated deep-dive of strategies clearing the user-set CAGR threshold |
| `full_catalog.html` | Sortable table of every backtested signal |
| `backtests/` | One Python file per signal (uses shared `harness.py`) |
| `results/` | One JSON per signal: Sharpe, CAGR, MaxDD, t-stat, hit rate, etc. (gitignored) |
| `research/01[a-ab]_*.md` | Source notes per hunt phase (27 phases) |
| `build_report.py` | Regenerates `full_catalog.html` from `results/*.json` |
| `.github/workflows/pages.yml` | Auto-deploys `index.html` + `full_catalog.html` on push |

## Methodology

1. **Hunt** — Research agents search for signals across a specific territory (TikTok, academic papers, FinTwit, alt-data, etc.)
2. **Dedup** — Each return is audited NEW / DUPE / RELATED-VARIANT against the existing catalog
3. **Backtest** — Surviving signals get implemented as standalone Python scripts using a shared harness; results saved as JSON
4. **Curate** — Strategies clearing the CAGR threshold are written up as deep-dive cards in `index.html`

## Honest disclaimers

- Backtests use free data with no transaction costs, slippage, or borrow costs
- Many "wins" depend on small event samples (N=10–30) — narrative risk is real
- Crypto strategies in particular ride a standout decade; future returns may differ
- "Beats SPY/BTC on risk-adjusted basis" usually means lower drawdown, not higher absolute return
- The Fed-put era has inverted many traditional distress signals (margin debt, swap lines, LEI hours) — they now mark buying opportunities rather than selloffs
- Past performance is not predictive. **Nothing in this repo is investment advice.**

## Reproducing the backtests

```bash
git clone https://github.com/sooflee/ekans.git
cd ekans
python3 -m venv .venv
.venv/bin/pip install yfinance pandas numpy scipy pandas-datareader pyarrow

# Run any backtest (each is standalone)
.venv/bin/python backtests/A14_btc_halving.py

# Regenerate full_catalog.html from all results/*.json
.venv/bin/python build_report.py
```
