"""
T9 AUDUSD vs iron ore proxy divergence.

Rule: 20-day rolling correlation < 0.20 AND AUDUSD has outperformed iron ore
proxy by > 5% over the trailing 30 days -> long iron ore proxy 6 weeks.

Data:
  - AUDUSD: FRED DEXUSAL (USD per AUD; we use directly).
  - Iron ore proxy: VALE (yfinance) is the primary proxy. SGX FEF futures
    are not on yfinance and EIA/FRED do not carry the 62%-Fe Platts/MB
    benchmark daily without scraping a Trading Economics page.

Substitution note: VALE equity is correlated to iron ore but also carries
Brazilian risk / royalty / FX / management noise. BHP would be a backup.
We use VALE as the "iron ore proxy" leg of the trade.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import (
    load_prices, load_fred, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    try:
        aud = load_fred("DEXUSAL", start="2005-01-01").iloc[:, 0].rename("AUDUSD")
        vale = load_prices(["VALE"], start="2005-01-01").iloc[:, 0].rename("VALE")
    except Exception as e:
        return mark_failed("T9_aud_iron_ore_divergence", f"data load: {e}")

    df = pd.concat([aud, vale], axis=1).ffill().dropna()
    aud_r = df["AUDUSD"].pct_change()
    vale_r = df["VALE"].pct_change()

    corr20 = aud_r.rolling(20).corr(vale_r)
    rel30 = (1 + aud_r).rolling(30).apply(np.prod, raw=True) - 1 - \
            ((1 + vale_r).rolling(30).apply(np.prod, raw=True) - 1)

    sig = (corr20 < 0.20) & (rel30 > 0.05)
    print(f"trigger days: {int(sig.sum())} / {len(sig.dropna())}")

    HOLD = 30  # 6 weeks
    pos = pd.Series(0.0, index=df.index)
    for d in sig[sig].index:
        i = df.index.searchsorted(d) + 1
        if i >= len(df.index):
            continue
        end = min(i + HOLD, len(df.index))
        pos.iloc[i:end] = 1.0

    pnl = (pos.shift(1) * vale_r).dropna()
    bench = vale_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="T9 AUDUSD-iron ore divergence -> long VALE")
    print_metrics(m)
    save_result("T9_aud_iron_ore_divergence", m, extra={
        "status": "ok",
        "rule": ("20d corr(AUDUSD, VALE) < 0.20 AND AUDUSD - VALE 30d return > +5% -> "
                 "long VALE 6 weeks."),
        "mechanism": ("FX has historically led iron-ore equity moves; divergence is "
                      "a mean-reversion signal toward the macro driver."),
        "source": "FRED DEXUSAL, yfinance VALE (SGX FEF not freely available)",
        "n_triggers_raw_days": int(sig.sum()),
        "caveats": ("VALE substituted for SGX FEF 62% Fe futures. VALE has equity risk "
                    "(idiosyncratic Brazil, royalty, ESG); a pure iron-ore play would "
                    "require Singapore Exchange data."),
    })


if __name__ == "__main__":
    main()
