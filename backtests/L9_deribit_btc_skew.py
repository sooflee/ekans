"""
L9 Deribit BTC 25-delta skew — no free historical option chain.

The Deribit public REST API (get_book_summary_by_currency, ticker, etc.)
returns only the *current* live snapshot. Historical option chains (daily
end-of-day IV by strike/expiry) require the paid Deribit history API
(https://docs.deribit.com/#test-historical-data) or a third-party feed
(Tardis.dev, Amberdata, Genesis Volatility).

Constructing a clean 25-delta skew time series for BTC therefore requires
a paid data source. Without that, every recent value is a single point —
no time series for backtest.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "L9_deribit_btc_skew",
        ("Deribit public REST endpoints return live snapshots only; "
         "historical EOD option chains by strike/expiry are paid (Deribit "
         "history API / Tardis / Amberdata). 25Δ skew cannot be backtested "
         "without that data."),
        extra={
            "rule": ("25Δ put/call skew > 2σ above 90d mean -> long BTC for 5 days "
                     "(intended rule; not testable here)."),
            "mechanism": "Crowded protective put demand often marks short-term capitulation lows.",
            "source_attempted": "https://www.deribit.com/api/v2/public/get_book_summary_by_currency (live snapshot only).",
        },
    )


if __name__ == "__main__":
    main()
