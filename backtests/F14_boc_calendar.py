"""
F14 Bank of Canada scheduled meeting calendar.

BoC holds 8 fixed-date meetings per year. We curate dates 2010-2025
(~120 events). For each, we approximate OIS-implied policy using a simple
1y CORRA OIS proxy (FRED has BoC overnight; OIS proxy uses 1Y Canada
swap-rate / 1Y T-bill from FRED IR3TIB01CAM156N or DTB1YR). When BoC
overnight > OIS+25bp the policy is "tight" and we short USDCAD 3 days
before; when overnight < OIS-25bp we long USDCAD 3 days before. Otherwise no
position.

NOTE: The OIS-implied 1y curve is not directly on FRED. As a proxy we use
the 1y Canada T-bill yield (FRED IR3TTS01CAM156N) vs BoC policy rate. Where
1y < overnight by 25bp+, the market is pricing cuts -> short CAD.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import load_prices, load_fred, compute_metrics, print_metrics, save_result, mark_failed

# Curated BoC scheduled meeting dates (rate decisions). 2010-2025.
# Source: BoC website "Key Interest Rate" historical decisions.
MEETINGS = [
    # 2010
    "2010-01-19", "2010-03-02", "2010-04-20", "2010-06-01", "2010-07-20",
    "2010-09-08", "2010-10-19", "2010-12-07",
    # 2011
    "2011-01-18", "2011-03-01", "2011-04-12", "2011-05-31", "2011-07-19",
    "2011-09-07", "2011-10-25", "2011-12-06",
    # 2012
    "2012-01-17", "2012-03-08", "2012-04-17", "2012-06-05", "2012-07-17",
    "2012-09-05", "2012-10-23", "2012-12-04",
    # 2013
    "2013-01-23", "2013-03-06", "2013-04-17", "2013-05-29", "2013-07-17",
    "2013-09-04", "2013-10-23", "2013-12-04",
    # 2014
    "2014-01-22", "2014-03-05", "2014-04-16", "2014-06-04", "2014-07-16",
    "2014-09-03", "2014-10-22", "2014-12-03",
    # 2015
    "2015-01-21", "2015-03-04", "2015-04-15", "2015-05-27", "2015-07-15",
    "2015-09-09", "2015-10-21", "2015-12-02",
    # 2016
    "2016-01-20", "2016-03-09", "2016-04-13", "2016-05-25", "2016-07-13",
    "2016-09-07", "2016-10-19", "2016-12-07",
    # 2017
    "2017-01-18", "2017-03-01", "2017-04-12", "2017-05-24", "2017-07-12",
    "2017-09-06", "2017-10-25", "2017-12-06",
    # 2018
    "2018-01-17", "2018-03-07", "2018-04-18", "2018-05-30", "2018-07-11",
    "2018-09-05", "2018-10-24", "2018-12-05",
    # 2019
    "2019-01-09", "2019-03-06", "2019-04-24", "2019-05-29", "2019-07-10",
    "2019-09-04", "2019-10-30", "2019-12-04",
    # 2020
    "2020-01-22", "2020-03-04", "2020-03-13", "2020-03-27", "2020-04-15",
    "2020-06-03", "2020-07-15", "2020-09-09", "2020-10-28", "2020-12-09",
    # 2021
    "2021-01-20", "2021-03-10", "2021-04-21", "2021-06-09", "2021-07-14",
    "2021-09-08", "2021-10-27", "2021-12-08",
    # 2022
    "2022-01-26", "2022-03-02", "2022-04-13", "2022-06-01", "2022-07-13",
    "2022-09-07", "2022-10-26", "2022-12-07",
    # 2023
    "2023-01-25", "2023-03-08", "2023-04-12", "2023-06-07", "2023-07-12",
    "2023-09-06", "2023-10-25", "2023-12-06",
    # 2024
    "2024-01-24", "2024-03-06", "2024-04-10", "2024-06-05", "2024-07-24",
    "2024-09-04", "2024-10-23", "2024-12-11",
    # 2025
    "2025-01-29", "2025-03-12", "2025-04-16", "2025-06-04", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
]


def main():
    sid = "F14_boc_calendar"
    try:
        # USDCAD via Yahoo "CAD=X" (USD per CAD inverse) — actually CAD=X is USDCAD.
        # We long USDCAD = short CAD; convention: if signal says "short USDCAD"
        # that's pos = -1 (CAD strengthens).
        import yfinance as yf
        usdcad = yf.download("CAD=X", start="2009-06-01", end="2026-06-01",
                             progress=False, auto_adjust=True)
        if isinstance(usdcad.columns, pd.MultiIndex):
            usdcad = usdcad["Close"].iloc[:, 0]
        else:
            usdcad = usdcad["Close"]
        usdcad.name = "USDCAD"
        cad_ret = usdcad.pct_change()

        # Policy rate (BoC overnight) and 1y T-bill from FRED.
        # IRSTCB01CAM156N = Canada overnight rate (monthly); use IR3TIB01CAM156N or
        # daily we use INTGSBCAM193N? Simpler: use BoC overnight via FRED series
        # IRSTCI01CAM156N (Immediate Rates - Call Money Canada).
        # 1y bill: IR3TTS01CAM156N (Treasury bill rate Canada 3-month) — but we
        # want 1y. We use IRLTLT01CAM156N? That's 10y. Use
        # IR3TTS01CAM156N (3m) as policy-tracking proxy minus IRLTLT01CAM156N (10y)
        # for curve. Better: use IRSTCB01CAM156N for overnight and
        # IR3TIB01CAM156N for 3m interbank.
        try:
            fred = load_fred(["IRSTCI01CAM156N", "IR3TIB01CAM156N"], start="2009-01-01")
            fred.columns = ["overnight", "ir3m"]
        except Exception:
            # Fallback set
            try:
                fred = load_fred(["IRSTCB01CAM156N", "IR3TTS01CAM156N"], start="2009-01-01")
                fred.columns = ["overnight", "ir3m"]
            except Exception as e:
                return mark_failed(sid, f"FRED series for Canada rates unavailable: {e}")

        fred = fred.dropna(how="all")
        # spread = overnight - 3m bill. Positive => market pricing easing.
        # NOTE: Spec asks "overnight vs 1y OIS"; we proxy with 3m.
        fred["spread_bp"] = (fred["overnight"] - fred["ir3m"]) * 100  # pct -> bp
        fred = fred.fillna(method="ffill")

        # For each meeting date, get the spread at the most-recent available
        # monthly observation prior to the meeting.
        idx = cad_ret.dropna().index
        pos = pd.Series(0.0, index=idx)
        used = []
        for d in MEETINGS:
            Dts = pd.Timestamp(d)
            loc = idx.searchsorted(Dts, side="left")
            if loc >= len(idx) or loc < 5:
                continue
            # Get spread observation prior to meeting
            # fred is monthly; take last monthly value <= meeting date
            f_idx = fred.index[fred.index <= Dts]
            if len(f_idx) == 0:
                continue
            spr = float(fred.loc[f_idx[-1], "spread_bp"])
            # Apply spec sign: overnight > OIS+25bp (tight) => short USDCAD (CAD
            # benefits). overnight < OIS-25bp (loose) => long USDCAD.
            # We invert: short USDCAD = pos -1.
            position = 0.0
            if spr > 25:
                position = -1.0  # short USDCAD
            elif spr < -25:
                position = 1.0  # long USDCAD
            if position == 0.0:
                continue
            start = max(loc - 3, 0)
            end = loc
            pos.iloc[start:end] = position
            used.append({"meeting": d, "spread_bp": spr, "pos": position})

        # Align series strictly.
        cad_ret_a = cad_ret.reindex(pos.index)
        port = pos.shift(1) * cad_ret_a
        active = (pos != 0).reindex(pos.index, fill_value=False)
        active_shift = active.shift(1).fillna(False).astype(bool)
        port_active = port[active_shift.values].dropna()
        if len(port_active) < 10:
            return mark_failed(sid, f"Too few active days ({len(port_active)})")

        # Use SPY benchmark for reference only; this is FX, no clean benchmark.
        m = compute_metrics(port_active, name="F14 BoC meeting calendar USDCAD signal")
        print_metrics(m)
        save_result(sid, m, extra={
            "status": "ok",
            "rule": "3 trading days before each BoC scheduled meeting: short USDCAD if BoC "
                    "overnight - 3m_bill > +25bp (tight); long USDCAD if < -25bp (loose).",
            "mechanism": "Pre-meeting positioning into expected policy surprise.",
            "universe": "USDCAD",
            "n_meetings": len(MEETINGS),
            "n_events_traded": len(used),
            "spread_proxy": "BoC overnight (IRSTCI01CAM156N) - Canada 3m interbank (IR3TIB01CAM156N)",
            "data_caveat": "3m bill used as OIS proxy; FRED Canadian-OIS 1y series not directly available.",
            "events_sample": used[:20],
            "source": "BoC fixed-announcement-date schedule + FRED Canadian rates",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
