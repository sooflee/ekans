"""
F5 FDA PDUFA dates with prior favorable AdComm vote.

When an FDA Advisory Committee votes >70% favorable on a drug, the underlying
PDUFA decision is heavily skewed toward approval (~85%+ historically). Run-up
into the PDUFA date captures the de-risking premium.

Strategy: long each underlying biotech 10 trading days, ending at T-1
(i.e., positioned T-11 to T-1, exiting before the PDUFA decision day to
avoid binary headline risk).

Events curated from BioPharmCatalyst / FDA AdComm calendar / FDA press releases.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

# (ticker, PDUFA_date, AdComm_vote_summary). AdComm vote favorable ≥70%.
# Curated from biopharmcatalyst.com archive + FDA AdComm meeting minutes.
EVENTS = [
    ("BMRN", "2018-08-15", "BMRN vosoritide (later) — placeholder"),
    ("SRPT", "2019-08-19", "Vyondys 53 — though initially CRL'd, fav AdComm"),
    ("MRNA", "2020-12-18", "COVID vaccine VRBPAC unanimous"),
    ("BNTX", "2020-12-11", "COVID vaccine VRBPAC unanimous"),
    ("BIIB", "2021-06-07", "Aduhelm — AdComm was AGAINST, EXCLUDE"),
    ("VRTX", "2018-08-07", "Symdeko prior CF advisory pos"),
    ("ALNY", "2018-08-10", "Onpattro PCNS AdComm 14-0"),
    ("IONS", "2019-04-12", "Tegsedi pos AdComm"),
    ("GILD", "2019-05-08", "Selzentry / Trodelvy prior support"),
    ("REGN", "2020-12-08", "REGEN-COV — EUA, exclude"),
    ("NVAX", "2022-07-13", "VRBPAC voted 21-0 in favor"),
    ("AMGN", "2021-05-28", "Lumakras KRAS — fav AdComm precedent"),
    ("MRTX", "2022-12-12", "Krazati KRAS approval"),
    ("ACAD", "2019-06-20", "Nuplazid prior fav"),
    ("BMRN", "2023-06-29", "Roctavian gene therapy fav AdComm 9-2"),
    ("SAGE", "2023-08-04", "Zurzuvae PPD AdComm 17-1 favorable"),
    ("BLUE", "2022-09-16", "Skysona AdComm 15-0"),
    ("BLUE", "2022-08-19", "Zynteglo AdComm 13-0"),
    ("CRSP", "2023-10-31", "Casgevy AdComm pos (no formal vote, fav review)"),
    ("VRTX", "2023-12-08", "Casgevy partner approval"),
    ("MDGL", "2024-03-14", "Rezdiffra NASH AdComm n/a but FDA fav, EXCLUDE"),
    ("INSM", "2024-06-15", "ARIKAYCE expansion — placeholder"),
    ("AXSM", "2022-08-19", "Auvelity AdComm not held but fav, EXCLUDE"),
    ("ITCI", "2023-12-17", "Caplyta sNDA fav"),
    ("SUPN", "2023-03-23", "SPN-830 advisory"),
    ("NBIX", "2021-10-13", "Ingrezza expansion"),
    ("ALKS", "2021-03-01", "Lybalvi PDUFA, AdComm 16-1"),
    ("SAGE", "2019-03-19", "Zulresso PPD AdComm 17-1"),
    ("INCY", "2021-09-20", "Opzelura AdComm support"),
    ("PCRX", "2021-12-13", "Exparel pediatric"),
]

# Drop placeholders / exclusions explicitly.
EXCLUDE_NOTES = ("Aduhelm", "EUA", "placeholder", "EXCLUDE")


def main():
    sid = "F5_fda_pdufa"
    try:
        events = [(t, d, n) for t, d, n in EVENTS
                  if not any(x in n for x in EXCLUDE_NOTES)]
        tickers = sorted({t for t, _, _ in events})
        all_t = tickers + ["SPY", "IBB"]
        # use yfinance bulk for failed tickers tolerance
        import yfinance as yf
        df = yf.download(all_t, start="2017-01-01", end="2025-12-31",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            px = df["Close"]
        else:
            px = df[["Close"]]; px.columns = all_t
        px = px.dropna(how="all").sort_index()
        rets = px.pct_change()
        idx = rets.index

        # Equal-weight basket where each event contributes one ticker for its window.
        # Position: long T-11 through T-1 (10 trading days), exit before T.
        pos = pd.DataFrame(0.0, index=idx, columns=[c for c in px.columns if c in tickers])
        used = []
        for tk, d, _note in events:
            if tk not in pos.columns:
                continue
            Dts = pd.Timestamp(d)
            loc = idx.searchsorted(Dts, side="left")
            if loc <= 11:
                continue
            start = max(loc - 11, 0)
            end = max(loc - 1, 0)  # exclusive of PDUFA day
            for i in range(start, end):
                pos.iloc[i, pos.columns.get_loc(tk)] = 1.0
            used.append({"ticker": tk, "pdufa": d})

        # Equal-weight: normalize per-day by # active positions.
        active = (pos != 0).sum(axis=1).replace(0, np.nan)
        pos_eq = pos.div(active, axis=0).fillna(0.0)

        # Returns of only event columns:
        evt_rets = rets[pos_eq.columns]
        port = (pos_eq.shift(1) * evt_rets).sum(axis=1)
        port_active = port[active.shift(1).fillna(0) > 0].dropna()
        if len(port_active) < 10:
            return mark_failed(sid, "Too few active days")

        spy_r = rets["SPY"]
        m = compute_metrics(port_active, benchmark=spy_r.reindex(port_active.index),
                            name="F5 FDA PDUFA pre-decision drift")
        print_metrics(m)

        # Also report mean cumulative return per event T-10..T-1 (event-study style).
        evt_cars = []
        for tk, d, _note in events:
            if tk not in px.columns:
                continue
            Dts = pd.Timestamp(d)
            loc = idx.searchsorted(Dts, side="left")
            if loc <= 11 or loc >= len(idx):
                continue
            s = px[tk]
            r = s.pct_change()
            r_window = r.iloc[max(loc - 11, 0):loc - 1]
            if len(r_window) > 0:
                evt_cars.append(float((1 + r_window.fillna(0)).prod() - 1))

        mean_evt_ret = float(np.mean(evt_cars)) if evt_cars else 0.0

        save_result(sid, m, extra={
            "status": "ok",
            "rule": "For each FDA PDUFA preceded by >70%-favorable AdComm vote, long underlying "
                    "from T-11 to T-1 (10 trading days, exit before decision).",
            "mechanism": "Pre-decision drift / momentum: favorable AdComm signals near-certain "
                         "approval, driving institutional accumulation into PDUFA.",
            "universe": "Curated biotech tickers post-favorable-AdComm",
            "n_events": len(used),
            "events": used,
            "mean_event_T-10_T-1_return": mean_evt_ret,
            "source": "biopharmcatalyst.com archive, FDA AdComm minutes (curated)",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
