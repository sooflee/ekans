"""
F8 ECB PEPP/APP weekly reinvestment publication.

ECB publishes APP+PEPP holdings weekly on Tuesdays. The strategy would compare
the latest weekly purchase vs a 4-week rolling average and trade BTP/EWP
accordingly.

NOTE: PEPP active purchases ended March 2022 and reinvestments ended in
June 2024 (net zero). The relevant weekly purchase series is not directly
on FRED with the cadence required (FRED only has aggregate ECB balance sheet),
and scraping ecb.europa.eu CSV pages is out of scope here.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    sid = "F8_ecb_pepp"
    return mark_failed(sid, extra={
        "status": "fail",
        "rule": "ECB weekly APP+PEPP purchase 4w-rolling regime trade in Italian/Spanish "
                "equity proxies.",
        "mechanism": "Reinvestment intensity supports periphery spreads and risk assets.",
        "data_caveat": "ECB weekly net purchase series (APP_BO_W, PEPP_BO_W) is published on "
                       "the ECB Statistical Data Warehouse but not on FRED at the required "
                       "weekly cadence; PEPP reinvestments terminated June 2024 so the "
                       "signal has near-zero live N going forward. Marking failed rather "
                       "than fabricating purchase amounts.",
        "source": "ECB SDW (https://sdw.ecb.europa.eu/)",
    }, reason="ECB weekly purchase data not available via FRED at required cadence; "
              "PEPP reinvestment program terminated June 2024.")


if __name__ == "__main__":
    main()
