"""
Helper to load Caldara-Iacoviello daily Geopolitical Risk index from the author site.
Caches xls download to data/gpr_daily.xls; returns a DataFrame indexed by date with
columns GPRD (daily GPR), GPRD_ACT, GPRD_THREAT, GPRD_MA30, GPRD_MA7.
"""
from pathlib import Path
import urllib.request
import ssl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
CACHE = DATA / "gpr_daily_recent.xls"


def load_gpr(force=False):
    if force or not CACHE.exists():
        ctx = ssl.create_default_context()
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
            CACHE.write_bytes(r.read())
    df = pd.read_excel(CACHE, engine="xlrd")
    df["date"] = pd.to_datetime(df["DAY"].astype(str), format="%Y%m%d")
    df = df.set_index("date").sort_index()
    keep = [c for c in ["GPRD", "GPRD_ACT", "GPRD_THREAT", "GPRD_MA7", "GPRD_MA30"] if c in df.columns]
    return df[keep].astype(float)


if __name__ == "__main__":
    g = load_gpr()
    print(g.shape, g.columns.tolist())
    print(g.head())
    print(g.tail())
