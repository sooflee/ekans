"""
Daily signal check — 2026-06-07.
Re-evaluates the price/market-data-driven LIVE signals against fresh data and
compares to the stored status in index.html's META block. On-chain / monthly-macro
/ event-driven signals are not programmatically checkable here and are left as-is.
"""
import warnings, datetime as dt
warnings.filterwarnings("ignore")
import yfinance as yf
import pandas as pd, numpy as np

TODAY = "2026-06-07"
FOMC = ["2026-04-29", "2026-06-17", "2026-07-29"]

def dl(tkr, start="2024-01-01"):
    df = yf.download(tkr, start=start, end=TODAY, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df["close"].dropna()

print(f"Fetching market data through {TODAY} ...")
T = {}
for t in ["SPY","XLU","EFA","EEM","TLT","GLD","IEF","MTUM","ACWX","UUP",
          "^VIX","^VIX3M","^MOVE","BTC-USD","QQQ"]:
    s = dl(t, "2024-06-01")
    T[t] = s
    asof = s.index[-1].date() if s is not None and len(s) else None
    last = float(s.iloc[-1]) if s is not None and len(s) else None
    print(f"  {t:9s} -> {('%.2f'%last) if last else 'NA':>9}  asof {asof}")

def ret(s, days):
    if s is None or len(s) <= days: return None
    return float(s.iloc[-1] / s.iloc[-days-1] - 1.0)

def sma(s, n):
    if s is None or len(s) < n: return None
    return float(s.iloc[-n:].mean())

def rsi(s, n=2):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return float((100 - 100/(1+rs)).iloc[-1])

results = {}  # id -> (computed_status, detail)

# F09: SPY 50d SMA vs 200d SMA  (need 200d -> longer history)
spy_long = dl("SPY", "2024-06-01")
s50, s200 = sma(spy_long, 50), sma(spy_long, 200)
results["F09"] = ("active" if s50 > s200 else "inactive",
                  f"50d {s50:.1f} {'>' if s50>s200 else '<'} 200d {s200:.1f}")

# C02: SPY 4wk (20d) return vs XLU 4wk return -> long SPY if SPY leads
r_spy, r_xlu = ret(T["SPY"], 20), ret(T["XLU"], 20)
results["C02"] = ("active" if r_spy > r_xlu else "inactive",
                  f"SPY 4wk {r_spy*100:+.1f}% vs XLU {r_xlu*100:+.1f}%")

# F08: SPY RSI(2) -> mean-reversion BUY only when oversold (<5..10); else inactive
r2 = rsi(T["SPY"], 2)
results["F08"] = ("active" if r2 < 10 else "inactive", f"RSI(2)={r2:.0f}")

# B01: VIX/VIX3M < 0.92 -> long (contango/calm)
v, v3 = float(T["^VIX"].iloc[-1]), float(T["^VIX3M"].iloc[-1])
ratio = v/v3
results["B01"] = ("active" if ratio < 0.92 else "inactive", f"VIX/VIX3M={ratio:.2f}")

# C11: MOVE/VIX < 6 -> long
mv = float(T["^MOVE"].iloc[-1]); mvr = mv/v
results["C11"] = ("active" if mvr < 6 else "inactive", f"MOVE/VIX={mvr:.2f}")

# C14: SPY, TLT, GLD all positive 12m -> all-asset risk-on
r12 = {k: ret(T[k], 252) for k in ["SPY","TLT","GLD"]}
all_pos = all(x is not None and x > 0 for x in r12.values())
results["C14"] = ("active" if all_pos else "inactive",
                  " ".join(f"{k} {v*100:+.0f}%" for k,v in r12.items()))

# C05: count of {SPY,EFA,EEM,GLD,IEF} above 10mo (~210d) SMA -> active if >=3
uni = ["SPY","EFA","EEM","GLD","IEF"]
above = []
for k in uni:
    s = dl(k, "2024-06-01")
    m = sma(s, 210)
    if m is not None and float(s.iloc[-1]) > m: above.append(k)
results["C05"] = ("active" if len(above) >= 3 else "inactive",
                  f"{len(above)}/5 above 10mo SMA: {','.join(above)}")

# C06: SPY & ACWX 12m momentum both positive (vs ~0 T-bill) -> equity-on
r_spy12, r_acwx12 = ret(T["SPY"], 252), ret(T["ACWX"], 252)
results["C06"] = ("active" if (r_spy12 > 0 and r_acwx12 > 0) else "inactive",
                  f"SPY 12m {r_spy12*100:+.1f}% ACWX {r_acwx12*100:+.1f}%")

# C07: SPY top-ranked on 12m momentum vs {EFA,EEM,TLT,GLD,IEF}
mom = {k: ret(T[k], 252) for k in ["SPY","EFA","EEM","TLT","GLD","IEF"]}
top = max(mom, key=lambda k: mom[k])
results["C07"] = ("active" if top == "SPY" else "inactive",
                  f"top mom={top} ({mom[top]*100:+.0f}%); SPY {mom['SPY']*100:+.0f}%")

# D02: MTUM YTD positive -> momentum leading
ytd = T["MTUM"][T["MTUM"].index >= "2026-01-01"]
mtum_ytd = float(ytd.iloc[-1]/ytd.iloc[0]-1) if len(ytd) > 1 else None
results["D02"] = ("active" if mtum_ytd > 0 else "inactive", f"MTUM YTD {mtum_ytd*100:+.1f}%")

# PL87: dollar weak (UUP below 52wk highs by a margin) -> exporters long
uup = dl("UUP", "2024-06-01")
uup_peak = float(uup.iloc[-252:].max()); uup_now = float(uup.iloc[-1])
uup_off = uup_now/uup_peak - 1
results["PL87"] = ("active" if uup_off < -0.02 else "inactive",
                   f"UUP {uup_off*100:+.1f}% from 52wk peak")

# H11: BTC-NDX 60d corr & BTC-GLD 60d corr (low corr -> default risk-on sizing)
btc = T["BTC-USD"]; qqq = T["QQQ"]; gld = T["GLD"]
def corr60(a, b):
    df = pd.concat([a.pct_change(), b.pct_change()], axis=1).dropna().iloc[-60:]
    return float(df.iloc[:,0].corr(df.iloc[:,1]))
c_bn, c_bg = corr60(btc, qqq), corr60(btc, gld)
results["H11"] = ("active" if (c_bn < 0.5 and c_bg < 0.5) else "inactive",
                  f"BTC-NDX {c_bn:.2f}, BTC-GLD {c_bg:.2f}")

# --- Calendar-based ---
today = dt.date(2026, 6, 7)
# A16: June is a seasonally weak month -> defensive
results["A16"] = ("defensive", "June (seasonally weak) -> cash")
# A07: trade window is the ~few days around an FOMC meeting; next is Jun 17
days_to_fomc = min((dt.date.fromisoformat(f) - today).days for f in FOMC
                   if dt.date.fromisoformat(f) >= today)
results["A07"] = ("active" if abs(days_to_fomc) <= 3 else "inactive",
                  f"next FOMC in {days_to_fomc}d")
# A06: FOMC-cycle week parity (even weeks since last meeting tend bullish)
last_fomc = max(dt.date.fromisoformat(f) for f in FOMC if dt.date.fromisoformat(f) <= today)
wk = ((today - last_fomc).days)//7
results["A06"] = ("active" if wk % 2 == 0 else "inactive",
                  f"week {wk} since FOMC {last_fomc}")

# Stored statuses from index.html META (current, post 2026-06-03)
STORED = {
    "F09":"active","C02":"active","F08":"inactive","B01":"active","C11":"active",
    "C14":"active","C05":"active","C06":"active","C07":"active","D02":"active",
    "PL87":"inactive","H11":"active","A16":"defensive","A07":"inactive","A06":"inactive",
}

print("\n=== DAILY CHECK", TODAY, "===")
changes = []
for sid in STORED:
    comp, detail = results[sid]
    old = STORED[sid]
    flag = "  " if comp == old else "**"
    if comp != old:
        changes.append({"id": sid, "from": old, "to": comp})
    print(f"{flag} {sid:5s} stored={old:9s} computed={comp:9s}  | {detail}")

print(f"\nChecked {len(STORED)} price/calendar-driven signals. {len(changes)} change(s).")
for c in changes:
    print(f"  CHANGE {c['id']}: {c['from']} -> {c['to']}")
