# Phase 1V — Sources with Verified / Verifiable Track Records
# Curated list of people and places online who have been consistently correct enough
# that mining their public output is worth a research round. Organized by the *kind* of
# track record they have (mathematically auditable → loosely verifiable → reputational).
# Use for: next signal hunt. Read the caveats at the bottom before treating anyone as oracle.

---

## Tier 1 — Mathematically Auditable Track Records (highest signal quality)

These have public PnL, calibration scores, or resolved binary forecasts. You can verify
the track record yourself before mining their thesis pipeline.

### V-1 Polymarket top whales (wallet-verified PnL)
- **What**: Public wallet addresses with $1M+ lifetime profit, sortable by all-time / monthly / weekly.
- **Where**: `polymarket.com/leaderboard/overall/monthly/profit`, plus third-party trackers `polymarketanalytics.com/traders`, `polycopy.app/best-polymarket-traders` (500K+ tracked wallets, daily updated), `predicting.top` (real-time, cross-platform with Kalshi).
- **Notable wallets**: "French Whale" (Théo) — ~$85M on 2024 US election; `HyperLiquid0xb` — $1.4M+ profit; `WindWalk3` — $1.1M+ mostly RFK Jr health policy markets; `1j59y6nk` — ~$1.4M sports/games.
- **Signal-mineability**: HIGH. Watch what categories specific whales concentrate in. If a whale who specializes in macro/political markets keeps loading the same side, that's an info-asymmetry signal. Cross-reference with their X/Twitter for thesis context.
- **Caveat**: Survivor bias is severe. Whales split across 4+ wallets to hide losses (Théo did). Leaderboard hides the graveyard. The right read is "if this *specific identified persona* keeps winning in *one specific category*, it's probably skill, not luck."
- **Originality**: 9 (very underexplored as a signal source)

### V-2 Manifold Markets — top-calibrated forecasters
- **What**: Manifold ranks users by *calibration* (% accuracy at each confidence bucket) not just profit. Better noise-controlled than Polymarket since traders are anonymous and don't have wallet-splitting incentive.
- **Where**: `manifold.markets/leaderboards`, `manifold.markets/calibration` (platform-wide cal chart).
- **Signal-mineability**: MEDIUM-HIGH. Top calibrators on macro/finance markets are usable as a "wisdom of the verified crowd" filter. Pull their public position history → use as Bayesian prior.
- **Originality**: 9

### V-3 Metaculus pro forecasters (ACX/Metaculus contest)
- **What**: $10K prize pool curated by Scott Alexander, ~600 questions, calibration scored by Brier.
- **Where**: `metaculus.com`, ACX contest archive at `astralcodexten.com`.
- **Signal-mineability**: MEDIUM. Better for "is X likely to happen by date Y" macro events than for tradable rules. Useful as an *event-flag service*: when top forecasters disagree sharply with a Polymarket consensus, the gap itself is the trade.
- **Originality**: 7

### V-4 Good Judgment Project Superforecasters
- **What**: Tetlock-vetted forecasters who outperformed CIA analysts by 30% in IARPA tournaments. The same outperformance held through 2024-25. *Feb 2025: FT reported they continue to beat futures markets on FOMC predictions.*
- **Where**: `goodjudgment.com` (mostly paywalled), but they publish to The Economist's "World Ahead" annual issue (2026 issue free-readable for: in-orbit Starship refueling odds, US midterms balance, US tariff trajectory, Russia-Ukraine, Nobel Peace Prize).
- **Signal-mineability**: HIGH for *what to bet against rates futures*. If GJP superforecasters disagree with Fed Funds futures on the next FOMC, fade the futures.
- **Originality**: 9 (an actual implementable rule: GJP-vs-Fed-Funds-futures basis trade)

### V-5 TipRanks / WallStreetZen top individual analysts
- **What**: Per-analyst hit rate and avg return on rated picks, computed automatically over years.
- **Notable analysts (2025 data)**:
  - **Sam Slutsky** (LifeSci Capital) — 67.7% success, 62.4% avg return — biotech/small-cap
  - **Gerard Cassidy** (RBC) — 88% success (91/103), 11.5% avg return — bank stocks
  - **Nick McKay** (Wedbush) — 88% success (49/56), **83.9% avg annual return**
- **Where**: `tipranks.com/experts/analysts`, `wallstreetzen.com/analysts`, Extel/Institutional Investor All-America rankings (Evercore ISI #1 for 4th straight year, 14 analysts in #1 sector slot).
- **Signal-mineability**: HIGH. Sleeve: copy buy ratings from a basket of top-5 verified analysts per sector, rebalance monthly. "Star analyst" effect is documented in literature (Nature 2023 paper).
- **Originality**: 7 (well-known concept, but per-analyst basket construction is uncommon)

---

## Tier 2 — Public Predictions, Loosely Verifiable (good signal candidates)

These don't have hard accuracy scores, but their public output is timestamped, archived,
and falsifiable. Useful as signal-source material once you do your own attribution.

### V-6 Annual prediction-review writers (calibrated by self-audit)
- **Scott Alexander / Astral Codex Ten** — annual ACX predictions with calibration audit. ~70%+ accuracy at his stated confidence buckets. Strong on AI, slowly broadening to macro.
  - Where: `astralcodexten.com`
- **Zvi Mowshowitz** — has evaluated ACX 2022, 2023 predictions; bought 30/lost 12 in 2023 (~71%). Monthly Roundup (now at #42 May 2026) has buried macro/policy signal. Slow but high-information.
  - Where: `thezvi.substack.com`
- **Matt Levine** (Bloomberg Money Stuff) — daily, free. Not formal predictions, but has called several SEC/derivatives stories years ahead (Robinhood gamification, AT1 wipeout, basis trade systemic risk). Mine for *event-class warnings*.
- **Tyler Cowen** (Marginal Revolution) — daily, free. Has a "this will be more important than people think" filter that's been historically prescient (AI 2018, GLP-1 2022, India 2020).
- **Dan Wang** annual letter — best China-watcher in English. Read 2024 and 2025 letters for *what's about to happen in Chinese industrial policy*.
- **Patrick Collison** annual blog post — tech/healthcare/state-capacity bets. Not market-tradable directly but excellent for trend-setting.
- **Originality**: 8 — these writers are read but not systematically mined for signals.

### V-7 Macro analysts with documented base-rate work
- **Lyn Alden** (`lynalden.com`) — fiscal-deficit-driven asset allocation. Premium newsletter. Called the 2022-23 reflation accurately. Currently long BTC for 2025. Already partially in catalog (P12 Steno overlaps). NOT YET MINED for her bank reserves / TGA models.
- **James Lavish** (Bitcoin Layer newsletter) — macro-into-bitcoin specialist. Called credit-cycle vs BTC rotations. Mineable for **Treasury General Account** as BTC predictor.
- **Jim Bianco** (Bianco Research) — bond-market specialist. Called 5%-on-10Y in 2023 against street consensus. Mineable for **dealer Treasury positioning** + **MOVE-vs-VIX divergence** rules.
- **Russell Napier** — deflation→inflation regime watcher; called the inflation regime change in 2021 (he wrote *Anatomy of the Bear*). Mineable for **bank-credit-growth-vs-CPI lag** rules.
- **David Rosenberg** (Rosenberg Research) — recession-call specialist. Mixed track record (early calls in 2022-24 wrong on timing) — but his composite recession-probability model is publicly described.
- **Felix Zulauf** (Zulauf Consulting) — secular-cycle specialist. Got the 2020 covid bottom direction right; mixed since. Mineable for his **inter-asset correlation regime model**.
- **Kyle Bass** — concentrated, occasional, but the calls he goes public on (Cyprus, JGB, China property) have been right more than chance. Lower update cadence.
- **Originality**: 6-9 depending on extracted rule

### V-8 Volatility / options specialists (small-shop hedge funds)
- **Christopher Cole** (Artemis Capital) — long-vol convex equity manager. *Allegory of the Hawk and the Serpent* paper has tradable framework. Mineable for **long-vol carry-neutralized basket construction**.
- **Kris Sidial** (Ambrus Group) — featured Hedge Fund Journal "Tomorrow's Titans 2024." "Carry-neutral" tail-risk strategy. Math-driven vol-skew specialist. Mineable for **skew dispersion** rules.
- **Benn Eifert** (QVR Advisors) — vol-arb derivatives. Tweets a lot of free real-time vol commentary. Mineable for **single-stock vs index vol arb** + **dealer hedging pressure** detection.
- **Cem Karsan** (Kai Volatility) — *already in catalog* (P1-P3, OPEX/vanna). Don't re-mine.
- **Originality**: 8 (vol-skew rules are systematically under-implemented retail-side)

### V-9 Quant-research shops that publish openly
- **Verdad Capital** (Dan Rasmussen) — weekly research email, free. Specializes in international microcap value, levered small-cap. Has a publicly archived call record. Mineable for **EV/EBIT international microcap basket** rules.
- **Empirical Research Partners** (Rochester Cahan) — institutional research, but excerpts leak. Strong on **breadth indicators**, **sector dispersion regime models**.
- **AQR Cliff's Perspectives** (`aqr.com/Insights`) — Cliff Asness writes long pieces with falsifiable claims. AQR's 2025 study of 196 dip-buying variants is now appendix-classic (F13 in your catalog cites it). Mineable for **value-momentum combination weights**.
- **Alpha Architect** (Wes Gray) — quantitative momentum + value books and Robust Asset Allocation. Free podcast + blog. Mineable for **momentum-with-volatility-overlay** rules.
- **ReSolve Asset Management** (Adam Butler, Rodrigo Gordillo, Corey Hoffstein) — Return Stacking + Robust Equity Momentum Index. Free podcast (Riffs). Mineable for **ensemble strategy construction**, **gold+momentum overlay** rules.
- **Newfound Research** (Corey Hoffstein) — three-axes-of-diversification framework. Mineable for **timing/luck attribution** sleeve.
- **Originality**: 7-8

### V-10 Crypto on-chain analysts (verifiable through blockchain itself)
- **Willy Woo** — *Bitcoin Forecast* paid newsletter, 1.1M followers. NVT, MVRV-like indicators his lineage. **Already in catalog** (H06 MVRV credits Glassnode but Woo invented the framework). Worth mining for his **realized-cap-vs-market-cap dispersion** + **macro liquidity overlays**.
- **Checkmate / James Check** (Glassnode lead) — weekly *The Week On-Chain* free reports. STH/LTH realized price, SOPR, MVRV bands all from his framework. **Mine for**: LTH supply-shock thresholds (when LTH supply hits N-year highs and reverses → tradable BTC long).
- **Murad Mahmudov** — has called multiple BTC cycle tops/bottoms publicly with timestamps. Specializes in memecoin supercycle thesis (mixed track record — right on 2020-21, wrong on 2024 timing).
- **PlanB** (Stock-to-Flow creator) — *was* the highest-profile quant. **S2F model failed 2022-25** — broken clock. Useful only as null-finding warning.
- **Mert Mumtaz** (Helius CEO) — Solana-native analyst with technical bona fides. Mineable for **on-chain throughput regime → SOL multiple** rules.
- **Justin Bons** — alt-L1 specialist with timestamped Twitter calls back to 2017. Mixed track record on token launches; better on protocol/governance directions.
- **Hyperliquid leaderboard whales** — `app.hyperliquid.xyz/leaderboard` — public perp PnL. Top traders ($1-10M+ all-time PnL) are tradable signal sources for **crypto perp basis** + **funding-rate divergence**.
- **Originality**: 7-9

### V-11 Forecaster *consortia* that publish track records
- **CXO Advisory** (defunct since 2021 but archive useful) — graded 60+ pundits 2005-2020. Found median accuracy ~47% (worse than coin flip). Useful inverse signal: identify the *worst*-rated pundits, fade them.
- **Hedgeye** — Keith McCullough's firm. Publishes daily "GIP model" macro quadrant calls. Subscription, but public calls on Twitter and YouTube are timestamped. Mineable for **quad-shift turning points** as regime overlay.
- **Bridgewater Daily Observations** — leaks to financial press regularly. Karen Karniol-Tambour has freely accessible interviews. Mineable for **inflation-vs-growth regime framework**.
- **Originality**: 8

---

## Tier 3 — Reputational / Anecdotal (low signal, but worth scanning)

Famous for a few good calls; track record is reputation, not data. Treat output as
*ideas to backtest*, not as forecasts to trust.

- **Michael Burry** — Big Short legend, "Cassandra B.C." Twitter prolific then deletes. Several wrong calls 2022-24 (he was early/wrong on tech short). His 13F is publicly tradable, but the famous calls (Big Short, BBBY) were *trade structure*, not stock-picking. Mineable for: **buying CDS on systemic-risk indices**.
- **Doomberg** — anonymous Substack on energy. Has been ahead of consensus on Ural discount, European gas crisis, copper supercycle. Substack only, paid.
- **Concoda** (Conor Sen) — Federal Reserve plumbing + bank reserves. Free Substack. Has called several FOMC pivot moments early. Mineable for **bank reserves-vs-MMF cycles**.
- **Peter Zeihan** — geopolitics. Loud, mixed track record. Strong on long-cycle demographics, wrong on near-term geopolitical timing. **Skip for trading**; useful for thesis cross-check only.
- **Raoul Pal** (Real Vision founder) — big-picture macro/crypto. "Banana Zone" framework. Mixed — was right on liquidity in 2020-21, wrong on aggressive crypto timing 2024.
- **The Felder Report** (Jesse Felder) — *already in catalog* (Margin Debt/GDP signal). Don't re-mine the same source.

---

## Tier 4 — Channels / Platforms to Watch (source aggregators, not individuals)

- **Excess Returns podcast** (Justin Carbonneau, Matt Zeigler, Jack Forehand) — interviews quant practitioners. Mike Green appeared (you have him in catalog), but ~200 other guests = pool of untapped rule-providers.
- **The Meb Faber Show** — Meb Faber interviews managers. 500+ episodes. Mineable for **named systematic strategies** that aren't well-known.
- **Top Traders Unplugged** (Niels Kaastrup-Larsen) — CTA-focused; Karsan ep 268 sourced your P1-P3 OPEX rules. ~700 episodes = signal pool.
- **MacroVoices** (Erik Townsend) — interviews macro fund managers; Howell, Gromen, Bianco all featured. Already mined for P5/P6, but ~500 episodes of untapped guests.
- **Forward Guidance** (Felix Jauvin / Blockworks) — bond/macro. Frequent Howell, Lavish.
- **Bankless** — crypto-policy focused. Lyn Alden's Trump/tariff macro pred episode is a representative pull.
- **The Investor's Podcast Network** — long-form macro mastermind episodes.
- **Hidden Forces** (Demetri Kofinas) — niche geopolitics/finance crossover. Excellent guest-quality, low coverage.
- **Capital Allocators** (Ted Seides) — interviews CIOs of endowments/family offices. ~400 episodes — mineable for **factor-allocation regime triggers**.
- **Two Quants and a Financial Planner** (Ben Felix, Cameron Passmore) — academic-quant grounded.
- **Originality**: 6-8 (well-known shows; mining for *unnamed* strategies inside episodes is the edge)

---

## Tier 5 — Niche / Underexplored

These are less-known but punch above their weight on specific domains:

- **Trade the Cycle / Brian Feroldi** — equity quality + cycle. Free YouTube + Substack.
- **Slime Mold Time Mold** — esoteric meta-science blog. Does annual reviews; called several public-health story turns early (obesity, lithium).
- **Dwarkesh Patel** podcast — does long interviews with researchers/policymakers. Higher-quality than typical podcaster.
- **Sebastian Mallaby books** — *More Money Than God*, *The Power Law*. Track-record review of fund managers; signal source for *what works in alternative assets*.
- **NGM Research / Niels Jensen** — boutique macro. Has called several rate-regime turns.
- **CrossBorder Capital** (Michael Howell — already P6) → also follow his *liquidity recession* model for **non-BTC equity allocation**.
- **Game of Trades (Twitter/X)** — chart-pattern person with timestamped public calls back to 2020. Mixed but better than median.
- **Andreas Steno Larsen** — *already in catalog* (P12). Don't re-mine.
- **13F sleuth services**: `whalewisdom.com`, `hedgefollow.com`, `dataroma.com` — track ALL 13F filers' changes. The *consistency-of-changes* signal (which fund keeps adding to which position over multiple quarters with positive returns) is mineable.
- **CFTC COT report persistent winners** — small specs vs large specs vs commercials. The *commercial-positioning vs specs* spread is classical but well-mined; the *new* angle is following *specific named non-commercial money-manager accounts* via swap-dealer category.
- **SEC Form 13F-HR** crowd-sourced trackers — Dataroma tracks superinvestors (Buffett, Klarman, Pabrai, Ackman). Cumulative-N-quarter adders.

---

## Caveats — before mining any of the above

1. **Survivor bias is the dominant force.** Anyone famous now had a few right calls. The bottom of the leaderboard exists. Always demand auditable timestamped output, not "they said it in 2020."
2. **Goalpost shifting / unfalsifiable predictions** — many macro figures bury timing. "Recession imminent" said quarterly for 4 years is not a track record. Filter for **specific date + specific magnitude + specific instrument**.
3. **"Broken clock" risk** — PlanB-S2F was right 2018-21 then wrong 2022+. The model was a calendar artifact, not signal. Demand cross-cycle persistence.
4. **Selection bias on social media** — losing forecasters delete tweets. Use Wayback Machine or third-party scraping (e.g., Polymarket third-party trackers store snapshots).
5. **Reflexivity** — once a signal becomes famous (Burry's CDS, Karsan's vanna), the alpha is already arbed. Best new sources are those *with* track records but *low* X follower counts.
6. **Always dedupe vs existing catalog** ([[feedback_dedup_signals]]): every signal extracted from these sources must be tagged NEW / DUPE / RELATED-VARIANT against the 314-row catalog before being passed to backtest.

---

## Top picks for the next hunt round

If picking ONE source to deep-dive next:
1. **Polymarket whale wallet analysis** — V-1. Identify 5-10 wallets with $500K+ PnL concentrated in *one* category (politics, macro, crypto); reverse-engineer their predictive edge.
2. **Hyperliquid top-perp-PnL leaderboard** — V-10 sub-bullet. Track crypto-native traders with timestamped public position history.
3. **GJP-vs-Fed-Funds-futures basis** — V-4. Concrete tradable rule: when superforecaster median materially disagrees with Fed Funds futures on next FOMC, fade futures.
4. **Per-analyst basket from TipRanks top-5/sector** — V-5. Concrete tradable: buy-rating-replication portfolio with monthly rebalance, scored by 12m forward return.
5. **Glassnode / Checkmate LTH supply-shock thresholds** — V-10. When LTH supply hits N-year extremes and reverses → mean-reversion BTC long.

---

# Process notes
- Total people/sources scouted: ~60+ distinct names.
- Skip list (already in catalog): Karsan, Green, Gromen, Howell, Constan, Johnson, Cowen, Lee, Steno Larsen, Felder.
- Next research file naming: 01w_* for whatever round comes after this.
