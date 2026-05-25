# Phase 1AE — AI / Data-Center Power Demand Signals
# 5 signals. All NEW.

### AE-1 PJM Base Residual Auction Clearing Price Spike ★
- **Rule**: PJM BRA clearing price increases >100% vs prior → long CEG/VST/NRG/PSEG, short NEE/SO; 90d hold.
- **Data**: pjm.com/markets-and-operations/rpm (Excel archives since 2007) + yfinance
- **Mechanism**: BRA sets capacity revenues 3yr forward; spike signals tightening reserves from DC load; generators see guaranteed revenue uplift not yet in models.
- **CAGR**: 25-60% in-position; Originality 9

### AE-2 Hyperscaler Capex Guidance Ratchet → Power-Infra Lag ★
- **Rule**: 2+ of {MSFT,GOOG,AMZN,META} guide capex UP >20% YoY same earnings season → long {ETN,VRT,PWR,HUBB,AMSC,MOD} 90d.
- **Data**: 10-Q filings (EDGAR) + yfinance; fires ~4x/yr
- **Mechanism**: Hyperscaler capex → equipment orders lag 1-2 quarters; supply chain stocks lag because sell-side updates on their own earnings cycle.
- **CAGR**: 20-45% in-position; Originality 8

### AE-3 NRC Pre-Application Activity → Nuclear Basket △
- **Rule**: New NRC ADAMS doc mentioning "data center"/"colocation"/hyperscaler → long CEG/VST/CCJ/SMR/OKLO/LEU 120d.
- **Data**: nrc.gov/reading-rm/adams.html keyword search + yfinance
- **CAGR**: 30-80% (CEG +100% in 12mo from TMI restart); N limited pre-2023; Originality 9

### AE-4 State PUC Data-Center Rate-Case Filing ✗
- **Rule**: Utility files rate-base addition citing >200MW DC load → long that utility 180d.
- **Data**: VA SCC, OH PUC, TX PUC dockets (free but multi-state formats)
- **CAGR**: 15-30%; Originality 8; HARD — fragmented state data

### AE-5 FERC Pipeline Certificate in DC Corridors ✗
- **Rule**: FERC CP certificate targeting county with >500MW DC load → long midstream operator 60d.
- **Data**: elibrary.ferc.gov + DC geographic cross-reference
- **CAGR**: 15-35%; Originality 8; HARD — cross-referencing two data sources
