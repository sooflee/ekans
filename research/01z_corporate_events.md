# Phase 1Z — Corporate-Event Database Signals
# 10 signals (DOJ, SEC enforcement, PACER, State AG, NLRB, CFPB). Dedup: all NEW.

### Z-1 DPA-Filing Two-Leg Drift
- **Rule**: Short on DOJ DPA press release day, hold 180 trading days; then flip long 120 days.
- **Asset**: Single-name (parent ticker)
- **Data**: justice.gov/news + EDGAR CIK mapping
- **Mechanism**: Monitor costs + capex restrictions; reverts once 1st annual monitor report clears
- **CAGR**: 14-18%; Originality 9

### Z-2 Compliance-Monitor Overhang Pair
- **Rule**: Pair: short single-name vs long GICS sub-industry ETF on DOJ monitor appointment; hold monitor term (3-5y).
- **Data**: DOJ corporate-enforcement-policy + agreement Schedule C parse
- **CAGR**: 10-12%; Originality 10

### Z-3 Whistleblower-Bounty Tape Reading
- **Rule**: When SEC OWB posts > $10M award (issuer redacted), short candidate set screened from prior-quarter 10-Q "loss contingencies" matching the redacted fact pattern; 90d.
- **Asset**: Basket short of 3-5 candidate tickers
- **Data**: sec.gov/about/offices/owb/owb-awards + EDGAR full-text
- **CAGR**: 12-15%; Originality 10 (forensic reverse-engineering)

### Z-4 Wells Notice → Litigation Release Gap
- **Rule**: Long issuer between Wells Notice (in 10-Q) and SEC litigation release; exit on litigation release.
- **Data**: EDGAR "Wells Notice" full-text + sec.gov/litigation/litreleases
- **CAGR**: 10-14%; Originality 8

### Z-5 Stanford SCAC First-Filed Drift
- **Rule**: Short defendant T+1 of first-filed securities class action; cover after 60 trading days.
- **Data**: securities.stanford.edu/filings.html
- **Mechanism**: Coffee-Klausner — institutional risk officers force position cuts before day-60 lead-plaintiff selection
- **CAGR**: 11-14%; Originality 8

### Z-6 Antitrust Complaint Reversion
- **Rule**: Long defendant on close of DOJ/FTC antitrust complaint filing; exit T+90.
- **Data**: justice.gov/atr + ftc.gov/legal-library
- **Mechanism**: Government wins <50% in federal antitrust over last decade; sell-off overprices loss prob
- **CAGR**: 10-13%; Originality 8

### Z-7 Merger-Litigation Deal-Break Predictor
- **Rule**: When pending US deal accumulates > 4 federal merger-objection complaints within 30 days of S-4 filing, short target until deal close/term.
- **Data**: PACER RECAP via courtlistener.com (free) + EDGAR S-4
- **CAGR**: 12-16%; Originality 9

### Z-8 Multi-State AG Coalition Filing
- **Rule**: Short company when ≥ 10 state AGs jointly file/sign on a complaint; 60d hold.
- **Data**: naag.org/policy-letters + state AG press feeds
- **CAGR**: 10-13%; Originality 9

### Z-9 NLRB Section 8(a) Complaint
- **Rule**: Short S&P 1500 employer on NLRB Regional Director formal Section 8(a) COMPLAINT (not mere charge); 90d.
- **Data**: nlrb.gov/cases-decisions filtered by status = "Complaint and Notice of Hearing"
- **CAGR**: 10-12%; Originality 10

### Z-10 CFPB Consent-Order Bank Pair
- **Rule**: Pair: short named issuer + long KBE on CFPB consent order/civil penalty post; 120d.
- **Data**: consumerfinance.gov/enforcement/actions
- **CAGR**: 10-13%; Originality 8
