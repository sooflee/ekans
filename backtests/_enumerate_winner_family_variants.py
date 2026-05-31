"""Generate new pipeline strategies by enumerating FRED series in families
where we've already found winners. Each new strategy mimics the winning pattern.

Adds new strategy entries to pipeline/strategies_queue.json with status='ready'.
The batch backtester (_generate_and_run_pl.py) handles bad series IDs gracefully.

Template per family:
- PPI commodity: WPU* trough recovery -> long materials/industrials
- CPI subindex: CUSR0000S* trough -> long XLY/XLP
- JOLTS: industry openings surge -> long sector ETF
- Manufacturers orders: A*SNO surge -> long XLI/ITA
- Commodity prices: P*USDM spike -> long buyer-side
- Bank credit: senior loan officer / delinquency cycle -> long banks
"""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / 'pipeline' / 'strategies_queue.json'

# Curated candidate FRED series IDs by family.
# Includes both winners' siblings and known FRED series IDs.
# Bad IDs will fail gracefully in the batch backtester.
CANDIDATES = {
    'PPI commodity (WPU)': [
        # Farm products
        ('WPS011', 'PPI: Farm Products'),
        ('WPU012', 'PPI: Cereal/Bakery'),
        ('WPU013', 'PPI: Fruits/Vegetables'),
        ('WPU014', 'PPI: Sugar'),
        ('WPU015', 'PPI: Oilseeds'),
        ('WPU016', 'PPI: Hay/Seeds'),
        # Processed foods
        ('WPU021', 'PPI: Meats'),
        ('WPU023', 'PPI: Dairy Products'),
        ('WPU024', 'PPI: Processed Fruits/Vegetables'),
        ('WPU026', 'PPI: Beverages'),
        # Textiles/leather
        ('WPU031', 'PPI: Synthetic Fibers'),
        ('WPU033', 'PPI: Apparel'),
        ('WPU041', 'PPI: Hides/Leather'),
        # Fuels
        ('WPU051', 'PPI: Coal'),
        ('WPU055', 'PPI: Gas Fuels'),
        ('WPU057', 'PPI: Petroleum Products Refined'),
        # Chemicals
        ('WPU061', 'PPI: Industrial Chemicals'),
        ('WPU063', 'PPI: Drugs/Pharmaceuticals'),
        ('WPU065', 'PPI: Plastic Resins'),
        # Rubber/plastics
        ('WPU071', 'PPI: Rubber/Plastic Products'),
        # Lumber
        ('WPU081', 'PPI: Lumber'),
        ('WPU082', 'PPI: Millwork'),
        ('WPU083', 'PPI: Plywood'),
        # Pulp/paper
        ('WPU0911', 'PPI: Pulp/Paper Materials'),
        ('WPU0912', 'PPI: Paper'),
        ('WPU0913', 'PPI: Paperboard'),
        # Metals
        ('WPU101', 'PPI: Iron/Steel'),
        ('WPU1017', 'PPI: Steel Mill Products'),
        ('WPU102', 'PPI: Nonferrous Metals'),
        ('WPU1025', 'PPI: Copper'),
        ('WPU1026', 'PPI: Aluminum'),
        # Machinery
        ('WPU111', 'PPI: Agricultural Machinery'),
        ('WPU112', 'PPI: Construction Machinery'),
        ('WPU114', 'PPI: General Purpose Machinery'),
        ('WPU116', 'PPI: Special Industry Machinery'),
        ('WPU117', 'PPI: Electrical Machinery'),
        # Furniture
        ('WPU121', 'PPI: Household Furniture'),
        ('WPU122', 'PPI: Commercial Furniture'),
        ('WPU124', 'PPI: Household Appliances'),
        # Non-metallic minerals
        ('WPU132', 'PPI: Concrete'),
        ('WPU133', 'PPI: Clay'),
        ('WPU134', 'PPI: Glass'),
        # Transportation
        ('WPU141', 'PPI: Motor Vehicles'),
        ('WPU144', 'PPI: Railroad Equipment'),
        # Service-only PPI siblings
        ('PCUOMINMIN', 'PPI: Mining'),
        ('PCUAUNS', 'PPI: All Industries'),
    ],
    'CPI subindex (CUSR0000)': [
        # Food
        ('CUSR0000SAF11', 'CPI: Food at Home'),
        ('CUSR0000SAF111', 'CPI: Cereals/Bakery'),
        ('CUSR0000SAF112', 'CPI: Meats/Poultry/Fish/Eggs'),
        ('CUSR0000SAF113', 'CPI: Dairy'),
        ('CUSR0000SAF114', 'CPI: Fruits/Vegetables'),
        ('CUSR0000SAF115', 'CPI: Nonalcoholic Beverages'),
        ('CUSR0000SAF116', 'CPI: Other Food at Home'),
        # Housing
        ('CUSR0000SAH1', 'CPI: Shelter'),
        ('CUSR0000SAH21', 'CPI: Household Energy'),
        ('CUSR0000SAH3', 'CPI: Household Furnishings'),
        # Medical
        ('CUSR0000SAM1', 'CPI: Medical Care Commodities'),
        # Services
        ('CUSR0000SAS2', 'CPI: Shelter (Services)'),
        ('CUSR0000SAS4', 'CPI: Transportation Services'),
        # Transportation
        ('CUSR0000SETA01', 'CPI: New Vehicles'),
        ('CUSR0000SETB', 'CPI: Motor Fuel'),
        ('CUSR0000SETB01', 'CPI: Gasoline'),
        ('CUSR0000SETD', 'CPI: Motor Vehicle Maintenance/Repair'),
        ('CUSR0000SETE', 'CPI: Motor Vehicle Insurance'),
        ('CUSR0000SETG01', 'CPI: Airline Fares'),
        ('CUSR0000SETG02', 'CPI: Other Intercity Transportation'),
        # Education/communication
        ('CUSR0000SEEA', 'CPI: Education'),
        ('CUSR0000SEEB', 'CPI: Communication'),
        # Recreation
        ('CUSR0000SEAA', 'CPI: Video Audio'),
        ('CUSR0000SEAB', 'CPI: Pets/Pet Products'),
        ('CUSR0000SEAC', 'CPI: Sporting Goods'),
        ('CUSR0000SEAD', 'CPI: Photo Equipment'),
    ],
    'JOLTS by industry': [
        # Openings (JOL)
        ('JTS1000JOL', 'JOLTS: Total Nonfarm Openings'),
        ('JTS2300JOL', 'JOLTS: Construction Openings'),
        ('JTS3000JOL', 'JOLTS: Manufacturing Openings'),
        ('JTS3300JOL', 'JOLTS: Durable Goods Mfg Openings'),
        ('JTS4000JOL', 'JOLTS: Trade/Trans/Util Openings'),
        ('JTS4400JOL', 'JOLTS: Retail Trade Openings'),
        ('JTS4500JOL', 'JOLTS: Wholesale Trade Openings'),
        ('JTS5100JOL', 'JOLTS: Information Openings'),
        ('JTS5500JOL', 'JOLTS: Finance/Insurance Openings'),
        ('JTS6000JOL', 'JOLTS: Education Services Openings'),
        ('JTS6500JOL', 'JOLTS: Health Care Openings'),
        ('JTS7000JOL', 'JOLTS: Leisure/Hospitality Openings'),
        ('JTS7200JOL', 'JOLTS: Accommodation/Food Openings'),
        # Quits (QUR/QUL)
        ('JTS1000QUR', 'JOLTS: Total Quits Rate'),
        ('JTS3000QUR', 'JOLTS: Manufacturing Quits Rate'),
        ('JTS4400QUR', 'JOLTS: Retail Quits Rate'),
        ('JTS6500QUR', 'JOLTS: Health Care Quits Rate'),
        # Layoffs (LDL)
        ('JTS1000LDL', 'JOLTS: Total Layoffs/Discharges'),
        ('JTS3000LDL', 'JOLTS: Manufacturing Layoffs'),
        ('JTS5100LDL', 'JOLTS: Information Layoffs'),
        # Hires (HIL)
        ('JTS1000HIL', 'JOLTS: Total Hires'),
        ('JTS3000HIL', 'JOLTS: Manufacturing Hires'),
    ],
    'Manufacturers orders': [
        ('A33SNO', 'Mfg: Durable Goods Primary Metals New Orders'),
        ('A36SNO', 'Mfg: Electrical Equip New Orders'),
        ('A37SNO', 'Mfg: Transportation Equipment New Orders'),
        ('A37BNO', 'Mfg: Motor Vehicles New Orders'),
        ('A33SUO', 'Mfg: Primary Metals Unfilled Orders'),
        ('A34SUO', 'Mfg: Machinery Unfilled Orders'),
        ('A36SUO', 'Mfg: Electrical Equip Unfilled Orders'),
        ('A37SUO', 'Mfg: Transportation Equipment Unfilled Orders'),
        ('AMTMNO', 'Total Manufacturing New Orders'),
        ('AMTMUO', 'Total Manufacturing Unfilled Orders'),
        ('AMTMVS', 'Total Manufacturing Shipments'),
        ('A311SVS', 'Mfg: Food Shipments'),
        ('A31SVS', 'Mfg: Wood Shipments'),
        ('A34SVS', 'Mfg: Machinery Shipments'),
        ('DGORDER', 'Durable Goods New Orders'),
        ('ACOGNO', 'Computers/Electronics New Orders'),
        ('ANXAVS', 'Nondefense Ex-Aircraft Capital Goods Shipments'),
    ],
    'Commodity prices (monthly)': [
        ('PWHEAMTUSDM', 'Global Wheat Price'),
        ('PRICENPQUSDM', 'Global Rice Price'),
        ('POATSUSDM', 'Global Oats Price'),
        ('PCOPPUSDM', 'Global Copper Price'),
        ('PLEADUSDM', 'Global Lead Price'),
        ('PALUMUSDM', 'Global Aluminum Price'),
        ('PIORECRUSDM', 'Global Iron Ore Price'),
        ('PTINUSDM', 'Global Tin Price'),
        ('PURANUSDM', 'Global Uranium Price'),
        ('POLVOILUSDM', 'Global Olive Oil Price'),
        ('PPALMUSDM', 'Global Palm Oil Price'),
        ('PSUNOUSDM', 'Global Sunflower Oil Price'),
        ('PRAWMUSDM', 'Global Raw Materials Price'),
        ('PRUBBUSDM', 'Global Rubber Price'),
        ('PWOOLCUSDM', 'Global Coarse Wool Price'),
        ('PORANGEUSDM', 'Global Orange Price'),
        ('PCOCOAUSDM', 'Global Cocoa Price'),
        ('PTEAUSDM', 'Global Tea Price'),
        ('PNGASUSUSDM', 'US Natural Gas Price'),
        ('PNGASEUUSDM', 'EU Natural Gas Price'),
        ('PNGASJPUSDM', 'Japan Natural Gas Price'),
        ('POILBREUSDM', 'Brent Crude Price'),
        ('POILDUBUSDM', 'Dubai Crude Price'),
        ('POILWTIUSDM', 'WTI Crude Price'),
        ('PCOALAUUSDM', 'Australian Coal Price'),
        ('PCOALSAUSDM', 'South African Coal Price'),
        ('PSALMUSDM', 'Global Salmon Price'),
        ('PSUGAISAUSDM', 'Sugar (ISA) Price'),
        ('PSUGAUSAUSDM', 'Sugar (USA) Price'),
    ],
    'Bank credit/delinquency': [
        ('DRBLACBS', 'Delinquency Rate: Business Loans'),
        ('DRTSCILM', 'SLOOS: Tightening Standards C&I Large Mid Firms'),
        ('DRTSCIS', 'SLOOS: Tightening Standards C&I Small Firms'),
        ('DRTSCIBS', 'SLOOS: Tightening Standards Consumer Credit Card'),
        ('SUBLPDCISC', 'SLOOS: Demand C&I Loans Small Firms'),
        ('SUBLPDCNSC', 'SLOOS: Demand Consumer Loans'),
        ('DRISCFLCC', 'SLOOS: Stronger Demand Consumer Credit Card'),
        ('DRISCFLCONS', 'SLOOS: Stronger Demand Consumer Loans'),
        ('TOTLL', 'Total Loans/Leases All Commercial Banks'),
        ('TOTBKCR', 'Total Bank Credit'),
        ('LOANS', 'Loans/Leases All Commercial Banks'),
        ('REALLN', 'Real Estate Loans'),
        ('CONSUMER', 'Consumer Loans Commercial Banks'),
        ('CCLACBM027SBOG', 'Consumer Loans: Credit Cards'),
        ('NONREVSL', 'Total Nonrevolving Credit'),
        ('REVOLSL', 'Total Revolving Credit'),
    ],
}

# Each family also gets a template for the strategy rule
FAMILY_TEMPLATES = {
    'PPI commodity (WPU)': {
        'tickers': ['XLB', 'XLI', 'IYM', 'SPY'],
        'horizon': '4-8 weeks',
        'rule_tpl': "Load FRED {sid} ({title}, monthly). When 3-month change inflects positive after 6+ months of decline (trough recovery), long XLB+XLI+IYM for 42 trading days. Benchmark vs SPY.",
        'category': 'B',
    },
    'CPI subindex (CUSR0000)': {
        'tickers': ['XLY', 'XLP', 'SPY'],
        'horizon': '4-8 weeks',
        'rule_tpl': "Load FRED {sid} ({title}, monthly). When YoY change troughs after 2+ months of deflation, long XLY+XLP for 42 trading days. Benchmark vs SPY.",
        'category': 'C',
    },
    'JOLTS by industry': {
        'tickers': ['XLI', 'XLY', 'XLV', 'XLF', 'SPY'],
        'horizon': '4-12 weeks',
        'rule_tpl': "Load FRED {sid} ({title}, monthly). When YoY change exceeds +20% vs overall JOLTS flat or declining, long XLI+XLY+XLV+XLF for 60 trading days. Benchmark vs SPY.",
        'category': 'C',
    },
    'Manufacturers orders': {
        'tickers': ['XLI', 'ITA', 'XLB', 'SPY'],
        'horizon': '4-8 weeks',
        'rule_tpl': "Load FRED {sid} ({title}, monthly). When value surges >2 standard deviations above 12-month rolling mean, long XLI+ITA+XLB for 42 trading days. Benchmark vs SPY.",
        'category': 'B',
    },
    'Commodity prices (monthly)': {
        'tickers': ['XLE', 'XLB', 'DBA', 'SPY'],
        'horizon': '4-12 weeks',
        'rule_tpl': "Load FRED {sid} ({title}, monthly). When 3-month return exceeds +30% (commodity spike), wait 10 trading days, then long XLE+XLB+DBA for 42 trading days. Benchmark vs SPY.",
        'category': 'B',
    },
    'Bank credit/delinquency': {
        'tickers': ['KRE', 'XLF', 'KBE', 'SPY'],
        'horizon': '3-6 months',
        'rule_tpl': "Load FRED {sid} ({title}, monthly). When metric peaks (local maximum) after 2+ quarters of rising, long KRE+XLF+KBE for 126 trading days as credit cycle inflects. Benchmark vs SPY.",
        'category': 'C',
    },
}


def make_strategy(family, sid, title, strat_num):
    tpl = FAMILY_TEMPLATES[family]
    family_slug = family.split('(')[0].strip().lower().replace(' ', '_').replace('/', '_')
    signal_id = f"PL{strat_num}_{family_slug}_{sid.lower()}"[:120]
    return {
        'strategy_id': f"PL{strat_num:03d}",
        'idea_id': f"PL{strat_num:03d}",
        'signal_id': signal_id,
        'name': f"FRED {sid} {title} -> Family Variant",
        'rule': tpl['rule_tpl'].format(sid=sid, title=title),
        'tickers': tpl['tickers'],
        'horizon': tpl['horizon'],
        'category': tpl['category'],
        'asset_class': 'equities',
        'originality': 6,
        'bt_feasibility': 7,
        'data_sources_concrete': {
            'prices': f"yfinance {', '.join(tpl['tickers'])}",
            'fundamental': f"FRED {sid}",
        },
        'backtest_approach': f"Family enumeration variant: sibling of winners in {family}. Uses generic z-score-based trigger.",
        'implementation_notes': "Auto-generated from family-mining round. Multiple-comparisons risk applies — require IS+OOS Sharpe > 0.5 to count as a real winner.",
        'status': 'ready',
        'created_at': datetime.now(timezone.utc).isoformat(),
        '_family_enum': True,
        '_source_family': family,
    }


def main():
    with open(QUEUE) as f:
        queue = json.load(f)

    existing_ids = {s.get('strategy_id') for s in queue}
    existing_signal_ids = {s.get('signal_id') for s in queue}

    # Also dedup by FRED series ID in name
    existing_series_in_names = set()
    import re
    for s in queue:
        for txt in [s.get('name', ''), s.get('rule', '')]:
            for m in re.findall(r'FRED\s+([A-Z][A-Z0-9_]{2,})', txt):
                existing_series_in_names.add(m)

    max_pl = 0
    for s in queue:
        sid = s.get('strategy_id', '')
        if sid.startswith('PL') and sid[2:].isdigit():
            max_pl = max(max_pl, int(sid[2:]))
    print(f"Highest existing PL id: PL{max_pl:03d}")
    next_id = max(max_pl + 1, 600)

    added = []
    skipped_dupe = 0
    for family, candidates in CANDIDATES.items():
        for fred_id, title in candidates:
            if fred_id in existing_series_in_names:
                skipped_dupe += 1
                continue
            strat = make_strategy(family, fred_id, title, next_id)
            if strat['strategy_id'] in existing_ids or strat['signal_id'] in existing_signal_ids:
                skipped_dupe += 1
                continue
            queue.append(strat)
            added.append((strat['strategy_id'], fred_id, family))
            existing_ids.add(strat['strategy_id'])
            existing_signal_ids.add(strat['signal_id'])
            existing_series_in_names.add(fred_id)
            next_id += 1

    print(f"Added: {len(added)} new strategies, skipped {skipped_dupe} dupes")
    by_family = {}
    for sid, fid, fam in added:
        by_family.setdefault(fam, []).append(fid)
    for fam, ids in by_family.items():
        print(f"  {fam}: {len(ids)} new")

    with open(QUEUE, 'w') as f:
        json.dump(queue, f, indent=2, default=str)
    print(f"\nSaved to {QUEUE}")


if __name__ == '__main__':
    main()
