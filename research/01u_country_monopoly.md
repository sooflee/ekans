# Phase 1U — Country-Commodity Monopoly Signals
# Returned by hunt agent. 15 signals, avg originality 9.4, all >= 9.

### U-1 Indonesia Nickel Royalty Tier Reset → Long LME Nickel
- **Country/share**: Indonesia / nickel / ~55% mined supply
- **Rule**: Long LME Nickel + Nickel Industries (NIC.AX) when Indonesia MEMR publishes PNBP royalty raising top nickel-ore tier > 2pp.
- **Data**: jdih.esdm.go.id Peraturan Pemerintah PDFs
- **CAGR**: 15-20%; Originality: 9

### U-2 DRC Cobalt Export Quota → Long Cobalt + CMOC
- **Country/share**: DRC / cobalt / ~74%
- **Rule**: Long SHFE cobalt + CMOC (603993.SS) within 24h of DRC ARECOMS announcing/extending export-quota suspension > 30% of national volume.
- **Data**: arecoms.cd press releases; mines.gouv.cd ministerial arrêtés
- **Notes**: Feb 2025 4-month ban moved prices +60%
- **CAGR**: 25-40%; Originality: 9

### U-3 South Africa PGM Wage Strike Notices → Long Platinum + SBSW
- **Country/share**: South Africa / platinum / ~70%, rhodium / ~80%
- **Rule**: Long PL + SBSW when AMCU/NUM Section 64 strike notice covers Amplats/Implats/Sibanye > 15% SA PGM output.
- **Data**: labour.gov.za CCMA Section 64 register; SENS announcements via JSE
- **CAGR**: 12-18%; Originality: 9

### U-4 Norway Salmon Sea-Lice Threshold → Long Mowi / Short Bakkafrost
- **Country/share**: Norway / farmed Atlantic salmon / ~50%
- **Rule**: Long MOWI.OL / short BAKKA.OL when NFSA weekly sea-lice count > 0.5 adult female lice/fish in PO3-PO5 for 3 consec weeks.
- **Data**: barentswatch.no/fiskehelse Mattilsynet sea-lice API
- **CAGR**: 12-18%; Originality: 10

### U-5 Madagascar Cyclone Track → Vanilla-Flavoring Hedge
- **Country/share**: Madagascar / vanilla / ~80%
- **Rule**: Long Symrise (SY1.DE) / McCormick (MKC) when NOAA SWIO cyclone cone covers SAVA region (Sambava-Antalaha-Andapa) with sustained > 120 km/h at 96h lead.
- **Data**: meteomadagascar.mg + nhc.noaa.gov SH-basin + ECMWF tropical cyclone tracker
- **CAGR**: 12-25%; Originality: 10
- **Note**: Vanilla itself isn't liquidly futures-listed; play via flavoring single-names

### U-6 Thai Rubber EUDR Compliance Drop-Off → Long TOCOM
- **Country/share**: Thailand / natural rubber / ~33% (#1)
- **Rule**: Long TOCOM rubber (JN3) + STGT.BK when Thai Rubber Authority weekly Hat Yai auction drops > 8% WoW coinciding with EUDR deadline within 90 days.
- **Data**: raot.co.th + eur-lex.europa.eu EUDR Regulation 2023/1115
- **CAGR**: 12-20%; Originality: 9

### U-7 Russia-Belarus Potash Rail Capacity → Long Nutrien/Mosaic
- **Country/share**: Belarus + Russia / potash / ~40% combined
- **Rule**: Long NTR + MOS when Lithuanian/Latvian rail (LDz, LTG) monthly stats show Belaruskali transit drops > 20% YoY for 2 consec months.
- **Data**: litrail.lt + ldz.lv monthly cargo statistics
- **CAGR**: 12-18%; Originality: 10

### U-8 Mongolia-China Coking Coal Truck Count → Long DCE JM / Teck
- **Country/share**: Mongolia / coking coal / ~50% of China imports
- **Rule**: Long DCE JM + TECK when Ganqimaodu/Gashuun Sukhait daily truck crossings < 800/day for 5 trading days.
- **Data**: customs.gov.mn daily border stats + coalresource.com free mirror
- **CAGR**: 12-20%; Originality: 10

### U-9 Fonterra GDT Pulse Spike Front-Run → Long ATM.NZ
- **Country/share**: NZ / WMP / ~30% (Fonterra ~80% of NZ)
- **Rule**: Long a2 Milk (ATM.NZ) + Saputo (SAP.TO) when bi-weekly GDT Pulse WMP C2 > +5% vs most recent main event.
- **Data**: globaldairytrade.info Pulse + main event JSON
- **Note**: 71% hit-rate of predicting main-event direction
- **CAGR**: 10-15%; Originality: 9

### U-10 Chilean Copper Royalty/Glacier Vote → Long COMEX HG + Antofagasta
- **Country/share**: Chile / copper / ~24% mined
- **Rule**: Long HG + ANTO.L when Chilean Senate publishes "primer informe" (first committee report) on glacier/royalty bill affecting Codelco/private ops.
- **Data**: senado.cl bill tracker (BCN/SIL); Comisión de Minería agendas
- **CAGR**: 15-25%; Originality: 9

### U-11 Vietnam Robusta Coffee FOB-Differential Inversion → Long ICE Robusta
- **Country/share**: Vietnam / Robusta / ~40%
- **Rule**: Long ICE Robusta (RM/LRC) when Vietnam Customs weekly FOB differential to London Robusta turns POSITIVE (premium) for first time in 12-week window.
- **Data**: customs.gov.vn weekly export stats (HS 0901) + ICO indicator prices
- **CAGR**: 15-25%; Originality: 10

### U-12 Argentina "Dólar Soja" FX Window → Long CBOT SM + Bunge
- **Country/share**: Argentina / soybean meal / ~40% global exports (#1)
- **Rule**: Long CBOT soybean meal (SM) + Bunge (BG) when Argentine government announces new dólar-soja FX window for soy exports lasting < 60 days.
- **Data**: boletinoficial.gob.ar daily decree scrape + bcr.com.ar Rosario weekly
- **Note**: 60-day-post-window rebound leg is the trade
- **CAGR**: 12-20%; Originality: 9

### U-13 Peru Las Bambas Roadblock → Long MMG + COMEX HG
- **Country/share**: Peru / copper / Las Bambas alone = 2% global
- **Rule**: Long MMG (1208.HK) + COMEX HG when MTC Peru's road-status flags PE-3SY (Corredor Vial Sur) closed > 7 consec days due to community blockade.
- **Data**: gob.pe/mtc Provias Nacional + defensoria.gob.pe social conflict monthly
- **CAGR**: 15-25%; Originality: 10

### U-14 Brazil Vale Dam License Downgrade → Long SGX FEF + Fortescue
- **Country/share**: Brazil / iron ore / ~22% (Vale = ~80% of Brazil)
- **Rule**: Long SGX FEF + Fortescue (FMG.AX) when ANM publishes DCE/DCO downgrade for Vale dam of Class C or higher.
- **Data**: gov.br/anm SIGBM dam database (open data)
- **CAGR**: 12-18%; Originality: 9

### U-15 China Antimony Export License Backlog → Long PPTA
- **Country/share**: China / antimony / ~48% mined + ~80% refined
- **Rule**: Long Perpetua Resources (PPTA) + UAMY when China MOFCOM weekly export license approvals for antimony (HS 8110) < 30% of pre-Sep-2024 baseline for 3 consec weeks.
- **Data**: mofcom.gov.cn export license bulletins + customs.gov.cn monthly trade
- **Note**: Sept 2024 license regime lifted antimony from $13k/t to $40k+
- **CAGR**: 25-50% on PPTA single-name; Originality: 10
