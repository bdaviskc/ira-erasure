#!/usr/bin/env python3
"""
Full Macroeconomic Impact Model — IRA vs. OBBBA
IRA Erasure Project, 2025

Extends the sectoral investment model (model.py) with a complete
macroeconomic framework:

  PART I  — Leontief Input-Output Model
              7 productive sectors + households (Type II)
              A matrix calibrated to BEA 2022 industry accounts
              Computed using pymrio.calc_L (Leontief inverse)
              Outputs: total economic output, GDP, sector-level employment

  PART II — Electricity Price Feedback
              Capacity gap (GW) → wholesale price premium (merit-order)
              Price premium → economy-wide electricity cost increase
              GDP drag from diverted expenditure + competitiveness loss

  PART III — Government Fiscal Accounting
              Tax credit expenditures (from model_output.json)
              Tax revenues: income, payroll, corporate, sales
              Net fiscal position by scenario
              Credit-to-investment efficiency ratio

References:
  BEA (2023) Industry Economic Accounts / Use Tables (2022)
  Leontief (1941) The Structure of American Economy
  BEA RIMS II User's Handbook (2023)
  Bivens, EPI (2019) "Updated Employment Multipliers for the U.S. Economy"
  Garrett-Peltier (2017) "Green versus Brown"
  Baker, Bloom & Davis (2016) Economic Policy Uncertainty Index
  Zandi, Moody's Analytics (2011) fiscal multiplier estimates
"""

import json
import numpy as np
import pandas as pd
import pymrio

# ============================================================
# LOAD SECTORAL MODEL OUTPUTS
# ============================================================
with open('model_output.json') as f:
    base = json.load(f)

YEARS  = base['ira']['years']
N_YRS  = len(YEARS)
SCENS  = ['ira', 'obbba']

# ============================================================
# PART I — INPUT-OUTPUT MODEL
# ============================================================
"""
Sector definitions (indices 0-6 = productive, index 7 = households):

  0  Clean energy installation  (solar/wind/storage construction & install)
  1  Clean manufacturing        (EVs, batteries, turbines, panels — factory)
  2  Primary materials          (steel, aluminum, copper, cement, composites)
  3  Electronics & components   (inverters, controls, chips, wiring)
  4  Professional services      (engineering, finance, legal, R&D)
  5  Transportation & logistics (freight, supply chain, site logistics)
  6  All other industries       (food, retail, healthcare, housing, energy, etc.)
  7  Households                 [extended sector for Type II induced effects]

Calibration:
  A[i,j] = value purchased from sector i per $1 of sector j's gross output
  Source: BEA 2022 Summary Use Table, NAICS-level I-O accounts
    Construction (NAICS 23): col 0
    Manufacturing (NAICS 31-33, clean-energy-intensive mix): col 1
    Primary metals + nonmetallic minerals (NAICS 331-327): col 2
    Electrical equipment + computers (NAICS 334-335): col 3
    Professional services (NAICS 54): col 4
    Transportation + warehousing (NAICS 48-49): col 5
    All other (aggregated): col 6

Value-added shares (VA = 1 - sum of column):
  Construction:   0.47   Clean mfg:      0.38   Materials:   0.53
  Electronics:    0.49   Prof svcs:      0.73   Transport:   0.58
  Other:          0.65   Households:     n/a (final demand sector)

Import shares per sector (fraction of indirect purchases that leak abroad):
  Calibrated to BEA Import Matrix 2022; higher for electronics (chips)
"""

SECTORS = [
    'clean_install',   # 0
    'clean_mfg',       # 1
    'materials',       # 2
    'electronics',     # 3
    'prof_svcs',       # 4
    'transport',       # 5
    'other',           # 6
    'households',      # 7
]
N = len(SECTORS)

A = np.zeros((N, N))

# ── Column 0: Clean energy installation (construction-like) ──────────────────
A[2, 0] = 0.14   # primary metals & composites (structural steel, aluminum, copper wire)
A[3, 0] = 0.07   # electronics (panels, inverters, turbine controls)
A[4, 0] = 0.09   # professional services (engineering, permitting, project finance)
A[5, 0] = 0.06   # transportation (equipment delivery, crew logistics)
A[6, 0] = 0.17   # other (utilities on site, insurance, fuel, minor supplies)
# VA = 0.47 | total intermediate = 0.53

# ── Column 1: Clean manufacturing (EVs, batteries, turbines, panels — factory)
A[2, 1] = 0.15   # primary metals (battery casings, structural, copper/aluminum bus bars)
A[3, 1] = 0.18   # electronics (battery cells, chips, motors, control systems)
A[4, 1] = 0.08   # professional services (R&D, patent licensing, finance, legal)
A[5, 1] = 0.05   # transportation (inbound parts, outbound finished goods)
A[6, 1] = 0.16   # other (chemicals, plastics, utilities, indirect labor)
# VA = 0.38 | total intermediate = 0.62

# ── Column 2: Primary materials (steel, aluminum, cement, copper, rare earths)
A[2, 2] = 0.12   # self-use (scrap metal, recycled inputs, alloys)
A[3, 2] = 0.02   # electronics (process control systems)
A[4, 2] = 0.04   # professional services (testing, compliance, engineering)
A[5, 2] = 0.07   # transportation (ore/scrap inbound, finished metal outbound)
A[6, 2] = 0.22   # other (energy-intensive: electricity, natural gas, mining inputs)
# VA = 0.53 | total intermediate = 0.47

# ── Column 3: Electronics & components (inverters, chips, transformers, cable)
A[2, 3] = 0.08   # primary metals (copper, rare earths, silicon)
A[3, 3] = 0.14   # self-use (subcomponents, circuit boards from circuit boards)
A[4, 3] = 0.10   # professional services (software, engineering, IP licensing)
A[5, 3] = 0.04   # transportation
A[6, 3] = 0.15   # other (chemicals, utilities, plastics)
# VA = 0.49 | total intermediate = 0.51

# ── Column 4: Professional & business services (engineering, legal, finance)
A[2, 4] = 0.01   # materials (office supplies)
A[3, 4] = 0.03   # electronics (computers, software, servers)
A[4, 4] = 0.06   # self-use (accountants hire lawyers, consultants hire subs)
A[5, 4] = 0.02   # transportation (business travel, delivery)
A[6, 4] = 0.15   # other (office space, utilities, food services)
# VA = 0.73 | total intermediate = 0.27

# ── Column 5: Transportation & logistics (freight, warehousing, delivery)
A[2, 5] = 0.05   # primary metals (vehicle parts, containers)
A[3, 5] = 0.02   # electronics (navigation, telematics)
A[4, 5] = 0.04   # professional services (insurance, dispatch, finance)
A[5, 5] = 0.03   # self-use (transport buys transport services)
A[6, 5] = 0.28   # other (fuel is largest input; also utilities, maintenance)
# VA = 0.58 | total intermediate = 0.42

# ── Column 6: All other industries (aggregated: retail, healthcare, utilities)
A[2, 6] = 0.04
A[3, 6] = 0.02
A[4, 6] = 0.07
A[5, 6] = 0.04
A[6, 6] = 0.18   # self-use within the aggregated "other" block
# VA = 0.65 | total intermediate = 0.35

# ── Column 7: Households (final demand sector for Type II induced effects) ────
# What households spend per $1 of household income (= MPC × domestic content)
#   MPC (marginal propensity to consume): 0.72  (CBO, Zandi estimates)
#   Import share on consumption: 0.15 (BEA; some leakage on goods consumption)
#   Savings rate adjustment: 0.08 (households save a portion)
#   Effective domestic spending propensity: 0.72 × (1 - 0.15) × (1 - 0.08) = 0.563
#   Allocation to domestic sectors (must sum to 0.563):
A[0, 7] = 0.017  # construction: home improvement, some new housing
A[1, 7] = 0.011  # clean mfg: consumer EVs, appliances
A[2, 7] = 0.005  # materials
A[3, 7] = 0.038  # electronics: phones, computers, TVs
A[4, 7] = 0.045  # professional: healthcare, education, finance, legal
A[5, 7] = 0.039  # transportation: vehicle operating costs, transit
A[6, 7] = 0.408  # other: food, clothing, retail, housing, entertainment
# Sum ≈ 0.563 (= MPC × (1-imports) × (1-savings))

# ── Household income row (row 7) ──────────────────────────────────────────────
# What each sector pays households per $1 of its output
# = VA share × labor share of VA × (1 - retained_earnings_fraction)
# Retained earnings / capital depreciation reduce the portion flowing to disposable income
# Estimated retained/depreciation share: ~20% of VA, reducing household income coefficient
A[7, 0] = 0.47 * 0.62 * 0.80   # = 0.233  construction
A[7, 1] = 0.38 * 0.55 * 0.80   # = 0.167  clean manufacturing
A[7, 2] = 0.53 * 0.50 * 0.80   # = 0.212  primary materials
A[7, 3] = 0.49 * 0.57 * 0.80   # = 0.224  electronics
A[7, 4] = 0.73 * 0.78 * 0.80   # = 0.456  professional services
A[7, 5] = 0.58 * 0.65 * 0.80   # = 0.302  transportation
A[7, 6] = 0.65 * 0.70 * 0.80   # = 0.364  other industries
A[7, 7] = 0.0                    # households don't pay households

# ── Compute Leontief inverse using pymrio.calc_L ─────────────────────────────
A_df = pd.DataFrame(A, index=SECTORS, columns=SECTORS)
L_df = pymrio.calc_L(A_df)
L    = L_df.values

# ── Validate: verify Type I and Type II multipliers are in published range ───
type1_multipliers = {}
type2_multipliers = {}
A_type1 = A.copy()
A_type1[7, :] = 0   # suppress household row for Type I
A_type1[:, 7] = 0   # suppress household column for Type I
L1 = np.linalg.inv(np.eye(N) - A_type1)

for j, s in enumerate(SECTORS[:7]):
    type1_multipliers[s] = L1[:, j].sum()
    type2_multipliers[s] = L[:, j].sum()

print("INPUT-OUTPUT MULTIPLIER VALIDATION")
print("─" * 55)
print(f"{'Sector':<22} {'Type I':>8} {'Type II':>8}  {'Expected range'}")
print("─" * 55)
expected = {
    'clean_install': '1.7–2.1',
    'clean_mfg':     '1.9–2.5',
    'materials':     '1.5–1.9',
    'electronics':   '1.7–2.1',
    'prof_svcs':     '1.2–1.6',
    'transport':     '1.5–1.9',
    'other':         '1.3–1.6',
}
for s in SECTORS[:7]:
    print(f"  {s:<20} {type1_multipliers[s]:>8.3f} {type2_multipliers[s]:>8.3f}  {expected[s]}")

# ── Employment & wage coefficients (per $M of sector output, 2023$) ───────────
# Source: BEA 2022 QCEW, DOE Employment Survey, RIMS II documentation
# These are direct-sector coefficients (the I-O multiplier handles total effects)
employment_coeff = {   # jobs per $M direct output
    'clean_install': 5.8,
    'clean_mfg':     4.2,
    'materials':     3.5,
    'electronics':   3.8,
    'prof_svcs':     8.5,
    'transport':     7.2,
    'other':         6.8,
    'households':    0.0,
}
avg_wage = {           # average annual wage $K (2023$, all job types in sector)
    'clean_install': 63,
    'clean_mfg':     74,
    'materials':     68,
    'electronics':   88,
    'prof_svcs':     92,
    'transport':     58,
    'other':         52,
    'households':    0,
}

# Convert to vectors for matrix operations
e = np.array([employment_coeff[s] for s in SECTORS])  # jobs per $M
w = np.array([avg_wage[s] for s in SECTORS])            # avg wage $K

# Total employment per $M of final demand (full I-O effect):
# ê = e @ L  (where e is a row vector of direct employment coefficients)
# ê[j] = total jobs generated per $M of final demand in sector j
e_total = e @ L   # shape (N,)

# ── Map clean energy investment to I-O final demand vectors ─────────────────
# Each $1B of clean energy investment is split across I-O sectors as final demand.
# The split reflects where the investment dollar lands in the supply chain:
#   Installation (solar/wind/storage/grid) → primarily "clean_install" final demand
#   Manufacturing (EVs, batteries, panels) → primarily "clean_mfg" final demand
#
# Within each, some investment goes directly to intermediate sectors
# (e.g., panel procurement = electronics sector demand)
# We use a simplified mapping: 100% to the primary sector, let I-O propagate

INVESTMENT_MAPPING = {
    # Map each clean energy investment sector to a SINGLE I-O final demand sector.
    # The I-O model (via L = (I-A)^{-1}) handles all downstream supply chain purchases
    # internally. Splitting demand across multiple sectors would double-count: e.g., putting
    # solar investment into both 'clean_install' AND 'electronics' would cause electronics
    # to be stimulated twice — once via direct demand, and again via construction's internal
    # purchases of electronics (already encoded in A[3,0] = 0.07).
    'solar':     {'clean_install': 1.0},
    'wind':      {'clean_install': 1.0},
    # Storage: roughly split between installation and domestic battery/component manufacturing
    'storage':   {'clean_install': 0.55, 'clean_mfg': 0.45},
    'clean_mfg': {'clean_mfg': 1.0},
    'grid':      {'clean_install': 1.0},
}

SECTOR_IDX = {s: i for i, s in enumerate(SECTORS)}

def investment_to_demand_vector(model_sector: str, investment_B: float) -> np.ndarray:
    """Convert $B of investment in a model sector to an I-O final demand vector."""
    f = np.zeros(N)
    for io_sector, share in INVESTMENT_MAPPING[model_sector].items():
        f[SECTOR_IDX[io_sector]] += investment_B * 1000  # convert $B to $M
    return f

# ── Import leakage by sector (fraction of supply chain that goes to imports) ──
IMPORT_LEAKAGE = {
    'clean_install': 0.20,   # some panels/components from abroad
    'clean_mfg':     0.22,   # battery cells still partly imported
    'materials':     0.08,   # mostly domestic (US steel, concrete)
    'electronics':   0.32,   # significant semiconductor imports
    'prof_svcs':     0.04,   # almost entirely domestic
    'transport':     0.06,
    'other':         0.10,
    'households':    0.12,
}
import_vec = np.array([IMPORT_LEAKAGE[s] for s in SECTORS])

# ── Run I-O computation for each scenario ────────────────────────────────────
def run_io_model(scenario_key: str) -> dict:
    data = base[scenario_key]
    MODEL_SECTORS = ['solar', 'wind', 'storage', 'clean_mfg', 'grid']

    results = {
        'gdp_B':                  [],   # GDP ($B): value added across all waves
        'total_output_B':         [],   # Total economy-wide output ($B)
        'total_jobs_all_waves':   [],   # All jobs (direct + indirect + induced)
        'wage_income_B':          [],   # Total wage bill ($B)
        'sector_output_B':        {s: [] for s in SECTORS[:7]},
    }

    for i in range(N_YRS):
        # Aggregate demand vector across all clean energy sectors this year
        f_total = np.zeros(N)
        for ms in MODEL_SECTORS:
            inv = data['sector_investment'][ms][i]  # $B
            f_total += investment_to_demand_vector(ms, inv)

        # Total output vector (all sectors): x = L @ f
        x_total = L @ f_total   # $M, shape (N,)

        # GDP = value added = (I - A) @ x, summed across sectors
        # Equivalently: GDP = sum over sectors of (VA_share × x)
        va_shares = np.array([
            0.47, 0.38, 0.53, 0.49, 0.73, 0.58, 0.65, 1.0
        ])
        gdp_vec    = va_shares * x_total           # $M per sector
        total_gdp  = gdp_vec[:7].sum()             # households sector excluded (avoid double-count)

        # Apply import leakage (reduce to domestic GDP only)
        # Import leakage reduces the indirect/induced layers
        direct_output = f_total.copy()
        indirect_induced = x_total - direct_output
        domestic_x = direct_output + indirect_induced * (1 - import_vec)
        domestic_gdp = (va_shares * domestic_x)[:7].sum()

        # Total employment (all waves): e_total[j] × f[j] for each demand sector
        # This gives jobs generated per unit of final demand, propagated through L
        total_jobs = int((e_total[:7] @ f_total[:7]).item())

        # Wage income: $M
        # Uses sector-level average wages applied to employment per sector
        wages_M = sum(
            int((e_total[j] * f_total[j]).item()) * w[j]
            for j in range(7)
        ) / 1000  # $K * jobs / 1000 = $M

        results['gdp_B'].append(round(domestic_gdp / 1000, 1))        # $M → $B
        results['total_output_B'].append(round(domestic_x[:7].sum() / 1000, 1))
        results['total_jobs_all_waves'].append(total_jobs)
        results['wage_income_B'].append(round(wages_M / 1000, 1))     # $M → $B

        for j, s in enumerate(SECTORS[:7]):
            results['sector_output_B'][s].append(round(domestic_x[j] / 1000, 1))

    return results

print("\n")
ira_io   = run_io_model('ira')
obbba_io = run_io_model('obbba')

# ============================================================
# PART II — ELECTRICITY PRICE FEEDBACK
# ============================================================
"""
Mechanism:
  Clean energy capacity gap (OBBBA vs. IRA) → fewer zero-marginal-cost MWh
  → more expensive natural gas peakers on the margin → higher wholesale prices

Merit-order effect calibration:
  Academic studies (Cludius et al. 2014; MacCormack et al. 2010; Hull et al. 2019):
  US context: ~100 GW of additional renewable capacity reduces average
  wholesale electricity price by $3–5/MWh (at current market penetration)
  Using $4.0/MWh per 100 GW (conservative mid-range)

Retail pass-through: ~40% (fixed transmission/distribution costs dampen
  wholesale-to-retail transmission; from FERC studies)

US electricity consumption: ~4,000 TWh/yr (EIA 2023, growing ~0.5%/yr)

GDP drag: electricity cost premium creates a redistribution from productive
  sectors to fossil fuel producers. Net GDP drag ≈ 35% of total cost premium
  (accounts for the ~65% that stays in the domestic economy as fossil fuel
  producer income, but represents a reallocation from manufacturing/services
  to extractive sector with lower employment multiplier and export leakage)
"""
MERIT_ORDER_RATE     = 4.0   # $/MWh per 100 GW (conservative)
RETAIL_PASSTHROUGH   = 0.40
ELEC_CONSUMPTION_TWH = 4000  # TWh/yr (2024 base, grows slightly)
ELEC_CONSUMPTION_GR  = 0.005 # 0.5%/yr growth
NET_GDP_DRAG_RATE    = 0.35  # fraction of cost premium that is net GDP drag

ira_cumcap   = base['ira']['cumulative_capacity_gw']
obbba_cumcap = base['obbba']['cumulative_capacity_gw']

elec_feedback = {
    'capacity_gap_gw':    [],
    'wholesale_premium':  [],   # $/MWh
    'retail_premium':     [],   # $/MWh
    'total_cost_premium': [],   # $B/yr
    'net_gdp_drag':       [],   # $B/yr
    'household_premium':  [],   # $/yr per household (assuming 130M HH, 18 MWh/yr each)
    'industrial_premium': [],   # $B/yr (commercial + industrial = 2,400 TWh/yr)
}

for i, year in enumerate(YEARS):
    cap_gap       = max(0, ira_cumcap[i] - obbba_cumcap[i])
    ws_premium    = cap_gap / 100 * MERIT_ORDER_RATE
    rt_premium    = ws_premium * RETAIL_PASSTHROUGH
    consumption   = ELEC_CONSUMPTION_TWH * (1 + ELEC_CONSUMPTION_GR) ** (year - 2024)
    total_premium = rt_premium * consumption / 1000      # $/MWh × TWh / 1000 = $B
    gdp_drag      = total_premium * NET_GDP_DRAG_RATE
    # Household share (~38% of total consumption = 1,500 TWh / 4,000 TWh)
    # Average US household uses ~18 MWh/yr (EIA)
    hh_premium    = rt_premium * 18   # $/MWh × 18 MWh/yr = $/yr per HH
    # Industrial + commercial premium (remaining ~62%)
    ind_premium   = total_premium * 0.62

    elec_feedback['capacity_gap_gw'].append(round(cap_gap, 1))
    elec_feedback['wholesale_premium'].append(round(ws_premium, 2))
    elec_feedback['retail_premium'].append(round(rt_premium, 2))
    elec_feedback['total_cost_premium'].append(round(total_premium, 1))
    elec_feedback['net_gdp_drag'].append(round(gdp_drag, 1))
    elec_feedback['household_premium'].append(round(hh_premium, 0))
    elec_feedback['industrial_premium'].append(round(ind_premium, 1))

# ============================================================
# PART III — GOVERNMENT FISCAL ACCOUNTING
# ============================================================
"""
Tax revenues generated by economic activity under each scenario.

Rates (US federal + state/local composite):
  Federal income tax (effective rate, blended all brackets): 18%
  State/local income: 4.5%
  Payroll taxes (employer + employee, capped):              12%  of wages
  Corporate income (effective, incl. pass-through):         13%  of gross profits
    (Profits estimated at ~15% of total investment for capital returns)
  Sales/excise on consumer spending from induced income:     3%  of induced GDP
  Property tax on new infrastructure (annualized):          0.8% of new capacity value
    ($1M/MW installed × 0.8% × GW additions)

Total effective fiscal feedback rate on GDP generated: ~21%
(Lower than nominal because not all GDP is taxable income at those rates;
 capital depreciation, retained earnings, and government transfers reduce the base)
"""

FED_INCOME_RATE     = 0.180
STATE_INCOME_RATE   = 0.045
PAYROLL_RATE        = 0.120   # on wages
CORPORATE_RATE      = 0.130   # on profits
PROFIT_MARGIN       = 0.150   # assumed profit as % of investment
SALES_RATE          = 0.030   # on consumption-driven (induced) GDP
PROPERTY_RATE       = 0.008   # on new installed value
PROPERTY_MW_VALUE_M = 1.0     # $M per MW → $1B per GW

def fiscal_accounting(scenario_key: str, io_results: dict) -> dict:
    data     = base[scenario_key]
    results  = {
        'tax_credits_B':        [],   # credit expenditure
        'income_tax_B':         [],   # income taxes (fed + state/local)
        'payroll_tax_B':        [],   # payroll taxes
        'corporate_tax_B':      [],   # corporate/business income tax
        'sales_tax_B':          [],   # consumption-driven sales taxes
        'property_tax_B':       [],   # property taxes on new infrastructure
        'total_tax_revenue_B':  [],   # sum of all taxes
        'net_fiscal_B':         [],   # revenues - credits
        'cumulative_net_B':     [],   # running total net fiscal
    }
    cum_net = 0.0

    for i in range(N_YRS):
        credits   = data['tax_credit_expenditure'][i]
        wages_B   = io_results['wage_income_B'][i]
        gdp_B     = io_results['gdp_B'][i]
        inv_total = data['total_investment'][i]
        cap_add   = data['capacity_additions_gw'][i]

        income_tax   = wages_B * (FED_INCOME_RATE + STATE_INCOME_RATE)
        payroll_tax  = wages_B * PAYROLL_RATE
        corp_tax     = inv_total * PROFIT_MARGIN * CORPORATE_RATE
        # Sales tax applies to the consumption/induced portion (~30% of GDP)
        induced_gdp  = gdp_B * 0.30
        sales_tax    = induced_gdp * SALES_RATE
        # Property tax: annualized tax on new infrastructure value
        property_tax = cap_add * PROPERTY_MW_VALUE_M * PROPERTY_RATE  # GW × $B/GW × rate

        total_rev = income_tax + payroll_tax + corp_tax + sales_tax + property_tax
        net_fiscal = total_rev - credits
        cum_net   += net_fiscal

        results['tax_credits_B'].append(round(credits, 1))
        results['income_tax_B'].append(round(income_tax, 1))
        results['payroll_tax_B'].append(round(payroll_tax, 1))
        results['corporate_tax_B'].append(round(corp_tax, 1))
        results['sales_tax_B'].append(round(sales_tax, 1))
        results['property_tax_B'].append(round(property_tax, 1))
        results['total_tax_revenue_B'].append(round(total_rev, 1))
        results['net_fiscal_B'].append(round(net_fiscal, 1))
        results['cumulative_net_B'].append(round(cum_net, 1))

    return results

ira_fiscal   = fiscal_accounting('ira',   ira_io)
obbba_fiscal = fiscal_accounting('obbba', obbba_io)

# ============================================================
# COMBINED OUTPUTS & PRINT SUMMARY
# ============================================================

# Incorporate GDP drag from electricity prices into net GDP
ira_gdp_net   = ira_io['gdp_B']   # No electricity drag in IRA scenario
obbba_gdp_net = [obbba_io['gdp_B'][i] - elec_feedback['net_gdp_drag'][i]
                 for i in range(N_YRS)]

print("\n" + "=" * 90)
print("FULL MACROECONOMIC MODEL — IRA vs. OBBBA")
print("=" * 90)
print(f"\n{'Year':<6} {'IRA GDP($B)':>12} {'OBBBA GDP':>12} {'Δ GDP':>10} "
      f"{'IRA Jobs(M)':>12} {'OBBBA Jobs':>12} {'Elec Drag($B)':>14}")
print("-" * 90)
for i, y in enumerate(YEARS):
    d_gdp  = ira_gdp_net[i] - obbba_gdp_net[i]
    print(f"{y:<6} "
          f"${ira_gdp_net[i]:>10.0f}B "
          f"${obbba_gdp_net[i]:>10.0f}B "
          f"${d_gdp:>8.0f}B "
          f"{ira_io['total_jobs_all_waves'][i]/1e6:>12.2f}M "
          f"{obbba_io['total_jobs_all_waves'][i]/1e6:>12.2f}M "
          f"${elec_feedback['net_gdp_drag'][i]:>12.1f}B")

ira_10y_gdp     = sum(ira_gdp_net)
obbba_10y_gdp   = sum(obbba_gdp_net)
ira_10y_jobs    = sum(ira_io['total_jobs_all_waves'])
obbba_10y_jobs  = sum(obbba_io['total_jobs_all_waves'])
ira_10y_wages   = sum(ira_io['wage_income_B'])
obbba_10y_wages = sum(obbba_io['wage_income_B'])
total_elec_drag = sum(elec_feedback['net_gdp_drag'])

print("\n" + "=" * 90)
print("CUMULATIVE 2024–2035 MACROECONOMIC SUMMARY")
print("=" * 90)
print(f"{'Metric':<38} {'IRA':>18} {'OBBBA':>18} {'Δ (lost under OBBBA)':>20}")
print("-" * 90)
print(f"{'Total Economy-Wide GDP ($B)':38} ${ira_10y_gdp:>16.0f} ${obbba_10y_gdp:>16.0f}  −${ira_10y_gdp - obbba_10y_gdp:.0f}B")
print(f"{'Total Jobs, All Waves (M job-yrs)':38} {ira_10y_jobs/1e6:>17.1f} {obbba_10y_jobs/1e6:>17.1f}  −{(ira_10y_jobs-obbba_10y_jobs)/1e6:.1f}M")
print(f"{'Total Wage Income ($B)':38} ${ira_10y_wages:>16.0f} ${obbba_10y_wages:>16.0f}  −${ira_10y_wages - obbba_10y_wages:.0f}B")
print(f"{'Electricity GDP Drag (OBBBA only)':38} {'—':>18} ${total_elec_drag:>16.1f}B  — (additional drag)")

print("\n" + "=" * 90)
print("FISCAL ACCOUNTING (2024–2035 CUMULATIVE)")
print("=" * 90)
ira_tot_rev   = sum(ira_fiscal['total_tax_revenue_B'])
obbba_tot_rev = sum(obbba_fiscal['total_tax_revenue_B'])
ira_tot_cred  = sum(ira_fiscal['tax_credits_B'])
obbba_tot_cred = sum(obbba_fiscal['tax_credits_B'])
ira_net       = ira_fiscal['cumulative_net_B'][-1]
obbba_net     = obbba_fiscal['cumulative_net_B'][-1]

print(f"{'Metric':<38} {'IRA':>18} {'OBBBA':>18}")
print("-" * 75)
print(f"{'Tax Credit Expenditures ($B)':38} ${ira_tot_cred:>16.0f} ${obbba_tot_cred:>16.0f}")
print(f"{'Total Tax Revenues Generated ($B)':38} ${ira_tot_rev:>16.0f} ${obbba_tot_rev:>16.0f}")
print(f"{'Net Fiscal Position ($B)':38} ${ira_net:>+16.0f} ${obbba_net:>+16.0f}")
print()
credit_per_gdp_ira   = ira_tot_cred   / (ira_10y_gdp   + 1e-6)
credit_per_gdp_obbba = obbba_tot_cred / (obbba_10y_gdp + 1e-6)
inv_mult_ira   = sum(base['ira']['total_investment'])   / (ira_tot_cred   + 1e-6)
inv_mult_obbba = sum(base['obbba']['total_investment']) / (obbba_tot_cred + 1e-6)
print(f"  IRA credit efficiency:   $1 of credit generates ${inv_mult_ira:.2f} in private investment + ${ira_10y_gdp/ira_tot_cred:.2f} in total GDP")
print(f"  OBBBA credit efficiency: $1 of credit generates ${inv_mult_obbba:.2f} in private investment + ${obbba_10y_gdp/obbba_tot_cred:.2f} in total GDP")

print("\n" + "=" * 90)
print("2030 ELECTRICITY PRICE FEEDBACK SNAPSHOT")
print("=" * 90)
idx30 = YEARS.index(2030)
print(f"  Capacity gap vs IRA:           {elec_feedback['capacity_gap_gw'][idx30]:.0f} GW")
print(f"  Wholesale price premium:       +${elec_feedback['wholesale_premium'][idx30]:.1f}/MWh")
print(f"  Retail price premium:          +${elec_feedback['retail_premium'][idx30]:.2f}/MWh")
print(f"  Total economy cost premium:    +${elec_feedback['total_cost_premium'][idx30]:.1f}B/yr")
print(f"  Net GDP drag:                  −${elec_feedback['net_gdp_drag'][idx30]:.1f}B/yr")
print(f"  Average household premium:     +${elec_feedback['household_premium'][idx30]:.0f}/yr")
print(f"  Industrial + commercial:       +${elec_feedback['industrial_premium'][idx30]:.1f}B/yr")

# ============================================================
# SAVE JSON OUTPUT
# ============================================================
output = {
    'metadata': {
        'model': 'IRA Erasure Full Macroeconomic Model v1.0',
        'methodology': [
            'Leontief I-O: 8-sector extended A matrix (7 productive + households)',
            'A matrix calibrated to BEA 2022 summary Use tables',
            'Leontief inverse computed via pymrio.calc_L',
            'Type II effects: A matrix extended with household income/spending rows',
            'Import leakage applied to indirect/induced output',
            'Employment: sector direct coefficients propagated through L',
            'Electricity: merit-order effect at $4/MWh per 100 GW, 40% retail pass-through',
            'Fiscal: income, payroll, corporate, sales, and property taxes modeled',
        ]
    },
    'years': YEARS,
    'ira_io':   ira_io,
    'obbba_io': obbba_io,
    'electricity_feedback': elec_feedback,
    'ira_fiscal':   ira_fiscal,
    'obbba_fiscal': obbba_fiscal,
    'summary': {
        'gdp_lost_B':              round(ira_10y_gdp - obbba_10y_gdp),
        'jobs_lost_M':             round((ira_10y_jobs - obbba_10y_jobs) / 1e6, 1),
        'wages_lost_B':            round(ira_10y_wages - obbba_10y_wages),
        'elec_drag_total_B':       round(total_elec_drag),
        'ira_net_fiscal_B':        round(ira_net),
        'obbba_net_fiscal_B':      round(obbba_net),
        'ira_inv_per_credit':      round(inv_mult_ira, 2),
        'obbba_inv_per_credit':    round(inv_mult_obbba, 2),
        'ira_gdp_per_credit':      round(ira_10y_gdp / ira_tot_cred, 2),
        'obbba_gdp_per_credit':    round(obbba_10y_gdp / obbba_tot_cred, 2),
    },
    'io_multipliers': {
        'type1': type1_multipliers,
        'type2': {s: float(type2_multipliers[s]) for s in SECTORS[:7]},
    }
}

with open('macro_output.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nSaved: macro_output.json")
