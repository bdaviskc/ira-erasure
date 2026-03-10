#!/usr/bin/env python3
"""
IRA vs. OBBBA Clean Energy Economy Macroeconomic Projection Model
IRA Erasure Project — Original Analysis, 2025

Methodology:
  Investment-based sectoral model with credit-rate elasticity, calibrated to
  2023 actuals. Produces annual projections 2024–2035 for two scenarios:
    (1) IRA as-passed (credits intact per original statutory schedule)
    (2) OBBBA enacted (accelerated phase-outs + policy uncertainty premium)

Key assumptions documented inline. All outputs in constant 2023 dollars.
"""

import json
import math

# ============================================================
# CALIBRATION ANCHORS (historical actuals)
# ============================================================
# Sources:
#   Investment: BloombergNEF Energy Transition Investment Trends 2024
#   Jobs: E2 Clean Jobs America 2024; DOE IRA Tracker
#   Capacity: EIA Electric Power Monthly; BNEF
ACTUALS = {
    2021: {'investment': 115, 'jobs': 3_200_000, 'capacity_gw': 322},
    2022: {'investment': 141, 'jobs': 3_400_000, 'capacity_gw': 350},
    2023: {'investment': 303, 'jobs': 3_500_000, 'capacity_gw': 388},
}

# ============================================================
# CREDIT RATE SCHEDULES
# ============================================================

def ira_credit(sector_id: str, year: int) -> float:
    """
    IRA as-enacted credit rates by sector.
    Clean electricity PTC (45Y) / ITC (48E): full rate through 2032,
    then statutory phase-down for projects NOT meeting domestic content.
    Rates expressed as fraction of project cost (ITC equivalent).

    Notes on conversions:
      PTC for wind: ~2.6¢/kWh (2024$) ≈ 26% of project cost at LCOE
      ITC for solar/storage: statutory 30%
      45X AMPC (advanced manufacturing): effective ~25–30% of project cost
    """
    FULL = {
        'solar':     0.30,  # ITC 48E
        'wind':      0.26,  # PTC 45Y, converted to ITC-equivalent
        'storage':   0.30,  # ITC 48E
        'clean_mfg': 0.28,  # 45X Advanced Manufacturing PTC, blended
        'grid':      0.06,  # 48C qualifying advanced energy project credit
    }
    base = FULL.get(sector_id, 0.0)
    if year <= 2032:
        return base
    # Phase-down schedule (per IRA Section 48E(b))
    phase = {2033: 0.75, 2034: 0.50, 2035: 0.25}
    return base * phase.get(year, 0.0)


def obbba_credit(sector_id: str, year: int) -> float:
    """
    OBBBA (One Big Beautiful Bill Act, 2025) accelerated phase-outs.

    Key changes modeled:
      Solar & Wind: construction-start deadline moved to end of 2025;
        projects beginning in 2026+ ineligible → credit effectively 0 for
        new projects by 2027. Existing pipeline partially captured 2026.
      Storage: 3-year step-down from 2026 (partial grandfathering of
        projects already in development).
      Clean Mfg (45X): Phase-down accelerated; 50% cut by 2027, sunset 2029.
      Grid (48C): Largely preserved; small phase-down from 2028.

    Note: FEOC restrictions also add ~5–8% compliance cost overhead to
    affected sectors (modeled as implicit credit reduction on affected
    portion of supply chain).
    """
    SCHEDULES = {
        'solar': {
            2024: 0.30, 2025: 0.30,
            2026: 0.10,  # Only projects with 2025 construction start
            2027: 0.02,  # Grandfathered tail
            2028: 0.00, 2029: 0.00, 2030: 0.00, 2031: 0.00, 2032: 0.00,
        },
        'wind': {
            2024: 0.26, 2025: 0.26,
            2026: 0.08,  # Only projects with 2025 construction start
            2027: 0.00,
            2028: 0.00, 2029: 0.00, 2030: 0.00, 2031: 0.00, 2032: 0.00,
        },
        'storage': {
            2024: 0.30, 2025: 0.30,
            2026: 0.22,
            2027: 0.12,
            2028: 0.04,
            2029: 0.00, 2030: 0.00, 2031: 0.00, 2032: 0.00,
        },
        'clean_mfg': {
            2024: 0.28, 2025: 0.24,
            2026: 0.18,
            2027: 0.12,
            2028: 0.06,
            2029: 0.02,
            2030: 0.00, 2031: 0.00, 2032: 0.00,
        },
        'grid': {
            2024: 0.06, 2025: 0.06, 2026: 0.06, 2027: 0.05,
            2028: 0.04, 2029: 0.02, 2030: 0.00, 2031: 0.00, 2032: 0.00,
        },
    }
    schedule = SCHEDULES.get(sector_id, {})
    if year in schedule:
        return schedule[year]
    if year > max(schedule.keys(), default=2024):
        return 0.0
    return ira_credit(sector_id, year)

# ============================================================
# SECTOR DEFINITIONS
# ============================================================
# Each sector has:
#   investment_2023: $B actual (sum calibrated to $303B)
#   organic_growth:  annual rate from technology cost decline + demand (IRA absent)
#   credit_elasticity: semi-elasticity — 1pp credit change → X% investment change
#     Literature range 0.3–0.7; higher for sectors with thin returns (wind/solar)
#     Sources: Gechert & Rannenberg (2018 meta-analysis); Zwick & Mahon (2017)
#   jobs_per_B:      job-years per $B invested (direct + 1st-tier indirect)
#     Source: NREL JEDI model, BLS QCEW, DOE jobs tracker
#   mfg_jobs_per_B:  manufacturing-specific job-years per $B
#   om_jobs_per_gw:  permanent O&M jobs added per GW of installed capacity
#   union_share:     fraction earning prevailing wage / in union contract
#     Sources: BLS union membership data; Princeton/MIT prevailing wage studies
#   gw_per_B:        GW installed per $B (0 for non-generation assets)

SECTORS = [
    {
        'id': 'solar',
        'name': 'Solar Power',
        'investment_2023': 120,
        'organic_growth': 0.055,   # 5.5%/yr: module cost decline + demand
        'credit_elasticity': 0.55,
        'jobs_per_B': 4800,
        'mfg_jobs_per_B': 950,
        'om_jobs_per_gw': 500,
        'union_share': 0.14,
        # gw_per_B calibrated to 2023 actuals: ~38 GW added / $120B total investment
        # (total investment includes supply chain + mfg, not just installation capex)
        'gw_per_B': 0.32,
    },
    {
        'id': 'wind',
        'name': 'Wind Power',
        'investment_2023': 58,
        'organic_growth': 0.040,
        'credit_elasticity': 0.60,  # Higher: thinner margins, very credit-sensitive
        'jobs_per_B': 4200,
        'mfg_jobs_per_B': 1400,
        'om_jobs_per_gw': 380,
        'union_share': 0.22,
        # Calibrated: ~8 GW added / $58B → 0.14 GW/B
        'gw_per_B': 0.14,
    },
    {
        'id': 'storage',
        'name': 'Battery Storage',
        'investment_2023': 45,
        'organic_growth': 0.130,    # 13%/yr: very rapid cost decline
        'credit_elasticity': 0.45,
        'jobs_per_B': 3500,
        'mfg_jobs_per_B': 2100,
        'om_jobs_per_gw': 280,
        'union_share': 0.13,
        # Calibrated: ~12 GW added / $45B → 0.27 GW/B
        'gw_per_B': 0.27,
    },
    {
        'id': 'clean_mfg',
        'name': 'Clean Manufacturing',
        'investment_2023': 55,
        'organic_growth': 0.090,    # 9%/yr: EV + battery domestic buildout
        'credit_elasticity': 0.50,
        'jobs_per_B': 3000,
        'mfg_jobs_per_B': 2600,
        'om_jobs_per_gw': 0,
        'union_share': 0.27,
        'gw_per_B': 0,
    },
    {
        'id': 'grid',
        'name': 'Grid Modernization',
        'investment_2023': 25,
        'organic_growth': 0.055,
        'credit_elasticity': 0.25,  # Lower: driven more by reliability need
        'jobs_per_B': 3800,
        'mfg_jobs_per_B': 700,
        'om_jobs_per_gw': 0,
        'union_share': 0.40,        # High: IBEW-organized utility work
        'gw_per_B': 0,
    },
]

# Verify calibration
assert sum(s['investment_2023'] for s in SECTORS) == 303, "Sector sum must equal 2023 actual"

# ============================================================
# POLICY UNCERTAINTY DISCOUNT
# ============================================================
# OBBBA's accelerated phase-outs suppress investment beyond the direct
# credit-rate effect. Project finance requires multi-year revenue certainty;
# uncertainty raises the cost of capital and delays FIDs.
#
# Source basis: Baker, Bloom & Davis (2016) Economic Policy Uncertainty
# index — a 1-SD EPU spike correlates with ~1.5–2% capex decline. OBBBA
# creates an estimated 0.5–1 SD shock to clean energy policy uncertainty.
# Modeled as multiplicative discount on organic growth component only.
UNCERTAINTY_DISCOUNT = {
    'ira':   {y: 1.00 for y in range(2024, 2036)},
    'obbba': {
        2024: 0.97,   # Pre-enactment uncertainty begins
        2025: 0.91,   # Enactment shock
        2026: 0.87,   # Peak uncertainty as credits phase out
        2027: 0.85,   # Trough: investors waiting for new policy
        2028: 0.87,
        2029: 0.90,
        2030: 0.93,
        2031: 0.95,
        2032: 0.97,
        2033: 0.98,
        2034: 0.99,
        2035: 1.00,
    },
}

# ============================================================
# INVESTMENT MODEL
# ============================================================
def project_investment(sector: dict, scenario: str, year: int,
                        prev_inv: float, prev_rate: float) -> float:
    """
    Annual investment projection.

    I(t) = [I(t-1) × (1 + g) + I(t-1) × ε × Δr] × u(t)

    Where:
      g   = organic growth rate
      ε   = credit elasticity
      Δr  = change in credit rate (current - prior year)
      u   = policy uncertainty discount

    The credit-change term captures the investment response to a step-change
    in subsidy. Investment anticipates announced credit changes (we assume
    investors form rational expectations one year ahead for the phase-out).
    """
    if scenario == 'ira':
        cur_rate = ira_credit(sector['id'], year)
        # One-year-ahead anticipation: look at next year's rate change
        next_rate = ira_credit(sector['id'], year + 1)
    else:
        cur_rate = obbba_credit(sector['id'], year)
        next_rate = obbba_credit(sector['id'], year + 1)

    g = sector['organic_growth']
    e = sector['credit_elasticity']
    u = UNCERTAINTY_DISCOUNT[scenario][year]

    # Organic component
    organic = prev_inv * (1 + g)

    # Credit effect: respond to current-year rate change from prior year
    delta_r = cur_rate - prev_rate
    credit_effect = prev_inv * e * delta_r

    # Anticipation effect: partial forward-looking adjustment (30% weight)
    # Investors begin pulling back or accelerating based on announced future rates
    next_delta = next_rate - cur_rate
    anticipation = prev_inv * e * next_delta * 0.30

    # Apply uncertainty discount to organic growth only (credit effect is direct)
    new_inv = (organic * u) + credit_effect + anticipation

    # Floor: investment doesn't collapse entirely (sunk capital, contracts)
    floor = prev_inv * 0.15
    return max(new_inv, floor)

# ============================================================
# SIMULATION
# ============================================================
def run_model(scenario: str, years: list) -> dict:
    results = {
        'scenario': scenario,
        'years': years,
        'total_investment': [],
        'sector_investment': {s['id']: [] for s in SECTORS},
        'new_jobs': [],
        'mfg_jobs': [],
        'union_jobs': [],
        'capacity_additions_gw': [],
        'cumulative_capacity_gw': [],
        'om_jobs_added': [],
        'total_employment': [],
        'tax_credit_expenditure': [],
    }

    # Initialize state from 2023 actuals
    state = {}
    for s in SECTORS:
        state[s['id']] = {
            'investment': float(s['investment_2023']),
            'credit_rate': ira_credit(s['id'], 2023),
        }

    cumulative_capacity_gw = 388.0   # GW installed through end-2023
    cumulative_om_jobs = 380_000     # Standing O&M workforce end-2023

    for year in years:
        year_inv = 0.0
        year_new_jobs = 0
        year_mfg_jobs = 0
        year_union_jobs = 0
        year_cap_add = 0.0
        year_om_add = 0
        year_tax_credit = 0.0

        for s in SECTORS:
            sid = s['id']
            prev_inv = state[sid]['investment']
            prev_rate = state[sid]['credit_rate']

            new_inv = project_investment(s, scenario, year, prev_inv, prev_rate)

            if scenario == 'ira':
                cur_rate = ira_credit(sid, year)
            else:
                cur_rate = obbba_credit(sid, year)

            state[sid]['investment'] = new_inv
            state[sid]['credit_rate'] = cur_rate

            # Derived outputs
            new_jobs   = int(new_inv * s['jobs_per_B'])
            mfg_jobs   = int(new_inv * s['mfg_jobs_per_B'])
            union_jobs = int(new_jobs * s['union_share'])
            cap_add    = new_inv * s['gw_per_B']
            om_add     = int(cap_add * s['om_jobs_per_gw'])
            tax_credit = new_inv * cur_rate

            year_inv        += new_inv
            year_new_jobs   += new_jobs
            year_mfg_jobs   += mfg_jobs
            year_union_jobs += union_jobs
            year_cap_add    += cap_add
            year_om_add     += om_add
            year_tax_credit += tax_credit

            results['sector_investment'][sid].append(round(new_inv, 1))

        cumulative_capacity_gw += year_cap_add
        cumulative_om_jobs     += year_om_add
        total_employment = year_new_jobs + cumulative_om_jobs

        results['total_investment'].append(round(year_inv, 1))
        results['new_jobs'].append(year_new_jobs)
        results['mfg_jobs'].append(year_mfg_jobs)
        results['union_jobs'].append(year_union_jobs)
        results['capacity_additions_gw'].append(round(year_cap_add, 1))
        results['cumulative_capacity_gw'].append(round(cumulative_capacity_gw, 1))
        results['om_jobs_added'].append(year_om_add)
        results['total_employment'].append(total_employment)
        results['tax_credit_expenditure'].append(round(year_tax_credit, 1))

    return results


# ============================================================
# RUN BOTH SCENARIOS
# ============================================================
YEARS = list(range(2024, 2036))

ira   = run_model('ira',   YEARS)
obbba = run_model('obbba', YEARS)

# ============================================================
# PRINT SUMMARY TABLE
# ============================================================
print("\nIRA vs. OBBBA MACROECONOMIC PROJECTION MODEL")
print("Original analysis — IRA Erasure Project, 2025")
print("=" * 92)
print(f"\n{'Year':<6} {'IRA Inv':>10} {'OBBBA':>10} {'Δ Inv':>10} "
      f"{'IRA Jobs':>12} {'OBBBA Jobs':>12} {'IRA GW':>8} {'OBBBA GW':>8}")
print("-" * 92)

for i, y in enumerate(YEARS):
    d_inv  = ira['total_investment'][i]  - obbba['total_investment'][i]
    print(f"{y:<6} "
          f"${ira['total_investment'][i]:>8.0f}B "
          f"${obbba['total_investment'][i]:>8.0f}B "
          f"-${d_inv:>7.0f}B "
          f"{ira['new_jobs'][i]:>12,} "
          f"{obbba['new_jobs'][i]:>12,} "
          f"{ira['capacity_additions_gw'][i]:>8.1f} "
          f"{obbba['capacity_additions_gw'][i]:>8.1f}")

# Cumulative 2024–2035
ira_total_inv    = sum(ira['total_investment'])
obbba_total_inv  = sum(obbba['total_investment'])
ira_total_jobs   = sum(ira['new_jobs'])
obbba_total_jobs = sum(obbba['new_jobs'])
ira_total_mfg    = sum(ira['mfg_jobs'])
obbba_total_mfg  = sum(obbba['mfg_jobs'])
ira_total_union  = sum(ira['union_jobs'])
obbba_total_union= sum(obbba['union_jobs'])
ira_total_gw     = sum(ira['capacity_additions_gw'])
obbba_total_gw   = sum(obbba['capacity_additions_gw'])
ira_total_credit = sum(ira['tax_credit_expenditure'])
obbba_total_credit = sum(obbba['tax_credit_expenditure'])

print("\n" + "=" * 92)
print("CUMULATIVE 2024–2035")
print("=" * 92)
print(f"{'Metric':<35} {'IRA':>18} {'OBBBA':>18} {'Δ (lost)':>16}")
print("-" * 92)
print(f"{'Total Investment':35} ${ira_total_inv:>15.0f}B ${obbba_total_inv:>15.0f}B  -${ira_total_inv - obbba_total_inv:.0f}B")
print(f"{'Job-Years (all clean energy)':35} {ira_total_jobs:>18,} {obbba_total_jobs:>18,}  -{ira_total_jobs - obbba_total_jobs:,}")
print(f"{'Manufacturing Job-Years':35} {ira_total_mfg:>18,} {obbba_total_mfg:>18,}  -{ira_total_mfg - obbba_total_mfg:,}")
print(f"{'Prevailing-Wage Job-Years':35} {ira_total_union:>18,} {obbba_total_union:>18,}  -{ira_total_union - obbba_total_union:,}")
print(f"{'Capacity Added (GW)':35} {ira_total_gw:>18.0f} {obbba_total_gw:>18.0f}  -{ira_total_gw - obbba_total_gw:.0f} GW")
print(f"{'Tax Credit Expenditure':35} ${ira_total_credit:>15.0f}B ${obbba_total_credit:>15.0f}B  +${ira_total_credit - obbba_total_credit:.0f}B (gov savings)")

# 2030 snapshot
idx_2030 = YEARS.index(2030)
print("\n" + "=" * 92)
print("2030 ANNUAL SNAPSHOT")
print("=" * 92)
print(f"{'Metric':<35} {'IRA':>18} {'OBBBA':>18} {'Δ':>16}")
print("-" * 92)
print(f"{'Annual Investment ($B)':35} ${ira['total_investment'][idx_2030]:>17.0f} ${obbba['total_investment'][idx_2030]:>17.0f}  -${ira['total_investment'][idx_2030] - obbba['total_investment'][idx_2030]:.0f}B")
print(f"{'Job-Years Created':35} {ira['new_jobs'][idx_2030]:>18,} {obbba['new_jobs'][idx_2030]:>18,}  -{ira['new_jobs'][idx_2030] - obbba['new_jobs'][idx_2030]:,}")
print(f"{'Installed Capacity (GW, cumul.)':35} {ira['cumulative_capacity_gw'][idx_2030]:>18.0f} {obbba['cumulative_capacity_gw'][idx_2030]:>18.0f}  -{ira['cumulative_capacity_gw'][idx_2030] - obbba['cumulative_capacity_gw'][idx_2030]:.0f} GW")

# ============================================================
# SAVE JSON OUTPUT
# ============================================================
output = {
    'metadata': {
        'model': 'IRA Erasure Macroeconomic Projection Model v1.0',
        'base_year': 2023,
        'calibration': ACTUALS,
        'notes': [
            'Investment in constant 2023 USD',
            'Job-years = direct + first-tier indirect; not permanent headcount',
            'Organic growth rates exclude credit effects',
            'Credit elasticities from Gechert & Rannenberg (2018) meta-analysis',
            'Uncertainty discount based on Baker/Bloom/Davis EPU framework',
            'OBBBA credit schedule based on legislative text and Bloomberg Tax analysis',
        ]
    },
    'ira': ira,
    'obbba': obbba,
    'summary': {
        'investment_lost_B':      round(ira_total_inv - obbba_total_inv),
        'jobs_lost':               ira_total_jobs - obbba_total_jobs,
        'mfg_jobs_lost':           ira_total_mfg - obbba_total_mfg,
        'union_jobs_lost':         ira_total_union - obbba_total_union,
        'capacity_lost_gw':        round(ira_total_gw - obbba_total_gw),
        'tax_credits_saved_B':     round(ira_total_credit - obbba_total_credit),
        'snap_2030_inv_delta_B':   round(ira['total_investment'][idx_2030] - obbba['total_investment'][idx_2030]),
        'snap_2030_jobs_delta':    ira['new_jobs'][idx_2030] - obbba['new_jobs'][idx_2030],
        'snap_2030_cap_delta_gw':  round(ira['cumulative_capacity_gw'][idx_2030] - obbba['cumulative_capacity_gw'][idx_2030]),
    }
}

with open('model_output.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nSaved: model_output.json")
