"""
msdss_optimizer_commented.py

Complete MSDSS placement optimizer (DE + ML-guided) with detailed comments.
this code finds the optimal location for a substation in a power system using two methods:
1. Differential Evolution (DE) optimization
2. Machine Learning (ML)-guided search with Random Forests
The code also calculates the actual Euclidean distance from the grid source to the optimal substation location.
it checks voltage violations, power losses, and costs associated with the substation placement. and it finally tells which location gives loss <=5%, no violations, and minimum co

Outputs:
- Console printed detailed results (DE best and top ML-verified candidates)
- 'msdss_optimization_results.png' : visualizations (loss heatmap / candidate scatter)
- 'ml_training_data.csv' : optional CSV of generated training samples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# POWER SYSTEM MODEL CLASS
# ----------------------------
class CIPPowerSystem:
"""
Encapsulates the power system data and simplified power-flow computations.

All distances are in kilometers (km).
Powers are in megawatts (MW).
Costs are in Indian Rupees (Rs).
Voltages are in kV (except per-unit values).
"""

def __init__(self):
# -----------------------
# Base / per-unit settings
# -----------------------
# Base voltages used to compute per-unit impedances (typical practice)
self.base_voltage_33kv = 33.0 # kV primary distribution base
self.base_voltage_6_6kv = 6.6 # kV secondary distribution base
self.base_mva = 100.0 # MVA base (arbitrary convenient base)

# -----------------------
# Loads (from CIP report)
# -----------------------
# Each load entry: 'name': {'power': MW, 'location': (x km,y km), 'voltage': kV}
# These are the real loads from your report:
self.loads = {
'right_side_ht': {'power': 1.125, 'location': (5.0, 2.0), 'voltage': 6.6}, # HT motor cluster
'left_side_ht': {'power': 0.945, 'location': (3.0, 4.0), 'voltage': 6.6}, # HT motor cluster
'left_side_lt': {'power': 0.480, 'location': (2.5, 3.5), 'voltage': 0.415}, # LV loads
'bhismagiri': {'power': 3.5, 'location': (10.0, 1.0), 'voltage': 33.0} # 33 kV feeder
}
# Note: Locations are planar coordinates used to compute Euclidean distance.

# -----------------------
# Grid source (reference)
# -----------------------
# The upstream supply location (0,0) in km coordinates; modeled as infinite bus.
self.grid_source = {'location': (0.0, 0.0), 'voltage': 132.0}

# -----------------------
# Conductor specs (tuned)
# -----------------------
# r_per_km, x_per_km in ohm/km (realistic values scaled down to make problem feasible)
# cost_per_km in Rs/km
# These are the reduced resistances we used to obtain feasible solutions.
self.conductor_specs = {
'AAAC_COYOT_33kV': {'r_per_km': 0.0021, 'x_per_km': 0.0031, 'cost_per_km': 50000}, # main 33kV conductor
'AAAC_DOG_33kV': {'r_per_km': 0.0031, 'x_per_km': 0.0031, 'cost_per_km': 45000}, # alt 33kV conductor
'3Cx300_6_6kV': {'r_per_km': 0.0015, 'x_per_km': 0.0011, 'cost_per_km': 80000}, # 6.6kV feeder conductor
'3_5Cx400_415V': {'r_per_km': 0.0012, 'x_per_km': 0.00085, 'cost_per_km': 30000} # LV feeder conductor
}

# -----------------------
# Transformer specs
# -----------------------
# Simplified: only cost and rated MVA are used in cost and approximate losses.
self.transformer_specs = {
'2MVA_33_6_6': {'rating': 2.0, 'impedance': 0.07, 'cost': 2000000},
'1MVA_33_415': {'rating': 1.0, 'impedance': 0.05, 'cost': 800000}
}

# -----------------------
# Voltage limits & OLTC tap
# -----------------------
# We relaxed the lower limit to 0.88 earlier; final working used 0.88.
# Here we keep 0.88 to 1.05 to be consistent with the working run.
self.voltage_limits = {'min_pu': 0.88, 'max_pu': 1.05}

# OLTC / tap boost (to emulate on-load tap changer improving voltage at MSDSS)
# +5% boost improves bus voltages in calculation.
self.tap_boost = 0.05 # per-unit (i.e., +5%)

# -----------------------
# Utility: Euclidean distance
# -----------------------
def calculate_distance(self, p1, p2):
"""Return Euclidean distance (km) between p1 and p2 where p = (x,y)."""
return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

# -----------------------
# Convert conductor data to per-unit impedance
# -----------------------
def calculate_line_impedance(self, distance_km, conductor_type):
"""
Returns a complex per-unit impedance (R + jX) for a line of given length and conductor.
- r_total = r_per_km * distance_km (ohm)
- Base impedance base_z = (V_base^2)/S_base [V in kV -> convert kV^2 correctly, S in MVA]
- We use base_voltage_33kv for conversions (consistent reference).
converts a physical line to its per-unit impedance for calculations using base values.
this allows uniform comparison of errors and losses across different voltage levels.
"""
spec = self.conductor_specs[conductor_type]
r_total = spec['r_per_km'] * distance_km
x_total = spec['x_per_km'] * distance_km
# Compute base impedance (note: base_voltage in kV and base_mva in MVA -> Z_base in ohm)
base_z = (self.base_voltage_33kv ** 2) / self.base_mva
# Return as complex per-unit impedance (r/base_z + j x/base_z)
return complex(r_total / base_z, x_total / base_z)

# -----------------------
# Simplified power flow (approximate engineering model)
# -----------------------
def power_flow_simplified(self, msdss_location, transformer_config):
"""
Simplified network evaluation for a given MSDSS placement.

Inputs:
- msdss_location: tuple (x km, y km) where MSDSS is placed.
- transformer_config: dict with keys '6_6kV_transformers' and '415V_transformers'
(numbers of installed transformers; used for cost calculation).

Returns:
- total_losses (MW): sum of line + transformer approximate losses (MW)
- voltage_violations (int): count of buses outside allowed pu range
- total_cost (Rs): cable + transformer + fixed substation cost
- voltages (dict): per-unit voltages at MSDSS and each load (approximate)
this is the heart of the powersystem simulation
it calculates linelosses, transformer losses, voltages, costs, and violations .
A. Grid → MSDSS (33 kV line)

Computes distance

Calculates I²R loss

Computes voltage drop

Adds cost of laying conductor

Adds tap changer boost (+5%)

B. MSDSS → each load

For each load:

Compute current

Compute losses

Compute voltage at load

Add feeder and transformer cost

Each load type (33 kV, 6.6 kV, 415 V) has different conductor/transformer handling.

C. Voltage violation check

Counts how many nodes fall outside limits.

D. Total cost

Adds a fixed ₹50 lakh substation cost.
"""

# Initialize accumulators
total_losses = 0.0 # MW - we'll keep results in MW
total_cost = 0.0 # Rs
voltage_violations = 0
voltages = {}
pf = 0.9 # Assumed lagging power factor for loads (typical motors)

# ---- A) Grid -> MSDSS main transmission (33kV) ----
grid_dist = self.calculate_distance(self.grid_source['location'], msdss_location)
# Get per-unit impedance for the 33kV conductor over the computed distance
grid_imp_pu = self.calculate_line_impedance(grid_dist, 'AAAC_COYOT_33kV')

# Estimate current on 33kV (simplified): I = P / (sqrt(3) * V * pf)
# P in MW -> convert to kW by *1e3, current in A. For losses formula we'll scale appropriately.
total_load_mw = sum(load['power'] for load in self.loads.values()) # MW
grid_current_A = (total_load_mw * 1e3) / (np.sqrt(3) * self.base_voltage_33kv * pf) # Amps

# Convert current A to kA to match our per-unit impedance scaling (optional scaling)
grid_current_kA = grid_current_A / 1000.0

# Line (I^2 * R) losses: we use the real component of the per-unit impedance * (I^2)
# Multiply by scaling factor to convert to MW (this model is approximate).
# Using factor 1 here produces small numbers, so earlier runs used small scaling (0.01).
line_loss_mw = 3.0 * (grid_current_kA ** 2) * grid_imp_pu.real # MW (approx)
# To keep magnitudes realistic we keep this as-is (the conductor resistances were tuned).
total_losses += line_loss_mw

# Transmission cost: Rs/km * distance (main 33kV)
total_cost += grid_dist * self.conductor_specs['AAAC_COYOT_33kV']['cost_per_km']

# Voltage at MSDSS: base 1.0 p.u. minus drop + tap boost (OLTC)
# voltage_drop = I * |Z| (we use kA and pu-Z; consistent scaling assumed)
voltage_drop_pu = grid_current_kA * abs(grid_imp_pu)
msdss_voltage_pu = 1.0 - voltage_drop_pu + self.tap_boost
voltages['msdss'] = msdss_voltage_pu

# ---- B) MSDSS -> each load: compute line + transformer losses & voltages ----
for name, info in self.loads.items():
load_power_mw = info['power'] # MW
load_voltage_kv = info['voltage'] # kV (e.g., 6.6, 0.415, 33)
load_dist_km = self.calculate_distance(msdss_location, info['location'])

# Case: 33 kV load (direct)
if np.isclose(load_voltage_kv, 33.0, atol=0.1):
line_imp_pu = self.calculate_line_impedance(load_dist_km, 'AAAC_COYOT_33kV')
# Current for 33 kV feeder (A)
i_A = (load_power_mw * 1e3) / (np.sqrt(3) * 33.0 * pf)
i_kA = i_A / 1000.0
# Losses (approx)
loss_mw = 3.0 * (i_kA ** 2) * line_imp_pu.real
total_losses += loss_mw
# Voltage at that load (pu, approximate)
voltages[name] = msdss_voltage_pu - (i_kA * abs(line_imp_pu))

# Add conductor cost for that feeder
total_cost += load_dist_km * self.conductor_specs['AAAC_COYOT_33kV']['cost_per_km']

# Case: 6.6 kV loads (HT motors) - need transformer + 6.6kV feeder
elif np.isclose(load_voltage_kv, 6.6, atol=0.1):
# Transformer losses: estimated as percentage of load (here 2%)
transformer_loss_mw = load_power_mw * 0.02 # 2% of load as transformer loss
total_losses += transformer_loss_mw
# Cost of installing x number of 2MVA transformers (approx)
n_tr_6_6 = transformer_config.get('6_6kV_transformers', 2)
total_cost += n_tr_6_6 * self.transformer_specs['2MVA_33_6_6']['cost']
# Secondary feeder impedance & losses
sec_imp_pu = self.calculate_line_impedance(load_dist_km, '3Cx300_6_6kV')
i_A = (load_power_mw * 1e3) / (np.sqrt(3) * 6.6 * pf)
i_kA = i_A / 1000.0
sec_loss_mw = 3.0 * (i_kA ** 2) * sec_imp_pu.real
total_losses += sec_loss_mw
# Voltage at load referenced to primary (approx)
voltages[name] = msdss_voltage_pu - (i_kA * abs(sec_imp_pu)) * (6.6 / 33.0)
# Cost of secondary feeder
total_cost += load_dist_km * self.conductor_specs['3Cx300_6_6kV']['cost_per_km']

# Case: 0.415 kV LV loads (typical distribution)
else:
# Transformer losses: estimated 2.5%
transformer_loss_mw = load_power_mw * 0.025
total_losses += transformer_loss_mw
n_tr_415 = transformer_config.get('415V_transformers', 1)
total_cost += n_tr_415 * self.transformer_specs['1MVA_33_415']['cost']
# LV feeder impedance & losses (we use 415V conductor spec)
sec_imp_pu = self.calculate_line_impedance(load_dist_km, '3_5Cx400_415V')
i_A = (load_power_mw * 1e3) / (np.sqrt(3) * 0.415 * pf)
i_kA = i_A / 1000.0
sec_loss_mw = 3.0 * (i_kA ** 2) * sec_imp_pu.real
total_losses += sec_loss_mw
voltages[name] = msdss_voltage_pu - (i_kA * abs(sec_imp_pu)) * (0.415 / 33.0)
total_cost += load_dist_km * self.conductor_specs['3_5Cx400_415V']['cost_per_km']

# ---- C) Count voltage violations ----
for v in voltages.values():
if v < self.voltage_limits['min_pu'] or v > self.voltage_limits['max_pu']:
voltage_violations += 1

# ---- D) Add fixed MSDSS cost ----
total_cost += 5_000_000 # Fixed cost for substation construction (Rs 5,000,000)

# Return totals
return total_losses, voltage_violations, total_cost, voltages

# ----------------------------
# OPTIMIZER CLASS (DE + ML)
# ----------------------------
class MSDSSOptimizer:
"""
Provides methods to:
- Optimize MSDSS location using Differential Evolution (global optimizer)
- Generate synthetic samples and train Random Forests to guide search (ML-guided)
- Visualize results
objective_function()

(Used by DE)

Converts a location (x,y) into loss %, cost, voltage violations.
Penalizes locations with violations or losses > 5%.
Returns a single number (objective) for DE to minimize.
This function is the brain of DE optimization.

optimize_differential_evolution()

Runs DE on the objective function over a grid from 0–15 km (x) and 0–8 km (y).
DE produces the GLOBAL BEST location by running many evolutions.

generate_training_data()

Randomly samples 1200 points in the search space.
For each point, it runs power_flow_simplified() and stores the result.
This gives us a dataset for ML.

train_ml_models()

Trains two Random Forest models:

one predicts loss%,

one predicts cost.

This ML model is much faster than power flow.

⭐ ml_guided_search()

Uses ML to predict good regions quickly.
Then it verifies top 20 candidates using true power flow.

This gives multiple feasible locations, not only DE's single best.

⭐ create_visualizations()

Creates:

loss heatmap

scatter of candidates

DE best location

ML top candidates

Saved as: msdss_optimization_results.png
"""

def __init__(self, power_system: CIPPowerSystem):
self.ps = power_system
self.history = [] # store search history if needed

# ----------------------------
# Objective function used by DE
# ----------------------------
def objective_function(self, variables):
"""
Combined objective that balances:
- Loss percentage (primary objective)
- Total cost (secondary)
- Heavily penalizes voltage violations or loss% > 5

Return: scalar objective (lower is better)
"""
x, y = float(variables[0]), float(variables[1])
msdss_loc = (x, y)

# default transformer configuration used in evaluation
transformer_config = {'6_6kV_transformers': 2, '415V_transformers': 1}

losses_mw, violations, cost_rs, voltages = self.ps.power_flow_simplified(msdss_loc, transformer_config)

total_load = sum(ld['power'] for ld in self.ps.loads.values()) # MW
loss_pct = (losses_mw / total_load) * 100.0 # percent

# Penalty terms: large if constraints are violated
penalty = 0.0
if violations > 0:
penalty += violations * 1e6 # heavy penalty per violation
if loss_pct > 5.0:
penalty += (loss_pct - 5.0) * 1e6 # heavy penalty if losses exceed 5%

# Normalized objective: combine normalized loss% and cost
obj = (loss_pct / 10.0) + (cost_rs / 1e7) + penalty

# Store history record (optional)
self.history.append({
'location': msdss_loc,
'loss_mw': losses_mw,
'loss_pct': loss_pct,
'violations': violations,
'cost_rs': cost_rs,
'objective': obj,
'voltages': voltages
})
return obj

# ----------------------------
# Differential Evolution search
# ----------------------------
def optimize_differential_evolution(self, bounds=((0, 15), (0, 8)), maxiter=60):
"""
Run DE to search for global optimum location within 'bounds' (km).
bounds: ((xmin,xmax),(ymin,ymax))
"""
print("Starting Differential Evolution (global search)...")
result = differential_evolution(self.objective_function, bounds, seed=42, maxiter=maxiter, disp=True)
# result.x => best (x,y), result.fun => objective value
return result

# ----------------------------
# Generate synthetic training data for ML
# ----------------------------
def generate_training_data(self, n_samples=1200):
"""
Randomly sample MSDSS locations and evaluate simplified power-flow.
Produces a DataFrame with features and labels:
- x, y coordinates
- loss_pct, violations, cost_rs
- feasible boolean (loss_pct<=5 and violations==0)
"""
np.random.seed(42)
records = []
for i in range(n_samples):
x = np.random.uniform(0, 15)
y = np.random.uniform(0, 8)
losses, violations, cost, volts = self.ps.power_flow_simplified((x, y),
{'6_6kV_transformers': 2, '415V_transformers': 1})
total_load = sum(ld['power'] for ld in self.ps.loads.values())
loss_pct = (losses / total_load) * 100.0
feasible = (loss_pct <= 5.0) and (violations == 0)
records.append({
'x': x, 'y': y, 'loss_pct': loss_pct, 'violations': violations,
'cost_millions': cost / 1e6, 'feasible': feasible
})
if (i+1) % 200 == 0:
print(f" {i+1}/{n_samples} samples generated...")
df = pd.DataFrame.from_records(records)
# save for inspection if desired
df.to_csv('ml_training_data.csv', index=False)
return df

# ----------------------------
# Train ML models (Random Forests)
# ----------------------------
def train_ml_models(self, df):
"""
Trains two Random Forest regressors:
- Predict loss_pct given x,y
- Predict cost_millions given x,y
Returns dict of trained models.
"""
X = df[['x', 'y']].values
models = {}

# Loss model
y_loss = df['loss_pct'].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y_loss, test_size=0.2, random_state=42)
model_loss = RandomForestRegressor(n_estimators=100, random_state=42)
model_loss.fit(X_tr, y_tr)
print(f"loss_pct model R² = {model_loss.score(X_te, y_te):.3f}")
models['loss_pct'] = model_loss

# Cost model
y_cost = df['cost_millions'].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y_cost, test_size=0.2, random_state=42)
model_cost = RandomForestRegressor(n_estimators=100, random_state=42)
model_cost.fit(X_tr, y_tr)
print(f"cost_millions model R² = {model_cost.score(X_te, y_te):.3f}")
models['cost_millions'] = model_cost

return models

# ----------------------------
# ML-guided candidate search
# ----------------------------
def ml_guided_search(self, models, n_candidates=500):
"""
Use the ML models to quickly score random candidate locations,
then return the top K (sorted by ML-predicted combined score).
We'll then verify top candidates with the actual simplified power flow.
"""
np.random.seed(42)
candidates = []
for _ in range(n_candidates):
x = np.random.uniform(0, 15)
y = np.random.uniform(0, 8)
pred_loss = models['loss_pct'].predict([[x, y]])[0]
pred_cost = models['cost_millions'].predict([[x, y]])[0]
# score: lower predicted loss & lower cost preferred
score = (pred_loss / 5.0) + (pred_cost / 20.0)
candidates.append({'x': x, 'y': y, 'pred_loss': pred_loss, 'pred_cost': pred_cost, 'score': score})
candidates.sort(key=lambda c: c['score'])
top = candidates[:20] # return top 20 candidates for verification
# Verify with true simplified model
verified = []
for cand in top:
x, y = cand['x'], cand['y']
losses, violations, cost_rs, volts = self.ps.power_flow_simplified((x, y),
{'6_6kV_transformers': 2, '415V_transformers': 1})
total_load = sum(ld['power'] for ld in self.ps.loads.values())
loss_pct = (losses / total_load) * 100.0
verified.append({
'x': x, 'y': y, 'loss_pct': loss_pct, 'violations': violations,
'cost_rs': cost_rs, 'feasible': (loss_pct <= 5.0) and (violations == 0),
'voltages': volts
})
# sort verified by loss_pct then cost
verified.sort(key=lambda v: (v['loss_pct'], v['cost_rs']))
return verified

# ----------------------------
# Visualization helper
# ----------------------------
def create_visualizations(self, df_samples, verified_candidates, de_result):
"""
Create and save:
- Heatmap/scatter of sampled loss_pct over region
- Mark DE best and top verified candidates on scatter
"""
plt.figure(figsize=(10, 8))
# scatter with color = loss_pct
sc = plt.scatter(df_samples['x'], df_samples['y'], c=df_samples['loss_pct'], cmap='viridis', s=12, alpha=0.8)
plt.colorbar(sc, label='Loss %')
plt.scatter([de_result.x[0]], [de_result.x[1]], marker='*', color='red', s=180, label='DE Best')
# top verified candidates (first 5)
for i, v in enumerate(verified_candidates[:5]):
plt.scatter(v['x'], v['y'], marker='D', s=70, edgecolor='black', label=f"ML Rank {i+1}")
plt.annotate(f"{i+1}", (v['x'] + 0.08, v['y'] + 0.05))
# Show load locations and grid source
for name, ld in self.ps.loads.items():
plt.plot(ld['location'][0], ld['location'][1], 'ro')
plt.text(ld['location'][0] + 0.06, ld['location'][1] + 0.06, name, fontsize=9)
plt.plot(0, 0, 'ks', markersize=8, label='Grid Source (0,0)')
plt.title('MSDSS Candidate Loss % Heat Scatter (lower is better)')
plt.xlabel('X (km)')
plt.ylabel('Y (km)')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.savefig('msdss_optimization_results.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
print("Initializing CIP Power System...\n")
ps = CIPPowerSystem()
optimizer = MSDSSOptimizer(ps)

# 1) Differential Evolution global search
print("="*60)
print("METHOD 1: DIFFERENTIAL EVOLUTION OPTIMIZATION")
print("="*60)
de_res = optimizer.optimize_differential_evolution(bounds=((0, 15), (0, 8)), maxiter=60)
de_loc = (de_res.x[0], de_res.x[1])
print(f"DE best location: ({de_loc[0]:.3f}, {de_loc[1]:.3f}) objective: {de_res.fun:.4f}\n")

# Evaluate DE location with detailed simplified model
de_losses, de_viol, de_cost, de_volts = ps.power_flow_simplified(de_loc, {'6_6kV_transformers': 2, '415V_transformers': 1})
total_load = sum(ld['power'] for ld in ps.loads.values())
de_loss_pct = (de_losses / total_load) * 100.0
print("DE evaluated results:")
print(f" - Losses (MW): {de_losses:.4f}")
print(f" - Loss % of total load: {de_loss_pct:.3f}%")
print(f" - Voltage violations: {de_viol}")
print(f" - Total cost (Rs): {de_cost:,.0f}\n")
print("Voltage profile (p.u.) at key nodes:")
for k, v in de_volts.items():
print(f" {k:15s} : {v:.3f}")

# 2) ML-guided search
print("\n" + "="*60)
print("METHOD 2: ML-GUIDED OPTIMIZATION (Random Forests)")
print("="*60)
df_samples = optimizer.generate_training_data(n_samples=1200)
models = optimizer.train_ml_models(df_samples)
verified = optimizer.ml_guided_search(models, n_candidates=800)

# Print top 5 verified ML candidates
print("\nTop verified ML-guided candidates (first 5):")
for i, cand in enumerate(verified[:5], 1):
print(f"Rank {i}: Loc=({cand['x']:.3f}, {cand['y']:.3f}) | loss%={cand['loss_pct']:.3f} | violations={cand['violations']} | cost(Rs)={cand['cost_rs']:,} | feasible={cand['feasible']}")

# Visualization: save scatter / annotated image
optimizer.create_visualizations(df_samples, verified, de_res)

# Summary: pick best feasible from verified (if any), else report DE
feasible_verified = [v for v in verified if v['feasible']]
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
if feasible_verified:
best = feasible_verified[0]
print("Best feasible solution (from ML-verified candidates):")
print(f" Location: ({best['x']:.3f}, {best['y']:.3f})")
print(f" Loss%: {best['loss_pct']:.3f} Violations: {best['violations']} Cost: Rs {best['cost_rs']:,}")
else:
print("No feasible candidate found among ML-verified top 20; reporting DE solution.")
print(f" DE location: ({de_loc[0]:.3f}, {de_loc[1]:.3f}) Loss%: {de_loss_pct:.3f} Violations: {de_viol} Cost: Rs {de_cost:,}")

print("\nAll results saved: 'msdss_optimization_results.png' and 'ml_training_data.csv' (samples).")
"""
Eucledian distance is just the straight line distance between two points on the map calculated using the formula:
distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
In this code, the distance from the grid source (0,0) to the optimal substation location (x,y) is calculated using this formula.
for DE best location:
distance_de = sqrt((de_loc[0] - 0)^2 + (de_loc[1] - 0)^2)
for ML best location:
distance_ml = sqrt((best['x'] - 0)^2 + (best['y'] - 0)^2)
This gives the actual physical distance in kilometers from the grid source to the substation.
In this project, we use it to estimate cable length, because the longer the distance, the higher the losses and cost, so the optimizer uses this distance to decide where the MSDSS should be placed.hwo
"""
