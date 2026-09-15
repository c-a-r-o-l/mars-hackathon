# Mars Fleet Dispatcher

Track B (Vehicles & Mobility) hackathon build. A dispatcher for the
settlement's delivery fleet: for each trip, predict how much energy every
vehicle would need, eliminate the ones that cannot complete it, and assign
the smallest suitable vehicle (right-sizing). If no vehicle qualifies,
refuse the trip.

## Files

| File | What it is |
|---|---|
| `dispatcher.py` | The whole build: energy model, rule recovery, dispatcher, results, chart |
| `index.html` | Trip Checker demo — fully self-contained (model ported to JS, verified bit-identical on all 1,500 trips); opens directly in a browser |
| `server.py` | Optional backend variant of the same demo (Python stdlib only) |
| `results/summary.txt` | Printed results of all steps |
| `results/dispatch.png` | Two-panel chart: success rates + outcome stacks |
| `critique.md` | Independent adversarial review — every claim verified against the data |
| `track-b2-mobility-ai4mars/track-b2-mobility-ai4mars/routes.csv` | 1,500 trips (only data used, with vehicles.csv) |
| `track-b2-mobility-ai4mars/track-b2-mobility-ai4mars/vehicles.csv` | Fleet specs (4 vehicles) |

Only `routes.csv` and `vehicles.csv` are loaded. `terrain_grid.csv` and the
image/label packs are not used.

## Run

```
python3 -m venv venv
./venv/bin/pip install numpy pandas scikit-learn matplotlib
./venv/bin/python dispatcher.py
```

## Demo

The Trip Checker is a small web page where you type in a trip (distance,
cargo weight, ground type) and it says which rover the dispatcher sends and
whether the trip will succeed — in plain language, for non-technical
audiences.

- **Just open `index.html` in a browser** — the model is embedded in the
  page (exported from `dispatcher.py`, verified bit-identical to Python on
  all 1,500 trips). No server, no internet needed.
- Alternatively, `./venv/bin/python server.py` serves the same page with a
  Python backend at http://localhost:8000.

The page also carries a one-screen summary of the project and the honest
framing (the 87.1% is labelled as a simulation).

## What it does, step by step

### Step 1 - Energy model (the ML)

Predicts `total_energy_wh` (385-14,015 Wh) with a
`GradientBoostingRegressor` (sklearn defaults), 80/20 split, seed 42.

- **Features (14):** `distance_km`, `num_cells`, `avg_slope_deg`,
  `max_slope_deg`, `avg_rock_density`, `pct_high_hazard_cells`,
  `payload_kg`, plus vehicle `mass_kg`, `battery_capacity_wh`,
  `max_payload_kg`, `base_efficiency_mult`, plus one-hot
  `dominant_terrain`.
- **Excluded as target-leaking:** `energy_margin_pct` (computed from the
  target), `safety_score` (r=-0.99 with `pct_high_hazard_cells`),
  `mission_success` (the outcome), `route_type` (planner label, not
  observable before the trip).

| Model | Held-out MAE | R² |
|---|---|---|
| Full model (14 features) | **193.3 Wh** | **0.9851** |
| Linear, `distance_km` only | 789.3 Wh | — |

### Step 2 - Recovered success rule

Grid search over thresholds reproduces the real `mission_success` exactly
(1,500/1,500 rows, 100%):

```
mission_success = (energy_margin_pct > 5) AND (safety_score > 0.5)
```

The rule reproduces the real `mission_success` column exactly (1,500/1,500),
consistent with this being the generator's rule. The counterfactual
evaluation below uses the same definition of success as the observed data -
not an invented one.

### Step 3 - Dispatcher (rules, not ML)

For each of the 1,500 trips:

1. Predict energy for all 4 vehicles (model refit on all 1,500 rows after
   the held-out validation above).
2. Drop vehicles where `payload_kg > max_payload_kg`.
3. Drop vehicles whose predicted energy fails the recovered margin rule
   (`margin > 5%` against that vehicle's battery).
4. Among the survivors, assign the one with the smallest
   `battery_capacity_wh` (pure right-sizing - don't send a 950 kg hauler
   when a 180 kg scout can safely do the job).
5. No survivor -> **NO-VEHICLE** (refused).

Apply the Step 2 rule to each assignment for predicted success. Unsafe
routes (`safety_score <= 0.5`) are still dispatched - vehicle choice cannot
fix terrain risk - but their safety failure **counts as a failure against
the assigned vehicle**. The refusal and the safety failures are reported
separately, never merged.

### Step 4 - Results (also in `results/summary.txt`)

| Metric | Value |
|---|---|
| Observed success rate (measured) | **63.9%** |
| Simulated dispatcher success rate | **87.1%** |
| Improvement in simulation | **+23.2 pp** |
| ...among dispatched trips | 87.4% (n=1,494) |
| NO-VEHICLE (refused) | **6** - no vehicle clears payload/battery with >5% margin |
| ROUTE-INFEASIBLE (dispatched, fails) | **188** - safety below threshold, counted as FAILURES against the assigned vehicle |

The dispatcher figure is a counterfactual simulation. Alternative vehicle
assignments were never actually driven, so these outcomes are predicted
under the dataset's own success rule, not observed.

NO-VEHICLE refusals are **not failures** - in operations they would be
re-planned (split load, different route), not driven.

**Failure breakdown** (the observed 542 failed trips):

| Bucket | Count |
|---|---|
| Fixed by reassignment | **348** |
| Remaining, energy-driven (NO-VEHICLE) | 6 |
| Remaining, safety-driven (ROUTE-INFEASIBLE) | 188 |

188 of the 194 remaining failures are safety-driven and unfixable by ANY
vehicle assignment: the dispatcher sits at the ceiling of what assignment
can fix. The rest needs route re-planning or infrastructure, not better
dispatch.

65 of the 542 observed failures breach both the energy-margin and the
safety threshold. These are counted as safety-driven (safety-first
convention), so "energy-fail 353" means failures that are not also safety
failures; a raw margin<=5 count gives 418.

**Held-out trips (Step 4b):** the same dispatcher, but with the energy model
trained only on the other 1,200 routes and applied to the 300 held-out
routes from the Step 1 split — trips the model never saw. Observed success
on those 300: 61.7%; dispatched predicted success: 85.0% (NO-VEHICLE 0,
safety-fail 45). Close to the 87.1% of the full run, which is what you want
to see: the dispatcher decision rule itself generalises, it is not just
refitting noise. Shown as the "HELD-OUT 300" group in the chart.

Assignments before -> after:

| Vehicle | Before | After |
|---|---|---|
| light_scout | 342 | 517 |
| cargo_hauler | 394 | 282 |
| crew_transport | 353 | 588 |
| swarm_builder | 411 | **107** |

324 trips moved off swarm_builder (0 refused). The vehicle that succeeds
only 21.7% of the time (1.2 kWh battery) now keeps just the light trips its
battery can actually clear.

### Step 5 - Chart (`results/dispatch.png`)

Two panels, large fonts for 3-metre readability:

- **Panel 1:** grouped bars, success rate by vehicle + overall, observed
  vs dispatched, plus a **HELD-OUT 300** pair: the dispatcher run on the
  300 held-out trips with the model trained only on the other 1,200
  (61.7% -> 85.0%). Dispatched per-vehicle rates cover **all trips
  assigned to that vehicle, with safety failures counted as failures** -
  e.g. cargo_hauler 82.6% (it inherits the heavy unsafe routes only it
  can carry), swarm_builder 97.2%, overall 63.9% -> 87.1%.
- **Panel 2:** stacked outcomes - success / energy-fail / safety-fail,
  observed vs dispatched. Energy-fail collapses from 353 to 6; the
  safety-fail segment barely moves (189 -> 188: route-level, not fixable
  by dispatch). This makes the safety residual visible instead of hiding
  it in the success rate.

## Validated vs not validated

```
VALIDATED:     energy model, trained on 1,200 routes, tested on 300 unseen.
               MAE 193.3 Wh, R^2 0.985.
NOT VALIDATED: counterfactual assignments, predicted with a model refit on
               all 1,500 rows, applied back to those same 1,500 routes.
               This is a demonstration of decisions, not an independent test.
```

## Data finding: no detectable vehicle signal in the data

The spec assumed each vehicle would need a different amount of energy per
trip. The data shows no detectable vehicle signal: vehicle-feature
importance is negligible (~1.8e-5 combined), vehicle-identity dummies do
not improve held-out MAE (193.5 vs 193.3 Wh), GBM residuals differ by only
30 Wh across vehicles (within-vehicle std is 130-183 Wh), and a model
trained on three vehicles predicts the fourth at 211-383 Wh vs 193.3 Wh
within-data (the high end is payload-range extrapolation when cargo_hauler
is held out). Inferred from model behaviour; the generator source is not
in the repo.

The vehicles' specs matter only through payload capacity and battery
margin - not through the energy column.

**How this changed the dispatch rule:** because trip energy is
vehicle-independent, no energy comparison between vehicles is meaningful
for the same route. The dispatch rule is pure right-sizing: among
vehicles that can carry the payload and clear the margin, pick the
smallest `battery_capacity_wh`. No tie band, no energy ranking.

## Honest framing

- The energy model is **validated on held-out real routes** (300 trips the
  model never saw): MAE 193.3 Wh, R² 0.985.
- The reassignment benefit is a **projection**: we predict energy for
  vehicle/route pairs that were never actually driven. That is
  interpolation within the training distribution - every vehicle appears on
  342-411 routes spanning the full grid - but it is a prediction, not an
  observation.
- The dispatcher rate uses the exact success rule recovered from the real
  data, applied to those predicted energies. It is a simulated
  counterfactual, not a field result.
- Safety failures are not hidden: the 188 unsafe routes are dispatched
  anyway, counted as failures against the assigned vehicle (87.4% among
  dispatched, 87.1% over all 1,500 trips), and shown as their own segment
  in the chart. The 6 NO-VEHICLE refusals are stated separately as
  refusals, not failures.
