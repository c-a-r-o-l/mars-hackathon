# Mars Hackathon — Vehicles & Mobility Synthetic Dataset

Synthetic data for the **Vehicles & Mobility** track: *"Rovers, transporters, and
autonomous fleets that have to choose safe routes and spend every watt
carefully."* Built for the route-scoring / autonomous-planning use case
("plan an autonomous route scored by safety and energy cost").

Fully synthetic and reproducible (seeded, `numpy`/`pandas` only — no external
downloads needed), so it's ready to use the moment hacking starts. Swap in
real HiRISE/MOLA rasters later via `rasterio`/`GDAL` if you have time; the
schema below is designed to look like what you'd get from real terrain data.

## Files

### `terrain_grid.csv` (3,600 rows — a 60×60 cell grid, 20 m/cell, ~1.2 km × 1.2 km)

One row per terrain cell.

| Column | Description |
|---|---|
| `cell_id` | Unique cell index |
| `grid_x`, `grid_y` | Grid coordinates (0–59) |
| `elevation_m` | Synthetic elevation (fractal noise + crater bowls/rims) |
| `slope_deg` | Local slope, derived from the elevation gradient |
| `terrain_type` | `regolith_plain`, `rock_field`, `dune_field`, `dust_pit`, `crater_floor`, `crater_rim`, `lava_tube_entrance` |
| `rock_density` | 0–1, surface rock coverage |
| `dust_depth_cm` | Loose dust depth |
| `solar_exposure_pct` | 0–100, flavour feature (panel sizing / microgrid crossover) |
| `surface_temp_c` | Flavour feature |
| `hazard_prob` | Probability this cell causes a stall/tip/stuck event |
| `safety_score` | `1 − hazard_prob` (convenience column) |
| `traversal_energy_wh_per_m` | **Target-ish**: Wh consumed per metre crossing this cell |

Use this to train a **per-cell cost model** (predict `traversal_energy_wh_per_m`
or `hazard_prob`/`safety_score` from slope/rock/dust/terrain_type), which is
exactly the kind of cost surface an A*/Dijkstra route planner needs.

### `vehicles.csv` (4 rows — reference table)

`vehicle_type`, `mass_kg`, `battery_capacity_wh`, `max_payload_kg`,
`base_efficiency_mult`, `description`. Four classes: `light_scout`,
`cargo_hauler`, `crew_transport`, `swarm_builder` — useful for the
payload-vs-range angle mentioned in the brief.

### `routes.csv` (1,500 rows — candidate point-to-point routes)

Each row is one candidate route across the grid for one vehicle, with
aggregated terrain stats and outcome labels.

| Column | Description |
|---|---|
| `route_id` | Unique route index |
| `vehicle_type` | Joins to `vehicles.csv` |
| `route_type` | `optimized` (energy+hazard-aware path) vs `naive` (near-straight-line, noisy) — optimized routes succeed noticeably more often, so this is a learnable signal |
| `start_x/y`, `end_x/y` | Endpoints on the grid |
| `num_cells`, `distance_km` | Route length |
| `avg_slope_deg`, `max_slope_deg` | Terrain difficulty along the path |
| `avg_rock_density`, `pct_high_hazard_cells` | Hazard exposure |
| `dominant_terrain` | Most common terrain type on the route |
| `payload_kg`, `payload_fraction_of_max` | Load carried |
| `battery_capacity_wh` | From the assigned vehicle |
| `total_energy_wh` | **Target**: predicted energy consumption for the trip |
| `energy_margin_pct` | `(battery − energy) / battery × 100` |
| `safety_score` | **Target**: 0–1, route-level safety (penalises both average hazard and hazard spikes) |
| `mission_success` | **Target**: binary — `1` if the route both has energy margin and is safe enough |

~64% of routes succeed overall; `optimized` routes succeed at a meaningfully
higher rate than `naive` ones — a good baseline signal for a classifier.

## Suggested ML angles (matches the brief's "Where ML helps")

- **Route/cost scoring (regression):** train on `terrain_grid.csv` to predict
  `traversal_energy_wh_per_m` or `hazard_prob` from `slope_deg`,
  `rock_density`, `dust_depth_cm`, `terrain_type` (one-hot). Use the model as
  the cost surface for your own A*/Dijkstra planner in Nav2/Gazebo or a
  simple grid search.
- **Mission success classifier:** train on `routes.csv` to predict
  `mission_success` from route + vehicle features. Compare `optimized` vs
  `naive` route performance as a sanity check / demo talking point.
- **Payload vs. range trade-off (regression):** predict `total_energy_wh` or
  `energy_margin_pct` from `payload_kg`, `distance_km`, `avg_slope_deg`,
  vehicle specs — directly answers "how much cargo can this rover carry this
  far, safely."
- **Terrain classification:** if you want an imagery angle, render
  `terrain_grid.csv` as a false-colour raster (elevation/slope/rock_density
  as channels) with `matplotlib`/PIL and treat cropped tiles as a synthetic
  "imagery" classification task for `terrain_type`.

## Quick start

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier

terrain = pd.read_csv("terrain_grid.csv")
routes = pd.read_csv("routes.csv")

# --- per-cell energy cost model ---
X = pd.get_dummies(terrain[["slope_deg", "rock_density", "dust_depth_cm", "terrain_type"]])
y = terrain["traversal_energy_wh_per_m"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
cost_model = GradientBoostingRegressor(random_state=42).fit(Xtr, ytr)
print("cost model R^2:", cost_model.score(Xte, yte))

# --- mission success classifier ---
feat_cols = ["distance_km", "avg_slope_deg", "max_slope_deg", "avg_rock_density",
             "pct_high_hazard_cells", "payload_fraction_of_max", "battery_capacity_wh"]
Xr = routes[feat_cols]
yr = routes["mission_success"]
Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42, stratify=yr)
clf = RandomForestClassifier(random_state=42).fit(Xtr, ytr)
print("success classifier accuracy:", clf.score(Xte, yte))
```

## Regenerating / tweaking

`generate_dataset.py` is included. It's seeded (`RNG_SEED = 42`) for
reproducibility — change the seed, grid size, crater count, or the
`TERRAIN_ENERGY_MULT`/`TERRAIN_HAZARD_BASE` dictionaries to reshape the
terrain difficulty, then rerun with:

```
python3 generate_dataset.py
```
