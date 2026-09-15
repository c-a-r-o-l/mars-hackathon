"""Mars Fleet Dispatcher - Track B (Vehicles & Mobility) hackathon build.

Step 1 (energy model): predict total_energy_wh from route + vehicle features.
Steps 2-5 (rule recovery, dispatcher, results, chart) follow in this file.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

DATA_DIR = "track-b2-mobility-ai4mars/track-b2-mobility-ai4mars"

FORBIDDEN = ["energy_margin_pct", "safety_score", "mission_success", "route_type"]


def load_data():
    routes = pd.read_csv(f"{DATA_DIR}/routes.csv")
    vehicles = pd.read_csv(f"{DATA_DIR}/vehicles.csv")

    # sanity: battery_capacity_wh appears in both files - confirm they agree
    merged_check = routes.merge(
        vehicles[["vehicle_type", "battery_capacity_wh"]],
        on="vehicle_type", suffixes=("_route", "_veh"))
    mismatches = (merged_check["battery_capacity_wh_route"]
                  != merged_check["battery_capacity_wh_veh"]).sum()
    print(f"battery_capacity_wh mismatches between routes and vehicles: {mismatches}")

    routes = routes.merge(
        vehicles[["vehicle_type", "mass_kg", "max_payload_kg",
                  "base_efficiency_mult"]],
        on="vehicle_type", how="left")
    return routes, vehicles


def build_features(routes):
    route_feats = ["distance_km", "num_cells", "avg_slope_deg", "max_slope_deg",
                   "avg_rock_density", "pct_high_hazard_cells", "payload_kg"]
    veh_feats = ["mass_kg", "battery_capacity_wh", "max_payload_kg",
                 "base_efficiency_mult"]

    # one-hot dominant_terrain: carries the rock-field cost signal that
    # avg_rock_density only partly captures. Fit on the full table so the
    # test split always has the same columns (crater_floor has only 14 rows).
    terrain_dummies = pd.get_dummies(routes["dominant_terrain"], dtype=float)

    X = pd.concat([routes[route_feats], routes[veh_feats], terrain_dummies],
                  axis=1)
    return X


def step1_energy_model(routes):
    X = build_features(routes)
    y = routes["total_energy_wh"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(f"\nfeatures ({X.shape[1]}): {list(X.columns)}")
    print(f"train rows: {len(X_tr)}, test rows: {len(X_te)}")

    # full model
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mae = np.mean(np.abs(pred - y_te))
    r2 = model.score(X_te, y_te)
    print(f"\n[full model] GradientBoostingRegressor (sklearn defaults)")
    print(f"  MAE = {mae:.1f} Wh")
    print(f"  R^2 = {r2:.4f}")

    # distance-only linear baseline
    base = LinearRegression()
    base.fit(X_tr[["distance_km"]], y_tr)
    pred_base = base.predict(X_te[["distance_km"]])
    mae_base = np.mean(np.abs(pred_base - y_te))
    print(f"[baseline]  LinearRegression on distance_km only")
    print(f"  MAE = {mae_base:.1f} Wh")

    return model, mae, r2, mae_base


def vehicle_signal_diagnostic(routes):
    """Reproducible evidence for the 'no detectable vehicle signal' finding
    (see README / results summary). Fits on all 1,500 rows unless noted."""
    print("\n=== DATA FINDING: vehicle signal diagnostic ===")

    X_all = build_features(routes)
    y_all = routes["total_energy_wh"].values

    # 1. vehicle-feature importances at 5 dp + sum + share of total
    model = GradientBoostingRegressor(random_state=42).fit(X_all, y_all)
    veh_feats = ["mass_kg", "battery_capacity_wh", "max_payload_kg",
                 "base_efficiency_mult"]
    idx = [X_all.columns.get_loc(f) for f in veh_feats]
    imp = model.feature_importances_[idx]
    for f, v in zip(veh_feats, imp):
        print(f"  importance {f:<20} {v:.5f}")
    print(f"  sum of vehicle-feature importances: {imp.sum():.5f} "
          f"({imp.sum()/model.feature_importances_.sum()*100:.3f}% of total)")

    # 2. held-out MAE with vehicle-identity dummies vs without
    X_d = pd.concat([X_all, pd.get_dummies(routes["vehicle_type"],
                                           dtype=float)], axis=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X_d, y_all, test_size=0.2,
                                              random_state=42)
    mae_d = np.abs(GradientBoostingRegressor(random_state=42)
                   .fit(X_tr, y_tr).predict(X_te) - y_te).mean()
    X_tr0, X_te0, y_tr0, y_te0 = train_test_split(X_all, y_all,
                                                  test_size=0.2,
                                                  random_state=42)
    mae_0 = np.abs(GradientBoostingRegressor(random_state=42)
                   .fit(X_tr0, y_tr0).predict(X_te0) - y_te0).mean()
    print(f"  held-out MAE without dummies: {mae_0:.1f} Wh; "
          f"with dummies: {mae_d:.1f} Wh")

    # 3. GBM residuals by vehicle (fit on all 1,500)
    resid = y_all - model.predict(X_all)
    r = routes.assign(gbm_resid=resid)
    print("  GBM residual mean/std by vehicle (fit on all 1,500):")
    for vt, grp in r.groupby("vehicle_type"):
        print(f"    {vt:<14} mean {grp.gbm_resid.mean():+7.1f}  "
              f"std {grp.gbm_resid.std():7.1f}")

    # 4. linear route+payload-only residuals by vehicle (the ~502 Wh
    # spread is payload-nonlinearity, not a vehicle effect)
    route_cols = ["distance_km", "num_cells", "avg_slope_deg",
                  "max_slope_deg", "avg_rock_density",
                  "pct_high_hazard_cells", "payload_kg"]
    lin = LinearRegression().fit(X_all[route_cols], y_all)
    lin_resid = y_all - lin.predict(X_all[route_cols])
    r = r.assign(lin_resid=lin_resid)
    print("  linear (route+payload only) residual mean by vehicle:")
    means = {}
    for vt, grp in r.groupby("vehicle_type"):
        means[vt] = grp.lin_resid.mean()
        print(f"    {vt:<14} mean {means[vt]:+7.1f}")
    spread = max(means.values()) - min(means.values())
    print(f"  linear residual spread across vehicles: {spread:.1f} Wh")


def step2_recover_success_rule(routes):
    """Find the energy_margin_pct / safety_score thresholds that best
    reproduce mission_success. Coarse grid to locate the region, then snap
    to clean round-number thresholds (float grids accumulate error and
    produce phantom misclassifications)."""
    margin = routes["energy_margin_pct"].values
    safety = routes["safety_score"].values
    y = routes["mission_success"].values
    n = len(y)

    def accuracy(m_th, s_th):
        pred = ((margin > m_th) & (safety > s_th)).astype(int)
        return (pred == y).mean()

    print("\n=== Step 2: recover success rule ===")
    # coarse grid
    best = max(
        ((accuracy(m, s), m, s)
         for m in np.arange(-30, 31, 1.0)
         for s in np.arange(0.10, 0.96, 0.01)),
        key=lambda t: t[0])
    print(f"coarse best: acc={best[0]:.4f} margin>{best[1]:.1f} safety>{best[2]:.2f}")

    # snap to clean round thresholds near the coarse optimum (both strict
    # and non-strict inequalities), pick the exact reproducer
    m0, s0 = round(best[1]), round(best[2], 2)
    candidates = []
    for m in (m0 - 1, m0, m0 + 1):
        for s in (round(s0 - 0.02, 2), s0, round(s0 + 0.02, 2)):
            for m_op in ("gt", "ge"):
                for s_op in ("gt", "ge"):
                    pred = ((margin > m if m_op == "gt" else margin >= m)
                            & (safety > s if s_op == "gt" else safety >= s))
                    candidates.append(((pred == y).mean(), m, m_op, s, s_op))
    acc, m_th, m_op, s_th, s_op = max(candidates, key=lambda t: t[0])
    m_sym = ">" if m_op == "gt" else ">="
    s_sym = ">" if s_op == "gt" else ">="
    print(f"recovered rule: mission_success = (energy_margin_pct {m_sym} {m_th})"
          f" AND (safety_score {s_sym} {s_th})")

    pred = (((margin > m_th) if m_op == "gt" else (margin >= m_th))
            & ((safety > s_th) if s_op == "gt" else (safety >= s_th))).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    print(f"confusion vs real mission_success: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"accuracy = {acc:.4f}  ({int(round(acc*n))}/{n} rows reproduced)")

    return m_th, m_op, s_th, s_op


def step3_dispatch(routes, vehicles, m_th, m_op, s_th, s_op):
    """For each of the 1,500 trips: predict energy for all 4 vehicles,
    drop payload-overloaded and margin-failing vehicles, assign the
    smallest suitable vehicle, else NO-VEHICLE. Apply the Step 2 rule to the
    assignment for predicted success.

    DATA FINDING (discovered while building this step): in this dataset,
    trip energy depends on the route + payload only - not on the vehicle.
    The GBM assigns 0.000 importance to all four vehicle features, adding
    vehicle identity dummies does not improve held-out MAE (193.5 vs
    193.3 Wh), and GBM residuals by vehicle differ by only 30 Wh (within-
    vehicle std is 130-183 Wh). The linear-model offsets by vehicle were
    an artifact of payload nonlinearity (vehicles occupy different payload
    ranges). Trip energy is a function of route + payload only, so there
    is no meaningful energy comparison between vehicles for the same
    route. Dispatch rule (pure right-sizing): among vehicles that can
    carry the payload and clear the margin, pick the smallest
    battery_capacity_wh. No tie band, no energy ranking."""
    print("\n=== Step 3: dispatcher ===")

    # Refit the energy model on all 1,500 rows for dispatch use.
    # It was validated on the held-out 20% in Step 1 (MAE 193.3 Wh).
    X_all = build_features(routes)
    y_all = routes["total_energy_wh"].values
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_all, y_all)
    print("energy model refit on all 1,500 rows (validated on held-out in Step 1)")

    veh_specs = vehicles.set_index("vehicle_type")

    # per-vehicle predicted energy across all trips (4 predict calls)
    pred_energy = {}
    for vt in veh_specs.index:
        X_v = X_all.copy()
        X_v["mass_kg"] = veh_specs.loc[vt, "mass_kg"]
        X_v["battery_capacity_wh"] = veh_specs.loc[vt, "battery_capacity_wh"]
        X_v["max_payload_kg"] = veh_specs.loc[vt, "max_payload_kg"]
        X_v["base_efficiency_mult"] = veh_specs.loc[vt, "base_efficiency_mult"]
        pred_energy[vt] = model.predict(X_v)

    def passes_margin(margin_val):
        return margin_val > m_th if m_op == "gt" else margin_val >= m_th

    def passes_safety(safety_val):
        return safety_val > s_th if s_op == "gt" else safety_val >= s_th

    assigned = []          # per trip: vehicle_type or "NO-VEHICLE"
    assigned_pred_success = []
    assigned_pred_energy = []
    assigned_pred_margin = []
    n_no_vehicle = 0
    n_safety_fail_dispatched = 0
    for i, row in routes.iterrows():
        qualifiers = []
        for vt in veh_specs.index:
            batt = veh_specs.loc[vt, "battery_capacity_wh"]
            max_payload = veh_specs.loc[vt, "max_payload_kg"]
            if row["payload_kg"] > max_payload:
                continue                      # payload constraint
            energy = pred_energy[vt][i]
            margin = (batt - energy) / batt * 100
            if not passes_margin(margin):
                continue                      # energy-margin constraint
            qualifiers.append((energy, batt, vt, margin))
        if not qualifiers:
            n_no_vehicle += 1
            assigned.append("NO-VEHICLE")
            assigned_pred_success.append(0)
            assigned_pred_energy.append(np.nan)
            assigned_pred_margin.append(np.nan)
        else:
            # pure right-sizing: smallest battery among qualifiers. No
            # energy comparison between vehicles - trip energy is
            # vehicle-independent in this dataset (see data finding).
            qualifiers.sort(key=lambda q: q[1])
            energy, batt, vt, margin = qualifiers[0]
            assigned.append(vt)
            assigned_pred_energy.append(energy)
            assigned_pred_margin.append(margin)
            # margin passes by construction; unsafe routes are dispatched
            # anyway (vehicle choice cannot fix terrain) but their safety
            # failure counts as a failure against the assigned vehicle
            ok = passes_safety(row["safety_score"])
            assigned_pred_success.append(int(ok))
            if not ok:
                n_safety_fail_dispatched += 1

    routes = routes.assign(
        assigned_vehicle=assigned,
        assigned_pred_energy_wh=assigned_pred_energy,
        assigned_pred_margin_pct=assigned_pred_margin,
        assigned_pred_success=assigned_pred_success)

    # diagnostic: what do the NO-VEHICLE trips look like?
    no_veh = routes[routes["assigned_vehicle"] == "NO-VEHICLE"]
    print(f"NO-VEHICLE trips: {n_no_vehicle}; examples "
          f"(distance_km, payload_kg, pred_energy_cargo):")
    for i in no_veh.index[:3]:
        row = routes.loc[i]
        batt = veh_specs.loc["cargo_hauler", "battery_capacity_wh"]
        e = pred_energy["cargo_hauler"][i]
        print(f"  route {i}: {row['distance_km']:.2f} km, "
              f"{row['payload_kg']:.0f} kg, cargo pred energy {e:.0f} Wh "
              f"(margin {(batt-e)/batt*100:.1f}%)")
    return routes, n_no_vehicle, n_safety_fail_dispatched


def step4_results(routes, n_no_vehicle, n_safety_fail_dispatched, m_th,
                  held=None):
    """Aggregate and print + save results/summary.txt."""
    import os
    os.makedirs("results", exist_ok=True)

    n = len(routes)
    observed_rate = routes["mission_success"].mean()
    # predicted success over the SAME 1,500 trips (NO-VEHICLE trips score 0
    # here, but refusals are not failures - stated in the summary)
    dispatch_rate_all = routes["assigned_pred_success"].mean()
    veh_types = ["light_scout", "cargo_hauler", "crew_transport",
                 "swarm_builder"]
    dispatched = routes[routes["assigned_vehicle"].isin(veh_types)]
    dispatch_rate_dispatched = dispatched["assigned_pred_success"].mean()

    before = routes["vehicle_type"].value_counts()
    after = dispatched["assigned_vehicle"].value_counts()

    moved_off_swarm = ((routes["vehicle_type"] == "swarm_builder")
                       & (routes["assigned_vehicle"].isin(veh_types))
                       & (routes["assigned_vehicle"] != "swarm_builder")).sum()
    swarm_refused = ((routes["vehicle_type"] == "swarm_builder")
                     & (~routes["assigned_vehicle"].isin(veh_types))).sum()

    # observed-side failure decomposition (safety-first: a row failing both
    # counts as safety-driven, since no vehicle assignment could fix it)
    obs_failures = int((routes["mission_success"] == 0).sum())
    obs_safety_fail = int((routes["safety_score"] <= 0.5).sum())
    obs_energy_fail = obs_failures - obs_safety_fail
    fixed = obs_failures - (n_no_vehicle + n_safety_fail_dispatched)

    lines = []
    add = lines.append
    add("MARS FLEET DISPATCHER - RESULTS")
    add("=" * 60)
    add("")
    add(f"Trips evaluated:                    {n}")
    add(f"Observed success rate (measured):    {observed_rate*100:.1f}%")
    add(f"Simulated dispatcher success rate:   {dispatch_rate_all*100:.1f}%")
    add(f"Improvement in simulation:           "
        f"+{(dispatch_rate_all-observed_rate)*100:.1f} pp")
    add(f"  ...among dispatched trips: {dispatch_rate_dispatched*100:.1f}%  "
        f"(n={len(dispatched)})")
    add("")
    add("The dispatcher figure is a counterfactual simulation. Alternative")
    add("vehicle assignments were never actually driven, so these outcomes are")
    add("predicted under the dataset's own success rule, not observed.")
    add("")
    add("Refusals and safety failures (reported separately, never merged):")
    add(f"  NO-VEHICLE:       {n_no_vehicle:>4}  - refused: no vehicle clears "
        f"payload/battery with >{m_th:g}% margin")
    add(f"  ROUTE-INFEASIBLE: {n_safety_fail_dispatched:>4}  - safety_score "
        f"below threshold; dispatched anyway but")
    add("                         counted as FAILURES against the assigned "
        "vehicle (no vehicle choice can fix it)")
    add("  NO-VEHICLE refusals are NOT failures. In operations they would be")
    add("  re-planned (split load, different route), not driven.")
    add("")
    add("Failure breakdown (the observed 542 failed trips, decomposed):")
    add(f"  fixed by reassignment:               {fixed}")
    add(f"  remaining - energy-driven (NO-VEHICLE):       {n_no_vehicle}")
    add(f"  remaining - safety-driven (ROUTE-INFEASIBLE): {n_safety_fail_dispatched}")
    add(f"  -> {n_safety_fail_dispatched}/{n_no_vehicle + n_safety_fail_dispatched} "
        f"of the remaining failures are safety-driven and")
    add("     unfixable by ANY vehicle assignment: the dispatcher is at the")
    add("     ceiling of what assignment can fix. The rest needs route")
    add("     re-planning or infrastructure, not better dispatch.")
    add("")
    add("65 of the 542 observed failures breach both the energy-margin and")
    add("the safety threshold. These are counted as safety-driven (safety-first")
    add("convention), so 'energy-fail 353' means failures that are not also")
    add("safety failures; a raw margin<=5 count gives 418.")
    if held is not None:
        add("")
        add(f"Dispatcher on HELD-OUT trips (model trained on the other 1,200 only):")
        add(f"  observed success on those {held['n']} trips:  "
            f"{held['obs_ok']} ({held['obs_ok']/held['n']*100:.1f}%)")
        add(f"  dispatched predicted success: "
            f"{held['disp_ok']} ({held['disp_ok']/held['n']*100:.1f}%)")
        add(f"  NO-VEHICLE {held['no_veh']} | safety-fail {held['safety_fail']}")
    add("")
    add("Assignments per vehicle (before -> after):")
    for vt in veh_types:
        add(f"  {vt:<14} {before.get(vt, 0):>5} -> {after.get(vt, 0):>5}")
    add("")
    add(f"Trips moved off swarm_builder: {moved_off_swarm}")
    add(f"swarm_builder trips refused:   {swarm_refused}")
    add("")
    add("DATA FINDING - no detectable vehicle signal in the data:")
    add("  Vehicle-feature importance is negligible (~1.8e-5 combined across")
    add("  mass_kg, battery_capacity_wh, max_payload_kg, base_efficiency_mult);")
    add("  vehicle-identity dummies do not improve held-out MAE (193.5 vs")
    add("  193.3 Wh); GBM residuals differ by only 30 Wh across vehicles")
    add("  (within-vehicle std 130-183 Wh). Inferred from model behaviour;")
    add("  the generator source is not in the repo. There is therefore no")
    add("  meaningful energy comparison between vehicles for the same route.")
    add("  Dispatcher policy (pure right-sizing): among vehicles that can")
    add("  carry the payload and clear the margin, pick the smallest")
    add("  battery_capacity_wh. No tie band, no energy ranking.")

    text = "\n".join(lines)
    print("\n=== Step 4: results ===\n" + text)
    with open("results/summary.txt", "w") as f:
        f.write(text + "\n")
    print("\n[saved] results/summary.txt")

    return obs_energy_fail, obs_safety_fail


def heldout_dispatch_eval(routes, vehicles, m_th, m_op, s_th, s_op):
    """Dispatch evaluation on the 300 held-out routes. The energy model is
    trained ONLY on the 1,200 training routes (same 80/20 split, seed 42, as
    Step 1) - these 300 trips were never in the model's training set.
    Everything else is identical to Step 3 (payload filter, margin rule,
    smallest battery, safety counted as a failure)."""
    print("\n=== Step 4b: dispatcher on held-out (unseen) trips ===")
    X = build_features(routes)
    y = routes["total_energy_wh"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42)
    model = GradientBoostingRegressor(random_state=42).fit(X_tr, y_tr)
    veh_specs = vehicles.set_index("vehicle_type")

    def passes_margin(m):
        return m > m_th if m_op == "gt" else m >= m_th

    def passes_safety(s):
        return s > s_th if s_op == "gt" else s >= s_th

    n = len(X_te)
    ok = no_veh = safety_fail = 0
    for i in X_te.index:
        row = routes.loc[i]
        qualifiers = []
        for vt in veh_specs.index:
            batt = veh_specs.loc[vt, "battery_capacity_wh"]
            if row["payload_kg"] > veh_specs.loc[vt, "max_payload_kg"]:
                continue
            X_v = X.loc[[i]].copy()
            X_v["mass_kg"] = veh_specs.loc[vt, "mass_kg"]
            X_v["battery_capacity_wh"] = batt
            X_v["max_payload_kg"] = veh_specs.loc[vt, "max_payload_kg"]
            X_v["base_efficiency_mult"] = veh_specs.loc[vt, "base_efficiency_mult"]
            energy = model.predict(X_v[X.columns])[0]
            margin = (batt - energy) / batt * 100
            if passes_margin(margin):
                qualifiers.append((batt, vt))
        if not qualifiers:
            no_veh += 1
        else:
            vt = min(qualifiers)[1]          # smallest battery that qualifies
            if passes_safety(row["safety_score"]):
                ok += 1
            else:
                safety_fail += 1
    obs_ok = int(routes.loc[X_te.index, "mission_success"].sum())
    print(f"  held-out trips: {n}  (model trained on the other 1,200 only)")
    print(f"  observed success (real):      {obs_ok}  ({obs_ok/n*100:.1f}%)")
    print(f"  dispatched predicted success: {ok}  ({ok/n*100:.1f}%)")
    print(f"  NO-VEHICLE: {no_veh}   safety-fail: {safety_fail}")
    return dict(n=n, obs_ok=obs_ok, disp_ok=ok, no_veh=no_veh,
                safety_fail=safety_fail)


def step5_chart(routes, n_no_vehicle, n_safety_fail_dispatched,
                obs_energy_fail, obs_safety_fail, held):
    """results/dispatch.png: two panels.
    Panel 1: success rate by vehicle + overall, observed vs dispatched.
    Per-vehicle dispatched rate counts safety failures as failures.
    Panel 2: stacked outcome bars (success / energy-fail / safety-fail),
    observed vs dispatched - so the safety residual is visible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    veh_types = ["light_scout", "cargo_hauler", "crew_transport",
                 "swarm_builder"]
    veh_labels = [v.replace("_", " ") for v in veh_types]

    dispatched = routes[routes["assigned_vehicle"].isin(veh_types)]
    obs_by_veh = [routes.loc[routes["vehicle_type"] == v,
                             "mission_success"].mean() * 100 for v in veh_types]
    # among trips assigned to each vehicle: safety failures count as
    # failures, so this is not a flat 100%
    disp_by_veh = [dispatched.loc[dispatched["assigned_vehicle"] == v,
                                  "assigned_pred_success"].mean() * 100
                   for v in veh_types]
    obs_overall = routes["mission_success"].mean() * 100
    disp_overall = routes["assigned_pred_success"].mean() * 100

    obs_stacks = [int(routes["mission_success"].sum()),
                  obs_energy_fail, obs_safety_fail]
    disp_stacks = [int(dispatched["assigned_pred_success"].sum()),
                   n_no_vehicle, n_safety_fail_dispatched]

    plt.rcParams.update({"font.size": 15, "font.weight": "bold",
                         "axes.labelweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # ---- Panel 1: success rate, observed vs simulated ----
    ax = axes[0]
    x = np.arange(6)
    w = 0.35
    obs_vals = obs_by_veh + [obs_overall,
                             held["obs_ok"] / held["n"] * 100]
    disp_vals = disp_by_veh + [disp_overall,
                               held["disp_ok"] / held["n"] * 100]
    b1 = ax.bar(x - w / 2, obs_vals, w,
                label="Observed (measured)", color="#5b7db1")
    b2 = ax.bar(x + w / 2, disp_vals, w,
                label="Simulated dispatcher (predicted)", color="#e0a031")
    ax.set_xticks(x, veh_labels + ["OVERALL", "HELD-OUT 300"])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Success rate by vehicle: observed vs simulated dispatcher",
                 fontsize=18, pad=12)
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.1f}%",
                        (rect.get_x() + rect.get_width() / 2,
                         rect.get_height() + 2),
                        ha="center", fontsize=13)
    ax.legend(fontsize=14, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # ---- Panel 2: stacked outcomes ----
    ax = axes[1]
    colors = {"success": "#4c9a5d", "energy-fail": "#e0a031",
              "safety-fail": "#c94f4f"}
    bottom = np.zeros(2)
    for label, obs_v, disp_v, color in [
            ("success", obs_stacks[0], disp_stacks[0], colors["success"]),
            ("energy-fail", obs_stacks[1], disp_stacks[1], colors["energy-fail"]),
            ("safety-fail", obs_stacks[2], disp_stacks[2], colors["safety-fail"])]:
        bars = ax.bar([0, 1], [obs_v, disp_v], 0.5, bottom=bottom,
                      label=label, color=color)
        for rect, v in zip(bars, [obs_v, disp_v]):
            ax.annotate(f"{v:,}",
                        (rect.get_x() + rect.get_width() / 2,
                         rect.get_y() + rect.get_height() / 2),
                        ha="center", va="center", fontsize=14, color="white")
        bottom += np.array([obs_v, disp_v])
    ax.set_xticks([0, 1], ["Observed", "Dispatched (simulated)"])
    ax.set_ylabel("Trips (of 1,500)")
    ax.set_ylim(0, 1600)
    ax.set_title("Outcomes: success / energy-fail / safety-fail",
                 fontsize=18, pad=12)
    ax.legend(fontsize=14, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Mars Fleet Dispatcher", fontsize=22, y=1.02)
    fig.text(0.5, 0.005,
             "Counterfactual simulation: alternative vehicle assignments "
             "were never driven; outcomes are predicted under the dataset's "
             "success rule. HELD-OUT 300: the dispatcher run on 300 trips "
             "the energy model never trained on (same split as the held-out "
             "validation in Step 1).",
             ha="center", fontsize=13, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig("results/dispatch.png", bbox_inches="tight", dpi=120)
    print("\n[saved] results/dispatch.png")


def main():
    routes, vehicles = load_data()
    step1_energy_model(routes)
    vehicle_signal_diagnostic(routes)
    m_th, m_op, s_th, s_op = step2_recover_success_rule(routes)
    routes, n_no_vehicle, n_safety_fail = step3_dispatch(
        routes, vehicles, m_th, m_op, s_th, s_op)
    held = heldout_dispatch_eval(routes, vehicles, m_th, m_op, s_th, s_op)
    obs_energy_fail, obs_safety_fail = step4_results(
        routes, n_no_vehicle, n_safety_fail, m_th, held)
    step5_chart(routes, n_no_vehicle, n_safety_fail,
                obs_energy_fail, obs_safety_fail, held)


if __name__ == "__main__":
    main()
