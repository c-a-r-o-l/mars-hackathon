# Adversarial verification critique — Mars Fleet Dispatcher

Method: every number below was independently recomputed from
`track-b2-mobility-ai4mars/track-b2-mobility-ai4mars/{routes.csv,vehicles.csv}`
with `./venv/bin/python` (sklearn 1.9.1), plus line-by-line code inspection of
`dispatcher.py`. The project's own printed numbers were never trusted.

## 1. Verdict summary

| # | Claim | Verdict | Key evidence (recomputed) |
|---|---|---|---|
| A | Step 1 model | VERIFIED | 14 features = 7 route + 4 vehicle + 3 one-hot `dominant_terrain` (3 unique values, not 4). Split 1,200/300, seed 42. Held-out MAE 193.3 Wh, R² 0.9851; distance-only linear MAE 789.3 Wh. `energy_margin_pct`, `safety_score`, `mission_success`, `route_type` never enter `build_features`. |
| B | Success rule | VERIFIED | `(margin>5) & (safety>0.5)` reproduces 1,500/1,500 exactly (TP=958, TN=542, FP=FN=0). Code really does a coarse grid (best 0.9987 at margin>5.0, safety>0.50) then snaps; strict `>` matters — 3 rows sit at `safety_score == 0.5` exactly, 0 at `margin == 5`. |
| C1 | Vehicle features 0.000 importance | VERIFIED (with note) | Actual importances: mass 2e-6, battery 4.6e-5, max_payload 4e-6, base_efficiency 4e-6 — total 5.6e-5 (0.006%). They print as 0.000 at 3 dp, so the wording is true-as-printed; "negligible" is the precise word. |
| C2 | Vehicle dummies don't help | VERIFIED | Re-running split with 4 vehicle-identity dummies: MAE 193.5 vs 193.3 Wh. Route+payload-only GBM even scores R² 0.9858 (vs 0.9851 with vehicle features). |
| C3 | Residuals by vehicle | VERIFIED | GBM (fit all 1,500): per-vehicle mean residuals −7.5/+8.0/−16.9/+13.0 → spread 29.9 Wh; within-vehicle std 129.8–182.7 (claimed 130–183). Linear route+payload-only model: mean residuals +30.7/+5.7/+253.3/−248.6 → spread 501.9 Wh (claimed ~502). |
| C4 | "Generator's energy formula ignores vehicle specs" | PARTIALLY SUPPORTED | Not proven from source: `generate_dataset.py` is absent (track README says "is included" — it isn't, in any folder). Evidence from data is strong but inferential: cross-vehicle transfer MAE 214–238 Wh (in-payload-range) vs within-data 190.7 Wh; zero dummies gain. See §2. |
| D1 | 63.9% → 87.1%, +23.2 pp, 87.4% (n=1,494) | VERIFIED | Independent reimplementation of the dispatch loop: 958/1500 = 63.87%; predicted-success 1,306/1,500 = 87.07%; among-dispatched 1,306/1,494 = 87.42%. |
| D2 | NO-VEHICLE 6, ROUTE-INFEASIBLE 188 | VERIFIED | 6 refusals (all 6 are observed failures), 188 dispatched with `safety_score <= 0.5`, counted as predicted failures. |
| D3 | Breakdown 348 + 6 + 188 = 542 | VERIFIED | Direct count of observed-fail→predicted-success = 348; 542 = 348+6+188 exactly; 958+348 = 1,306 holds. No observed-successful trip is refused (checked: 0). |
| D4 | Assignments 342→517, 394→282, 353→588, 411→107; 324 moved; 0 refused | VERIFIED | All counts reproduce; 324 = swarm_builder rows reassigned to another vehicle, 0 swarm rows refused. |
| D5 | Chart values | VERIFIED (code) | Chart code computes 90.3 / 82.6 / 85.4 / 97.2 per-vehicle dispatched rates, overall 87.07%; stacks 958/353/189 observed, 1,306/6/188 dispatched. Caveat: PNG annotations use `{:.0f}` → displays "97%" not "97.2%", "64%" not "63.9%". See §5. |
| E | Internal consistency | VERIFIED | `results/summary.txt` is byte-identical to the printed Step 4 block. grep for "cheapest", "50 Wh", "tie", "281", "467", "502", "233", "104": only the current "No tie band, no energy ranking" phrasing exists. No stale figures anywhere. |
| F | Honesty/framing | VERIFIED | 87.1% is labeled "simulated"/"predicted"/"counterfactual" in README, summary.txt, chart legend AND chart footer. Safety failures shown as own segment and own line. "Validated vs not validated" box is accurate (300 held-out rows vs refit-on-all-1,500 applied back). |
| G | No circularity | VERIFIED | Success rule derived from real `energy_margin_pct`/`safety_score` columns. Dispatcher uses model only for energy; predicted success = (predicted margin > 5) AND real `safety_score > 0.5`. `safety_score` is a raw CSV column, never computed from the model. |
| H | Bug hunt | NO BUGS FOUND | All printed numbers match my independent recomputation. Index alignment is safe (post-merge RangeIndex 0..1499 vs positional predict arrays). Fragility nits in §4. |

## 2. Detailed findings (anything not fully verified)

### C4 — the generator claim is an inference, not proven
`generate_dataset.py` is missing from the repo (track README promises it).
The strongest statement the evidence supports: "In this dataset,
`total_energy_wh` shows no detectable dependence on vehicle specs — identity
dummies add zero predictive value, vehicle-feature importance is ~5.6e-5,
and a model trained on three vehicles predicts the fourth at near-within
MAE (214–238 vs 191 Wh, payload ranges overlapping)." The README header
"the generator's energy formula ignores vehicle specs" and the sentence
"the vehicle's specs ... never enter the energy column" assert a fact about
code nobody can see. Recommended: reword to "no detectable vehicle signal"
or add "inferred from the data; generator source not in repo". Same applies
to "This is the generator's real rule" (Step 2): 1,500/1,500 exact
reproduction makes this highly likely, but it is still an inference.
Extra nuance, in the project's favor: payloads 250–600 kg are carried only
by cargo_hauler, so vehicle-independence there is untestable — but the
dispatcher never compares vehicles in that band (only cargo_hauler can
qualify), so the untestable region cannot affect any assignment.

### C1 — exact importances
Not literally 0.0: mass_kg 2.0e-6, battery_capacity_wh 4.6e-5 (largest of
the four), max_payload_kg 4.0e-6, base_efficiency_mult 4.0e-6. Sum ≈ 0.006%
of importance. Print at 3 decimals = "0.000". Consequence: per-trip
predicted energy differs across vehicles by at most 21 Wh (max over 1,500
trips) — consistent with "no meaningful energy comparison between
vehicles", since 21 Wh ≪ 193 Wh MAE and ≪ 130–183 Wh within-vehicle std.

### The 65 double-fail rows (convention, not an error)
65 of the 542 failures have BOTH `margin <= 5` AND `safety <= 0.5`. The
code (and the 353/189 split) counts them as safety-driven ("safety-first",
stated in a code comment only). So "energy-fail 353" means "failures that
are not safety failures" — a raw `margin <= 5` count gives 418. Not wrong,
but the README/chart never say this; a judge recomputing `margin <= 5`
will get 418 and may think the chart is wrong. One clarifying line fixes it.

### Summary.txt vs printed output
Byte-identical apart from a leading `\n` added by `print` (verified).

## 3. "Nothing faked?" statement

No. Nothing is faked, hallucinated, or silently massaged. Every headline
number in README.md, results/summary.txt and the chart code reproduces
exactly from the raw CSVs via the documented pipeline, which I reimplemented
independently. The 87.1% is honestly labeled as a simulated counterfactual
in all three artifacts; safety failures are displayed, not hidden; the
validated/unvalidated distinction is accurate. The only claims that go
beyond what is strictly provable are (a) attributing the vehicle-independence
finding to "the generator's formula" when the generator source is absent,
and (b) calling the recovered rule "the generator's real rule" — both are
strong, well-supported inferences but are inferences. All other wording
matches the code and the data, including the unglamorous numbers (0 refused,
188 unfixable safety failures).

## 4. Improvements with very limited time

1. **Make the C-claim numbers reproducible (10–15 min) — HIGHEST VALUE.**
   The importances, dummies-MAE and residual analyses appear in README and
   summary but are computed NOWHERE in dispatcher.py. A judge running the
   script can't reproduce the "data finding". Add ~30 lines to Step 1/3
   that compute and print: the 4 importances (5 dp), dummies MAE, per-vehicle
   GBM residual means/std, linear route+payload-only residual means.
2. **Chart annotation precision (2 min).** Change `{:.0f}%` to `{:.1f}%`
   (and stacks are fine) so the PNG shows 63.9/87.1/97.2 matching the README,
   not 64/87/97. Currently the PNG visibly disagrees with the README's
   1-decimal numbers.
3. **One-line clarification of the 353 vs 418 energy-fail count (3 min).**
   Add to README/summary: "65 rows fail both margin and safety; they are
   counted as safety-driven (safety-first convention)". Preempts a judge
   recomputing `margin<=5` and getting 418.
4. **Soften the generator wording (5 min).** "Data finding: the generator's
   energy formula ignores vehicle specs" → "no detectable vehicle signal in
   the data (generator source not in repo; inference from model behavior)".
   Also soften "This is the generator's real rule" → "reproduces the real
   mission_success exactly (1,500/1,500), consistent with this being the
   generator's rule". Removes the only attackable certainty claims.
5. **Robustness one-liner for the 87.1% (5 min, already computed).** The
   result is not knife-edge: only 5 assigned trips have margin ≤ 6% and only
   3 refusals have best margin in (3,5]. Print "dispatch outcome robust to
   ±1 pp of energy-model error (8 trips near threshold)". Cheap defense of
   the counterfactual against "your model error flips the result".
6. **Make the failure decomposition self-evident (5 min).** Replace
   `fixed = obs_failures - (n_no_vehicle + n_safety_fail_dispatched)` with
   `fixed = ((mission_success==0) & (assigned_pred_success==1)).sum()` and
   assert the sum equals 542. Current formula is correct only because all 6
   refusals are observed failures (true today); the direct count can't go
   silently wrong.
7. **Use recovered thresholds in Step 4 (3 min).** `obs_safety_fail`
   hardcodes `safety_score <= 0.5` instead of `s_th/s_op` from Step 2.
   Works today (rule = >0.5) but is a latent inconsistency if thresholds
   ever change.
8. **One extra seed for the energy model (10 min, optional).** Rerun the
   80/20 MAE with seed 43 or 0 and report "193 ± x Wh across seeds" — kills
   "your 193.3 is seed luck" questions.

## 5. Chart discrepancies

I cannot visually inspect the PNG; verification is by reading `step5_chart`
and recomputing what it draws. Code-level values are all correct
(panel 1: 90.3/82.6/85.4/97.2/87.1 dispatched, 70.8/84.0/83.9/21.7/63.9
observed; panel 2: 958/353/189 vs 1,306/6/188). Two discrepancies between
what the chart displays and the README text, both caused by `{:.0f}`
rounding: percentages display as integers (97% for 97.2%, 64% for 63.9%,
87% for 87.1%), and observed swarm_builder 21.7% displays as 22%. Fix in
item 2 above. Chart footer correctly states the counterfactual nature.
