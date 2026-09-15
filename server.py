"""Mars Trip Checker - tiny demo web app (Python stdlib only, no new deps).

Trains the same energy model as dispatcher.py at startup, then serves
index.html and answers /predict with the dispatcher's decision for a
user-entered trip.

Run:  ./venv/bin/python server.py   ->   http://localhost:8000
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from dispatcher import load_data, build_features

ROUTES, VEHICLES = load_data()
X_ALL = build_features(ROUTES)
Y = ROUTES["total_energy_wh"].values

MODEL = GradientBoostingRegressor(random_state=42).fit(X_ALL, Y)
# route safety estimated from the hazard-cell share (r=-0.99 in the data)
SAFETY_FIT = LinearRegression().fit(
    X_ALL[["pct_high_hazard_cells"]], ROUTES["safety_score"].values)

# terrain presets: slope (deg), rock cover (0-1), share of high-hazard
# cells, dominant terrain type
TERRAIN = {
    "easy":  dict(name="Easy ground", slope=8.0, rock=0.40, hazard=0.10,
                  regolith=1.0, rockfield=0.0),
    "rough": dict(name="Rough ground", slope=12.0, rock=0.62, hazard=0.25,
                  regolith=1.0, rockfield=0.0),
    "rocky": dict(name="Rock field", slope=16.0, rock=0.85, hazard=0.60,
                  regolith=0.0, rockfield=1.0),
}
VEHICLE_NAMES = {
    "light_scout": "the scout rover",
    "cargo_hauler": "the big cargo hauler",
    "crew_transport": "the crew transporter",
    "swarm_builder": "the swarm robot",
}


def predict_trip(distance_km, payload_kg, terrain_key):
    t = TERRAIN[terrain_key]
    safety = float(SAFETY_FIT.predict([[t["hazard"]]])[0])
    row = {
        "distance_km": distance_km,
        "num_cells": max(3, round(distance_km * 42)),
        "avg_slope_deg": t["slope"],
        "max_slope_deg": t["slope"] * 2.2,
        "avg_rock_density": t["rock"],
        "pct_high_hazard_cells": t["hazard"],
        "payload_kg": payload_kg,
        "crater_floor": 0.0,
        "regolith_plain": t["regolith"],
        "rock_field": t["rockfield"],
    }
    vehicles = {}
    for vt, spec in VEHICLES.set_index("vehicle_type").iterrows():
        r = dict(row)
        r.update(mass_kg=spec.mass_kg, battery_capacity_wh=spec.battery_capacity_wh,
                 max_payload_kg=spec.max_payload_kg,
                 base_efficiency_mult=spec.base_efficiency_mult)
        energy = float(MODEL.predict(pd.DataFrame([r])[X_ALL.columns])[0])
        margin = (spec.battery_capacity_wh - energy) / spec.battery_capacity_wh * 100
        vehicles[vt] = {
            "name": VEHICLE_NAMES[vt],
            "max_payload_kg": float(spec.max_payload_kg),
            "battery_capacity_wh": float(spec.battery_capacity_wh),
            "energy_wh": round(energy),
            "battery_pct": round(energy / spec.battery_capacity_wh * 100),
            "can_carry": payload_kg <= spec.max_payload_kg,
            "can_finish": margin > 5,
        }
    qualifiers = [vt for vt, v in vehicles.items()
                  if v["can_carry"] and v["can_finish"]]
    assigned = min(qualifiers, key=lambda vt: vehicles[vt]["battery_capacity_wh"]) \
        if qualifiers else None
    if assigned is None:
        if any(v["can_carry"] for v in vehicles.values()):
            why = "range"       # someone can carry it, but nobody has the battery
        else:
            why = "too_heavy"   # no rover can carry the payload at all
    else:
        why = None
    return {
        "vehicles": vehicles,
        "assigned": assigned,
        "safe_route": safety > 0.5,
        "why": why,
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open("index.html", "rb") as f:
                self.wfile.write(f.read())
        elif parsed.path == "/predict":
            q = parse_qs(parsed.query)
            try:
                distance = min(5.0, max(0.2, float(q.get("distance", ["1.5"])[0])))
                payload = min(650.0, max(1.0, float(q.get("payload", ["20"])[0])))
                terrain = q.get("terrain", ["easy"])[0]
                if terrain not in TERRAIN:
                    raise ValueError(terrain)
            except (ValueError, IndexError):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "bad input"}')
                return
            result = predict_trip(distance, payload, terrain)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        print("  ", fmt % args)


if __name__ == "__main__":
    print("Mars Trip Checker -> http://localhost:8000  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
