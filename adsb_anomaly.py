"""
ADS-B track anomaly detection: physics rules vs Isolation Forest.

  python adsb_anomaly.py                      uses adsb_tracks.csv
  python adsb_anomaly.py --csv my.csv --runs 30 --fpr 0.01

Needs: pip install pandas scikit-learn

How it is evaluated
  1. Features are computed for every pair of consecutive messages of one aircraft
     (does the position change match the reported speed / heading / climb rate?).
  2. The first 60% of the recording (by time) is the clean training set.
  3. The last 40% gets synthetic attacks injected into a fraction of its messages
     (teleport, speed spoof, altitude spoof, heading spoof), repeated over many seeds.
  4. Rules and Isolation Forest are scored on the same data: recall per attack type
     and false-positive rate on the untouched messages.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

KINDS = ["teleport", "speed", "altitude", "heading"]
FEATURES = ["implied_speed", "speed_err", "vr_err", "heading_err", "accel", "turn_rate", "dt"]
NEED = ["t", "icao", "lat", "lon", "altitude", "speed", "heading", "vertical_rate"]


# ---------- geometry ----------
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def wrap180(x):
    return (x + 180) % 360 - 180


# ---------- data ----------
def load(path, home=(12.95, 77.62), max_range_km=450.0):
    df = pd.read_csv(path)
    df = df.dropna(subset=NEED).copy()
    far = haversine_km(home[0], home[1], df["lat"], df["lon"]) > max_range_km
    if far.any():
        print("dropped %d messages beyond %.0f km (decoder position glitches, kept out of training)"
              % (far.sum(), max_range_km))
    df = df[~far].copy()
    df["icao"] = df["icao"].astype(str)
    df = df.sort_values("t").reset_index(drop=True)
    return df


def make_features(df):
    """One row per consecutive message pair of the same aircraft (row = the later message)."""
    df = df.sort_values(["icao", "t"]).reset_index(drop=True)
    df["row_id"] = df.index
    prev = df.groupby("icao").shift(1)
    dt = df["t"] - prev["t"]
    dist_km = haversine_km(prev["lat"], prev["lon"], df["lat"], df["lon"])
    implied = dist_km / (dt / 3600.0) / 1.852  # knots
    brg = bearing_deg(prev["lat"], prev["lon"], df["lat"], df["lon"])

    f = pd.DataFrame(index=df.index)
    f["icao"], f["t"], f["row_id"], f["prev_id"] = df["icao"], df["t"], df["row_id"], prev["row_id"]
    f["dt"] = dt
    f["implied_speed"] = implied
    f["speed_err"] = implied - (df["speed"] + prev["speed"]) / 2
    alt_rate = (df["altitude"] - prev["altitude"]) / dt * 60.0  # ft/min
    f["vr_err"] = alt_rate - (df["vertical_rate"] + prev["vertical_rate"]) / 2
    herr = wrap180(brg - df["heading"]).abs()
    f["heading_err"] = herr.where(implied > 50, 0.0)  # heading is meaningless when almost stationary
    f["accel"] = (df["speed"] - prev["speed"]).abs() / dt
    f["turn_rate"] = wrap180(df["heading"] - prev["heading"]).abs() / dt
    f = f[(f["dt"] > 0.2) & (f["dt"] < 30)].replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["prev_id"])
    f["prev_id"] = f["prev_id"].astype(int)
    return f


# ---------- detectors ----------
def rule_flags(f, speed_tol, vr_tol, hdg_tol):
    return (
        (f["implied_speed"] > 800)
        | (f["speed_err"].abs() > speed_tol)
        | (f["vr_err"].abs() > vr_tol)
        | (f["heading_err"] > hdg_tol)
    ).to_numpy()


def fit_forest(train_f, fpr, seed):
    scaler = StandardScaler().fit(train_f[FEATURES])
    model = IsolationForest(n_estimators=300, random_state=seed).fit(scaler.transform(train_f[FEATURES]))
    train_scores = -model.score_samples(scaler.transform(train_f[FEATURES]))
    thr = np.quantile(train_scores, 1 - fpr)

    def predict(f):
        return (-model.score_samples(scaler.transform(f[FEATURES]))) > thr

    return predict


# ---------- attack injection ----------
def inject(df, rng, frac):
    df = df.copy()
    n = max(3, int(frac * len(df)))
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    kinds = {}
    for i in idx:
        k = KINDS[rng.integers(len(KINDS))]
        s = rng.choice([-1, 1])
        if k == "teleport":
            df.loc[i, "lat"] += s * rng.uniform(0.15, 0.8)
            df.loc[i, "lon"] += s * rng.uniform(0.15, 0.8)
        elif k == "speed":
            df.loc[i, "speed"] = max(0.0, df.loc[i, "speed"] + s * rng.uniform(150, 350))
        elif k == "altitude":
            df.loc[i, "altitude"] += s * rng.uniform(3000, 12000)
        else:
            df.loc[i, "heading"] = (df.loc[i, "heading"] + rng.uniform(90, 180)) % 360
        kinds[i] = k
    return df, kinds


def evaluate(test_df, detectors, runs, frac, seed):
    rows = []
    for r in range(runs):
        rng = np.random.default_rng(seed + r)
        tdf, kinds = inject(test_df.reset_index(drop=True), rng, frac)
        f = make_features(tdf)
        # row_id after make_features is the index of the sorted frame; map injected rows via (icao, t)
        tdf_sorted = tdf.sort_values(["icao", "t"]).reset_index(drop=True)
        inj_keys = {(tdf.loc[i, "icao"], tdf.loc[i, "t"]): k for i, k in kinds.items()}
        keys = list(zip(tdf_sorted["icao"], tdf_sorted["t"]))
        inj_ids = {j: inj_keys[key] for j, key in enumerate(keys) if key in inj_keys}

        is_att = f["row_id"].isin(inj_ids).to_numpy()
        is_follow = f["prev_id"].isin(inj_ids).to_numpy() & ~is_att  # ambiguous pair after an attack
        keep = ~is_follow
        kind = np.array([inj_ids.get(i, "") for i in f["row_id"]])

        for name, fn in detectors.items():
            flag = fn(f)
            clean = keep & ~is_att
            rows.append({"method": name, "kind": "clean (false positives)",
                         "hit": flag[clean].mean() if clean.any() else np.nan})
            for k in KINDS:
                m = keep & is_att & (kind == k)
                if m.any():
                    rows.append({"method": name, "kind": k, "hit": flag[m].mean()})
    res = pd.DataFrame(rows).groupby(["kind", "method"])["hit"].mean().unstack("method")
    return res.reindex(KINDS + ["clean (false positives)"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="adsb_tracks.csv")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--frac", type=float, default=0.08, help="fraction of test messages to attack")
    ap.add_argument("--fpr", type=float, default=0.01, help="target false-positive rate for the forest")
    ap.add_argument("--speed-tol", type=float, default=120.0)
    ap.add_argument("--vr-tol", type=float, default=2500.0)
    ap.add_argument("--hdg-tol", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--home", default="12.95,77.62", help="receiver lat,lon")
    ap.add_argument("--max-range-km", type=float, default=450.0)
    a = ap.parse_args()

    df = load(a.csv, tuple(float(x) for x in a.home.split(",")), a.max_range_km)
    print("messages with full data: %d, aircraft: %d, span: %.0f s" %
          (len(df), df["icao"].nunique(), df["t"].max() - df["t"].min()))
    cut = df["t"].quantile(0.6)
    train_df, test_df = df[df["t"] <= cut], df[df["t"] > cut]

    train_f = make_features(train_df)
    if len(train_f) < 30:
        raise SystemExit("Only %d training pairs. Log a longer recording (30+ min) and rerun." % len(train_f))
    print("training pairs: %d, test messages: %d\n" % (len(train_f), len(test_df)))

    print("clean-data feature summary (training set):")
    print(train_f[FEATURES].describe().loc[["mean", "std", "min", "max"]].round(1).to_string(), "\n")

    forest = fit_forest(train_f, a.fpr, a.seed)
    detectors = {
        "rules": lambda f: rule_flags(f, a.speed_tol, a.vr_tol, a.hdg_tol),
        "isolation_forest": forest,
    }
    res = evaluate(test_df, detectors, a.runs, a.frac, a.seed)
    print("Detection rate (recall) per attack type; last row = false-positive rate on clean messages")
    print((res * 100).round(1).astype(str).add("%").to_string())

    tf = make_features(test_df)
    out = tf.assign(rule_flag=rule_flags(tf, a.speed_tol, a.vr_tol, a.hdg_tol),
                    forest_flag=forest(tf))
    out.to_csv("anomaly_scores.csv", index=False)
    print("\nclean test pairs flagged: rules %d / forest %d of %d  (saved anomaly_scores.csv)" %
          (out["rule_flag"].sum(), out["forest_flag"].sum(), len(out)))


if __name__ == "__main__":
    main()
