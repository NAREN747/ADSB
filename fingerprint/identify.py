"""Use a trained fingerprint model: identify the transmitter of each burst and compare it with
the ICAO the message claims. A confident mismatch is a spoofing / ghost-aircraft indicator that
complements the kinematic checks in adsb_anomaly.py (teleport, speed, altitude, heading).

    python -m fingerprint.identify --run runs/cnn1 --data iq_bursts_new --out rf_identity.csv

Closed set: the model only knows the aircraft it was trained on. A claimed ICAO that is not in
the training set is reported as known=False and is never flagged.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from . import dataset as D
from .models import build


def load_run(run, device="cpu"):
    import torch
    cfg = json.load(open(os.path.join(run, "config.json")))
    feat_dim = len(cfg["feat_mu"]) if cfg["fuse_features"] else 0
    model = build(cfg["model"], len(cfg["classes"]), feat_dim)
    model.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=device))
    return model.to(device).eval(), cfg


def predict_proba(model, cfg, iq, start, device="cpu"):
    """iq: raw cropped bursts (N, L, 2) as stored by the capture block -> (N, n_classes) probabilities."""
    import torch
    X = D.preprocess(iq, start, remove_cfo=cfg["remove_cfo"])
    feats = None
    if cfg["fuse_features"]:
        from .features import extract_features
        feats = torch.from_numpy(((extract_features(X) - np.array(cfg["feat_mu"])) / np.array(cfg["feat_sd"])).astype(np.float32))
    xt = torch.from_numpy(np.ascontiguousarray(X.transpose(0, 2, 1)))
    out = []
    with torch.no_grad():
        for i in range(0, len(xt), 512):
            f = None if feats is None else feats[i:i + 512].to(device)
            out.append(torch.softmax(model(xt[i:i + 512].to(device), f), -1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="rf_identity.csv")
    ap.add_argument("--min-conf", type=float, default=0.9, help="predicted aircraft must have at least this probability")
    ap.add_argument("--max-claimed", type=float, default=0.05, help="claimed aircraft must have at most this probability")
    a = ap.parse_args()

    model, cfg = load_run(a.run)
    b = D.load_shards(a.data, crop=cfg["crop"], min_snr=0.0)
    prob = predict_proba(model, cfg, b.iq, b.start)
    classes = np.array(cfg["classes"])
    cidx = {c: i for i, c in enumerate(classes)}
    pred = prob.argmax(1)
    known = np.array([c in cidx for c in b.icao])
    p_claim = np.array([prob[i, cidx[c]] if c in cidx else np.nan for i, c in enumerate(b.icao)])
    p_top = prob.max(1)
    mismatch = known & (classes[pred] != b.icao) & (p_top >= a.min_conf) & (p_claim <= a.max_claimed)
    df = pd.DataFrame({"t": b.t, "icao_claimed": b.icao, "icao_rf": classes[pred], "p_rf": p_top,
                       "p_claimed": p_claim, "known": known, "rf_mismatch": mismatch, "snr": b.snr})
    df.sort_values("t").to_csv(a.out, index=False)
    print("%d bursts | known aircraft %d | RF/ICAO mismatches %d (%.2f%% of known) -> %s"
          % (len(df), known.sum(), mismatch.sum(), 100 * mismatch.sum() / max(known.sum(), 1), a.out))
    if known.any():
        print("claimed ICAO agrees with RF identity on %.1f%% of known bursts" % (100 * (classes[pred] == b.icao)[known].mean()))


if __name__ == "__main__":
    main()
