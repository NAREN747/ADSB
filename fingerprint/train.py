"""Train and evaluate an RF-fingerprint classifier (transmitter ID from raw IQ).

    python -m fingerprint.train --data iq_bursts --model cnn --out runs/cnn1
    python -m fingerprint.train --data iq_bursts --model rnn --out runs/rnn1
    python -m fingerprint.train --data iq_bursts --model cnn --fuse-features --out runs/cnn_fused

Always uses a chronological split (see dataset.time_split). Reports, next to the neural model,
  * a handcrafted-feature gradient-boosting baseline (--baseline), and
  * a LEAKAGE CHECK: a classifier that sees only channel descriptors (SNR, noise floor).
    If that check scores well above chance, part of the task can be solved without looking at the
    transmitter at all (aircraft separated by range/power), so treat the CNN number with suspicion.
"""
import argparse
import json
import os
import time

import numpy as np

from . import dataset as D
from .features import extract_features


def _metrics(y, pred, prob, snr, n_classes):
    from sklearn.metrics import f1_score
    out = {"accuracy": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro")),
           "chance": 1.0 / n_classes, "n": int(len(y))}
    if prob is not None and n_classes >= 3:
        top3 = np.argsort(-prob, axis=1)[:, :3]
        out["top3_accuracy"] = float((top3 == y[:, None]).any(axis=1).mean())
    for lo, hi in ((-99, 10), (10, 15), (15, 20), (20, 99)):
        m = (snr >= lo) & (snr < hi)
        if m.sum() >= 20:
            out["acc_snr_%s_%s" % (lo if lo > -99 else "min", hi if hi < 99 else "max")] = float((pred[m] == y[m]).mean())
    return out


def _gbm(Xtr, ytr, Xte):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(Xtr, ytr)
    return clf.predict(Xte), clf.predict_proba(Xte)


def run(a):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from .models import build

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    b = D.load_shards(a.data, crop=a.crop, min_snr=a.min_snr, dfs=a.dfs)
    b = D.filter_classes(b, a.min_bursts, a.max_bursts, a.seed)
    tr, va, te = D.time_split(b, gap_s=a.gap) if a.split == "time" else D.session_split(b)
    classes = sorted(set(b.icao[tr]) & set(b.icao[va]) & set(b.icao[te]))
    cmap = {c: i for i, c in enumerate(classes)}
    tr, va, te = [i[np.isin(b.icao[i], classes)] for i in (tr, va, te)]
    y = np.array([cmap.get(c, -1) for c in b.icao])
    print("aircraft: %d | train %d / val %d / test %d bursts | chance %.1f%%"
          % (len(classes), len(tr), len(va), len(te), 100 / len(classes)))

    X = D.preprocess(b.iq, b.start, remove_cfo=a.remove_cfo)
    feats = None
    if a.fuse_features or a.baseline:
        t0 = time.time(); feats = extract_features(X); print("handcrafted features: %s (%.0fs)" % (feats.shape, time.time() - t0))
    mu = sd = None
    if a.fuse_features:
        mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-6
        feats = (feats - mu) / sd

    metrics = {"classes": classes, "split": a.split}
    if a.baseline:
        p, pr = _gbm(feats[tr], y[tr], feats[te])
        metrics["baseline_handcrafted_gbm"] = _metrics(y[te], p, pr, b.snr[te], len(classes))
    chan = np.stack([b.snr, D.noise_floor_db(b.iq, b.start)], axis=1)
    p, pr = _gbm(chan[tr], y[tr], chan[te])
    metrics["leakage_check_channel_only"] = _metrics(y[te], p, pr, b.snr[te], len(classes))

    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    ds = {k: D.BurstTensorSet(X[i], y[i], feats[i] if a.fuse_features else None)
          for k, i in (("train", tr), ("val", va), ("test", te))}
    dl = DataLoader(ds["train"], batch_size=a.batch, shuffle=True, drop_last=True)
    model = build(a.model, len(classes), feats.shape[1] if a.fuse_features else 0).to(dev)
    cnt = np.bincount(y[tr], minlength=len(classes)).astype(np.float32)
    w = torch.tensor(cnt.sum() / (len(classes) * np.maximum(cnt, 1)), device=dev)
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr * 3, total_steps=a.epochs * len(dl))

    def predict(d):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(d), 512):
                batch = [t[i:i + 512].to(dev) for t in (d.x, d.f, d.y) if t is not None]
                out.append(torch.softmax(model(*batch[:-1]), -1).cpu())
        return torch.cat(out).numpy()

    os.makedirs(a.out, exist_ok=True)
    best, bad = -1, 0
    for ep in range(a.epochs):
        model.train(); tl = 0
        for batch in dl:
            batch = [t.to(dev) for t in batch]
            x = batch[0]
            if a.noise_aug > 0:
                x = x + a.noise_aug * torch.randn_like(x)
            loss = lossf(model(x, *batch[1:-1]), batch[-1])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step(); tl += loss.item()
        vp = predict(ds["val"]); vacc = float((vp.argmax(1) == y[va]).mean())
        print("epoch %2d  loss %.4f  val acc %.4f" % (ep + 1, tl / len(dl), vacc))
        if vacc > best:
            best, bad = vacc, 0
            torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop"); break

    model.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev))
    tp = predict(ds["test"])
    metrics["neural_test"] = _metrics(y[te], tp.argmax(1), tp, b.snr[te], len(classes))
    metrics["best_val_acc"] = best
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    json.dump({"model": a.model, "classes": classes, "crop": a.crop, "start": int(b.start),
               "remove_cfo": a.remove_cfo, "fuse_features": a.fuse_features,
               "feat_mu": None if mu is None else mu.tolist(), "feat_sd": None if sd is None else sd.tolist(),
               "n_in": int(X.shape[1])}, open(os.path.join(a.out, "config.json"), "w"))
    print("\n== test results (chronologically later bursts, never seen in training) ==")
    for k in ("neural_test", "baseline_handcrafted_gbm", "leakage_check_channel_only"):
        if k in metrics:
            print("%-28s acc %.3f  macroF1 %.3f  (chance %.3f)" % (k, metrics[k]["accuracy"], metrics[k]["macro_f1"], metrics[k]["chance"]))
    lk, nn_ = metrics["leakage_check_channel_only"]["accuracy"], metrics["neural_test"]["accuracy"]
    if lk > 2.5 * metrics["leakage_check_channel_only"]["chance"] and lk > 0.5 * nn_:
        print("WARNING: SNR/noise-floor alone already gets %.0f%%; the model may be learning channel, not hardware." % (100 * lk))
    print("saved to", a.out)
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="runs/run1")
    ap.add_argument("--model", choices=["cnn", "rnn"], default="cnn")
    ap.add_argument("--crop", choices=["frame", "full"], default="frame")
    ap.add_argument("--split", choices=["time", "session"], default="time")
    ap.add_argument("--gap", type=float, default=2.0, help="seconds dropped after each split boundary")
    ap.add_argument("--min-bursts", type=int, default=300); ap.add_argument("--max-bursts", type=int, default=4000)
    ap.add_argument("--min-snr", type=float, default=8.0); ap.add_argument("--dfs", type=int, nargs="*")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--noise-aug", type=float, default=0.0)
    ap.add_argument("--fuse-features", action="store_true"); ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--remove-cfo", action="store_true"); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
