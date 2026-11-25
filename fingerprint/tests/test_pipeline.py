"""Pipeline invariants: no train/test leakage in time, preprocessing invariances, model shapes."""
import os
import sys
import tempfile

import numpy as np


def run():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from fingerprint import dataset as D
    from fingerprint.features import extract_features, feature_names
    from fingerprint.synth import make_synthetic

    d = tempfile.mkdtemp()
    make_synthetic(d, n_tx=3, per_tx=150, window=600, seed=3, shard=200)
    b = D.load_shards(d)
    assert b.iq.shape[1:] == (272, 2) and b.start == 16

    # 1. chronological split: per aircraft, train < val < test with a guard gap
    tr, va, te = D.time_split(b, gap_s=2.0, min_per_split=5)
    for ic in np.unique(b.icao):
        t = lambda idx: b.t[idx][b.icao[idx] == ic]
        assert t(tr).max() < t(va).min() and t(va).max() < t(te).min()
        assert t(va).min() - t(tr).max() >= 0 and t(te).min() - t(va).max() >= 2.0 - 1e-9
    assert not (set(tr) & set(va)) and not (set(va) & set(te)) and not (set(tr) & set(te))

    # 2. preprocessing ignores gain and carrier phase (channel nuisances)
    raw = b.iq[:20]
    rot = (raw[..., 0] + 1j * raw[..., 1]) * 3.7 * np.exp(1j * 1.234)
    raw2 = np.stack([rot.real, rot.imag], -1).astype(np.float32)
    a, c = D.preprocess(raw, b.start), D.preprocess(raw2, b.start)
    assert np.abs(a - c).max() < 1e-3, np.abs(a - c).max()

    # 3. features finite, right width
    f = extract_features(a)
    assert f.shape == (20, len(feature_names())) and np.isfinite(f).all()

    # 4. models: logits and embeddings
    import torch
    from fingerprint.models import build
    x = torch.randn(4, 2, 272)
    for name in ("cnn", "rnn"):
        for fd in (0, 53):
            m = build(name, 5, fd).eval()
            ft = torch.randn(4, fd) if fd else None
            assert m(x, ft).shape == (4, 5) and m.embed(x, ft).shape == (4, 128)
    print("pipeline OK")


if __name__ == "__main__":
    run()
