"""Shard loading, chronological splits and IQ preprocessing for RF fingerprinting."""
import glob
import os
from dataclasses import dataclass

import numpy as np

FRAME = 240          # samples of one ADS-B frame at 2 MS/s (8 us preamble + 112 us data)
FS = 2e6


@dataclass
class Bursts:
    iq: np.ndarray        # (N, L, 2) float32, cropped around the frame
    t: np.ndarray         # (N,) float64 UTC seconds
    icao: np.ndarray      # (N,) str
    df: np.ndarray        # (N,) int
    snr: np.ndarray       # (N,) float32 dB
    session: np.ndarray   # (N,) str
    start: int            # index of the first preamble pulse inside the cropped window

    def __len__(self):
        return len(self.t)

    def take(self, idx):
        return Bursts(self.iq[idx], self.t[idx], self.icao[idx], self.df[idx],
                      self.snr[idx], self.session[idx], self.start)


def load_shards(path, crop="frame", margin=16, min_snr=0.0, dfs=None):
    """Load every iq_*.npz under `path` (file or directory).

    crop="frame": keep [pre-margin, pre+240+margin) of each window (default, ~272 samples).
    crop="full" : keep the whole captured window.
    """
    files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "iq_*.npz")))
    if not files:
        raise FileNotFoundError("no iq_*.npz shards in %s" % path)
    parts = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        pre, win = int(z["pre"]), int(z["window"])
        a, b = (max(0, pre - margin), min(win, pre + FRAME + margin)) if crop == "frame" else (0, win)
        n = len(z["t"])
        parts.append(Bursts(z["iq"][:, a:b, :].astype(np.float32), z["t"], z["icao"].astype(str),
                            z["df"].astype(int), z["snr"], np.full(n, str(z["session"])), pre - a))
    out = Bursts(np.concatenate([p.iq for p in parts]), np.concatenate([p.t for p in parts]),
                 np.concatenate([p.icao for p in parts]), np.concatenate([p.df for p in parts]),
                 np.concatenate([p.snr for p in parts]), np.concatenate([p.session for p in parts]),
                 parts[0].start)
    keep = out.snr >= min_snr
    if dfs:
        keep &= np.isin(out.df, list(dfs))
    return out.take(np.nonzero(keep)[0])


def filter_classes(b, min_bursts, max_bursts=None, seed=0):
    """Drop aircraft with too few bursts; cap busy ones evenly in time (not randomly dropping a tail)."""
    rng = np.random.default_rng(seed)
    keep = []
    for icao in np.unique(b.icao):
        idx = np.nonzero(b.icao == icao)[0]
        if len(idx) < min_bursts:
            continue
        if max_bursts and len(idx) > max_bursts:
            idx = idx[np.sort(rng.choice(len(idx), max_bursts, replace=False))]
        keep.append(idx)
    if not keep:
        raise ValueError("no aircraft with >= %d bursts" % min_bursts)
    return b.take(np.sort(np.concatenate(keep)))


def time_split(b, fracs=(0.7, 0.15, 0.15), gap_s=2.0, min_per_split=20):
    """Per-aircraft chronological split: earliest bursts train, then val, latest test.

    A guard gap of `gap_s` seconds is dropped after each boundary so neighbouring bursts
    (same channel, same instant) never straddle train and evaluation data.
    Aircraft that cannot fill all three splits are removed. Returns (train, val, test) indices.
    """
    tr, va, te = [], [], []
    for icao in np.unique(b.icao):
        idx = np.nonzero(b.icao == icao)[0]
        idx = idx[np.argsort(b.t[idx])]
        n = len(idx)
        c1, c2 = int(n * fracs[0]), int(n * (fracs[0] + fracs[1]))
        t1, t2 = b.t[idx[c1]], b.t[idx[min(c2, n - 1)]]
        i_tr = idx[b.t[idx] < t1]
        i_va = idx[(b.t[idx] >= t1 + gap_s) & (b.t[idx] < t2)]
        i_te = idx[b.t[idx] >= t2 + gap_s]
        if min(len(i_tr), len(i_va), len(i_te)) < min_per_split:
            continue
        tr.append(i_tr); va.append(i_va); te.append(i_te)
    if not tr:
        raise ValueError("no aircraft spans enough time for a chronological split; record longer")
    return tuple(np.concatenate(x) for x in (tr, va, te))


def session_split(b, test_sessions=1, min_per_split=20):
    """Train on the earlier recording sessions, test on the latest ones (aircraft must appear in both)."""
    sess = sorted(np.unique(b.session))
    if len(sess) < 2:
        raise ValueError("session split needs at least 2 recording sessions")
    test_set = set(sess[-test_sessions:])
    is_te = np.array([s in test_set for s in b.session])
    ok = [i for i in np.unique(b.icao)
          if (b.icao[~is_te] == i).sum() >= min_per_split and (b.icao[is_te] == i).sum() >= min_per_split]
    if not ok:
        raise ValueError("no aircraft appears in both train and test sessions")
    in_ok = np.isin(b.icao, ok)
    tr = np.nonzero(~is_te & in_ok)[0]
    te = np.nonzero(is_te & in_ok)[0]
    # carve the validation set from the end of the training sessions, per aircraft
    va = []
    for icao in ok:
        idx = tr[b.icao[tr] == icao]
        idx = idx[np.argsort(b.t[idx])]
        va.append(idx[int(len(idx) * 0.85):])
    va = np.concatenate(va)
    tr = np.setdiff1d(tr, va)
    return tr, va, te


# ------------------------------------------------------------------ preprocessing
def _on_mask(p):
    """Samples that carry a pulse: power above a quarter of the burst's 95th-percentile power."""
    ref = np.percentile(p, 95, axis=1, keepdims=True)
    return p > 0.25 * ref


def estimate_cfo(x, fs=FS):
    """Carrier offset (Hz) from the phase advance between adjacent pulse samples. x: (N, L) complex."""
    p = np.abs(x) ** 2
    m = _on_mask(p)
    pair = m[:, 1:] & m[:, :-1]
    ph = np.sum(np.where(pair, x[:, 1:] * np.conj(x[:, :-1]), 0), axis=1)
    return np.angle(ph) * fs / (2 * np.pi)


def preprocess(iq, start, remove_cfo=False, fs=FS):
    """Make bursts comparable across channels without erasing the transmitter's imperfections.

    1. unit power over the pulse samples (removes distance / gain),
    2. rotate so the preamble has phase 0 (the carrier phase at the receiver is arbitrary),
    3. optionally remove the carrier offset (off by default: it is a genuine transponder trait).
    iq: (N, L, 2) -> (N, L, 2) float32
    """
    x = (iq[..., 0] + 1j * iq[..., 1]).astype(np.complex64)
    p = np.abs(x) ** 2
    m = _on_mask(p)
    scale = np.sqrt(np.sum(p * m, axis=1) / np.maximum(m.sum(axis=1), 1))
    x = x / np.maximum(scale, 1e-9)[:, None]
    if remove_cfo:
        f = estimate_cfo(x, fs)
        n = np.arange(x.shape[1])[None, :]
        x = x * np.exp(-2j * np.pi * f[:, None] * n / fs)
    s0 = max(0, start - 2)
    pre = x[:, s0:start + 12]
    pw = np.abs(pre) ** 2
    strong = pw > 0.5 * pw.max(axis=1, keepdims=True)
    ang = np.angle(np.sum(pre * strong, axis=1))
    x = x * np.exp(-1j * ang)[:, None]
    return np.stack([x.real, x.imag], axis=-1).astype(np.float32)


def noise_floor_db(iq, start):
    """Median power in the samples before the burst, relative to the pulse power (a channel descriptor)."""
    x = iq[..., 0] + 1j * iq[..., 1]
    p = np.abs(x) ** 2
    pre = p[:, :max(4, start - 2)]
    m = _on_mask(p)
    pulse = np.sum(p * m, axis=1) / np.maximum(m.sum(axis=1), 1)
    return 10 * np.log10(np.median(pre, axis=1) / np.maximum(pulse, 1e-12) + 1e-12)


class BurstTensorSet:
    """Preprocessed tensors for one split. Items: (x[2, L], y) or (x, feats, y)."""

    def __init__(self, iq, y, feats=None):
        import torch
        self.x = torch.from_numpy(np.ascontiguousarray(iq.transpose(0, 2, 1)))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self.f = None if feats is None else torch.from_numpy(feats.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.x[i], self.y[i]) if self.f is None else (self.x[i], self.f[i], self.y[i])
