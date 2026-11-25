"""Synthetic ADS-B-like IQ shards for testing the pipeline WITHOUT hardware.

Each fake transmitter gets its own fixed impairments (carrier offset, I/Q imbalance, DC offset,
pulse rise time, amplitude droop, phase noise). Channel SNR drifts slowly over time so a random
split would leak; a chronological split must not. The shards use the same format as the capture block.

    python -m fingerprint.synth --out synth_bursts --tx 8 --per-tx 1500

Accuracy on this data says nothing about real-world performance: it only shows the code works.
"""
import argparse
import os

import numpy as np
from scipy import signal

OS = 8                                   # oversampling versus 2 MS/s
PREAMBLE = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]


def _frame_slots(rng):
    bits = rng.integers(0, 2, 112)
    data = np.stack([bits, 1 - bits], axis=1).ravel()
    return np.array(PREAMBLE + list(data), dtype=float)       # 240 half-symbols of 0.5 us


def _make_tx(rng, i):
    return dict(cfo=rng.uniform(-250e3, 250e3), gain_err=rng.uniform(-0.12, 0.12),
                phase_err=np.deg2rad(rng.uniform(-6, 6)), dc=rng.uniform(-0.03, 0.03, 2),
                rise=rng.integers(1, 4), droop=rng.uniform(0, 0.15), pn=rng.uniform(0.002, 0.02),
                snr0=rng.uniform(14, 24), per=rng.uniform(300, 1200), ph=rng.uniform(0, 6.28))


def _burst(tx, snr_db, rng, window, pre):
    env = np.repeat(_frame_slots(rng), OS)
    k = np.ones(tx["rise"] * 2 + 1) / (tx["rise"] * 2 + 1)
    env = np.convolve(env, k, mode="same")
    env *= 1 - tx["droop"] * np.linspace(0, 1, env.size) * (env > 0.5)
    n = env.size
    t = np.arange(n) / (2e6 * OS)
    ph = 2 * np.pi * tx["cfo"] * t + np.cumsum(rng.normal(0, tx["pn"], n)) + rng.uniform(0, 6.28)
    z = env * np.exp(1j * ph)
    i_, q_ = z.real, z.imag
    q2 = tx["gain_err"] * 0 + (1 + tx["gain_err"]) * (q_ * np.cos(tx["phase_err"]) + i_ * np.sin(tx["phase_err"]))
    z = (i_ + tx["dc"][0]) + 1j * (q2 + tx["dc"][1])
    sigma = 1.0
    amp = sigma * 10 ** (snr_db / 20) * np.sqrt(2)
    full = np.zeros((window + 8) * OS, dtype=complex)
    start = (pre + int(rng.integers(0, 2))) * OS + int(rng.integers(0, OS))
    full[start:start + n] = amp * z * np.exp(1j * rng.uniform(0, 6.28))
    full += sigma * (rng.normal(size=full.size) + 1j * rng.normal(size=full.size)) / np.sqrt(2)
    full = signal.decimate(full, OS, ftype="fir", zero_phase=True)[:window]
    return full


def make_synthetic(out_dir, n_tx=8, per_tx=1500, window=2400, pre=240, duration=3600.0, seed=0,
                   shard=500, session="synth"):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    txs = [_make_tx(rng, i) for i in range(n_tx)]
    icaos = ["%06x" % rng.integers(0x100000, 0xFFFFFF) for _ in range(n_tx)]
    rows = []
    for ti, tx in enumerate(txs):
        for tt in np.sort(rng.uniform(0, duration, per_tx)):
            snr = tx["snr0"] + 6 * np.sin(2 * np.pi * tt / tx["per"] + tx["ph"]) + rng.normal(0, 1.0)
            rows.append((tt, ti, snr))
    rows.sort()
    buf = {k: [] for k in ("iq", "t", "icao", "df", "snr", "off")}
    idx = 0
    for n, (tt, ti, snr) in enumerate(rows):
        x = _burst(txs[ti], snr, rng, window, pre)
        scale = 0.25 / np.sqrt(np.mean(np.abs(x[pre:pre + 240]) ** 2))
        x = np.clip(np.round(x * scale * 127) / 127, -1, 1)
        buf["iq"].append(np.stack([x.real, x.imag], -1).astype(np.float16))
        buf["t"].append(1.7e9 + tt); buf["icao"].append(icaos[ti]); buf["df"].append(17)
        buf["snr"].append(snr); buf["off"].append(n * window)
        if len(buf["t"]) == shard or n == len(rows) - 1:
            np.savez(os.path.join(out_dir, "iq_%s_%05d.npz" % (session, idx)),
                     iq=np.stack(buf["iq"]), t=np.array(buf["t"]), icao=np.array(buf["icao"], "<U6"),
                     df=np.array(buf["df"], np.int16), snr=np.array(buf["snr"], np.float32),
                     sample_offset=np.array(buf["off"], np.int64), pre=pre, window=window, fs=2e6,
                     session=session)
            idx += 1
            buf = {k: [] for k in buf}
    return icaos


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synth_bursts")
    ap.add_argument("--tx", type=int, default=8)
    ap.add_argument("--per-tx", type=int, default=1500)
    ap.add_argument("--window", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(make_synthetic(a.out, a.tx, a.per_tx, a.window, seed=a.seed))
