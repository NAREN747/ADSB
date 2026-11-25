"""Handcrafted RF-fingerprint features: statistical, spectral, wavelet and higher-order statistics.

Input is preprocessed IQ (N, L, 2) from dataset.preprocess(remove_cfo=False), so the carrier
offset is still present and is itself one of the features.
"""
import numpy as np
import pywt
from scipy import signal, stats

from .dataset import FS, _on_mask, estimate_cfo


def _cumulants(x):
    """Normalised complex cumulants (zero-mean x): |C20|, |C40|, C42 over C21 powers."""
    x = x - x.mean()
    c21 = np.mean(np.abs(x) ** 2)
    c20 = np.mean(x ** 2)
    c40 = np.mean(x ** 4) - 3 * c20 ** 2
    c42 = np.mean(np.abs(x) ** 4) - np.abs(c20) ** 2 - 2 * c21 ** 2
    c21 = max(c21, 1e-12)
    return [abs(c20) / c21, abs(c40) / c21 ** 2, c42.real / c21 ** 2]


def _one(x, fs):
    a = np.abs(x)
    m = _on_mask(a[None] ** 2)[0]
    xo, ao = x[m], a[m]
    if len(xo) < 8:
        xo, ao = x, a
    I, Q = xo.real, xo.imag
    f = []

    # statistical: envelope shape, DC, I/Q imbalance
    f += [ao.mean(), ao.std(), stats.skew(ao), stats.kurtosis(ao), ao.max() / max(ao.mean(), 1e-9)]
    f += [I.mean(), Q.mean(), I.std(), Q.std(), I.std() / max(Q.std(), 1e-9),
          np.corrcoef(I, Q)[0, 1] if I.std() > 0 and Q.std() > 0 else 0.0]
    f += [a[~m].mean() if (~m).any() else 0.0, a[~m].std() if (~m).any() else 0.0]   # off-pulse leakage / noise

    # spectral: CFO, instantaneous-frequency spread, PSD shape
    cfo = estimate_cfo(x[None], fs)[0]
    pair = m[1:] & m[:-1]
    dphi = np.angle(x[1:] * np.conj(x[:-1]))[pair]
    f += [cfo, dphi.std() * fs / (2 * np.pi) if len(dphi) > 3 else 0.0]
    fr, psd = signal.welch(x, fs=fs, nperseg=min(64, len(x)), return_onesided=False, detrend=False)
    psd = np.fft.fftshift(psd)
    fr = np.fft.fftshift(fr)
    psd = psd / max(psd.sum(), 1e-12)
    cen = (fr * psd).sum()
    f += [cen / fs, np.sqrt(((fr - cen) ** 2 * psd).sum()) / fs,
          np.exp(np.mean(np.log(psd + 1e-12))) / max(psd.mean(), 1e-12)]
    f += list(10 * np.log10(np.add.reduceat(psd, np.linspace(0, len(psd), 17, dtype=int)[:-1]) + 1e-9))

    # wavelet: log energy per level for envelope and I channel
    for sig in (a, x.real):
        co = pywt.wavedec(sig - sig.mean(), "db4", level=4, mode="periodization")
        e = np.array([np.sum(c ** 2) for c in co]) + 1e-9
        f += list(np.log10(e)) + [-(e / e.sum() * np.log(e / e.sum())).sum()]

    # higher-order statistics
    f += _cumulants(xo)
    f += [stats.kurtosis(I), stats.kurtosis(Q), stats.skew(I), stats.skew(Q)]
    return np.array(f, dtype=np.float32)


def feature_names():
    n = ["amp_mean", "amp_std", "amp_skew", "amp_kurt", "crest", "I_mean", "Q_mean", "I_std", "Q_std",
         "IQ_std_ratio", "IQ_corr", "off_mean", "off_std", "cfo_hz", "inst_freq_std_hz",
         "psd_centroid", "psd_spread", "psd_flatness"]
    n += ["psd_bin%d" % i for i in range(16)]
    for s in ("env", "I"):
        n += ["wl_%s_L%d" % (s, i) for i in range(5)] + ["wl_%s_entropy" % s]
    n += ["c20", "c40", "c42", "I_kurt", "Q_kurt", "I_skew", "Q_skew"]
    return n


def extract_features(iq, fs=FS):
    """iq: (N, L, 2) preprocessed -> (N, F) float32 (NaN/inf replaced by 0)."""
    x = iq[..., 0] + 1j * iq[..., 1]
    out = np.stack([_one(x[i], fs) for i in range(len(x))])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
