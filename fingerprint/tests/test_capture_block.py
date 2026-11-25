"""Logic test for epy_iq_capture.blk using a mocked GNU Radio (no hardware, no gnuradio needed).

Feeds a synthetic sample stream in awkward chunk sizes, framer-style tags and decoder-style messages
(some arriving before the IQ window is complete, some after, one never, one with a wrong timestamp)
and checks that every saved burst carries the right IQ and the right ICAO.
"""
import collections
import glob
import importlib
import os
import sys
import tempfile
import types

import numpy as np

Tag = collections.namedtuple("Tag", "offset value")


def _install_mocks():
    gr = types.ModuleType("gnuradio.gr")

    class sync_block:
        def __init__(self, name, in_sig, out_sig):
            self._read = 0
            self._tags = []

        def nitems_read(self, i):
            return self._read

        def get_tags_in_range(self, i, s, e, key):
            return [t for t in self._tags if s <= t.offset < e]

        def message_port_register_in(self, p): pass
        def set_msg_handler(self, p, f): pass

    gr.sync_block = sync_block
    pkg = types.ModuleType("gnuradio"); pkg.gr = gr
    pmt = types.ModuleType("pmt")
    pmt.intern = lambda s: s
    pmt.car = lambda m: m[0]
    pmt.to_python = lambda x: x
    sys.modules.update({"gnuradio": pkg, "gnuradio.gr": gr, "pmt": pmt})


def run():
    _install_mocks()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(here))
    mod = importlib.import_module("fingerprint.epy_iq_capture")
    out = tempfile.mkdtemp()
    fs, window, pre = 2e6, 600, 240
    b = mod.blk(out_dir=out, window=window, pre=pre, samp_rate=fs, shard_size=100)

    rng = np.random.default_rng(0)
    total = 20000
    stream = (rng.normal(size=total) + 1j * rng.normal(size=total)).astype(np.complex64) * 0.01
    offs = [1500, 3100, 3990, 7000, 9100, 12000, 15000]     # 3990: burst whose window straddles chunk edges
    snrs = [12.5, 14.25, 9.75, 20.125, 11.0, 17.5, 8.0]
    icaos = ["aaaaaa", "bbbbbb", "cccccc", "dddddd", "eeeeee", "ffffff", "111111"]
    for o in offs:
        stream[o] = 0.5 + 0.25j + o * 1e-5                  # marker at the burst start
    T0 = 1.7e9
    b._tags = [Tag(o, ("SOB", s)) for o, s in zip(offs, snrs)]

    def msg(i, dt=0.0):
        return ({"timestamp": T0 + offs[i] / fs + dt, "icao": icaos[i], "df": 17, "snr": snrs[i]}, None)

    expected = {0, 1, 2, 3, 4, 6}                           # 5 = CRC failed -> never labelled
    early = {0: msg(0), 3: msg(3)}                          # labels that arrive BEFORE the iq is cut
    for m in early.values():
        b.handle_msg(m)
    pos, chunk_sizes, k = 0, [997, 1500, 333, 2048, 4001], 0
    sent_late = set()
    while pos < total:
        n = min(chunk_sizes[k % len(chunk_sizes)], total - pos); k += 1
        b._read = pos
        b.work([stream[pos:pos + n], np.zeros(n, np.float32)], None)
        pos += n
        for i in (1, 2, 4, 6):                              # labels arriving after their bursts passed
            if i not in sent_late and pos > offs[i] + 50:
                sent_late.add(i); b.handle_msg(msg(i))
    b.handle_msg(({"timestamp": T0 + 99.0, "icao": "bad000", "df": 17, "snr": snrs[5]}, None))  # right snr, wrong time
    b.stop()

    files = sorted(glob.glob(os.path.join(out, "iq_*.npz")))
    assert files, "no shard written"
    z = np.load(files[0])
    got = dict(zip(z["sample_offset"].tolist(), zip(z["icao"].tolist(), range(len(z["t"])))))
    assert set(got) == {offs[i] for i in expected}, (sorted(got), stats(b))
    for i in expected:
        icao, row = got[offs[i]]
        assert icao == icaos[i], (icao, icaos[i])
        mk = z["iq"][row, pre].astype(np.float32)
        assert abs(mk[0] - (0.5 + offs[i] * 1e-5)) < 2e-3 and abs(mk[1] - 0.25) < 2e-3, "IQ window misaligned"
        assert z["iq"].shape[1:] == (window, 2)
        assert abs(z["t"][row] - (T0 + offs[i] / fs)) < 1e-6
    assert b.stats["saved"] == len(expected), b.stats
    assert b.stats["mismatch"] == 1, b.stats            # the wrong-time label was rejected, not saved
    print("capture block OK:", b.stats)


def stats(b):
    return b.stats


if __name__ == "__main__":
    run()
