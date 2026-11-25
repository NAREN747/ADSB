"""
Embedded Python Block: ADS-B IQ burst capture for RF fingerprinting.

Wire it like this (see ADSB_fingerprint.grc):
    rtlsdr_source_0 : 0      ->  this block : 0   (complex IQ, same stream the decoder chain uses)
    adsb_framer_0   : 0      ->  this block : 1   (float; carries the framer's "burst" tags)
    adsb_decoder_0  : decoded ->  this block : decoded  (message port)

How a burst gets its label
    * The framer tags the first preamble pulse of every candidate burst with ("SOB", snr).
      That tag's offset is an absolute sample index, so this block can cut the matching
      IQ window out of its own copy of the stream: [offset - pre, offset - pre + window).
    * The decoder only publishes a 'decoded' message when the CRC check passed. Its
      metadata carries the same snr value (a double, copied unchanged through demod and
      decoder). Matching on that exact value pairs each decoded ICAO with its IQ window.
    * Bursts whose CRC failed never get a label and are discarded, so every saved
      burst is CRC-verified.

Output: shards  <out_dir>/iq_<session>_<nnnnn>.npz  with
    iq (N, window, 2) float16   I/Q, sample [:, pre] is the first preamble pulse
    t (N,) float64              UTC time from the demodulator
    icao (N,) str               CRC-verified ICAO address
    df (N,) int16               downlink format
    snr (N,) float32            framer SNR estimate in dB
    sample_offset (N,) int64    absolute sample index of the burst start
    pre, window, fs, session    scalars
"""
import os
import threading
import time

import numpy as np
import pmt
from gnuradio import gr

FRAME_SAMPLES_2MSPS = 240  # 8 us preamble + 112 us data at 2 MS/s


class blk(gr.sync_block):
    def __init__(self, out_dir="iq_bursts", window=2400, pre=240, samp_rate=2e6,
                 max_per_icao=3000, shard_size=500, min_snr_db=0.0):
        gr.sync_block.__init__(self, name="ADSB IQ Capture",
                               in_sig=[np.complex64, np.float32], out_sig=None)
        self.out_dir = os.path.abspath(str(out_dir))
        self.window = int(window)
        self.pre = int(pre)
        self.fs = float(samp_rate)
        self.samp_rate = self.fs
        self.max_per_icao = int(max_per_icao)
        self.shard_size = int(shard_size)
        self.min_snr_db = float(min_snr_db)
        if self.pre + FRAME_SAMPLES_2MSPS > self.window:
            raise ValueError("window must be at least pre + 240 samples")
        os.makedirs(self.out_dir, exist_ok=True)

        self.session = time.strftime("%Y%m%d-%H%M%S")
        self.lock = threading.Lock()

        # sample ring buffer (own copy of the IQ stream)
        self.buf = np.zeros(0, dtype=np.complex64)
        self.buf0 = 0                      # absolute index of buf[0]
        self.pending = []                  # bursts whose window is not complete yet
        self.ready = {}                    # snr -> finished IQ window waiting for its label
        self.labels = {}                   # snr -> decoded message waiting for its IQ window

        # time-base consistency check: decoded timestamp - offset/fs is constant
        self.t_base = None
        self.bad_streak = 0

        self.rows = {k: [] for k in ("iq", "t", "icao", "df", "snr", "off")}
        self.per_icao = {}
        self.shard_idx = 0
        self.last_flush = time.time()
        self.stats = dict(tags=0, saved=0, no_label=0, no_iq=0, mismatch=0, capped=0)

        self.message_port_register_in(pmt.intern("decoded"))
        self.set_msg_handler(pmt.intern("decoded"), self.handle_msg)
        print("ADSB IQ Capture: writing shards to %s (window=%d, pre=%d)"
              % (self.out_dir, self.window, self.pre))

    # ------------------------------------------------------------------ stream side
    def work(self, input_items, output_items):
        x = input_items[0]
        n = len(x)
        start = self.nitems_read(0)
        end = start + n

        with self.lock:
            if self.buf.size == 0:
                self.buf0 = start
            self.buf = np.concatenate([self.buf, x.astype(np.complex64, copy=False)])

            for tag in self.get_tags_in_range(1, start, end, pmt.intern("burst")):
                try:
                    snr = float(pmt.to_python(tag.value)[1])
                except Exception:
                    continue
                if not np.isfinite(snr) or snr < self.min_snr_db:
                    continue
                self.stats["tags"] += 1
                off = int(tag.offset)
                self.pending.append({"off": off, "snr": snr,
                                     "a": off - self.pre, "b": off - self.pre + self.window})

            still = []
            for p in self.pending:
                if p["b"] > end:
                    still.append(p)
                    continue
                if p["a"] < self.buf0:
                    self.stats["no_iq"] += 1       # tag arrived after its samples were trimmed
                    continue
                seg = self.buf[p["a"] - self.buf0: p["b"] - self.buf0]
                p["iq"] = np.stack([seg.real, seg.imag], axis=-1).astype(np.float16)
                p["created"] = time.time()
                self._pair_with_label(p)
            self.pending = still

            keep_from = min([p["a"] for p in self.pending] + [end - self.pre - 64])
            cut = keep_from - self.buf0
            if cut > 0:
                self.buf = self.buf[cut:]
                self.buf0 = keep_from

            self._expire()
            if time.time() - self.last_flush > 60:
                self._flush()
        return n

    # ---------------------------------------------------------------- message side
    def handle_msg(self, msg):
        try:
            meta = pmt.to_python(pmt.car(msg))
            snr = float(meta["snr"])
            lab = {"t": float(meta["timestamp"]), "icao": str(meta["icao"]).strip(),
                   "df": int(meta.get("df", -1)), "snr": snr, "created": time.time()}
        except Exception:
            return
        if not lab["icao"] or not np.isfinite(snr):
            return
        with self.lock:
            p = self.ready.pop(snr, None)
            if p is None:
                self.labels[snr] = lab
            else:
                self._commit(p, lab)

    # ------------------------------------------------------------------- internals
    def _pair_with_label(self, p):
        lab = self.labels.pop(p["snr"], None)
        if lab is None:
            self.ready[p["snr"]] = p
        else:
            self._commit(p, lab)

    def _commit(self, p, lab):
        base = lab["t"] - p["off"] / self.fs
        if self.t_base is None:
            self.t_base = base
        if abs(base - self.t_base) > 2e-3:          # same snr value, wrong burst
            self.stats["mismatch"] += 1
            self.bad_streak += 1
            if self.bad_streak > 20:                # first match was the wrong one: re-learn
                self.t_base, self.bad_streak = None, 0
            return
        self.bad_streak = 0
        icao = lab["icao"]
        if self.per_icao.get(icao, 0) >= self.max_per_icao:
            self.stats["capped"] += 1
            return
        self.per_icao[icao] = self.per_icao.get(icao, 0) + 1
        self.rows["iq"].append(p["iq"])
        self.rows["t"].append(lab["t"])
        self.rows["icao"].append(icao)
        self.rows["df"].append(lab["df"])
        self.rows["snr"].append(p["snr"])
        self.rows["off"].append(p["off"])
        self.stats["saved"] += 1
        if len(self.rows["t"]) >= self.shard_size:
            self._flush()

    def _expire(self, max_age=10.0):
        now = time.time()
        for store, key in ((self.ready, "no_label"), (self.labels, "no_iq")):
            for k in [k for k, v in store.items() if now - v["created"] > max_age]:
                del store[k]
                if key == "no_label":
                    self.stats[key] += 1       # CRC failed (or never decoded): expected, not an error

    def _flush(self):
        self.last_flush = time.time()
        if not self.rows["t"]:
            return
        path = os.path.join(self.out_dir, "iq_%s_%05d.npz" % (self.session, self.shard_idx))
        np.savez(path,
                 iq=np.stack(self.rows["iq"]),
                 t=np.asarray(self.rows["t"], dtype=np.float64),
                 icao=np.asarray(self.rows["icao"], dtype="<U6"),
                 df=np.asarray(self.rows["df"], dtype=np.int16),
                 snr=np.asarray(self.rows["snr"], dtype=np.float32),
                 sample_offset=np.asarray(self.rows["off"], dtype=np.int64),
                 pre=self.pre, window=self.window, fs=self.fs, session=self.session)
        print("ADSB IQ Capture: %s  (%d bursts) | %s" % (os.path.basename(path), len(self.rows["t"]), self.stats))
        self.shard_idx += 1
        self.rows = {k: [] for k in self.rows}

    def stop(self):
        with self.lock:
            self._flush()
        return True
