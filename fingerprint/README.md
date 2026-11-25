# RF fingerprinting for the ADSB project

Identify an aircraft transponder from the raw IQ of its ADS-B burst (hardware imperfections),
then compare that identity with the ICAO the message claims.

```
RTL-SDR -> framer -> demod -> decoder (CRC) ──┐
    │  IQ            burst tags                │ decoded ICAO + snr
    └──────────────► epy_iq_capture ◄──────────┘
                          │  iq_*.npz shards (CRC-verified, ICAO-labelled)
                          ▼
        train.py  (CNN / LSTM / handcrafted baseline, chronological split)
                          ▼
        identify.py  ->  rf_identity.csv   (rf_mismatch = spoofing indicator)
```

## 1. Capture (GNU Radio)
Open `ADSB_fingerprint.grc` (your `ADSB.grc` plus one embedded block, **ADSB IQ Capture**, wired to
`rtlsdr_source:0`, `adsb_framer:0` and `adsb_decoder:decoded`). Run it as usual. Shards appear in `iq_bursts/`.
If GRC shows the block without ports, double-click it and press OK so it re-reads the source.
A 2400-sample window is 1.2 ms but a frame is only 240 samples; training crops to the frame
(`--crop frame`, default). You can lower `window` to 600 to save disk (~10 KB per burst at 2400).

## 2. Train
```
pip install -r fingerprint/requirements.txt
python -m fingerprint.train --data iq_bursts --model cnn --out runs/cnn1 --baseline
python -m fingerprint.train --data iq_bursts --model rnn --out runs/rnn1
python -m fingerprint.train --data iq_bursts --model cnn --fuse-features --out runs/fused
```
Useful flags: `--split session` (train on earlier recordings, test on the latest), `--min-bursts`,
`--min-snr`, `--remove-cfo` (ablation: how much of the accuracy is carrier offset?), `--crop full`.

The split is chronological **per aircraft** with a 2 s guard gap. The report prints the neural result,
the handcrafted baseline and a **leakage check** (a model that sees only SNR and noise floor).
If the leakage check is not far above chance level, the neural score reflects the transmitter.

## 3. Use
```
python -m fingerprint.identify --run runs/cnn1 --data iq_bursts_new --out rf_identity.csv
```
`rf_mismatch` is true when the RF identity is confidently a *different* known aircraft than the one claimed.
Join it with `anomaly_scores.csv` from `adsb_anomaly.py` on (icao, time) if you want one combined verdict.

## Test without hardware
```
python -m fingerprint.synth --out synth_bursts --tx 8
python -m fingerprint.train --data synth_bursts --model cnn --out runs/synth --baseline
python -m fingerprint.tests.test_capture_block
python -m fingerprint.tests.test_pipeline
```
Synthetic transmitters are easy to separate; those scores only show that the code runs.

## Limits to keep in mind
* Closed set: only aircraft seen in training can be recognised. A new ICAO is `known=False`, never flagged.
* An aircraft needs many bursts spread over time (hundreds, across at least several minutes). Most aircraft
  cross your range once, so collect for hours or days; scheduled flights that return daily are ideal for `--split session`.
* One RTL-SDR at 2 MS/s with 8-bit samples hides fine detail (rise time, phase noise). Carrier offset
  is the likely dominant cue; it includes your dongle's own offset, so keep the dongle, antenna and gain fixed
  between training and use, and let it warm up.
* Real-data accuracy has not been measured: it depends on your recordings.
