"""Example runner: load a recording and apply shielding.

Usage: python run_example.py [path/to/recording/0]
"""
from __future__ import annotations

import os
import sys

import numpy as np

from nextmind_reader import load_eeg, read_inf
from shielding import apply_shielding_pipeline


def main(rec_path: str):
    # prefer preprocessed if available
    paths = [os.path.join(rec_path, "eeg_preprocessed.raw"), os.path.join(rec_path, "eeg.raw")]
    raw = None
    for p in paths:
        if os.path.exists(p):
            raw = p
            break
    if raw is None:
        print("No EEG raw file found in", rec_path)
        return 2

    print("Loading:", raw)
    data, info = load_eeg(raw)
    print("Shape:", data.shape, "info:", info)

    # Try to get sample rate from eeg.inf if present
    inf = {}
    infpath = os.path.join(rec_path, "eeg.inf")
    if os.path.exists(infpath):
        inf = read_inf(infpath)

    fs = float(inf.get("samplerate", inf.get("sample_rate", 250)))
    print("Assumed sampling rate:", fs)

    cleaned, mask = apply_shielding_pipeline(data, fs)
    print("Cleaned shape:", cleaned.shape)
    print("Good samples fraction:", mask.mean())

    out_npz = os.path.join(rec_path, "eeg_cleaned.npz")
    np.savez_compressed(out_npz, cleaned=cleaned, mask=mask, info=info)
    print("Saved cleaned data to", out_npz)
    return 0


if __name__ == "__main__":
    rec = sys.argv[1] if len(sys.argv) > 1 else "."
    sys.exit(main(rec))
