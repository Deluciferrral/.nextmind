"""Light-weight NextMind recording reader.

This module provides functions to discover recordings under a .nextmind workspace
and to load `.raw` and `.inf` files with robust fallbacks.

Assumptions and behavior:
- `.inf` files may be JSON or simple text key=value lines; we try both.
- `.raw` EEG files are binary samples. If the metadata doesn't specify shape/dtype
  we try common dtypes (float32, int16) and common channel counts (8,16,32,64)
  and pick the first combination where the file length is divisible by the
  channel count.

This is intentionally defensive so it can be used on unknown or partial exports.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np


def find_recordings(root: str) -> list:
    """Return list of recording subfolder paths (e.g., recording/0, recording/1)."""
    recdir = os.path.join(root, "recording")
    if not os.path.isdir(recdir):
        return []
    items = []
    for name in sorted(os.listdir(recdir)):
        p = os.path.join(recdir, name)
        if os.path.isdir(p):
            items.append(p)
    return items


def read_inf(path: str) -> Dict:
    """Try to read an .inf metadata file and parse JSON or key=value pairs.

    If parsing fails, returns {'raw_bytes': b'...'} as a fallback.
    """
    if not os.path.exists(path):
        return {}
    # read as bytes and try multiple decodings/parsers
    with open(path, "rb") as f:
        raw = f.read()

    # try JSON
    try:
        text = raw.decode("utf-8")
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # try simple text key=value lines
    try:
        text = raw.decode("utf-8", errors="ignore")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        out = {}
        for L in lines:
            if "=" in L:
                k, v = L.split("=", 1)
                out[k.strip()] = v.strip()
        if out:
            return out
    except Exception:
        pass

    # fallback: return raw bytes (hex string to keep JSON-serializable)
    return {"raw_bytes_hex": raw.hex()[:1024]}


def _infer_dtype_and_channels(raw_path: str) -> Tuple[np.dtype, int]:
    """Heuristic: try a few dtypes and channel counts, return a plausible pair."""
    filesize = os.path.getsize(raw_path)
    # candidate dtypes (numpy dtype and byte size)
    candidates = [(np.float32, 4), (np.int16, 2), (np.int32, 4)]
    channel_options = [8, 16, 32, 64, 4]

    for dtype, bsz in candidates:
        for ch in channel_options:
            if filesize % (bsz * ch) == 0:
                return dtype, ch

    # fallback
    return np.float32, 8


def load_eeg(raw_path: str, n_channels: Optional[int] = None, dtype: Optional[np.dtype] = None) -> Tuple[np.ndarray, Dict]:
    """Load EEG raw file into a NumPy array shaped (n_samples, n_channels).

    Parameters
    - raw_path: path to .raw file
    - n_channels: optional, force number of channels
    - dtype: optional numpy dtype to use (e.g., np.float32)

    Returns (data, info). `info` contains inferred dtype, n_channels and n_samples.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(raw_path)

    if dtype is None or n_channels is None:
        inferred_dtype, inferred_ch = _infer_dtype_and_channels(raw_path)
        dtype = dtype or inferred_dtype
        n_channels = n_channels or inferred_ch

    # read raw data
    data = np.fromfile(raw_path, dtype=dtype)
    if data.size == 0:
        raise ValueError("Empty EEG file: %s" % raw_path)

    if data.size % n_channels != 0:
        # If shape doesn't fit, try to re-infer channels with dtype fixed
        filesize = os.path.getsize(raw_path)
        bsz = np.dtype(dtype).itemsize
        possible = []
        for ch in [4, 8, 16, 32, 64]:
            if data.size % ch == 0:
                possible.append(ch)
        if possible:
            n_channels = possible[0]
        else:
            # reshape anyway by truncating tail
            new_len = (data.size // n_channels) * n_channels
            data = data[:new_len]

    n_samples = data.size // n_channels
    data = data.reshape((n_samples, n_channels))

    info = {"dtype": str(dtype), "n_channels": n_channels, "n_samples": n_samples}
    return data, info


def load_events(raw_path: str) -> Tuple[np.ndarray, Dict]:
    """Try to load an events/raw file.

    This function tries float32 and int32 as common encodings, and returns the
    raw array and info.
    """
    if not os.path.exists(raw_path):
        return np.array([]), {}
    for dtype in (np.float32, np.int32, np.int16):
        try:
            arr = np.fromfile(raw_path, dtype=dtype)
            if arr.size > 0:
                return arr, {"dtype": str(dtype), "n_events": arr.size}
        except Exception:
            continue
    # fallback: return raw bytes length
    return np.array([]), {"raw_bytes": os.path.getsize(raw_path)}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Quick inspect a recording folder")
    p.add_argument("recording", nargs="?", default=".", help="path to recording folder (e.g., recording/0)")
    args = p.parse_args()

    rec = args.recording
    print("Inspecting:", rec)
    for name in ["eeg.inf", "eeg.raw", "eeg_preprocessed.raw", "event.raw", "stim.raw", "target_info.raw"]:
        path = os.path.join(rec, name)
        if os.path.exists(path):
            print(" -", name, "size=", os.path.getsize(path))
    # show example reading of eeg.raw if exists
    eeg_raw = os.path.join(rec, "eeg.raw")
    if os.path.exists(eeg_raw):
        try:
            d, info = load_eeg(eeg_raw)
            print("Loaded EEG:", info)
        except Exception as exc:
            print("Failed to load EEG:", exc)
