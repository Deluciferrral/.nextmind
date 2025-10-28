"""Shielding (filtering / simple artifact rejection) utilities for NextMind EEG.

These are small, dependency-light helpers built on numpy/scipy.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from typing import Optional, Tuple


def bandpass_filter(data: np.ndarray, fs: float, low: float = 1.0, high: float = 40.0, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass to multi-channel data.

    data: shape (n_samples, n_channels)
    """
    if data.ndim == 1:
        data = data[:, None]
    nyq = 0.5 * fs
    low_ = low / nyq
    high_ = high / nyq
    b, a = signal.butter(order, [low_, high_], btype="band")
    filtered = signal.filtfilt(b, a, data, axis=0)
    return filtered


def notch_filter(data: np.ndarray, fs: float, freq: float = 50.0, Q: float = 30.0) -> np.ndarray:
    """Apply an IIR notch (bandstop) filter at `freq` Hz.

    Uses second-order sections for numerical stability.
    """
    if data.ndim == 1:
        data = data[:, None]
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = signal.iirnotch(w0, Q)
    filtered = signal.filtfilt(b, a, data, axis=0)
    return filtered


def simple_threshold_artifact_rejection(data: np.ndarray, z_thresh: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    """Identify artifact samples by z-score across time per channel and mask them.

    Returns (cleaned_data, mask) where mask is boolean array shape (n_samples,) True=clean.
    Cleaning is done by linear interpolation across small gaps.
    """
    if data.ndim == 1:
        data = data[:, None]

    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    # avoid division by zero
    std[std == 0] = 1.0
    z = (data - mean) / std
    # any channel exceeding threshold marks that sample as bad
    bad = np.any(np.abs(z) > z_thresh, axis=1)
    mask = ~bad

    cleaned = data.copy()
    n_samples = cleaned.shape[0]
    # simple interpolation per channel across bad samples
    for ch in range(cleaned.shape[1]):
        arr = cleaned[:, ch]
        bad_idx = np.where(bad)[0]
        if bad_idx.size == 0:
            continue
        good_idx = np.where(~bad)[0]
        if good_idx.size < 2:
            # not enough good points to interpolate; fill with median
            arr[bad] = np.nanmedian(arr[~bad])
            cleaned[:, ch] = arr
            continue
        cleaned[:, ch] = np.interp(np.arange(n_samples), good_idx, arr[good_idx])

    return cleaned, mask


def apply_shielding_pipeline(data: np.ndarray, fs: float, band=(1.0, 40.0), notch_freq: Optional[float] = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience: bandpass -> notch -> artifact rejection.

    Returns (cleaned_data, mask)
    """
    bp = bandpass_filter(data, fs, low=band[0], high=band[1])
    if notch_freq is not None:
        bp = notch_filter(bp, fs, freq=notch_freq)
    cleaned, mask = simple_threshold_artifact_rejection(bp)
    return cleaned, mask
