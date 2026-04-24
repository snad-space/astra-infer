from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import ArrayLike

BANDS = np.array(["g", "r", "i"])
SEQUENCE_PER_BAND = {"g": 300, "r": 350, "i": 50}
assert list(BANDS) == list(SEQUENCE_PER_BAND.keys())

SEQUENCE_LENGTH = 700
assert sum(SEQUENCE_PER_BAND.values()) == SEQUENCE_LENGTH

MJD_OFFSET = 58_000.0

LG_EFF_WAVE = {"g": np.log10(4746.48), "r": np.log10(6366.38), "i": np.log10(7829.03)}
assert list(BANDS) == list(LG_EFF_WAVE.keys())


def run_onnx(
    onnx_file: str | Path, norm_mag: ArrayLike, time: ArrayLike, lg_wave: ArrayLike, mask: ArrayLike
) -> ArrayLike:
    session = ort.InferenceSession(onnx_file)
    norm_mag = norm_mag.astype(np.float32)
    time = time.astype(np.float32)
    lg_wave = lg_wave.astype(np.float32)
    mask = mask.astype(np.float32)
    (embeddings,) = session.run(None, {"input": norm_mag, "times": time, "band_info": lg_wave, "mask": mask})
    return embeddings


def normalize_mag(mag, magerr):
    weighted_mean = np.average(mag, weights=magerr**-2)
    norm_mag = mag - weighted_mean
    return norm_mag


def normalize_time(time):
    return time - MJD_OFFSET


def single_band_first_window(mag, time, band, band_name):
    boolmask = band == band_name
    mag, time, band = mag[boolmask], time[boolmask], band[boolmask]

    dtype = mag.dtype
    input_size = mag.size
    seq_size = SEQUENCE_PER_BAND[band_name]
    pad_size = seq_size - input_size
    lg_eff_wave = LG_EFF_WAVE[band_name]
    # Clipping
    if pad_size <= 0:
        return (
            mag[:seq_size],
            time[:seq_size],
            np.full(seq_size, lg_eff_wave, dtype=dtype),
            np.zeros(seq_size, dtype=dtype),
        )
    # Padding
    pad_val = np.zeros(pad_size, dtype=dtype)
    return (
        np.r_[mag, pad_val],
        np.r_[time, pad_val],
        np.r_[np.full(input_size, lg_eff_wave), pad_val],
        np.r_[np.zeros(input_size, dtype=dtype), np.ones(pad_size, dtype=dtype)],
    )


def first_window(mag, time, band):
    mag_ = []
    time_ = []
    lg_wave_ = []
    mask_ = []
    for band_name in BANDS:
        mag_b, time_b, lg_wave_b, mask_b = single_band_first_window(mag, time, band, band_name)
        mag_.append(mag_b)
        time_.append(time_b)
        lg_wave_.append(lg_wave_b)
        mask_.append(mask_b)
    return (
        np.concatenate(mag_)[None, :, None],
        np.concatenate(time_)[None, :, None],
        np.concatenate(lg_wave_)[None, :, None],
        np.concatenate(mask_)[None, :],
    )


def infer(onnx_file: str | Path, time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, band: ArrayLike):
    """Run inference for a single light curve"""

    norm_mag = normalize_mag(mag, magerr).astype(np.float32)
    norm_time = normalize_time(time).astype(np.float32)

    time_idx = np.argsort(norm_time)
    norm_time, norm_mag, band = norm_time[time_idx], norm_mag[time_idx], band[time_idx]

    first_window_inputs = first_window(norm_mag, time, band)

    return run_onnx(onnx_file, *first_window_inputs)
