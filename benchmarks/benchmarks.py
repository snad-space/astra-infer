"""Benchmarks for astra_infer pre-processing.

Parameterised over:
- ``n_obs``    : total number of observations in the light curve
                 (32   → heavy padding; 700 → exactly one sequence;
                  4096 → heavy clipping)
- ``band_mix`` : how observations are distributed across g / r / i bands
                 ("balanced" ≈ 43/50/7 %;
                  "g_only"   → all in g;
                  "ri_only"  → split evenly between r and i, none in g)
- ``presorted``: whether the caller guarantees time-sorted input
                 (``True`` skips the internal ``argsort`` step)

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html.
"""

import numpy as np

from astra_infer.infer import BANDS, first_window, normalize_mag, normalize_time, preprocess

_N_OBS = [32, 700, 4096]
_BAND_MIXES = ["balanced", "g_only", "ri_only"]
_PRESORTED = [False, True]


def _make_inputs(n_obs: int, band_mix: str, seed: int = 42):
    """Return ``(time, mag, magerr, band)`` arrays for the given parameters."""
    rng = np.random.default_rng(seed)

    time = rng.uniform(58_000, 59_000, n_obs)
    mag = rng.normal(loc=18.0, scale=1.0, size=n_obs)
    magerr = np.full(n_obs, 0.1)

    if band_mix == "balanced":
        # Mirrors the SEQUENCE_PER_BAND ratios (300 / 350 / 50 → 43/50/7 %)
        weights = np.array([300, 350, 50], dtype=float)
        weights /= weights.sum()
        band = rng.choice(BANDS, size=n_obs, p=weights)
    elif band_mix == "g_only":
        band = np.full(n_obs, "g")
    elif band_mix == "ri_only":
        band = rng.choice(["r", "i"], size=n_obs)
    else:
        raise ValueError(band_mix)

    return time, mag, magerr, band


class PreprocessingBenchmarks:
    """Individual pre-processing steps — no ONNX session required."""

    params = [_N_OBS, _BAND_MIXES, _PRESORTED]
    param_names = ["n_obs", "band_mix", "presorted"]

    def setup(self, n_obs, band_mix, presorted):
        """Prepare raw (and optionally pre-sorted) input arrays."""
        time, mag, magerr, band = _make_inputs(n_obs, band_mix)

        if presorted:
            norm_time = normalize_time(time).astype(np.float32)
            idx = np.argsort(norm_time)
            time, mag, magerr, band = time[idx], mag[idx], magerr[idx], band[idx]

        self.time = time
        self.mag = mag
        self.magerr = magerr
        self.band = band

    def time_preprocess(self, n_obs, band_mix, presorted):
        """Full pre-processing pipeline via the standalone preprocess()."""
        preprocess(self.time, self.mag, self.magerr, self.band, presorted=presorted)

    def time_first_window(self, n_obs, band_mix, presorted):
        """``first_window`` only (inputs already normalised and sorted)."""
        norm_mag = normalize_mag(self.mag, self.magerr).astype(np.float32)
        norm_time = normalize_time(self.time).astype(np.float32)
        first_window(norm_mag, norm_time, self.band)

    def time_normalize_mag(self, n_obs, band_mix, presorted):
        """Magnitude normalisation only."""
        normalize_mag(self.mag, self.magerr)

    def time_argsort(self, n_obs, band_mix, presorted):
        """Time-sort step only (skipped when ``presorted=True``)."""
        if presorted:
            return
        norm_time = normalize_time(self.time).astype(np.float32)
        np.argsort(norm_time)


