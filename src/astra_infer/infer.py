from collections.abc import Sequence
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
    """Run ONNX inference, creating a new session from *onnx_file* each call.

    For repeated inference prefer :class:`AstraInfer`, which holds a single
    session across calls.
    """
    session = ort.InferenceSession(onnx_file)
    return _run_session(session, norm_mag, time, lg_wave, mask)


def _run_session(
    session: ort.InferenceSession,
    norm_mag: ArrayLike,
    time: ArrayLike,
    lg_wave: ArrayLike,
    mask: ArrayLike,
) -> ArrayLike:
    norm_mag = norm_mag.astype(np.float32)
    time = time.astype(np.float32)
    lg_wave = lg_wave.astype(np.float32)
    mask = mask.astype(np.float32)
    (embeddings,) = session.run(None, {"input": norm_mag, "times": time, "band_info": lg_wave, "mask": mask})
    return embeddings


def normalize_mag(mag: ArrayLike, magerr: ArrayLike) -> ArrayLike:
    """Return inverse-variance weighted mean-subtracted magnitudes."""
    weighted_mean = np.average(mag, weights=magerr**-2)
    return mag - weighted_mean


def normalize_time(time: ArrayLike) -> ArrayLike:
    """Shift times by :data:`MJD_OFFSET` so values are near zero."""
    return time - MJD_OFFSET


def single_band_first_window(
    mag: ArrayLike, time: ArrayLike, band: ArrayLike, band_name: str
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Clip or pad a single band to its target sequence length.

    Returns ``(mag, time, lg_eff_wave, mask)`` each of length
    ``SEQUENCE_PER_BAND[band_name]``.  Padded positions have ``mask == 1``.
    """
    boolmask = band == band_name
    mag, time = mag[boolmask], time[boolmask]

    dtype = mag.dtype
    input_size = mag.size
    seq_size = SEQUENCE_PER_BAND[band_name]
    pad_size = seq_size - input_size
    lg_eff_wave = LG_EFF_WAVE[band_name]

    if pad_size <= 0:
        # Clipping
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


def first_window(
    mag: ArrayLike, time: ArrayLike, band: ArrayLike
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Assemble the first-window input tensors for all bands.

    Returns ``(norm_mag, time, lg_wave, mask)`` each shaped ``(1, 700, 1)``
    except *mask* which is ``(1, 700)``, ready to feed into the ONNX model.
    """
    mag_, time_, lg_wave_, mask_ = [], [], [], []
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


def preprocess(
    time: ArrayLike,
    mag: ArrayLike,
    magerr: ArrayLike,
    band: ArrayLike,
    *,
    presorted: bool = False,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Run the pre-processing pipeline and return ONNX-ready tensors.

    Parameters
    ----------
    time : array-like
        Observation times (MJD).
    mag : array-like
        Observed magnitudes.
    magerr : array-like
        Magnitude uncertainties.
    band : array-like
        Band labels (elements must be in ``BANDS``).
    presorted : bool, optional
        If ``True``, the caller guarantees that observations are already
        sorted by time in ascending order, skipping the internal
        ``argsort`` step.  Default is ``False``.

    Returns
    -------
    tuple of four ndarrays
        ``(norm_mag, norm_time, lg_wave, mask)`` shaped for the ONNX
        model: first three are ``(1, 700, 1)``, mask is ``(1, 700)``.
    """
    norm_mag = normalize_mag(mag, magerr).astype(np.float32)
    norm_time = normalize_time(time).astype(np.float32)

    if not presorted:
        time_idx = np.argsort(norm_time)
        norm_time = norm_time[time_idx]
        norm_mag = norm_mag[time_idx]
        band = band[time_idx]

    return first_window(norm_mag, norm_time, band)


def preprocess_batch(
    times: Sequence[ArrayLike],
    mags: Sequence[ArrayLike],
    magerrs: Sequence[ArrayLike],
    bands: Sequence[ArrayLike],
    *,
    presorted: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-process multiple light curves into stacked ONNX-ready tensors.

    Each light curve is processed independently via :func:`preprocess`, then
    the results are concatenated along the batch axis.

    Parameters
    ----------
    times, mags, magerrs, bands : sequences of array-like
        Per-light-curve observation arrays (equal length).
    presorted : bool, optional
        If ``True``, every light curve is assumed to be sorted by time.
        Default is ``False``.

    Returns
    -------
    tuple of four ndarrays
        ``(norm_mag, norm_time, lg_wave, mask)`` with shapes
        ``(N, 700, 1)``, ``(N, 700, 1)``, ``(N, 700, 1)``, ``(N, 700)``.
    """
    tensors = [
        preprocess(t, m, me, b, presorted=presorted)
        for t, m, me, b in zip(times, mags, magerrs, bands, strict=True)
    ]
    return (
        np.concatenate([t[0] for t in tensors], axis=0),
        np.concatenate([t[1] for t in tensors], axis=0),
        np.concatenate([t[2] for t in tensors], axis=0),
        np.concatenate([t[3] for t in tensors], axis=0),
    )


class AstraInfer:
    """Wraps an ONNX session together with the light-curve pre-processing pipeline.

    Parameters
    ----------
    onnx_file : str or Path
        Path to the ONNX model file.  The session is created once and reused
        for every call, avoiding repeated session-initialisation overhead.
    """

    def __init__(self, onnx_file: str | Path) -> None:
        self._session = ort.InferenceSession(onnx_file)

    def preprocess(
        self,
        time: ArrayLike,
        mag: ArrayLike,
        magerr: ArrayLike,
        band: ArrayLike,
        *,
        presorted: bool = False,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Run the pre-processing pipeline and return ONNX-ready tensors.

        Delegates to the module-level :func:`preprocess` function.
        """
        return preprocess(time, mag, magerr, band, presorted=presorted)

    def __call__(
        self,
        time: ArrayLike,
        mag: ArrayLike,
        magerr: ArrayLike,
        band: ArrayLike,
        *,
        presorted: bool = False,
    ) -> ArrayLike:
        """Run pre-processing and inference for a single light curve.

        Parameters
        ----------
        time : array-like
            Observation times (MJD).
        mag : array-like
            Observed magnitudes.
        magerr : array-like
            Magnitude uncertainties.
        band : array-like
            Band labels (elements must be in ``BANDS``).
        presorted : bool, optional
            If ``True``, the caller guarantees that observations are already
            sorted by time in ascending order, skipping the internal
            ``argsort`` step.  Default is ``False``.

        Returns
        -------
        embeddings : ndarray, shape (1, 512)
            Model output embeddings.
        """
        return _run_session(self._session, *self.preprocess(time, mag, magerr, band, presorted=presorted))

    def predict_batch(
        self,
        norm_mag: np.ndarray,
        norm_time: np.ndarray,
        lg_wave: np.ndarray,
        mask: np.ndarray,
        *,
        batch_size: int = 128,
    ) -> np.ndarray:
        """Run ONNX inference on pre-processed stacked tensors.

        Accepts the output of :func:`preprocess_batch` directly and splits it
        into chunks of up to ``batch_size`` for each ONNX call.

        Parameters
        ----------
        norm_mag, norm_time, lg_wave : ndarray, shape (N, 700, 1)
            Pre-processed input tensors as returned by :func:`preprocess_batch`.
        mask : ndarray, shape (N, 700)
            Padding mask as returned by :func:`preprocess_batch`.
        batch_size : int, optional
            Maximum number of light curves per ONNX call.  Default is 128.

        Returns
        -------
        embeddings : ndarray, shape (N, 512)
            One 512-d embedding per input light curve.
        """
        n = norm_mag.shape[0]
        results = []
        for start in range(0, n, batch_size):
            sl = slice(start, start + batch_size)
            results.append(_run_session(self._session, norm_mag[sl], norm_time[sl], lg_wave[sl], mask[sl]))
        return np.concatenate(results, axis=0)


def infer(
    onnx_file: str | Path,
    time: ArrayLike,
    mag: ArrayLike,
    magerr: ArrayLike,
    band: ArrayLike,
    *,
    presorted: bool = False,
) -> ArrayLike:
    """Run inference for a single light curve.

    .. note::
        A new ONNX session is created on every call.  For repeated inference
        use :class:`AstraInfer` directly so the session is reused.

    Parameters
    ----------
    onnx_file : str or Path
        Path to the ONNX model file.
    time : array-like
        Observation times (MJD).
    mag : array-like
        Observed magnitudes.
    magerr : array-like
        Magnitude uncertainties.
    band : array-like
        Band labels (elements must be in ``BANDS``).
    presorted : bool, optional
        If ``True``, the caller guarantees that observations are already sorted
        by time in ascending order, skipping the internal ``argsort`` step.
        Default is ``False``.

    Returns
    -------
    embeddings : ndarray, shape (1, 512)
        Model output embeddings.
    """
    return AstraInfer(onnx_file)(time, mag, magerr, band, presorted=presorted)
