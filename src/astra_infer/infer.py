from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnxruntime as ort
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    import pyarrow as pa

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


def preprocess_lc(
    time: ArrayLike,
    mag: ArrayLike,
    magerr: ArrayLike,
    band: ArrayLike,
    *,
    presorted: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-process a single light curve into ONNX-ready tensors.

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
        ``(norm_mag, norm_time, lg_wave, mask)`` shaped ``(1, 700, 1)``,
        ``(1, 700, 1)``, ``(1, 700, 1)``, ``(1, 700)``.
    """
    norm_mag = normalize_mag(mag, magerr).astype(np.float32)
    norm_time = normalize_time(time).astype(np.float32)

    if not presorted:
        time_idx = np.argsort(norm_time)
        norm_time = norm_time[time_idx]
        norm_mag = norm_mag[time_idx]
        band = band[time_idx]

    return first_window(norm_mag, norm_time, band)


_DEFAULT_FIELD_NAMES: dict[str, str] = {"time": "time", "mag": "mag", "magerr": "magerr", "band": "band"}
"""Default mapping from canonical field names to Arrow column/field names."""


def _is_arrow(obj: Any) -> bool:
    """Return True if *obj* is a recognised PyArrow container type."""
    try:
        import pyarrow as pa

        return isinstance(obj, (pa.ListArray, pa.ChunkedArray, pa.Table, pa.StructArray))
    except ImportError:
        return False


def _preprocess_many_arrow(
    lcs: pa.ListArray | pa.ChunkedArray | pa.Table | pa.StructArray,
    field_names: dict[str, str],
    *,
    presorted: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-process an Arrow input into stacked ONNX-ready tensors.

    Supports four layouts:

    * ``pa.ListArray`` with a struct value type (*list-of-struct*): the natural
      output of nested-pandas / Arrow list arrays.
    * ``pa.ChunkedArray`` of the same type — combined with
      :meth:`~pyarrow.ChunkedArray.combine_chunks` before processing.
    * ``pa.Table`` where every relevant column is a ``ListArray`` of
      per-observation values (*struct-of-lists*).
    * ``pa.StructArray`` whose fields are ``ListArray`` s of per-observation
      values (*struct-of-lists*).

    Numeric fields are extracted as zero-copy NumPy views via
    ``flat.to_numpy()``.  The string ``band`` column is sliced per-row with
    ``pa.Array.slice()`` to avoid materialising the full flat list.

    Parameters
    ----------
    lcs : pa.ListArray | pa.ChunkedArray | pa.Table | pa.StructArray
        Arrow container holding the light curves.
    field_names : dict[str, str]
        Mapping from canonical names (``"time"``, ``"mag"``, ``"magerr"``,
        ``"band"``) to the actual Arrow column / field names.
    presorted : bool, optional
        If ``True``, every light curve is assumed to be sorted by time.

    Returns
    -------
    tuple of four ndarrays
        ``(norm_mag, norm_time, lg_wave, mask)`` with shapes
        ``(N, 700, 1)``, ``(N, 700, 1)``, ``(N, 700, 1)``, ``(N, 700)``.
    """
    import pyarrow as pa

    if isinstance(lcs, pa.ChunkedArray):
        lcs = lcs.combine_chunks()

    if isinstance(lcs, pa.ListArray):
        # list-of-struct: offsets index into a flat StructArray
        offsets = lcs.offsets.to_numpy()
        flat_struct = lcs.values
        cols = {key: flat_struct.field(arrow_name) for key, arrow_name in field_names.items()}

    elif isinstance(lcs, (pa.Table, pa.StructArray)):
        # struct-of-lists: each field / column is a ListArray of per-observation values
        offsets = None
        cols = {}
        for key, arrow_name in field_names.items():
            col = lcs.column(arrow_name) if isinstance(lcs, pa.Table) else lcs.field(arrow_name)
            if isinstance(col, pa.ChunkedArray):
                col = col.combine_chunks()
            if offsets is None:
                offsets = col.offsets.to_numpy()
            cols[key] = col.values

    else:
        raise TypeError(f"Unsupported Arrow input type: {type(lcs).__name__}")

    time_flat = cols["time"].to_numpy(zero_copy_only=False)
    mag_flat = cols["mag"].to_numpy(zero_copy_only=False)
    magerr_flat = cols["magerr"].to_numpy(zero_copy_only=False)
    band_col = cols["band"]  # kept as Arrow; sliced per-row to avoid full materialisation

    tensors = []
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        band_i = np.asarray(band_col.slice(s, e - s).to_pylist())
        tensors.append(
            preprocess_lc(time_flat[s:e], mag_flat[s:e], magerr_flat[s:e], band_i, presorted=presorted)
        )

    return (
        np.concatenate([t[0] for t in tensors], axis=0),
        np.concatenate([t[1] for t in tensors], axis=0),
        np.concatenate([t[2] for t in tensors], axis=0),
        np.concatenate([t[3] for t in tensors], axis=0),
    )


def preprocess_many(
    lcs: Sequence[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]] | Any,
    *,
    field_names: dict[str, str] | None = None,
    presorted: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-process multiple light curves into stacked ONNX-ready tensors.

    Accepts either a sequence of ``(time, mag, magerr, band)`` tuples or a
    PyArrow container (``ListArray``, ``ChunkedArray``, ``Table``, or
    ``StructArray``).  See :func:`_preprocess_many_arrow` for details of the
    Arrow layouts that are supported.

    Parameters
    ----------
    lcs : sequence of (time, mag, magerr, band) tuples, or Arrow container
        One light curve per element / row.
    field_names : dict[str, str] | None, optional
        Mapping from canonical names (``"time"``, ``"mag"``, ``"magerr"``,
        ``"band"``) to the Arrow column / field names used in *lcs*.  Only
        relevant when *lcs* is an Arrow container.  If ``None``, the canonical
        names are used unchanged.  Passing a non-None value forces Arrow
        processing even when *lcs* is a plain sequence.
    presorted : bool, optional
        If ``True``, every light curve is assumed to be sorted by time.
        Default is ``False``.

    Returns
    -------
    tuple of four ndarrays
        ``(norm_mag, norm_time, lg_wave, mask)`` with shapes
        ``(N, 700, 1)``, ``(N, 700, 1)``, ``(N, 700, 1)``, ``(N, 700)``.
    """
    if field_names is not None or _is_arrow(lcs):
        return _preprocess_many_arrow(lcs, field_names or _DEFAULT_FIELD_NAMES, presorted=presorted)
    tensors = [preprocess_lc(*lc, presorted=presorted) for lc in lcs]
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

    def predict_lc(
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
        return _run_session(self._session, *preprocess_lc(time, mag, magerr, band, presorted=presorted))

    def predict_tensors(
        self,
        norm_mag: np.ndarray,
        norm_time: np.ndarray,
        lg_wave: np.ndarray,
        mask: np.ndarray,
        *,
        batch_size: int | None = 128,
    ) -> np.ndarray:
        """Run ONNX inference on stacked pre-processed tensors.

        Accepts the output of :func:`preprocess_many` directly.  When
        *batch_size* is set, the input is split into chunks so each ONNX call
        processes at most that many light curves.  Pass ``batch_size=None`` to
        run all light curves in a single ONNX call.

        Parameters
        ----------
        norm_mag, norm_time, lg_wave : ndarray, shape (N, 700, 1)
            Stacked pre-processed tensors as returned by :func:`preprocess_many`.
        mask : ndarray, shape (N, 700)
            Padding mask as returned by :func:`preprocess_many`.
        batch_size : int or None, optional
            Maximum number of light curves per ONNX call.  ``None`` runs all
            light curves in a single call.  Default is 128.

        Returns
        -------
        embeddings : ndarray, shape (N, 512)
            One 512-d embedding per input light curve.
        """
        if batch_size is None:
            return _run_session(self._session, norm_mag, norm_time, lg_wave, mask)
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
    return AstraInfer(onnx_file).predict_lc(time, mag, magerr, band, presorted=presorted)
