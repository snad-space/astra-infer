from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnxruntime as ort
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    import pyarrow as pa

BANDS = np.array(["g", "r", "i"])
"""ZTF photometric bands accepted by the model."""

SEQUENCE_PER_BAND = {"g": 300, "r": 350, "i": 50}
assert list(BANDS) == list(SEQUENCE_PER_BAND.keys())

SEQUENCE_LENGTH = 700
"""Total fixed sequence length fed to the model (300 g + 350 r + 50 i)."""
assert sum(SEQUENCE_PER_BAND.values()) == SEQUENCE_LENGTH

MJD_OFFSET = 58_000.0
"""MJD subtracted from observation times during normalisation."""

LG_EFF_WAVE = {"g": np.log10(4746.48), "r": np.log10(6366.38), "i": np.log10(7829.03)}
"""log10 effective wavelength (Å) for each ZTF band."""
assert list(BANDS) == list(LG_EFF_WAVE.keys())

_DEFAULT_FIELD_NAMES: dict[str, str] = {"time": "time", "mag": "mag", "magerr": "magerr", "band": "band"}


# ---------------------------------------------------------------------------
# Internal preprocessing helpers
# ---------------------------------------------------------------------------

def _normalize_mag(mag: ArrayLike, magerr: ArrayLike) -> ArrayLike:
    weighted_mean = np.average(mag, weights=magerr**-2)
    return mag - weighted_mean


def _normalize_time(time: ArrayLike) -> ArrayLike:
    return time - MJD_OFFSET


def _single_band_window(
    mag: ArrayLike, time: ArrayLike, band: ArrayLike, band_name: str
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    boolmask = band == band_name
    mag, time = mag[boolmask], time[boolmask]

    dtype = mag.dtype
    input_size = mag.size
    seq_size = SEQUENCE_PER_BAND[band_name]
    pad_size = seq_size - input_size
    lg_eff_wave = LG_EFF_WAVE[band_name]

    if pad_size <= 0:
        return (
            mag[:seq_size],
            time[:seq_size],
            np.full(seq_size, lg_eff_wave, dtype=dtype),
            np.zeros(seq_size, dtype=dtype),
        )

    pad_val = np.zeros(pad_size, dtype=dtype)
    return (
        np.r_[mag, pad_val],
        np.r_[time, pad_val],
        np.r_[np.full(input_size, lg_eff_wave), pad_val],
        np.r_[np.zeros(input_size, dtype=dtype), np.ones(pad_size, dtype=dtype)],
    )


def _first_window(
    mag: ArrayLike, time: ArrayLike, band: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mag_, time_, lg_wave_, mask_ = [], [], [], []
    for band_name in BANDS:
        mag_b, time_b, lg_wave_b, mask_b = _single_band_window(mag, time, band, band_name)
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


def _preprocess_one(
    time: ArrayLike,
    mag: ArrayLike,
    magerr: ArrayLike,
    band: ArrayLike,
    *,
    presorted: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    norm_mag = _normalize_mag(mag, magerr).astype(np.float32)
    norm_time = _normalize_time(time).astype(np.float32)

    if not presorted:
        idx = np.argsort(norm_time)
        norm_time = norm_time[idx]
        norm_mag = norm_mag[idx]
        band = band[idx]

    return _first_window(norm_mag, norm_time, band)


def _is_arrow(obj: Any) -> bool:
    try:
        import pyarrow as pa

        return isinstance(obj, (pa.ListArray, pa.ChunkedArray, pa.Table, pa.StructArray))
    except ImportError:
        return False


def _run_session(
    session: ort.InferenceSession,
    norm_mag: np.ndarray,
    norm_time: np.ndarray,
    lg_wave: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    (embeddings,) = session.run(
        None,
        {
            "input": norm_mag.astype(np.float32),
            "times": norm_time.astype(np.float32),
            "band_info": lg_wave.astype(np.float32),
            "mask": mask.astype(np.float32),
        },
    )
    return embeddings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Inputs:
    """Preprocessed ZTF light-curve tensors ready for ONNX inference.

    Holds the four fixed-length arrays that the Astra embedding model expects.
    All arrays share the same batch axis ``N`` (one row per light curve).
    Construct via :func:`preprocess_lc` (single light curve) or
    :func:`preprocess_many` (multiple light curves).

    The model expects photometry in the ZTF *g*, *r*, and *i* bands
    (see https://www.ztf.caltech.edu for the ZTF photometric system).

    Attributes
    ----------
    norm_mag : ndarray, shape (N, 700, 1)
        Inverse-variance weighted mean-subtracted magnitudes.
    norm_time : ndarray, shape (N, 700, 1)
        MJD times shifted by :data:`MJD_OFFSET`.
    lg_wave : ndarray, shape (N, 700, 1)
        log10 effective wavelength (Å) for each observation's ZTF band.
    mask : ndarray, shape (N, 700)
        Padding mask — ``1`` for padded positions, ``0`` for real observations.
    """

    norm_mag: np.ndarray
    norm_time: np.ndarray
    lg_wave: np.ndarray
    mask: np.ndarray


def _preprocess_arrow(
    lcs: pa.ListArray | pa.ChunkedArray | pa.Table | pa.StructArray,
    field_names: dict[str, str],
    *,
    presorted: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-row slices from an Arrow container and preprocess each."""
    import pyarrow as pa

    if isinstance(lcs, pa.ChunkedArray):
        lcs = lcs.combine_chunks()

    if isinstance(lcs, pa.ListArray):
        offsets = lcs.offsets.to_numpy()
        flat_struct = lcs.values
        cols = {key: flat_struct.field(arrow_name) for key, arrow_name in field_names.items()}
    elif isinstance(lcs, (pa.Table, pa.StructArray)):
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
    band_col = cols["band"]

    singles = []
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        band_i = np.asarray(band_col.slice(s, e - s).to_pylist())
        singles.append(
            _preprocess_one(time_flat[s:e], mag_flat[s:e], magerr_flat[s:e], band_i, presorted=presorted)
        )

    return (
        np.concatenate([s[0] for s in singles], axis=0),
        np.concatenate([s[1] for s in singles], axis=0),
        np.concatenate([s[2] for s in singles], axis=0),
        np.concatenate([s[3] for s in singles], axis=0),
    )


def preprocess_lc(
    time: ArrayLike,
    mag: ArrayLike,
    magerr: ArrayLike,
    band: ArrayLike,
    *,
    presorted: bool = False,
) -> Inputs:
    """Pre-process a single ZTF light curve into ONNX-ready :class:`Inputs`.

    Applies inverse-variance weighted mean subtraction, time normalisation
    (offset by :data:`MJD_OFFSET`), optional chronological sorting, and
    per-band clipping / zero-padding to produce fixed-length tensors.

    Parameters
    ----------
    time : array-like
        Observation times (MJD).
    mag : array-like
        PSF magnitudes.
    magerr : array-like
        1-σ magnitude uncertainties.
    band : array-like
        Band labels — each element must be one of ``{"g", "r", "i"}``
        (ZTF photometric bands, see https://www.ztf.caltech.edu).
    presorted : bool, optional
        Skip the internal time sort when the input is already sorted.
        Default is ``False``.

    Returns
    -------
    Inputs
        Preprocessed tensors with ``N = 1``.
    """
    return Inputs(*_preprocess_one(time, mag, magerr, band, presorted=presorted))


def preprocess_many(
    lcs: Sequence[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]] | Any,
    *,
    field_names: dict[str, str] | None = None,
    presorted: bool = False,
) -> Inputs:
    """Pre-process multiple ZTF light curves into a single stacked :class:`Inputs`.

    Applies the same per-curve preprocessing as :func:`preprocess_lc` to each
    light curve, then stacks the results along the batch axis.

    Accepts either a sequence of ``(time, mag, magerr, band)`` tuples or a
    PyArrow container.  Supported Arrow layouts:

    * ``pa.ListArray`` / ``pa.ChunkedArray`` with a struct value type
      (*list-of-struct*): each element is a list of per-observation structs.
    * ``pa.Table`` / ``pa.StructArray`` where each relevant column is a
      ``ListArray`` of per-observation values (*struct-of-lists*).

    Numeric Arrow columns are extracted as zero-copy NumPy views; the string
    ``band`` column is sliced per-row to avoid full materialisation.

    Parameters
    ----------
    lcs : sequence of (time, mag, magerr, band) tuples, or Arrow container
        One light curve per element / row.
    field_names : dict[str, str] or None, optional
        Mapping from canonical names (``"time"``, ``"mag"``, ``"magerr"``,
        ``"band"``) to the actual Arrow column / field names.  Only used for
        Arrow inputs; defaults to the canonical names.
    presorted : bool, optional
        Skip the internal time sort when every light curve is already sorted.
        Default is ``False``.

    Returns
    -------
    Inputs
        Preprocessed tensors with ``N`` equal to the number of light curves.
    """
    if field_names is not None or _is_arrow(lcs):
        return Inputs(*_preprocess_arrow(lcs, field_names or _DEFAULT_FIELD_NAMES, presorted=presorted))
    singles = [_preprocess_one(*lc, presorted=presorted) for lc in lcs]
    return Inputs(
        np.concatenate([s[0] for s in singles], axis=0),
        np.concatenate([s[1] for s in singles], axis=0),
        np.concatenate([s[2] for s in singles], axis=0),
        np.concatenate([s[3] for s in singles], axis=0),
    )


class Infer:
    """Astra embedding model for ZTF light curves.

    Wraps an ONNX session and exposes a single :meth:`predict` method that
    accepts preprocessed :class:`Inputs`.  The model was designed for
    photometry in the ZTF *g*, *r*, and *i* bands
    (see https://www.ztf.caltech.edu).

    Parameters
    ----------
    onnx_file : str or Path
        Path to the ONNX model file.  The session is created once and reused
        for every call.
    """

    def __init__(self, onnx_file: str | Path) -> None:
        self._session = ort.InferenceSession(onnx_file)

    def predict(
        self,
        inputs: Inputs,
        *,
        batch_size: int | None = 128,
    ) -> np.ndarray:
        """Run ONNX inference on preprocessed light-curve tensors.

        Parameters
        ----------
        inputs : Inputs
            Preprocessed tensors as returned by :func:`preprocess_lc` or
            :func:`preprocess_many`.
        batch_size : int or None, optional
            Maximum number of light curves per ONNX call.  ``None`` passes all
            light curves in a single call.  Default is 128.

        Returns
        -------
        embeddings : ndarray, shape (N, 512)
            One 512-d embedding per input light curve.
        """
        if batch_size is None:
            return _run_session(self._session, inputs.norm_mag, inputs.norm_time, inputs.lg_wave, inputs.mask)
        n = inputs.norm_mag.shape[0]
        results = []
        for start in range(0, n, batch_size):
            sl = slice(start, start + batch_size)
            results.append(
                _run_session(
                    self._session,
                    inputs.norm_mag[sl],
                    inputs.norm_time[sl],
                    inputs.lg_wave[sl],
                    inputs.mask[sl],
                )
            )
        return np.concatenate(results, axis=0)
