from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnxruntime as ort
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    pass

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

_SUBSAMPLING_METHODS = frozenset({"beginning", "end", "middle", "window", "sample"})
_STOCHASTIC_SUBSAMPLING = frozenset({"window", "sample"})


# ---------------------------------------------------------------------------
# Internal preprocessing helpers
# ---------------------------------------------------------------------------


def _normalize_mag(mag: ArrayLike, magerr: ArrayLike) -> ArrayLike:
    if len(mag) == 0:
        return np.empty(0, dtype=np.float32)
    weighted_mean = np.average(mag, weights=magerr**-2)
    return mag - weighted_mean


def _normalize_time(time: ArrayLike) -> ArrayLike:
    return time - MJD_OFFSET


def _cut_range(band: np.ndarray, m_total: int) -> tuple[int, int]:
    """Return valid (min_cut, max_cut) observation-index range for the window strategy."""
    min_cuts, max_cuts = [], []
    for b, seq in SEQUENCE_PER_BAND.items():
        half = seq // 2
        obs = np.where(band == b)[0]
        if 0 < half <= len(obs):
            min_cuts.append(int(obs[half - 1]) + 1)
            max_cuts.append(int(obs[-half]))
    min_cut = min(min_cuts, default=0)
    max_cut = max(max(max_cuts, default=m_total - 1), min_cut)
    return min_cut, max_cut


def _apply_strategy_to_bands(
    mag: np.ndarray,
    time: np.ndarray,
    band: np.ndarray,
    strategy: str,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply a subsampling method to all bands; return concatenated (700,) arrays.

    ``"beginning"`` and ``"end"`` select per-band (first / last *seq_size*
    observations).  ``"middle"`` cuts at the midpoint of the full observation
    array.  ``"window"`` draws a global cut index uniformly from the valid
    range that ensures at least one band has *seq_size // 2* observations on
    each side.  ``"sample"`` draws per-band.
    """
    dtype = mag.dtype
    m_total = len(time)

    cut_global = 0
    if strategy == "middle" and m_total > 0:
        cut_global = m_total // 2
    elif strategy == "window" and m_total > 0:
        min_cut, max_cut = _cut_range(band, m_total)
        cut_global = int(rng.integers(min_cut, max_cut + 1))

    result_mag, result_time, result_lg_wave, result_mask = [], [], [], []
    for band_name in BANDS:
        boolmask = band == band_name
        mag_b, time_b = mag[boolmask], time[boolmask]
        seq_size = SEQUENCE_PER_BAND[band_name]
        lg_eff_wave = LG_EFF_WAVE[band_name]
        m = len(mag_b)

        match strategy:
            case "beginning":
                mag_sel, time_sel = mag_b[:seq_size], time_b[:seq_size]
            case "end":
                mag_sel, time_sel = mag_b[-seq_size:], time_b[-seq_size:]
            case "middle" | "window":
                # Per-band cut index = number of this band's obs before
                # cut_global; centre a seq_size window around it.
                cut_b = int(boolmask[:cut_global].sum())
                start = max(0, min(cut_b - seq_size // 2, m - seq_size))
                mag_sel = mag_b[start : start + seq_size]
                time_sel = time_b[start : start + seq_size]
            case "sample":
                n_sel = min(m, seq_size)
                idx = np.sort(rng.choice(m, size=n_sel, replace=False)) if m > 0 else np.array([], dtype=int)
                mag_sel, time_sel = mag_b[idx], time_b[idx]
            case _:
                raise ValueError(
                    f"Unknown subsampling method: {strategy!r}. "
                    f"Expected one of {sorted(_SUBSAMPLING_METHODS)}."
                )

        input_size = len(mag_sel)
        pad_size = seq_size - input_size

        result_mag.append(mag_sel)
        result_time.append(time_sel)
        result_lg_wave.append(np.full(input_size, lg_eff_wave, dtype=dtype))
        result_mask.append(np.zeros(input_size, dtype=dtype))
        if pad_size > 0:
            pad_zeros = np.zeros(pad_size, dtype=dtype)
            result_mag.append(pad_zeros)
            result_time.append(pad_zeros)
            result_lg_wave.append(pad_zeros)
            result_mask.append(np.ones(pad_size, dtype=dtype))

    return (
        np.concatenate(result_mag),
        np.concatenate(result_time),
        np.concatenate(result_lg_wave),
        np.concatenate(result_mask),
    )


def _first_window(
    mag: ArrayLike, time: ArrayLike, band: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the ``'beginning'`` strategy; return ``(1, 700, 1)`` tensors."""
    m, t, lw, mk = _apply_strategy_to_bands(mag, time, band, "beginning", None)
    return m[None, :, None], t[None, :, None], lw[None, :, None], mk[None, :]


def _preprocess_one(
    time: ArrayLike,
    mag: ArrayLike,
    magerr: ArrayLike,
    band: ArrayLike,
    *,
    subsampling: list[str],
    rng: np.random.Generator | None,
    presorted: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-process one light curve for every subsampling method. Returns ``(S, 700, 1)`` tensors."""
    norm_mag = _normalize_mag(mag, magerr).astype(np.float32)
    norm_time = _normalize_time(time).astype(np.float32)

    if not presorted:
        idx = np.argsort(norm_time)
        norm_time = norm_time[idx]
        norm_mag = norm_mag[idx]
        band = band[idx]

    all_mag, all_time, all_lg_wave, all_mask = [], [], [], []
    for strategy in subsampling:
        m, t, lw, mk = _apply_strategy_to_bands(norm_mag, norm_time, band, strategy, rng)
        all_mag.append(m)
        all_time.append(t)
        all_lg_wave.append(lw)
        all_mask.append(mk)

    return (
        np.stack(all_mag)[:, :, None],  # (S, 700, 1)
        np.stack(all_time)[:, :, None],  # (S, 700, 1)
        np.stack(all_lg_wave)[:, :, None],  # (S, 700, 1)
        np.stack(all_mask),  # (S, 700)
    )


def _is_arrow(obj: Any) -> bool:
    if hasattr(obj, "__arrow_array__"):
        return True
    try:
        import pyarrow as pa

        return isinstance(
            obj,
            pa.ListArray | pa.LargeListArray | pa.FixedSizeListArray | pa.ChunkedArray,
        )
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
    All arrays share batch axis ``N`` (one row per light curve) and strategy
    axis ``S`` (one slice per subsampling strategy).
    Construct via :func:`preprocess_lc` (single light curve) or
    :func:`preprocess_many` (multiple light curves).

    The model expects photometry in the ZTF *g*, *r*, and *i* bands
    (see https://www.ztf.caltech.edu for the ZTF photometric system).

    Attributes
    ----------
    norm_mag : ndarray, shape (N, S, 700, 1)
        Inverse-variance weighted mean-subtracted magnitudes.
    norm_time : ndarray, shape (N, S, 700, 1)
        MJD times shifted by :data:`MJD_OFFSET`.
    lg_wave : ndarray, shape (N, S, 700, 1)
        log10 effective wavelength (Å) for each observation's ZTF band.
    mask : ndarray, shape (N, S, 700)
        Padding mask — ``1`` for padded positions, ``0`` for real observations.
    n_subsampling : int
        Number of subsampling methods *S*.
    """

    norm_mag: np.ndarray
    norm_time: np.ndarray
    lg_wave: np.ndarray
    mask: np.ndarray
    n_subsampling: int = 1


def _preprocess_arrow(
    lcs: Any,
    field_names: dict[str, str],
    *,
    subsampling: list[str],
    rng: np.random.Generator | None,
    presorted: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-row slices from an Arrow container and preprocess each."""
    import pyarrow as pa

    if hasattr(lcs, "__arrow_array__") and not isinstance(lcs, pa.Array | pa.ChunkedArray):
        lcs = pa.array(lcs)

    if isinstance(lcs, pa.ChunkedArray):
        lcs = lcs.combine_chunks()

    if not isinstance(lcs, pa.ListArray | pa.LargeListArray | pa.FixedSizeListArray):
        raise TypeError(f"Unsupported Arrow input type: {type(lcs).__name__}")

    cols = {key: lcs.values.field(arrow_name) for key, arrow_name in field_names.items()}
    null_cols = [key for key, col in cols.items() if col.null_count > 0]
    if null_cols:
        raise ValueError(f"Null values found in observation column(s): {null_cols}")
    time_flat = cols["time"].to_numpy(zero_copy_only=False)
    mag_flat = cols["mag"].to_numpy(zero_copy_only=False)
    magerr_flat = cols["magerr"].to_numpy(zero_copy_only=False)
    band_col = cols["band"]

    if isinstance(lcs, pa.FixedSizeListArray):
        list_size = lcs.type.list_size
        offsets = range(0, (len(lcs) + 1) * list_size, list_size)
    else:
        offsets = lcs.offsets.to_numpy()

    singles = []
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        start, end = int(start), int(end)
        band_i = band_col.slice(start, end - start).to_numpy(zero_copy_only=False)
        singles.append(
            _preprocess_one(
                time_flat[start:end],
                mag_flat[start:end],
                magerr_flat[start:end],
                band_i,
                subsampling=subsampling,
                rng=rng,
                presorted=presorted,
            )
        )

    s = len(subsampling)
    if len(singles) == 0:
        return (
            np.empty((0, s, SEQUENCE_LENGTH, 1), dtype=np.float32),
            np.empty((0, s, SEQUENCE_LENGTH, 1), dtype=np.float32),
            np.empty((0, s, SEQUENCE_LENGTH, 1), dtype=np.float32),
            np.empty((0, s, SEQUENCE_LENGTH), dtype=np.float32),
        )
    return (
        np.concatenate([s[0][None] for s in singles], axis=0),  # (N, S, 700, 1)
        np.concatenate([s[1][None] for s in singles], axis=0),
        np.concatenate([s[2][None] for s in singles], axis=0),
        np.concatenate([s[3][None] for s in singles], axis=0),  # (N, S, 700)
    )


def preprocess_lc(
    time: ArrayLike,
    mag: ArrayLike,
    magerr: ArrayLike,
    band: ArrayLike,
    *,
    subsampling: str | list[str] = "beginning",
    rng: int | np.random.Generator | None = None,
    presorted: bool = False,
) -> Inputs:
    """Pre-process a single ZTF light curve into ONNX-ready :class:`Inputs`.

    Applies inverse-variance weighted mean subtraction, time normalisation
    (offset by :data:`MJD_OFFSET`), optional chronological sorting, and
    per-band clipping / zero-padding to produce fixed-length tensors.
    One set of tensors is produced per entry in ``subsampling``.

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
    subsampling : str or list of str, optional
        Subsampling method or list of methods applied per band before
        zero-padding to the fixed sequence length.  Available methods:

        - ``"beginning"`` (default): select the chronologically first
          *seq_size* observations in each band independently.
        - ``"end"``: select the chronologically last *seq_size* observations
          in each band independently.
        - ``"middle"``: place a global cut at the midpoint of the full
          (all-band) sorted observation array.  Each band then selects a
          window of *seq_size* observations centred on its own count of
          observations before the cut, ensuring all bands share the same
          temporal reference point.
        - ``"window"``: same per-band windowing as ``"middle"``, but the
          global cut index is drawn uniformly at random from a valid range.
          The range is chosen so that at least one band has *seq_size // 2*
          observations on each side of the cut.  Requires ``rng``.
        - ``"sample"``: draw a random subset of up to *seq_size* observations
          in each band independently (without replacement), then sort them
          chronologically.  Requires ``rng``.

    rng : int, numpy.random.Generator, or None, optional
        Seed or random number generator used by stochastic subsampling methods
        (``"window"`` and ``"sample"``).  An integer is forwarded to
        :func:`numpy.random.default_rng`.  ``None`` picks an unpredictable
        seed.  Default is ``None``.
    presorted : bool, optional
        Skip the internal time sort when the input is already sorted.
        Default is ``False``.

    Returns
    -------
    Inputs
        Preprocessed tensors with shape ``(1, S, 700, …)`` where *S* is
        ``len(subsampling)``.
    """
    if isinstance(subsampling, str):
        subsampling = [subsampling]
    rng = np.random.default_rng(rng) if not _STOCHASTIC_SUBSAMPLING.isdisjoint(subsampling) else None
    result = _preprocess_one(time, mag, magerr, band, subsampling=subsampling, rng=rng, presorted=presorted)
    return Inputs(
        result[0][None],  # (1, S, 700, 1)
        result[1][None],
        result[2][None],
        result[3][None],  # (1, S, 700)
        n_subsampling=len(subsampling),
    )


def preprocess_many(
    lcs: Sequence[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]] | Any,
    *,
    field_names: dict[str, str] | None = None,
    subsampling: str | list[str] = "beginning",
    rng: int | np.random.Generator | None = None,
    presorted: bool = False,
) -> Inputs:
    """Pre-process multiple ZTF light curves into a single stacked :class:`Inputs`.

    Applies the same per-curve preprocessing as :func:`preprocess_lc` to each
    light curve, then stacks the results along the batch axis.

    Accepts either a sequence of ``(time, mag, magerr, band)`` tuples or a
    PyArrow container.  Supported Arrow layouts:

    * ``pa.ListArray`` / ``pa.LargeListArray`` / ``pa.FixedSizeListArray`` /
      ``pa.ChunkedArray`` — list-of-struct layout where each element is a list
      of per-observation structs.
    * Any object implementing the ``__arrow_array__`` protocol (converted via
      :func:`pyarrow.array`).

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
    subsampling : str or list of str, optional
        Subsampling method or list of methods.  See :func:`preprocess_lc`
        for available options.  Default is ``"beginning"``.
    rng : int, numpy.random.Generator, or None, optional
        Seed or random number generator for stochastic subsampling methods.
        See :func:`preprocess_lc`.  Default is ``None``.
    presorted : bool, optional
        Skip the internal time sort when every light curve is already sorted.
        Default is ``False``.

    Returns
    -------
    Inputs
        Preprocessed tensors with shape ``(N, S, 700, …)`` where *N* is the
        number of light curves and *S* is ``len(subsampling)``.
    """
    if isinstance(subsampling, str):
        subsampling = [subsampling]
    rng = np.random.default_rng(rng) if not _STOCHASTIC_SUBSAMPLING.isdisjoint(subsampling) else None

    if field_names is not None or _is_arrow(lcs):
        arrays = _preprocess_arrow(
            lcs,
            field_names or _DEFAULT_FIELD_NAMES,
            subsampling=subsampling,
            rng=rng,
            presorted=presorted,
        )
        return Inputs(*arrays, n_subsampling=len(subsampling))

    singles = [_preprocess_one(*lc, subsampling=subsampling, rng=rng, presorted=presorted) for lc in lcs]
    n_sub = len(subsampling)
    if len(singles) == 0:
        return Inputs(
            np.empty((0, n_sub, SEQUENCE_LENGTH, 1), dtype=np.float32),
            np.empty((0, n_sub, SEQUENCE_LENGTH, 1), dtype=np.float32),
            np.empty((0, n_sub, SEQUENCE_LENGTH, 1), dtype=np.float32),
            np.empty((0, n_sub, SEQUENCE_LENGTH), dtype=np.float32),
            n_subsampling=n_sub,
        )
    return Inputs(
        np.concatenate([s[0][None] for s in singles], axis=0),  # (N, S, 700, 1)
        np.concatenate([s[1][None] for s in singles], axis=0),
        np.concatenate([s[2][None] for s in singles], axis=0),
        np.concatenate([s[3][None] for s in singles], axis=0),  # (N, S, 700)
        n_subsampling=n_sub,
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
    providers : list of str, optional
        Execution providers passed to :class:`onnxruntime.InferenceSession`,
        e.g. ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.
        Defaults to the onnxruntime default (CPU).
    provider_options : list of dict, optional
        Per-provider option dicts, passed as the ``provider_options`` argument
        to :class:`onnxruntime.InferenceSession`.
    sess_options : onnxruntime.SessionOptions, optional
        Session-level options (thread counts, graph optimisation level, etc.)
        passed to :class:`onnxruntime.InferenceSession`.
    """

    def __init__(
        self,
        onnx_file: str | Path,
        *,
        providers: list[str] | None = None,
        provider_options: list[dict] | None = None,
        sess_options: ort.SessionOptions | None = None,
    ) -> None:
        self._session = ort.InferenceSession(
            onnx_file,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        )

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
            Maximum number of (light curve × strategy) slices per ONNX call.
            ``None`` passes all slices in a single call.  Default is 128.

        Returns
        -------
        embeddings : ndarray, shape (N, S, 512)
            One 512-d embedding per input light curve per strategy.
        """
        n_lcs = inputs.norm_mag.shape[0]
        n_strat = inputs.n_subsampling
        total = n_lcs * n_strat

        # Flatten the subsampling axis so ONNX sees a plain (N*S, 700, 1) batch.
        mag = inputs.norm_mag.reshape(total, SEQUENCE_LENGTH, 1)
        time = inputs.norm_time.reshape(total, SEQUENCE_LENGTH, 1)
        lg = inputs.lg_wave.reshape(total, SEQUENCE_LENGTH, 1)
        mask = inputs.mask.reshape(total, SEQUENCE_LENGTH)

        if batch_size is None:
            out = _run_session(self._session, mag, time, lg, mask)
        else:
            results = []
            for start in range(0, total, batch_size):
                sl = slice(start, start + batch_size)
                results.append(_run_session(self._session, mag[sl], time[sl], lg[sl], mask[sl]))
            out = np.concatenate(results, axis=0)

        return out.reshape(n_lcs, n_strat, -1)
