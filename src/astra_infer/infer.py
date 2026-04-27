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

_STRATEGIES = frozenset({"beginning", "end", "middle", "window", "sample"})


# ---------------------------------------------------------------------------
# Internal preprocessing helpers
# ---------------------------------------------------------------------------


def _resolve_rng(rng: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _normalize_mag(mag: ArrayLike, magerr: ArrayLike) -> ArrayLike:
    weighted_mean = np.average(mag, weights=magerr**-2)
    return mag - weighted_mean


def _normalize_time(time: ArrayLike) -> ArrayLike:
    return time - MJD_OFFSET


def _select_band_obs(
    mag_b: np.ndarray,
    time_b: np.ndarray,
    seq_size: int,
    strategy: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Select observations from a single band using the given strategy.

    Input arrays are already filtered to this band and time-sorted.
    Returns selected (mag, time) arrays of length at most ``seq_size``.
    Padding is applied separately by the caller.
    """
    m = len(mag_b)
    if m == 0 or strategy == "beginning":
        return mag_b[:seq_size], time_b[:seq_size]
    if strategy == "end":
        start = max(0, m - seq_size)
        return mag_b[start:], time_b[start:]
    if strategy == "middle":
        if m <= seq_size:
            return mag_b, time_b
        center = m // 2
        half = seq_size // 2
        start = max(0, min(center - half, m - seq_size))
        return mag_b[start : start + seq_size], time_b[start : start + seq_size]
    if strategy == "window":
        # Draw t_cut uniformly from [t_band[0], t_band[max(0, m-seq_size)]].
        # All observations at time >= t_cut are kept; at most seq_size are taken.
        max_start_idx = max(0, m - seq_size)
        t_cut = rng.uniform(time_b[0], time_b[max_start_idx])
        sel = time_b >= t_cut
        return mag_b[sel][:seq_size], time_b[sel][:seq_size]
    if strategy == "sample":
        # Draw min(m, seq_size) indices without replacement, keep chronological order.
        n_sel = min(m, seq_size)
        idx = np.sort(rng.choice(m, size=n_sel, replace=False))
        return mag_b[idx], time_b[idx]
    raise ValueError(f"Unknown strategy: {strategy!r}. Expected one of {sorted(_STRATEGIES)}.")


def _apply_strategy_to_bands(
    mag: np.ndarray,
    time: np.ndarray,
    band: np.ndarray,
    strategy: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply a subsampling strategy to all bands; return concatenated (700,) arrays."""
    dtype = mag.dtype
    result_mag, result_time, result_lg_wave, result_mask = [], [], [], []
    for band_name in BANDS:
        boolmask = band == band_name
        mag_b, time_b = mag[boolmask], time[boolmask]
        seq_size = SEQUENCE_PER_BAND[band_name]
        lg_eff_wave = LG_EFF_WAVE[band_name]

        mag_sel, time_sel = _select_band_obs(mag_b, time_b, seq_size, strategy, rng)

        input_size = len(mag_sel)
        pad_size = seq_size - input_size

        if pad_size <= 0:
            result_mag.append(mag_sel)
            result_time.append(time_sel)
            result_lg_wave.append(np.full(seq_size, lg_eff_wave, dtype=dtype))
            result_mask.append(np.zeros(seq_size, dtype=dtype))
        else:
            pad_val = np.zeros(pad_size, dtype=dtype)
            result_mag.append(np.r_[mag_sel, pad_val])
            result_time.append(np.r_[time_sel, pad_val])
            result_lg_wave.append(np.r_[np.full(input_size, lg_eff_wave, dtype=dtype), pad_val])
            result_mask.append(np.r_[np.zeros(input_size, dtype=dtype), np.ones(pad_size, dtype=dtype)])

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
    m, t, lw, mk = _apply_strategy_to_bands(mag, time, band, "beginning", np.random.default_rng(None))
    return m[None, :, None], t[None, :, None], lw[None, :, None], mk[None, :]


def _preprocess_one(
    time: ArrayLike,
    mag: ArrayLike,
    magerr: ArrayLike,
    band: ArrayLike,
    *,
    strategies: list[str],
    rng: np.random.Generator,
    presorted: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-process one light curve for every strategy. Returns ``(S, 700, 1)`` tensors."""
    norm_mag = _normalize_mag(mag, magerr).astype(np.float32)
    norm_time = _normalize_time(time).astype(np.float32)

    if not presorted:
        idx = np.argsort(norm_time)
        norm_time = norm_time[idx]
        norm_mag = norm_mag[idx]
        band = band[idx]

    all_mag, all_time, all_lg_wave, all_mask = [], [], [], []
    for strategy in strategies:
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
    try:
        import pyarrow as pa

        return isinstance(obj, pa.ListArray | pa.ChunkedArray | pa.Table | pa.StructArray)
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
    n_strategies : int
        Number of subsampling strategies *S*.
    """

    norm_mag: np.ndarray
    norm_time: np.ndarray
    lg_wave: np.ndarray
    mask: np.ndarray
    n_strategies: int = 1


def _preprocess_arrow(
    lcs: pa.ListArray | pa.ChunkedArray | pa.Table | pa.StructArray,
    field_names: dict[str, str],
    *,
    strategies: list[str],
    rng: np.random.Generator,
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
    elif isinstance(lcs, pa.Table | pa.StructArray):
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
            _preprocess_one(
                time_flat[s:e],
                mag_flat[s:e],
                magerr_flat[s:e],
                band_i,
                strategies=strategies,
                rng=rng,
                presorted=presorted,
            )
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
    strategies: str | list[str] = "beginning",
    rng: int | np.random.Generator | None = None,
    presorted: bool = False,
) -> Inputs:
    """Pre-process a single ZTF light curve into ONNX-ready :class:`Inputs`.

    Applies inverse-variance weighted mean subtraction, time normalisation
    (offset by :data:`MJD_OFFSET`), optional chronological sorting, and
    per-band clipping / zero-padding to produce fixed-length tensors.
    One set of tensors is produced per entry in ``strategies``.

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
    strategies : str or list of str, optional
        Subsampling strategy or list of strategies applied per band before
        padding.  Available strategies:

        - ``"beginning"`` (default): chronologically first observations.
        - ``"end"``: chronologically last observations.
        - ``"middle"``: centre of the observation sequence by index.
        - ``"window"``: random contiguous window — a cut time *t_cut* is
          drawn uniformly from ``[t_band[0], t_band[max(0, M−N)]]``
          (index-based, guaranteeing at least *N* observations remain).
        - ``"sample"``: random subsample without replacement, sorted to
          preserve chronological order.

    rng : int, numpy.random.Generator, or None, optional
        Seed or random number generator used by stochastic strategies
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
        ``len(strategies)``.
    """
    if isinstance(strategies, str):
        strategies = [strategies]
    rng_ = _resolve_rng(rng)
    result = _preprocess_one(time, mag, magerr, band, strategies=strategies, rng=rng_, presorted=presorted)
    return Inputs(
        result[0][None],  # (1, S, 700, 1)
        result[1][None],
        result[2][None],
        result[3][None],  # (1, S, 700)
        n_strategies=len(strategies),
    )


def preprocess_many(
    lcs: Sequence[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]] | Any,
    *,
    field_names: dict[str, str] | None = None,
    strategies: str | list[str] = "beginning",
    rng: int | np.random.Generator | None = None,
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
    strategies : str or list of str, optional
        Subsampling strategy or list of strategies.  See :func:`preprocess_lc`
        for available options.  Default is ``"beginning"``.
    rng : int, numpy.random.Generator, or None, optional
        Seed or random number generator for stochastic strategies.  See
        :func:`preprocess_lc`.  Default is ``None``.
    presorted : bool, optional
        Skip the internal time sort when every light curve is already sorted.
        Default is ``False``.

    Returns
    -------
    Inputs
        Preprocessed tensors with shape ``(N, S, 700, …)`` where *N* is the
        number of light curves and *S* is ``len(strategies)``.
    """
    if isinstance(strategies, str):
        strategies = [strategies]
    rng_ = _resolve_rng(rng)

    if field_names is not None or _is_arrow(lcs):
        arrays = _preprocess_arrow(
            lcs,
            field_names or _DEFAULT_FIELD_NAMES,
            strategies=strategies,
            rng=rng_,
            presorted=presorted,
        )
        return Inputs(*arrays, n_strategies=len(strategies))

    singles = [_preprocess_one(*lc, strategies=strategies, rng=rng_, presorted=presorted) for lc in lcs]
    return Inputs(
        np.concatenate([s[0][None] for s in singles], axis=0),  # (N, S, 700, 1)
        np.concatenate([s[1][None] for s in singles], axis=0),
        np.concatenate([s[2][None] for s in singles], axis=0),
        np.concatenate([s[3][None] for s in singles], axis=0),  # (N, S, 700)
        n_strategies=len(strategies),
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
        n_strat = inputs.n_strategies
        total = n_lcs * n_strat

        # Flatten the strategy axis so ONNX sees a plain (N*S, 700, 1) batch.
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

        return out.reshape(n_lcs, n_strat, 512)
