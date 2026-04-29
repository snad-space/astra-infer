import numpy as np
import pytest
from astra_infer.infer import (
    BANDS,
    SEQUENCE_LENGTH,
    Infer,
    preprocess_lc,
    preprocess_many,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lc(n, rng):
    """Return (time, mag, magerr, band) for a random light curve."""
    time = rng.uniform(58_000, 59_000, n)
    mag = rng.normal(loc=18, scale=1.0, size=n)
    magerr = np.full(n, 0.1)
    band = rng.choice(BANDS, size=n)
    return time, mag, magerr, band


# ---------------------------------------------------------------------------
# preprocess_lc — shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [32, 4098])
def test_inputs_from_lc_shape(n):
    """preprocess_lc returns tensors of the right shape for short and long LCs."""
    inputs = preprocess_lc(*_make_lc(n, np.random.default_rng(42)))

    assert inputs.norm_mag.shape == (1, 1, SEQUENCE_LENGTH, 1)
    assert inputs.norm_time.shape == (1, 1, SEQUENCE_LENGTH, 1)
    assert inputs.lg_wave.shape == (1, 1, SEQUENCE_LENGTH, 1)
    assert inputs.mask.shape == (1, 1, SEQUENCE_LENGTH)
    assert inputs.n_subsampling == 1


def test_inputs_from_lc_multiple_strategies_shape():
    """preprocess_lc with multiple strategies returns (1, S, 700, 1) tensors."""
    strategies = ["beginning", "end", "middle"]
    inputs = preprocess_lc(*_make_lc(300, np.random.default_rng(0)), subsampling=strategies)

    n_strat = len(strategies)
    assert inputs.norm_mag.shape == (1, n_strat, SEQUENCE_LENGTH, 1)
    assert inputs.mask.shape == (1, n_strat, SEQUENCE_LENGTH)
    assert inputs.n_subsampling == n_strat


def test_inputs_from_lc_single_string_strategy():
    """A single strategy string is accepted and gives S=1."""
    inputs = preprocess_lc(*_make_lc(100, np.random.default_rng(1)), subsampling="end")
    assert inputs.n_subsampling == 1
    assert inputs.norm_mag.shape == (1, 1, SEQUENCE_LENGTH, 1)


def test_inputs_from_lc_presorted():
    """presorted=True gives the same result as the default sort path."""
    rng = np.random.default_rng(7)
    n = 200
    time, mag, magerr, band = _make_lc(n, rng)

    idx = np.argsort(time)
    time_s, mag_s, magerr_s, band_s = time[idx], mag[idx], magerr[idx], band[idx]

    result_auto = preprocess_lc(time, mag, magerr, band)
    result_pre = preprocess_lc(time_s, mag_s, magerr_s, band_s, presorted=True)

    np.testing.assert_array_equal(result_auto.norm_mag, result_pre.norm_mag)
    np.testing.assert_array_equal(result_auto.norm_time, result_pre.norm_time)


# ---------------------------------------------------------------------------
# preprocess_lc — strategies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["beginning", "end", "middle", "window", "sample"])
def test_strategy_produces_valid_output(strategy):
    """Every strategy produces finite, correctly shaped tensors."""
    inputs = preprocess_lc(*_make_lc(400, np.random.default_rng(2)), subsampling=strategy, rng=0)
    assert inputs.norm_mag.shape == (1, 1, SEQUENCE_LENGTH, 1)
    assert np.all(np.isfinite(inputs.norm_mag[inputs.mask == 0]))


@pytest.mark.parametrize("strategy", ["beginning", "end", "middle"])
def test_deterministic_strategies_require_no_rng(strategy):
    """Deterministic strategies give identical results regardless of rng."""
    lc = _make_lc(400, np.random.default_rng(3))
    a = preprocess_lc(*lc, subsampling=strategy, rng=0)
    b = preprocess_lc(*lc, subsampling=strategy, rng=99)
    np.testing.assert_array_equal(a.norm_mag, b.norm_mag)


def test_beginning_end_differ_for_long_lc():
    """'beginning' and 'end' select different time windows for a long LC."""
    lc = _make_lc(800, np.random.default_rng(4))
    beg = preprocess_lc(*lc, subsampling="beginning")
    end = preprocess_lc(*lc, subsampling="end")
    assert not np.array_equal(beg.norm_mag, end.norm_mag)


def test_short_lc_end_same_as_beginning():
    """For a short LC (m_band < seq_size for every band), 'end' equals 'beginning'."""
    # 10 obs — far fewer than any per-band seq_size, so last == first per band.
    lc = _make_lc(10, np.random.default_rng(5))
    beg = preprocess_lc(*lc, subsampling="beginning")
    end = preprocess_lc(*lc, subsampling="end")
    np.testing.assert_array_equal(beg.norm_mag, end.norm_mag)


def test_window_strategy_reproducible_with_rng():
    """'window' gives identical results when given the same integer seed."""
    lc = _make_lc(800, np.random.default_rng(6))
    a = preprocess_lc(*lc, subsampling="window", rng=42)
    b = preprocess_lc(*lc, subsampling="window", rng=42)
    np.testing.assert_array_equal(a.norm_mag, b.norm_mag)


def test_window_strategy_varies_with_different_rng():
    """'window' generally produces different results for different seeds."""
    lc = _make_lc(800, np.random.default_rng(7))
    a = preprocess_lc(*lc, subsampling="window", rng=0)
    b = preprocess_lc(*lc, subsampling="window", rng=1)
    # Not guaranteed, but extremely unlikely to be equal for 800 obs.
    assert not np.array_equal(a.norm_mag, b.norm_mag)


def test_sample_strategy_reproducible_with_rng():
    """'sample' gives identical results when given the same integer seed."""
    lc = _make_lc(800, np.random.default_rng(8))
    a = preprocess_lc(*lc, subsampling="sample", rng=0)
    b = preprocess_lc(*lc, subsampling="sample", rng=0)
    np.testing.assert_array_equal(a.norm_mag, b.norm_mag)


def test_rng_generator_accepted():
    """An np.random.Generator instance is accepted for rng."""
    lc = _make_lc(400, np.random.default_rng(9))
    rng = np.random.default_rng(10)
    inputs = preprocess_lc(*lc, subsampling=["window", "sample"], rng=rng)
    assert inputs.n_subsampling == 2


def test_invalid_strategy_raises():
    """An unrecognised strategy name raises ValueError."""
    lc = _make_lc(100, np.random.default_rng(11))
    with pytest.raises(ValueError, match="Unknown subsampling method"):
        preprocess_lc(*lc, subsampling="bogus")


# ---------------------------------------------------------------------------
# Zero-length inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["beginning", "end", "middle", "window", "sample"])
def test_zero_obs_lc(strategy):
    """preprocess_lc with 0 observations returns a fully-padded (1,1,700,1) tensor."""
    inputs = preprocess_lc(
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        subsampling=strategy,
        rng=0,
    )
    assert inputs.norm_mag.shape == (1, 1, SEQUENCE_LENGTH, 1)
    assert np.all(inputs.mask == 1)


def test_empty_sequence_batch():
    """preprocess_many([]) returns (0, 1, 700, 1) tensors."""
    inputs = preprocess_many([])
    assert inputs.norm_mag.shape == (0, 1, SEQUENCE_LENGTH, 1)
    assert inputs.mask.shape == (0, 1, SEQUENCE_LENGTH)
    assert inputs.n_subsampling == 1


def test_empty_arrow_batch():
    """preprocess_many with a 0-row Arrow ListArray returns (0, 1, 700, 1) tensors."""
    import pyarrow as pa

    empty_struct = pa.StructArray.from_arrays(
        [pa.array([], type=pa.float64())] * 3 + [pa.array([], type=pa.string())],
        names=["time", "mag", "magerr", "band"],
    )
    empty_lcs = pa.ListArray.from_arrays(pa.array([0], type=pa.int32()), empty_struct)
    inputs = preprocess_many(empty_lcs)
    assert inputs.norm_mag.shape == (0, 1, SEQUENCE_LENGTH, 1)
    assert inputs.mask.shape == (0, 1, SEQUENCE_LENGTH)


def test_null_observation_values_raise():
    """Null values inside the flat observation arrays raise ValueError."""
    import pyarrow as pa

    struct = pa.StructArray.from_arrays(
        [
            pa.array([1.0, None, 3.0]),
            pa.array([1.0, 2.0, 3.0]),
            pa.array([0.1] * 3),
            pa.array(["g", "r", "i"]),
        ],
        names=["time", "mag", "magerr", "band"],
    )
    lcs = pa.ListArray.from_arrays(pa.array([0, 3], type=pa.int32()), struct)
    with pytest.raises(ValueError, match="time"):
        preprocess_many(lcs)


def test_null_list_element_treated_as_zero_length():
    """A null element in a ListArray is treated as a zero-length (fully-padded) LC."""
    import pyarrow as pa

    rng = np.random.default_rng(16)
    lc = _make_lc(50, rng)

    normal = pa.ListArray.from_arrays(
        pa.array([0, 50], type=pa.int32()),
        pa.StructArray.from_arrays(
            [pa.array(lc[0]), pa.array(lc[1]), pa.array(lc[2]), pa.array(lc[3], type=pa.string())],
            names=["time", "mag", "magerr", "band"],
        ),
    )
    # Build a 2-row array: row 0 = null, row 1 = the actual LC
    with_null = pa.concat_arrays([pa.array([None], type=normal.type), normal])

    inputs = preprocess_many(with_null)
    assert inputs.norm_mag.shape == (2, 1, SEQUENCE_LENGTH, 1)
    assert np.all(inputs.mask[0] == 1)  # null row is fully padded
    assert not np.all(inputs.mask[1] == 1)  # real row has observations


# ---------------------------------------------------------------------------
# preprocess_many
# ---------------------------------------------------------------------------


def test_inputs_from_lcs_shape():
    """preprocess_many stacks individual preprocess_lc results correctly."""
    rng = np.random.default_rng(3)
    lcs = [_make_lc(n, rng) for n in [32, 700, 4096]]

    stacked = preprocess_many(lcs)
    assert stacked.norm_mag.shape == (3, 1, SEQUENCE_LENGTH, 1)
    assert stacked.mask.shape == (3, 1, SEQUENCE_LENGTH)
    assert stacked.n_subsampling == 1

    for i, lc in enumerate(lcs):
        single = preprocess_lc(*lc)
        np.testing.assert_array_equal(single.norm_mag, stacked.norm_mag[[i]])
        np.testing.assert_array_equal(single.mask, stacked.mask[[i]])


def test_inputs_from_lcs_multiple_strategies():
    """preprocess_many with S strategies produces (N, S, 700, 1) tensors."""
    rng = np.random.default_rng(12)
    lcs = [_make_lc(200, rng) for _ in range(4)]
    strategies = ["beginning", "end"]

    inputs = preprocess_many(lcs, subsampling=strategies)
    assert inputs.norm_mag.shape == (4, 2, SEQUENCE_LENGTH, 1)
    assert inputs.n_subsampling == 2


# ---------------------------------------------------------------------------
# Infer.predict
# ---------------------------------------------------------------------------


def test_predict_empty_batch(onnx_file):
    """predict on an empty Inputs returns (0, 1, 512) without error."""
    embeddings = Infer(onnx_file).predict(preprocess_many([]))
    assert embeddings.shape == (0, 1, 512)


@pytest.mark.parametrize("n", [32, 4098])
def test_predict_single(onnx_file, n):
    """predict returns (1, 1, 512) finite embeddings for a single LC."""
    rng = np.random.default_rng(42)
    embeddings = Infer(onnx_file).predict(preprocess_lc(*_make_lc(n, rng)))

    assert embeddings.shape == (1, 1, 512)
    assert np.all(np.isfinite(embeddings))


def test_predict_multiple_strategies(onnx_file):
    """predict returns (1, S, 512) when multiple strategies are used."""
    strategies = ["beginning", "end", "middle"]
    inputs = preprocess_lc(*_make_lc(400, np.random.default_rng(0)), subsampling=strategies)
    embeddings = Infer(onnx_file).predict(inputs)

    assert embeddings.shape == (1, len(strategies), 512)
    assert np.all(np.isfinite(embeddings))


def test_predict_deterministic(onnx_file):
    """Session is reused; calling twice with the same input gives identical results."""
    rng = np.random.default_rng(0)
    model = Infer(onnx_file)
    inputs = preprocess_lc(*_make_lc(150, rng))

    np.testing.assert_array_equal(model.predict(inputs), model.predict(inputs))


@pytest.mark.parametrize("n_curves,batch_size", [(1, 128), (10, 3), (10, 128), (10, None)])
def test_predict_batch(onnx_file, n_curves, batch_size):
    """predict returns (N, 1, 512) and matches repeated single-curve calls."""
    rng = np.random.default_rng(1)
    lcs = [_make_lc(150, rng) for _ in range(n_curves)]

    model = Infer(onnx_file)
    batch_embeddings = model.predict(preprocess_many(lcs), batch_size=batch_size)

    assert batch_embeddings.shape == (n_curves, 1, 512)
    assert np.all(np.isfinite(batch_embeddings))

    for i, lc in enumerate(lcs):
        single = model.predict(preprocess_lc(*lc))
        np.testing.assert_array_equal(batch_embeddings[i], single[0])


@pytest.mark.parametrize("n_curves,n_strategies", [(3, 2), (5, 3)])
def test_predict_batch_multi_strategy(onnx_file, n_curves, n_strategies):
    """predict returns (N, S, 512) for batches with multiple strategies."""
    strategies = ["beginning", "end", "middle", "window", "sample"][:n_strategies]
    rng = np.random.default_rng(13)
    lcs = [_make_lc(300, rng) for _ in range(n_curves)]

    inputs = preprocess_many(lcs, subsampling=strategies, rng=42)
    embeddings = Infer(onnx_file).predict(inputs)

    assert embeddings.shape == (n_curves, n_strategies, 512)
    assert np.all(np.isfinite(embeddings))


# ---------------------------------------------------------------------------
# PyArrow helpers
# ---------------------------------------------------------------------------


def _make_list_struct(lcs):
    """Build a pa.ListArray (list-of-struct) from a list of (time, mag, magerr, band) tuples."""
    import pyarrow as pa

    all_time, all_mag, all_magerr, all_band = [], [], [], []
    offsets = [0]
    for time, mag, magerr, band in lcs:
        all_time.append(pa.array(time))
        all_mag.append(pa.array(mag))
        all_magerr.append(pa.array(magerr))
        all_band.append(pa.array(band, type=pa.string()))
        offsets.append(offsets[-1] + len(time))

    flat_struct = pa.StructArray.from_arrays(
        [
            pa.concat_arrays(all_time),
            pa.concat_arrays(all_mag),
            pa.concat_arrays(all_magerr),
            pa.concat_arrays(all_band),
        ],
        names=["time", "mag", "magerr", "band"],
    )
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), flat_struct)


# ---------------------------------------------------------------------------
# Arrow tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_arrow", [_make_list_struct], ids=["list_struct"])
def test_inputs_from_lcs_arrow_matches_sequence(make_arrow):
    """preprocess_many with Arrow input matches sequence-of-tuples result."""
    rng = np.random.default_rng(5)
    lcs = [_make_lc(n, rng) for n in [32, 700, 4096]]

    result_seq = preprocess_many(lcs)
    result_arrow = preprocess_many(make_arrow(lcs))

    np.testing.assert_array_equal(result_seq.norm_mag, result_arrow.norm_mag)
    np.testing.assert_array_equal(result_seq.norm_time, result_arrow.norm_time)
    np.testing.assert_array_equal(result_seq.lg_wave, result_arrow.lg_wave)
    np.testing.assert_array_equal(result_seq.mask, result_arrow.mask)


def test_inputs_from_lcs_arrow_chunked():
    """preprocess_many accepts a ChunkedArray of list-struct type."""
    import pyarrow as pa

    rng = np.random.default_rng(9)
    lcs = [_make_lc(n, rng) for n in [50, 200]]

    chunked = pa.chunked_array([_make_list_struct(lcs)])

    result_seq = preprocess_many(lcs)
    result_arrow = preprocess_many(chunked)

    np.testing.assert_array_equal(result_seq.norm_mag, result_arrow.norm_mag)


def test_inputs_from_lcs_arrow_nonzero_first_offset():
    """preprocess_many handles a sliced ListArray where offsets[0] != 0."""

    rng = np.random.default_rng(14)
    lcs = [_make_lc(n, rng) for n in [50, 100, 200]]

    full = _make_list_struct(lcs)
    sliced = full.slice(1)  # offsets[0] == offset of second original row, != 0

    result_seq = preprocess_many(lcs[1:])
    result_arrow = preprocess_many(sliced)

    np.testing.assert_array_equal(result_seq.norm_mag, result_arrow.norm_mag)
    np.testing.assert_array_equal(result_seq.mask, result_arrow.mask)


def _make_fixed_size_list_struct(lcs, list_size):
    """Build a pa.FixedSizeListArray (fixed-size-list-of-struct) padding each LC to list_size."""
    import pyarrow as pa

    all_time, all_mag, all_magerr, all_band = [], [], [], []
    for time, mag, magerr, band in lcs:
        n = len(time)
        pad = list_size - n
        all_time.append(pa.concat_arrays([pa.array(time), pa.array(np.zeros(pad))]))
        all_mag.append(pa.concat_arrays([pa.array(mag), pa.array(np.zeros(pad))]))
        all_magerr.append(pa.concat_arrays([pa.array(magerr), pa.array(np.full(pad, 0.1))]))
        all_band.append(
            pa.concat_arrays([pa.array(band, type=pa.string()), pa.array(["g"] * pad, type=pa.string())])
        )

    flat_struct = pa.StructArray.from_arrays(
        [
            pa.concat_arrays(all_time),
            pa.concat_arrays(all_mag),
            pa.concat_arrays(all_magerr),
            pa.concat_arrays(all_band),
        ],
        names=["time", "mag", "magerr", "band"],
    )
    return pa.FixedSizeListArray.from_arrays(flat_struct, type=pa.list_(flat_struct.type, list_size))


def test_inputs_from_lcs_arrow_fixed_size_list():
    """preprocess_many handles a FixedSizeListArray with correct row count."""

    rng = np.random.default_rng(15)
    list_size = 50
    lcs = [_make_lc(list_size, rng) for _ in range(3)]

    result_seq = preprocess_many(lcs)
    result_arrow = preprocess_many(_make_fixed_size_list_struct(lcs, list_size))

    np.testing.assert_array_equal(result_seq.norm_mag, result_arrow.norm_mag)
    np.testing.assert_array_equal(result_seq.mask, result_arrow.mask)


def test_inputs_from_lcs_arrow_custom_field_names():
    """preprocess_many accepts a custom field_names mapping."""
    import pyarrow as pa

    rng = np.random.default_rng(6)
    n = 100
    time, mag, magerr, band = _make_lc(n, rng)

    flat_struct = pa.StructArray.from_arrays(
        [pa.array(time), pa.array(mag), pa.array(magerr), pa.array(band, type=pa.string())],
        names=["mjd", "psf_mag", "psf_magerr", "fid"],
    )
    arrow_lcs = pa.ListArray.from_arrays(pa.array([0, n], type=pa.int32()), flat_struct)

    custom_names = {"time": "mjd", "mag": "psf_mag", "magerr": "psf_magerr", "band": "fid"}
    result_arrow = preprocess_many(arrow_lcs, field_names=custom_names)
    result_seq = preprocess_many([(time, mag, magerr, band)])

    np.testing.assert_array_equal(result_seq.norm_mag, result_arrow.norm_mag)
    np.testing.assert_array_equal(result_seq.mask, result_arrow.mask)
