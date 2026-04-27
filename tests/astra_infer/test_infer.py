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
    assert inputs.n_strategies == 1


def test_inputs_from_lc_multiple_strategies_shape():
    """preprocess_lc with multiple strategies returns (1, S, 700, 1) tensors."""
    strategies = ["beginning", "end", "middle"]
    inputs = preprocess_lc(*_make_lc(300, np.random.default_rng(0)), strategies=strategies)

    n_strat = len(strategies)
    assert inputs.norm_mag.shape == (1, n_strat, SEQUENCE_LENGTH, 1)
    assert inputs.mask.shape == (1, n_strat, SEQUENCE_LENGTH)
    assert inputs.n_strategies == n_strat


def test_inputs_from_lc_single_string_strategy():
    """A single strategy string is accepted and gives S=1."""
    inputs = preprocess_lc(*_make_lc(100, np.random.default_rng(1)), strategies="end")
    assert inputs.n_strategies == 1
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
    inputs = preprocess_lc(*_make_lc(400, np.random.default_rng(2)), strategies=strategy, rng=0)
    assert inputs.norm_mag.shape == (1, 1, SEQUENCE_LENGTH, 1)
    assert np.all(np.isfinite(inputs.norm_mag[inputs.mask == 0]))


@pytest.mark.parametrize("strategy", ["beginning", "end", "middle"])
def test_deterministic_strategies_require_no_rng(strategy):
    """Deterministic strategies give identical results regardless of rng."""
    lc = _make_lc(400, np.random.default_rng(3))
    a = preprocess_lc(*lc, strategies=strategy, rng=0)
    b = preprocess_lc(*lc, strategies=strategy, rng=99)
    np.testing.assert_array_equal(a.norm_mag, b.norm_mag)


def test_beginning_end_differ_for_long_lc():
    """'beginning' and 'end' select different time windows for a long LC."""
    lc = _make_lc(800, np.random.default_rng(4))
    beg = preprocess_lc(*lc, strategies="beginning")
    end = preprocess_lc(*lc, strategies="end")
    assert not np.array_equal(beg.norm_mag, end.norm_mag)


def test_short_lc_end_same_as_beginning():
    """For a short LC (m_band < seq_size for every band), 'end' equals 'beginning'."""
    # 10 obs — far fewer than any per-band seq_size, so last == first per band.
    lc = _make_lc(10, np.random.default_rng(5))
    beg = preprocess_lc(*lc, strategies="beginning")
    end = preprocess_lc(*lc, strategies="end")
    np.testing.assert_array_equal(beg.norm_mag, end.norm_mag)


def test_window_strategy_reproducible_with_rng():
    """'window' gives identical results when given the same integer seed."""
    lc = _make_lc(800, np.random.default_rng(6))
    a = preprocess_lc(*lc, strategies="window", rng=42)
    b = preprocess_lc(*lc, strategies="window", rng=42)
    np.testing.assert_array_equal(a.norm_mag, b.norm_mag)


def test_window_strategy_varies_with_different_rng():
    """'window' generally produces different results for different seeds."""
    lc = _make_lc(800, np.random.default_rng(7))
    a = preprocess_lc(*lc, strategies="window", rng=0)
    b = preprocess_lc(*lc, strategies="window", rng=1)
    # Not guaranteed, but extremely unlikely to be equal for 800 obs.
    assert not np.array_equal(a.norm_mag, b.norm_mag)


def test_sample_strategy_reproducible_with_rng():
    """'sample' gives identical results when given the same integer seed."""
    lc = _make_lc(800, np.random.default_rng(8))
    a = preprocess_lc(*lc, strategies="sample", rng=0)
    b = preprocess_lc(*lc, strategies="sample", rng=0)
    np.testing.assert_array_equal(a.norm_mag, b.norm_mag)


def test_rng_generator_accepted():
    """An np.random.Generator instance is accepted for rng."""
    lc = _make_lc(400, np.random.default_rng(9))
    rng = np.random.default_rng(10)
    inputs = preprocess_lc(*lc, strategies=["window", "sample"], rng=rng)
    assert inputs.n_strategies == 2


def test_invalid_strategy_raises():
    """An unrecognised strategy name raises ValueError."""
    lc = _make_lc(100, np.random.default_rng(11))
    with pytest.raises(ValueError, match="Unknown strategy"):
        preprocess_lc(*lc, strategies="bogus")


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
    assert stacked.n_strategies == 1

    for i, lc in enumerate(lcs):
        single = preprocess_lc(*lc)
        np.testing.assert_array_equal(single.norm_mag, stacked.norm_mag[[i]])
        np.testing.assert_array_equal(single.mask, stacked.mask[[i]])


def test_inputs_from_lcs_multiple_strategies():
    """preprocess_many with S strategies produces (N, S, 700, 1) tensors."""
    rng = np.random.default_rng(12)
    lcs = [_make_lc(200, rng) for _ in range(4)]
    strategies = ["beginning", "end"]

    inputs = preprocess_many(lcs, strategies=strategies)
    assert inputs.norm_mag.shape == (4, 2, SEQUENCE_LENGTH, 1)
    assert inputs.n_strategies == 2


# ---------------------------------------------------------------------------
# Infer.predict
# ---------------------------------------------------------------------------


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
    inputs = preprocess_lc(*_make_lc(400, np.random.default_rng(0)), strategies=strategies)
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

    inputs = preprocess_many(lcs, strategies=strategies, rng=42)
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


def _make_table(lcs):
    """Build a pa.Table (struct-of-lists) from a list of (time, mag, magerr, band) tuples."""
    import pyarrow as pa

    times, mags, magerrs, bands = zip(*lcs, strict=True)
    return pa.table(
        {
            "time": pa.array([list(t) for t in times]),
            "mag": pa.array([list(m) for m in mags]),
            "magerr": pa.array([list(me) for me in magerrs]),
            "band": pa.array([list(b) for b in bands]),
        }
    )


# ---------------------------------------------------------------------------
# Arrow tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_arrow", [_make_list_struct, _make_table], ids=["list_struct", "table"])
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
