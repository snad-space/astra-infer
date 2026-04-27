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
# preprocess_lc
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [32, 4098])
def test_inputs_from_lc_shape(n):
    """preprocess_lc returns tensors of the right shape for short and long LCs."""
    rng = np.random.default_rng(42)
    time = rng.uniform(58_000, 59_000, n)
    mag = rng.normal(loc=18, scale=1.0, size=n)
    magerr = np.full(n, 0.1)
    band = rng.choice(BANDS, size=n)

    inputs = preprocess_lc(time, mag, magerr, band)

    assert inputs.norm_mag.shape == (1, SEQUENCE_LENGTH, 1)
    assert inputs.norm_time.shape == (1, SEQUENCE_LENGTH, 1)
    assert inputs.lg_wave.shape == (1, SEQUENCE_LENGTH, 1)
    assert inputs.mask.shape == (1, SEQUENCE_LENGTH)


def test_inputs_from_lc_presorted():
    """presorted=True gives the same result as the default sort path."""
    rng = np.random.default_rng(7)
    n = 200

    time = rng.uniform(58_000, 59_000, n)
    mag = rng.normal(loc=18, scale=1.0, size=n)
    magerr = np.full(n, 0.1)
    band = rng.choice(BANDS, size=n)

    idx = np.argsort(time)
    time_s, mag_s, magerr_s, band_s = time[idx], mag[idx], magerr[idx], band[idx]

    result_auto = preprocess_lc(time, mag, magerr, band)
    result_pre = preprocess_lc(time_s, mag_s, magerr_s, band_s, presorted=True)

    np.testing.assert_array_equal(result_auto.norm_mag, result_pre.norm_mag)
    np.testing.assert_array_equal(result_auto.norm_time, result_pre.norm_time)


# ---------------------------------------------------------------------------
# preprocess_many
# ---------------------------------------------------------------------------


def test_inputs_from_lcs_shape():
    """preprocess_many stacks individual preprocess_lc results correctly."""
    rng = np.random.default_rng(3)
    lcs = [
        (rng.uniform(58_000, 59_000, n), rng.normal(18.0, 1.0, n), np.full(n, 0.1), rng.choice(BANDS, size=n))
        for n in [32, 700, 4096]
    ]

    stacked = preprocess_many(lcs)
    assert stacked.norm_mag.shape == (3, SEQUENCE_LENGTH, 1)
    assert stacked.mask.shape == (3, SEQUENCE_LENGTH)

    for i, lc in enumerate(lcs):
        single = preprocess_lc(*lc)
        np.testing.assert_array_equal(single.norm_mag, stacked.norm_mag[[i]])
        np.testing.assert_array_equal(single.mask, stacked.mask[[i]])


# ---------------------------------------------------------------------------
# Infer.predict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [32, 4098])
def test_predict_single(onnx_file, n):
    """predict returns (1, 512) finite embeddings for a single LC."""
    rng = np.random.default_rng(42)
    time = rng.uniform(58_000, 59_000, n)
    mag = rng.normal(loc=18, scale=1.0, size=n)
    magerr = np.full(n, 0.1)
    band = rng.choice(BANDS, size=n)

    embeddings = Infer(onnx_file).predict(preprocess_lc(time, mag, magerr, band))

    assert embeddings.shape == (1, 512)
    assert np.all(np.isfinite(embeddings))


def test_predict_deterministic(onnx_file):
    """Session is reused; calling twice with the same input gives identical results."""
    rng = np.random.default_rng(0)
    n = 150
    time = rng.uniform(58_000, 59_000, n)
    mag = rng.normal(loc=18, scale=1.0, size=n)
    magerr = np.full(n, 0.1)
    band = rng.choice(BANDS, size=n)

    model = Infer(onnx_file)
    inputs = preprocess_lc(time, mag, magerr, band)

    np.testing.assert_array_equal(model.predict(inputs), model.predict(inputs))


@pytest.mark.parametrize("n_curves,batch_size", [(1, 128), (10, 3), (10, 128), (10, None)])
def test_predict_batch(onnx_file, n_curves, batch_size):
    """predict returns (N, 512) and matches repeated single-curve calls."""
    rng = np.random.default_rng(1)
    n = 150

    lcs = [
        (rng.uniform(58_000, 59_000, n), rng.normal(18.0, 1.0, n), np.full(n, 0.1), rng.choice(BANDS, size=n))
        for _ in range(n_curves)
    ]

    model = Infer(onnx_file)
    batch_embeddings = model.predict(preprocess_many(lcs), batch_size=batch_size)

    assert batch_embeddings.shape == (n_curves, 512)
    assert np.all(np.isfinite(batch_embeddings))

    for i, lc in enumerate(lcs):
        single = model.predict(preprocess_lc(*lc))
        np.testing.assert_array_equal(batch_embeddings[i], single[0])


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
    lcs = [
        (rng.uniform(58_000, 59_000, n), rng.normal(18.0, 1.0, n), np.full(n, 0.1), rng.choice(BANDS, size=n))
        for n in [32, 700, 4096]
    ]

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
    lcs = [
        (rng.uniform(58_000, 59_000, n), rng.normal(18.0, 1.0, n), np.full(n, 0.1), rng.choice(BANDS, size=n))
        for n in [50, 200]
    ]

    chunked = pa.chunked_array([_make_list_struct(lcs)])

    result_seq = preprocess_many(lcs)
    result_arrow = preprocess_many(chunked)

    np.testing.assert_array_equal(result_seq.norm_mag, result_arrow.norm_mag)


def test_inputs_from_lcs_arrow_custom_field_names():
    """preprocess_many accepts a custom field_names mapping."""
    import pyarrow as pa

    rng = np.random.default_rng(6)
    n = 100
    time = rng.uniform(58_000, 59_000, n)
    mag = rng.normal(18.0, 1.0, n)
    magerr = np.full(n, 0.1)
    band = rng.choice(BANDS, size=n)

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
