import numpy as np
import pytest

from astra_infer.infer import (
    BANDS,
    SEQUENCE_LENGTH,
    AstraInfer,
    infer,
    preprocess_lc,
    preprocess_many,
    run_onnx,
)


def test_run_onnx(onnx_file):
    """run_onnx returns embeddings of the correct shape and all finite."""
    n_samples = 10
    shape = (n_samples, SEQUENCE_LENGTH, 1)
    norm_mag = np.zeros(shape)
    time = np.zeros(shape)
    band_info = np.zeros(shape)
    mask = np.zeros(shape[:-1])

    embeddings = run_onnx(onnx_file, norm_mag, time, band_info, mask)

    assert embeddings.shape == (n_samples, 512)
    assert np.all(np.isfinite(embeddings))


@pytest.mark.parametrize("n", [32, 4098])
def test_infer(onnx_file, n):
    """infer() handles both short (padded) and long (clipped) light curves."""
    rng = np.random.default_rng(42)

    time = rng.uniform(58_000, 59_000, n)
    mag = rng.normal(loc=18, scale=1.0, size=n)
    magerr = np.full(n, 0.1)
    band = rng.choice(BANDS, size=n)

    embeddings = infer(onnx_file, time, mag, magerr, band)

    assert embeddings.shape == (1, 512)
    assert np.all(np.isfinite(embeddings))


@pytest.mark.parametrize("n", [32, 4098])
def test_astra_infer_class(onnx_file, n):
    """AstraInfer reuses the session and produces consistent output."""
    rng = np.random.default_rng(0)

    time = rng.uniform(58_000, 59_000, n)
    mag = rng.normal(loc=18, scale=1.0, size=n)
    magerr = np.full(n, 0.1)
    band = rng.choice(BANDS, size=n)

    model = AstraInfer(onnx_file)
    embeddings = model(time, mag, magerr, band)

    assert embeddings.shape == (1, 512)
    assert np.all(np.isfinite(embeddings))
    # Calling twice with the same input gives identical results
    np.testing.assert_array_equal(embeddings, model(time, mag, magerr, band))


@pytest.mark.parametrize("n_curves,batch_size", [(1, 128), (10, 3), (10, 128)])
def test_predict_batch(onnx_file, n_curves, batch_size):
    """predict_batch returns (N, 512) and matches repeated single-curve calls."""
    rng = np.random.default_rng(1)
    n = 150

    lcs = [
        (rng.uniform(58_000, 59_000, n), rng.normal(18.0, 1.0, n), np.full(n, 0.1), rng.choice(BANDS, size=n))
        for _ in range(n_curves)
    ]

    model = AstraInfer(onnx_file)
    batch_embeddings = model.predict_batch(*preprocess_many(lcs), batch_size=batch_size)

    assert batch_embeddings.shape == (n_curves, 512)
    assert np.all(np.isfinite(batch_embeddings))

    # Must match individual calls
    for i, lc in enumerate(lcs):
        single = model(*lc)
        np.testing.assert_array_equal(batch_embeddings[i], single[0])


def test_presorted_matches_unsorted(onnx_file):
    """presorted=True gives the same result as the default sort path."""
    rng = np.random.default_rng(7)
    n = 200

    time = rng.uniform(58_000, 59_000, n)
    mag = rng.normal(loc=18, scale=1.0, size=n)
    magerr = np.full(n, 0.1)
    band = rng.choice(BANDS, size=n)

    # Sort inputs before calling with presorted=True
    idx = np.argsort(time)
    time_s, mag_s, magerr_s, band_s = time[idx], mag[idx], magerr[idx], band[idx]

    model = AstraInfer(onnx_file)
    result_auto = model(time, mag, magerr, band)
    result_pre = model(time_s, mag_s, magerr_s, band_s, presorted=True)

    np.testing.assert_array_equal(result_auto, result_pre)


def test_preprocess_many_matches_preprocess_lc(onnx_file):
    """preprocess_many stacks individual preprocess_lc results correctly."""
    rng = np.random.default_rng(3)
    lcs = [
        (rng.uniform(58_000, 59_000, n), rng.normal(18.0, 1.0, n), np.full(n, 0.1), rng.choice(BANDS, size=n))
        for n in [32, 700, 4096]
    ]

    stacked = preprocess_many(lcs)
    assert stacked[0].shape == (3, 700, 1)
    assert stacked[3].shape == (3, 700)

    for i, lc in enumerate(lcs):
        single = preprocess_lc(*lc)
        for s_arr, m_arr in zip(single, stacked, strict=True):
            np.testing.assert_array_equal(s_arr, m_arr[[i]])


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
        [pa.concat_arrays(all_time), pa.concat_arrays(all_mag),
         pa.concat_arrays(all_magerr), pa.concat_arrays(all_band)],
        names=["time", "mag", "magerr", "band"],
    )
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), flat_struct)


def _make_table(lcs):
    """Build a pa.Table (struct-of-lists) from a list of (time, mag, magerr, band) tuples."""
    import pyarrow as pa

    times, mags, magerrs, bands = zip(*lcs, strict=True)
    return pa.table({
        "time": pa.array([list(t) for t in times]),
        "mag": pa.array([list(m) for m in mags]),
        "magerr": pa.array([list(me) for me in magerrs]),
        "band": pa.array([list(b) for b in bands]),
    })


# ---------------------------------------------------------------------------
# Arrow tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make_arrow", [_make_list_struct, _make_table],
                         ids=["list_struct", "table"])
def test_preprocess_many_arrow_matches_sequence(make_arrow):
    """preprocess_many with Arrow input matches sequence-of-tuples result."""
    rng = np.random.default_rng(5)
    lcs = [
        (rng.uniform(58_000, 59_000, n), rng.normal(18.0, 1.0, n), np.full(n, 0.1), rng.choice(BANDS, size=n))
        for n in [32, 700, 4096]
    ]

    result_seq = preprocess_many(lcs)
    result_arrow = preprocess_many(make_arrow(lcs))

    for a, b in zip(result_seq, result_arrow, strict=True):
        np.testing.assert_array_equal(a, b)


def test_preprocess_many_arrow_chunked():
    """preprocess_many accepts a ChunkedArray of list-struct type."""
    import pyarrow as pa

    rng = np.random.default_rng(9)
    lcs = [
        (rng.uniform(58_000, 59_000, n), rng.normal(18.0, 1.0, n), np.full(n, 0.1), rng.choice(BANDS, size=n))
        for n in [50, 200]
    ]

    list_struct = _make_list_struct(lcs)
    chunked = pa.chunked_array([list_struct])

    result_seq = preprocess_many(lcs)
    result_arrow = preprocess_many(chunked)

    for a, b in zip(result_seq, result_arrow, strict=True):
        np.testing.assert_array_equal(a, b)


def test_preprocess_many_arrow_custom_field_names():
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

    for a, b in zip(result_seq, result_arrow, strict=True):
        np.testing.assert_array_equal(a, b)
