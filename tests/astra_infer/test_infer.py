import numpy as np
import pytest

from astra_infer.infer import BANDS, SEQUENCE_LENGTH, AstraInfer, infer, run_onnx


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
