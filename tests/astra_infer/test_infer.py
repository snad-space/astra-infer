import numpy as np
import pytest
from astra_infer.infer import BANDS, SEQUENCE_LENGTH, infer, run_onnx


def test_run_onnx(onnx_file):
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
    rng = np.random.default_rng(42)

    time = rng.uniform(58_000, 59_000, n)
    mag = rng.normal(loc=18, scale=1.0, size=n)
    magerr = np.full(n, 0.1)
    band = rng.choice(BANDS, size=n)

    embeddings = infer(onnx_file, time, mag, magerr, band)

    assert embeddings.shape == (1, 512)
    assert np.all(np.isfinite(embeddings))
