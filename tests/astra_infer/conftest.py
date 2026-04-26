from pathlib import Path

import pytest


@pytest.fixture
def onnx_file():
    """Path to the minimal test ONNX model (700 → 512 linear)."""
    return Path(__file__).parent / "test_data" / "test_model.onnx"
