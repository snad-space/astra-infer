from pathlib import Path

import pytest


@pytest.fixture
def onnx_file():
    return Path(__file__).parent / "test_data" / "best_contrastive.onnx"
