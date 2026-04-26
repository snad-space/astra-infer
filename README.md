
# astra_infer

Python package for running inference with Astra light-curve embedding models.
Given a multi-band photometric light curve (magnitudes, times, and band labels),
the package pre-processes the data and runs it through an ONNX embedding model,
returning a 512-dimensional embedding vector.

[![PyPI](https://img.shields.io/pypi/v/astra_infer?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/astra_infer/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/snad-space/astra_infer/smoke-test.yml)](https://github.com/snad-space/astra_infer/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/snad-space/astra_infer/branch/main/graph/badge.svg)](https://codecov.io/gh/snad-space/astra_infer)
[![Read The Docs](https://img.shields.io/readthedocs/astra-infer)](https://astra-infer.readthedocs.io/)
[![Benchmarks](https://img.shields.io/github/actions/workflow/status/snad-space/astra_infer/asv-main.yml?label=benchmarks)](https://snad-space.github.io/astra_infer/)

## Overview

The model expects ZTF-style light curves in three bands: **g**, **r**, and **i**.
Each call pre-processes the raw observations into fixed-length band sequences
(300 g + 350 r + 50 i = 700 total), then runs the ONNX model to produce a
512-dimensional embedding.

Pre-processing steps:
1. Inverse-variance weighted mean subtraction of magnitudes.
2. Time normalisation (offset by MJD 58 000).
3. Chronological sorting (can be skipped if input is already sorted).
4. Per-band clipping / zero-padding to the target sequence length.

## Installation

```bash
pip install astra_infer
```

## Quick start

```python
import numpy as np
from astra_infer import AstraInfer

# Load the model once — the ONNX session is kept alive for repeated calls
model = AstraInfer("path/to/model.onnx")

# Run inference on a light curve
embedding = model(time, mag, magerr, band)   # → ndarray shape (1, 512)
```

If your observations are already sorted by time you can skip the internal sort:

```python
embedding = model(time, mag, magerr, band, presorted=True)
```

A convenience function is also available for one-off calls (creates a new session
each time, so prefer `AstraInfer` for batch usage):

```python
from astra_infer import infer

embedding = infer("path/to/model.onnx", time, mag, magerr, band)
```

## Input format

| Array    | dtype   | Shape  | Description                                 |
|----------|---------|--------|---------------------------------------------|
| `time`   | float64 | (n,)   | Observation times in MJD                    |
| `mag`    | float64 | (n,)   | PSF magnitudes                              |
| `magerr` | float64 | (n,)   | 1-σ magnitude uncertainties                 |
| `band`   | str     | (n,)   | Band labels — each element in `{"g","r","i"}` |

## Development

```bash
pip install -e ".[dev]"
pytest
```
