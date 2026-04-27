
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

The model is designed for photometry from the
[Zwicky Transient Facility (ZTF)](https://www.ztf.caltech.edu) in three bands:
**g**, **r**, and **i**.  Each call pre-processes the raw observations into
fixed-length band sequences (300 g + 350 r + 50 i = 700 total), then runs the
ONNX model to produce a 512-dimensional embedding.

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
from astra_infer import Infer, preprocess_lc

# Load the model once — the ONNX session is kept alive for repeated calls
model = Infer("path/to/model.onnx")

# Pre-process, then infer
inputs = preprocess_lc(time, mag, magerr, band)
embedding = model.predict(inputs)   # → ndarray shape (1, 1, 512)
```

If your observations are already sorted by time you can skip the internal sort:

```python
inputs = preprocess_lc(time, mag, magerr, band, presorted=True)
```

## Subsampling strategies

Apply one or more subsampling strategies to select a window of observations
per band before padding.  `predict` always returns a 3-D array
``(N, S, 512)`` — one embedding per light curve per strategy.

```python
# Single strategy (default: "beginning")
inputs = preprocess_lc(time, mag, magerr, band, strategies="end")
embedding = model.predict(inputs)   # → (1, 1, 512)

# Multiple strategies in one call
inputs = preprocess_lc(
    time, mag, magerr, band,
    strategies=["beginning", "end", "middle", "window", "sample"],
    rng=42,          # seed for stochastic strategies
)
embeddings = model.predict(inputs)  # → (1, 5, 512)
```

Available strategies:

For all strategies except `"sample"`, a single global cut time is derived from
the full (all-band) sorted observation array and applied uniformly to every
band, so selected observations overlap in time across bands.

| Strategy | Description |
|----------|-------------|
| `"beginning"` (default) | Cut at the earliest observation |
| `"end"` | Cut at `t_global[max(0, M−K)]` where *K* = total obs actually selected |
| `"middle"` | Cut at `t_global[max(0, M−K) // 2]` — centred window |
| `"window"` | Cut time drawn uniformly from `[t_global[0], t_global[max(0, M−K)]]` |
| `"sample"` | Per-band random subsample without replacement |

## Batch inference

Pre-process multiple light curves at once, then run a single batched ONNX call:

```python
from astra_infer import Infer, preprocess_many

lcs = [(time1, mag1, magerr1, band1), (time2, mag2, magerr2, band2), ...]

inputs = preprocess_many(lcs)
embeddings = model.predict(inputs)              # → ndarray shape (N, 1, 512)
embeddings = model.predict(inputs, batch_size=None)  # single ONNX call

# Combine batch inference with multiple strategies
inputs = preprocess_many(lcs, strategies=["beginning", "end"], rng=0)
embeddings = model.predict(inputs)              # → ndarray shape (N, 2, 512)
```

`preprocess_many` also accepts PyArrow containers (*list-of-struct* or *struct-of-lists*):

```python
# pa.ListArray / pa.ChunkedArray with struct value type, or pa.Table / pa.StructArray
inputs = preprocess_many(arrow_lcs)

# Custom column names
inputs = preprocess_many(
    arrow_lcs,
    field_names={"time": "mjd", "mag": "psf_mag", "magerr": "psf_magerr", "band": "fid"},
)
```

## Input format

| Array    | Shape  | Description                                                    |
|----------|--------|----------------------------------------------------------------|
| `time`   | (n,)   | Observation times in MJD (any numeric dtype)                   |
| `mag`    | (n,)   | PSF magnitudes (any numeric dtype)                             |
| `magerr` | (n,)   | 1-σ magnitude uncertainties (any numeric dtype)                |
| `band`   | (n,)   | ZTF band labels — each element in `{"g","r","i"}` (see [ZTF](https://www.ztf.caltech.edu)) |

## Development

```bash
pip install -e ".[dev]"
pytest
```
