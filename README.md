# astra_infer

Python package for running inference with **ASTRA** light-curve embedding models.
Given a multi-band photometric light curve (magnitudes, magnitude errors, times, and band labels), the package pre-processes the data and runs it through an ONNX embedding model, returning a 512-dimensional embedding vector.

[![PyPI](https://img.shields.io/pypi/v/astra_infer?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/astra_infer/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/snad-space/astra_infer/smoke-test.yml)](https://github.com/snad-space/astra_infer/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/snad-space/astra_infer/branch/main/graph/badge.svg)](https://codecov.io/gh/snad-space/astra_infer)
[![Read The Docs](https://img.shields.io/readthedocs/astra-infer)](https://astra-infer.readthedocs.io/)
[![Benchmarks](https://img.shields.io/github/actions/workflow/status/snad-space/astra_infer/asv-main.yml?label=benchmarks)](https://snad-space.github.io/astra_infer/)

---

## 🌐 The ASTRA Ecosystem
The ASTRA project is split across four distinct repositories:

| Repository | Description | Link |
| :--- | :--- | :--- |
| **Development Code** | Core framework, data pipeline, and model training. | [GitHub: astra](https://github.com/TorshaMajumder/astra) |
| **Inference Code** | Lightweight, production-ready inference scripts and utilities (This Repo). | *[Current Repository]* |
| **Astronomy Dataset** | Light curve datasets formatted for ASTRA. | [Hugging Face: Dataset](https://huggingface.co/datasets/snad-space/astra-zubercaldr16_gaiadr3vclassre) |
| **Model Weights & ONNX** | Pre-trained & finetuned model checkpoints along with ONNX exports. | [Hugging Face: Models](https://huggingface.co/ashrot/astra-clr-base) |

---

## Overview

The model is designed for photometry from the Zwicky Transient Facility (ZTF; [Bellm et al., 2019](https://ui.adsabs.harvard.edu/abs/2019PASP..131a8002B/abstract)) [Zubercal DR16](http://atua.caltech.edu/ZTF/Zubercal.html) catalog in three bands:
**g**, **r**, and **i**. Each call pre-processes the raw observations into fixed-length band sequences (300 g + 350 r + 50 i = 700 total), then runs the ONNX model to produce a 512-dimensional embedding.

Pre-processing steps:
1. Inverse-variance weighted mean subtraction of magnitudes.
2. Time normalisation (offset by MJD 58 000).
3. Chronological sorting (can be skipped if input is already sorted).
4. Per-band clipping / zero-padding to the target sequence length.

---

## Installation

```bash
pip install astra_infer
```

`astra_infer` does **not** install an ONNX runtime automatically, because the right package depends on your hardware. Install one separately before calling `Infer`:

| Hardware | Package |
|----------|---------|
| CPU | `pip install onnxruntime` |
| NVIDIA GPU (CUDA) | `pip install onnxruntime-gpu` |
| Apple Silicon / macOS | `pip install onnxruntime-silicon` |
| Windows DirectML | `pip install onnxruntime-directml` |

See [onnxruntime.ai](https://onnxruntime.ai) for the full list of packages and installation options.

Alternatively, the `onnx` extra installs the CPU variant for you:

```bash
pip install "astra_infer[onnx]"
```

---

## Quick start

To load your model, you need the pre-trained `.onnx` file. You can download it manually from our [Hugging Face Models Repository](https://huggingface.co/ashrot/astra-clr-base) 

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

---

## Subsampling strategies

Apply one or more subsampling strategies to select a window of observations per band before padding. `predict` always returns a 3-D array ``(N, S, 512)`` — one embedding per light curve per strategy.

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

| Strategy | Description |
|----------|-------------|
| `"beginning"` (default) | Per-band: chronologically first observations |
| `"end"` | Per-band: chronologically last observations |
| `"middle"` | Global cut at the midpoint of all observations; per-band window of *N* centred around it |
| `"window"` | Global cut drawn uniformly from a valid range; per-band window of *N* centred around it |
| `"sample"` | Per-band random subsample without replacement |

---

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

---

## Input format

| Array    | Shape  | Description                                                    |
|----------|--------|----------------------------------------------------------------|
| `time`   | (n,)   | Observation times in MJD (any numeric dtype)                   |
| `mag`    | (n,)   | PSF magnitudes (any numeric dtype)                             |
| `magerr` | (n,)   | 1-σ magnitude uncertainties (any numeric dtype)                |
| `band`   | (n,)   | ZTF band labels — each element in `{"g","r","i"}` (see [ZTF](https://www.ztf.caltech.edu)) |

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

