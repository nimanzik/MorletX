# MorletX

***A lightweight, high-performance Python library for continuous wavelet transform (CWT) using Morlet wavelet filter bank, with GPU computing support.***

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**❗ Note this project is still in early development.**

## Overview

This Python library provides an implementation of the Morlet wavelet transform, a powerful time-frequency analysis method that offers an intuitive approach to understanding signal characteristics. The implementation is inspired by the pioneering work of [French geophysicist Jean Morlet](https://en.wikipedia.org/wiki/Jean_Morlet), leveraging his original, highly intuitive formulation which laid the foundation for the Continuous Wavelet Transform (CWT).

Unlike more abstract modern formulations of the CWT, Jean Morlet's method is deeply rooted in physical intuition, making it particularly accessible for comprehending how signals vary in frequency over time.

## Features

- **Intuitive Implementation**: Directly implements Morlet's original and physically intuitive formulation of the wavelet transform, as detailed in his 1982 papers.
- **High Performance**: Supports computations on both the CPU and GPU, enabling efficient processing of large datasets and high-frequency signals.
- **Pythonic Design**: Developed as a user-friendly Python library, making it accessible for researchers and developers.

## Installation

If you use `uv` as your Python package manager, you can install MorletX using the following command:

```bash
# Install as a Git dependency source using `uv` (recommended)
$ uv add git+https://github.com/nimanzik/MorletX
```

If you prefer to use `pip`, you need to install it from source:

```bash
# Install from source using `pip`
$ git clone https://github.com/nimanzik/MorletX.git
$ cd MorletX
$ pip install .
```

## Usage

To compute the scalogram (the output of the CWT):

```python
import numpy as np

from morletx.core import MorletFilterBank

>>> data = ... # some signal data
>>> fs = ...   # sampling frequency of the signal

>>> filter_bank = MorletFilterBank(
...     n_octaves=8,          # Number of octaves to cover
...     n_intervals=4,        # Number of intervals (filters) per octave
...     shape_ratio=5,        # Shape ratio of the Morlet wavelet
...     duration=2.0,         # Duration of the Morlet wavelet
...     sampling_freq=fs,     # Sampling frequency of the signal
...     array_engine="cupy",  # Choices: "numpy" or "cupy"
... )

>>> mode = "magnitude"        # Choices: "magnitude", "power", "complex"

>>> scalogram = filter_bank.transform(data, mode=mode, detach_from_device=True)
```

### Switching between CPU and GPU

Switching between CPU and GPU computation is as simple as changing the `array_engine` parameter to either `"numpy"` or `"cupy"`.

- For CPU computation (and storing the results as NumPy arrays):

    ```python
    >>> filter_bank = MorletFilterBank(..., array_engine="numpy")
    ```

- For GPU computation (and storing the results as CuPy arrays):

    ```python
    >>> filter_bank = MorletFilterBank(..., array_engine="cupy")
    ```

### Use CuPy for computation but get results as NumPy array

When using GPU for computation, the results will be returned as so-called "device arrays". To move the results to the host (CPU) memory, use the `detach_from_device` parameter:

```python
>>> filter_bank = MorletFilterBank(..., array_engine="cupy")
>>> scalogram_as_numpy = filter_bank.transform(..., detach_from_device=True)
```

By default, the `detach_from_device` parameter is set to `False`, meaning the results will be stored as device arrays when using GPU for computation (note that it has no effect on CPU computation).

### Use CuPy for computation and get results as PyTorch Tensor

Both PyTorch and CuPy support `__cuda_array_interface__`, so zero-copy data exchange between CuPy and PyTorch can be achieved at no cost.

The only requirement is that the tensor must be already on GPU before exchanging data. Therefore, make sure that `detach_from_device=False` (which is the default behavior) when doing the transformation.

PyTorch supports zero-copy data exchange through `DLPack`, so you can get the results as a PyTorch tensor as follows:

```python
>>> import torch

>>> filter_bank = MorletFilterBank(..., array_engine="cupy")
>>> scalogram_as_cupy = filter_bank.transform(..., detach_from_device=False)
>>> scalogram_as_torch = torch.from_dlpack(scalogram_as_cupy)
```

### Visualisation

There are quick-and-ready methods to visualise both the filter bank and the computed scalogram.

- To visualise the scalogram:

```python
>>> import matplotlib.pyplot as plt

>>> fig_sgram, ax_sgram = plt.subplots()
>>> filter_bank.plot_scalogram(ax=ax_sgram, scalogram=scalogram, mode=mode)
```

- To visualise the frequency responses of the filter bank:

```python
>>> fig_fbank, ax_fbank = plt.subplots()
>>> filter_bank.plot_responses(ax=ax_fbank, n_fft=512)
```

Here is an example of the computed scalogram for a signal with a sampling frequency of 16 kHz:

<div align="center">
<table>
<tr>
<td align="center">
<strong>Example Signal</strong><br>
<img src="docs/assets/images/01_example_signal.png" alt="Example Signal" width="100%">
</td>
<td align="center">
<strong>Morlet Filter Bank</strong><br>
<img src="docs/assets/images/02_filter_bank.png" alt="Morlet Filter Bank" width="100%">
</td>
<td align="center">
<strong>Computed scalogram</strong><br>
<img src="docs/assets/images/03_scalogram.png" alt="Computed scalogram" width="100%">
</td>
</tr>
</table>
</div>

## Shape Ratio ($\kappa$)

A significant innovation introduced by Morlet is the **shape ratio**, $\kappa$. This parameter defines the Gaussian time width at half-amplitude as an integer multiple of the wavelet's dominant period ($\Delta t = \kappa T_0$). This allows for the preservation of the wavelet's shape as its dominant period changes, providing a consistent analysis across frequencies.

❗ This section will be expanded and detailed in the future.

## Examples

<!-- markdownlint-disable-next-line MD033 -->
<a href="https://nimanzik.github.io/MorletX/assets/htmls/basic_example.html" target="_blank">This example</a> shows how to use `MorletX` to compute the wavelet transform of an acoustic Fin-Whale signal. The [marimo notebook](examples/marimo_notebooks/basic_example.py) for this example is also available for interactive exploration.

## Troubleshooting

Report any issues or bugs on [GitHub Issues](https://github.com/nimanzik/MorletX/issues).

## Contact

- Nima Nooshiri – [nima.nooshiri@gmail.com](mailto:nima.nooshiri@gmail.com)
