from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .logging_utils import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


def get_array_module(array_engine: Literal["numpy", "cupy"]) -> Any:
    """Get the array module based on the specified engine.

    It returns the `numpy` module if the engine is not recognized or not
    installed.

    Parameters
    ----------
    array_engine : {'numpy', 'cupy'}
        The name of the array module that will be used for computations.

    Returns
    -------
    array_module : np or cp
        Array module.
    """
    if array_engine == "cupy":
        try:
            import cupy as cp

            return cp
        except ImportError:
            logger.warning(
                "CuPy is not installed. Falling back to NumPy as the array module."
            )
    return np


def get_signal_module(array_engine: Literal["numpy", "cupy"]) -> Any:
    """Get the signal module based on the specified engine.

    It returns the `scipy.signal` module if the engine is not recognized or not
    installed.

    Parameters
    ----------
    array_engine : {'numpy', 'cupy'}
        The name of the array module that is used for computations.

    Returns
    -------
    signal_module : scipy.signal or cupyx.scipy.signal
        Signal module.
    """
    from scipy import signal

    if array_engine == "cupy":
        try:
            from cupyx.scipy import signal
        except ImportError:
            logger.warning(
                "CuPy is not installed. Falling back to SciPy as the signal module."
            )

    return signal


def get_centered_array(arr: NDArray, new_shape: tuple[int, ...]) -> NDArray:
    """Return the centered array with the new shape.

    Adapted from: https://github.com/scipy/scipy/blob/main/scipy/signal/_signaltools.py#L411

    Parameters
    ----------
    arr : ndarray
        Input array.
    new_shape : tuple of int
        Desired shape of the output array.

    Returns
    -------
    centered_arr : ndarray
        Centered array with the new shape.
    """
    output_shape = np.asarray(new_shape)
    current_shape = np.array(arr.shape)
    start_idx = (current_shape - output_shape) // 2
    end_idx = start_idx + output_shape
    slice_idxs = [slice(start_idx[k], end_idx[k]) for k in range(len(end_idx))]
    return arr[tuple(slice_idxs)]
