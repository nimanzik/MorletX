from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


def get_centered_array(arr: torch.Tensor, new_shape: tuple[int, ...]) -> torch.Tensor:
    """Return the center newshape portion of a tensor.

    Adapted from: https://github.com/scipy/scipy/blob/main/scipy/signal/_signaltools.py#L411

    Parameters
    ----------
    arr : torch.Tensor
        Input tensor.
    new_shape : tuple of int
        Desired shape of the output tensor.

    Returns
    -------
    centered_arr : torch.Tensor
        Centered tensor with the new shape.
    """
    new_shape_arr = torch.tensor(new_shape)
    current_shape = torch.tensor(arr.shape)
    start_idx = (current_shape - new_shape_arr) // 2
    end_idx = start_idx + new_shape_arr
    slice_idxs = [slice(int(start_idx[k]), int(end_idx[k])) for k in range(len(end_idx))]
    return arr[tuple(slice_idxs)]


def tukey_window(window_length: int, alpha: float = 0.5, device: torch.device | str | None = None) -> torch.Tensor:
    """Return a Tukey window (tapered cosine window).

    Parameters
    ----------
    window_length : int
        Number of points in the output window.
    alpha : float, default=0.5
        Shape parameter of the Tukey window, representing the fraction of the
        window inside the cosine tapered region.
        If zero, the Tukey window is equivalent to a rectangular window.
        If one, the Tukey window is equivalent to a Hann window.
    device : torch.device or str or None, default=None
        Device on which to create the window.

    Returns
    -------
    w : torch.Tensor
        The window, with the maximum value normalized to 1.
    """
    if alpha <= 0:
        return torch.ones(window_length, device=device)
    elif alpha >= 1.0:
        return torch.hann_window(window_length, periodic=False, device=device)

    # Normal case: 0 < alpha < 1
    n = torch.arange(window_length, device=device, dtype=torch.float64)
    width = int(alpha * (window_length - 1) / 2.0)

    # Create the window
    w = torch.ones(window_length, device=device, dtype=torch.float64)

    # Taper the first width samples
    n1 = n[:width + 1]
    w[:width + 1] = 0.5 * (1 + torch.cos(torch.pi * (-1 + 2.0 * n1 / alpha / (window_length - 1))))

    # Taper the last width samples
    n2 = n[window_length - width - 1:]
    w[window_length - width - 1:] = 0.5 * (1 + torch.cos(torch.pi * (-2.0 / alpha + 1 + 2.0 * n2 / alpha / (window_length - 1))))

    return w
