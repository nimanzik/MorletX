from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm

if TYPE_CHECKING:
    from matplotlib.axes import Axes as MplAxes
    from matplotlib.colors import Colormap
    from numpy.typing import NDArray


def plot_tf_plane(
    ax: MplAxes,
    freqs: NDArray,
    times: NDArray,
    xgram: NDArray,
    label: str,
    log_scale: bool = False,
    cmap: str | Colormap | None = None,
    auto_xlabel: bool = True,
    auto_ylabel: bool = True,
    auto_cbar: bool = True,
) -> MplAxes:
    """Plot the time-frequency (TF) plane.

    Parameters
    ----------
    ax : Axes or None, default=None
        The Matplotlib axes to plot the TF plane.
    freqs : ndarray
        The array of sample frequencies.
    times : ndarray
        The array of segment times.
    xgram : ndarray
        The TF representation of the signal.
    label : str
        The mode of the TF representation (e.g., 'psd', 'magnitude'). This is
        used as the label for the colorbar.
    log_scale : bool, default=False
        Whether to plot the TF plane in decibel (dB) scale.
    cmap : str or Colormap or None, default=None
        The colormap to use for the TF plane.

    Returns
    -------
    ax : MplAxes
        Matplotlib axes displaying the TF plane.
    """
    if xgram.ndim != 2:
        raise ValueError(
            f"Time-frequency representation input must be 2D, but got {xgram.ndim}D."
        )

    if freqs.shape + times.shape != xgram.shape:
        raise ValueError(
            f"The shapes of `freqs`, `times`, and input time-frequency representation "
            f"do not match: {freqs.shape} + {times.shape} != {xgram.shape}."
        )

    # Downsample the frequency dimension if it is too large
    freqs, xgram = _downsample_fdim(freqs, xgram)

    # Convert to decibel scale if needed
    if log_scale:
        xgram = _to_decibel(xgram)

    # Get the min and max values for the colorbar
    v_min, v_max = _get_vrange(xgram)
    cmap = cmap or cm.lipari

    if xgram.shape[-1] > 1000:
        # Use `imshow` for large data
        im = ax.imshow(
            xgram,
            cmap=cmap,
            aspect="auto",
            origin="lower",
            extent=(times[0], times[-1], 0, len(freqs) - 1),
            vmin=v_min,
            vmax=v_max,
        )
        cbar = plt.colorbar(im, ax=ax)
        y_ticks = np.linspace(0, len(freqs) - 1, num=8, dtype=int)
        y_tick_labels = [f"{freqs[i]:.0f}" for i in y_ticks]
        ax.set_yticks(ticks=y_ticks, labels=y_tick_labels)
    else:
        # Use `pcolormesh` for small data
        pc = ax.pcolormesh(
            times,
            freqs,
            xgram,
            cmap=cmap,
            shading="gouraud",
            vmin=v_min,
            vmax=v_max,
        )
        cbar = plt.colorbar(pc, ax=ax)

    if auto_xlabel:
        ax.set_xlabel("Time")
    if auto_ylabel:
        ax.set_ylabel("Frequency")

    if auto_cbar:
        clabel = label.upper() if label == "psd" else label.capitalize()
        cbar.set_label(f"{clabel} (dB)" if log_scale else clabel)

    return ax


def _downsample_plan(n_original: int, n_max: int) -> tuple[int, int]:
    n_mean = (n_original - 1) // n_max + 1
    n_ds = n_original // n_mean
    return n_ds, n_mean


def _downsample_fdim(freqs: NDArray, xgram: NDArray) -> tuple[NDArray, NDArray]:
    max_num_freqs = 500

    n_freqs, n_times = xgram.shape
    n_freqs_ds, n_freqs_mean = _downsample_plan(n_freqs, max_num_freqs)

    if n_freqs_mean == 1:
        return freqs, xgram

    n_freqs_trim = n_freqs_ds * n_freqs_mean
    return (
        freqs[:n_freqs_trim:n_freqs_mean],
        xgram[:n_freqs_trim].reshape(n_freqs_ds, n_freqs_mean, n_times).mean(axis=1),
    )


def _to_decibel(a: NDArray, tiny: float = 1e-10) -> NDArray:
    return 10 * np.log10(np.fmax(a, tiny))


def _get_vrange(xgram: NDArray) -> tuple[float, float]:
    mu = xgram.mean()
    sigma = xgram.std(ddof=1)
    v_min = max(mu - 3 * sigma, xgram.min())
    v_max = min(mu + 3 * sigma, xgram.max())
    return v_min, v_max
