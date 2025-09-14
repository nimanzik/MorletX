from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
from scipy.signal.windows import tukey

from .utils.array_utils import get_array_module
from .utils.fft_utils import _cwt_via_fft

if TYPE_CHECKING:
    from matplotlib.axes import Axes as MplAxes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure as MplFigure
    from matplotlib.figure import SubFigure as MplSubFigure
    from numpy.typing import ArrayLike, NDArray
    from plotly.graph_objs import Figure as PlotlyFigure

Ln2 = math.log(2.0)
PI = math.pi


class MorletWaveletGroup:
    """Base class for single and multi-scale complex Morlet-wavelets."""

    def __init__(
        self,
        center_freqs: float | Sequence[float] | NDArray[np.float64],
        shape_ratios: float | Sequence[float] | NDArray[np.float64],
        duration: float,
        sampling_freq: float,
        array_engine: Literal["numpy", "cupy"] = "numpy",
    ) -> None:
        """Initialize the complex Morlet-wavelet group.

        Parameters
        ----------
        center_freqs : float or array-like of float
            Center frequencies of the wavelets.
        shape_ratios : float or array-like of float
            Shape ratios of the wavelets (a.k.a. number of cycles).
        duration : float
            Time duration of the wavelets.
        sampling_freq : float
            Sampling frequency of the wavelets (should be the same as the
            signals to be analyzed).
        array_engine : {'numpy', 'cupy'}, default='numpy'
            The array module to use for computations.

        Raises
        ------
        ValueError
            - If the center frequencies are not positive or exceed the Nyquist.
            - If the shape ratios are not positive or have an incompatible
              shape with the center frequencies.

        Notes
        -----
        - The unit of the `duration` and `sampling_freq` must be compatible
          with each other, since this is not checked internally:

          | `duration`   | `sampling_freq` |
          |--------------|-----------------|
          | seconds      | Hz              |
          | milliseconds | kHz             |
          | microseconds | MHz             |
        """
        self._center_freqs_numpy = np.atleast_1d(center_freqs)
        self._shape_ratios_numpy = np.atleast_1d(shape_ratios)
        self.duration = duration
        self.sampling_freq = sampling_freq
        self.array_engine = array_engine
        self._check_center_freqs()
        self._check_shape_ratios()

    def _fetch_array_module(self) -> Any:
        """Return the array module for computations."""
        return get_array_module(self.array_engine)

    def _check_center_freqs(self) -> None:
        """Check the center frequencies of the wavelets."""
        if self._center_freqs_numpy.size == 0:
            raise ValueError("Center frequencies must not be empty.")

        if self._center_freqs_numpy.ndim != 1:
            raise ValueError("Center frequencies must be a 1D array-like object.")

        if not all(self._center_freqs_numpy > 0.0):
            raise ValueError("Center frequencies must be positive.")

        if not all(self._center_freqs_numpy < self.nyquist_freq):
            raise ValueError("Center frequencies must be less than the Nyquist.")

    def _check_shape_ratios(self) -> None:
        """Check the shape ratios of the wavelets."""
        if not all(self._shape_ratios_numpy > 0.0):
            raise ValueError("Shape ratios must be positive.")

        if (
            self._shape_ratios_numpy.size != 1
            and self._shape_ratios_numpy.shape != self._center_freqs_numpy.shape
        ):
            raise ValueError(
                "Shape ratios must be either a scalar or a 1D array-like "
                "object with the same length as the center frequencies."
            )

    def __len__(self) -> int:
        return self._center_freqs_numpy.size

    @property
    def center_freqs(self) -> NDArray:
        """Center frequencies of the wavelets."""
        xp = self._fetch_array_module()
        return xp.asarray(self._center_freqs_numpy)

    @property
    def shape_ratios(self) -> NDArray:
        """Shape ratios of the wavelets."""
        xp = self._fetch_array_module()
        return xp.asarray(self._shape_ratios_numpy)

    @property
    def nyquist_freq(self) -> float:
        """Nyquist frequency."""
        return 0.5 * self.sampling_freq

    @property
    def delta_t(self) -> float:
        """Sampling interval."""
        return 1.0 / self.sampling_freq

    @property
    def n_t(self) -> int:
        """Number of time points."""
        return int(round(self.duration * self.sampling_freq)) + 1

    @property
    def times(self) -> NDArray:
        """Time points."""
        xp = self._fetch_array_module()
        return xp.arange(self.n_t) * self.delta_t - 0.5 * self.duration

    @property
    def time_widths(self) -> NDArray:
        """Time widths of the wavelets.

        Returns
        -------
        time_widths : ndarray of shape (n_center_freqs,)
            Time widths of the wavelets. They are in the same units as the
            `duration`.
        """
        return self.shape_ratios / self.center_freqs

    @property
    def freq_widths(self) -> NDArray:
        """Frequency widths (bandwidths) of the wavelets.

        Returns
        -------
        freq_widths : ndarray of shape (n_center_freqs,)
            Frequency widths of the wavelets. They are in the same units as the
            `sampling_freq`.
        """
        return (4.0 * Ln2) / (PI * self.time_widths)

    @property
    def omega0s(self) -> NDArray:
        """Angular frequencies of the wavelets (Scipy's `omega0`)."""
        return (self.shape_ratios * PI) / math.sqrt(2.0 * Ln2)

    @property
    def scales(self) -> NDArray:
        """Scales of the wavelets."""
        return (self.omega0s * self.sampling_freq) / (2.0 * PI * self.center_freqs)

    @property
    def waveforms(self) -> NDArray:
        """Return the values of the wavelets in the time domain.

        Returns
        -------
        waveforms : complex ndarray of shape (n_center_freqs, n_times)
            Wavelets in the time domain.
        """
        xp = self._fetch_array_module()
        gaussian = xp.exp(-4.0 * Ln2 * (self.times / self.time_widths[:, None]) ** 2)
        oscillation = xp.exp(1j * 2.0 * PI * self.center_freqs[:, None] * self.times)
        return gaussian * oscillation

    @property
    def spectral_max_amps(self) -> NDArray:
        """Maximum amplitudes of the Fourier spectra of the wavelets."""
        return 0.5 * math.sqrt(PI / Ln2) * self.time_widths

    def transform(
        self,
        data: ArrayLike,
        demean: bool = True,
        tukey_alpha: float | None = 0.1,
        mode: Literal["power", "magnitude", "complex"] = "power",
        move_to_host: bool = False,
    ) -> NDArray:
        """Compute the wavelet transform of the input signal(s).

        Parameters
        ----------
        data : ndarray of shape (..., n_times)
            Input signal(s) to be analyzed.
        demean : bool, default=True
            Whether to demean the input signal(s) before computing the
            wavelet transform.
        tukey_alpha : float or None, default=0.1
            Alpha parameter for the Tukey window. If None, no windowing is
            applied.
        mode : {'power', 'magnitude', 'complex'}, default='power'
            Specifies the type of the returned values:
                - `'power'`: squared magnitude of the coefficients.
                - `'magnitude'`: absolute magnitude of the coefficients.
                - `'complex'`: complex-valued coefficients.
        move_to_host : bool, default=False
            Whether to detach the arrays from the device and move them to the
            host memory (applicable only if the array engine is CuPy).

        Returns
        -------
        coeffs : ndarray of shape (..., n_center_freqs, n_times)
            Wavelet-transform coefficients.

        Notes
        -----
        The shape of the output depends on the shape of the input signal(s):
            - `F`: number of center frequencies (wavelets)
            - `B`: batch size
            - `C`: number of channels
            - `L`: number of time points

            | Input shape | Output shape   |
            |-------------|----------------|
            | `(L,)`      | `(F, L)`       |
            | `(B, L)`    | `(B, F, L)`    |
            | `(C, L)`    | `(C, F, L)`    |
            | `(B, C, L)` | `(B, C, F, L)` |
        """
        _check_cwt_mode(mode)

        xp = self._fetch_array_module()
        data = xp.asarray(data)

        if demean:
            data = data - data.mean(axis=-1, keepdims=True)

        if tukey_alpha is not None:
            data = data * xp.asarray(tukey(data.shape[-1], tukey_alpha))

        wt_coeffs = _cwt_via_fft(data, self.waveforms, True, self.array_engine)
        wt_coeffs /= xp.sqrt(self.scales[:, None])  # Normalize by the scales

        mode_operations = {
            "power": lambda inp: xp.square(xp.abs(inp)),
            "magnitude": xp.abs,
            "complex": lambda inp: inp,
        }

        wt_coeffs = mode_operations[mode](wt_coeffs)

        if move_to_host and xp.__name__ == "cupy":
            return xp.asnumpy(wt_coeffs)
        return wt_coeffs

    def magnitude_responses(
        self, normalize: bool = True, move_to_host: bool = False
    ) -> tuple[NDArray, NDArray]:
        """Return the frequency responses of the wavelets.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to return the normalized responses.
        detach_from_device : bool, default=False
            Whether to detach the arrays from the device and move them to the
            host memory (applicable only if the array engine is CuPy).

        Returns
        -------
        freqs : ndarray of shape (n_freqs,)
            Frequency points.
        resps : ndarray of shape (n_center_freqs, n_freqs)
            Frequency responses of the wavelets.
        """
        xp = self._fetch_array_module()
        rfreqs = xp.fft.rfftfreq(n=self.n_t, d=self.delta_t)
        phase_diffs = 2.0 * PI * (rfreqs - self.center_freqs[:, None])
        resps = xp.exp(
            -1.0 * xp.square(self.time_widths[:, None] * phase_diffs) / (16.0 * Ln2)
        )

        if not normalize:
            resps *= self.spectral_max_amps[:, None]

        if move_to_host and xp.__name__ == "cupy":
            return xp.asnumpy(rfreqs), xp.asnumpy(resps)
        return rfreqs, resps

    def plot_responses_plotly(self, normalize: bool = True) -> PlotlyFigure:
        """Plot the frequency responses of the wavelets using Plotly.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to plot the normalized responses.

        Returns
        -------
        fig: PlotlyFigure
            Plotly figure displaying the frequency responses.
        """
        from plotly import graph_objects as go

        freqs, resps = self.magnitude_responses(normalize=normalize, move_to_host=True)

        fig = go.Figure()
        for resp in resps:
            fig.add_trace(go.Scatter(x=freqs, y=resp, showlegend=False))

        fig.update_xaxes(title_text="Frequency [Hz]")
        fig.update_yaxes(
            title_text="Magnitude, normalized" if normalize else "Magnitude"
        )

        return fig

    def plot_responses_mpl(self, normalize: bool = True) -> MplFigure:
        """Plot the frequency responses of the wavelets using Matplotlib.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to plot the normalized responses.

        Returns
        -------
        fig : MplFigure
            Matplotlib figure displaying the frequency responses.
        """
        import matplotlib.pyplot as plt

        freqs, resps = self.magnitude_responses(normalize=normalize, move_to_host=True)

        fig, ax = plt.subplots()
        for resp in resps:
            ax.plot(freqs, resp)

        ax.set(
            xlabel="Frequency [Hz]",
            ylabel="Magnitude, normalized" if normalize else "Magnitude",
        )
        fig.set_layout_engine("tight")
        return fig

    def plot_scalogram_mpl(
        self,
        data: ArrayLike,
        demean: bool = True,
        tukey_alpha: float | None = 0.1,
        mode: Literal["power", "magnitude"] = "power",
        log_scale: bool = False,
        cmap: str | Colormap | None = None,
        ax: MplAxes | None = None,
    ) -> MplFigure | MplSubFigure | None:
        """Plot the scalogram of the input signal(s).

        Parameters
        ----------
        data : ndarray of shape (..., n_times)
            Input signal(s) to be analyzed.
        demean : bool, default=True
            Whether to demean the input signal(s) before computing the wavelet
            transform.
        tukey_alpha : float or None, default=0.1
            Alpha parameter for the Tukey window. If None, no windowing is
            applied.
        mode : {'power', 'magnitude', 'complex'}, default='power'
            Specifies the type of the returned values:
                - `'power'`: squared magnitude of the coefficients.
                - `'magnitude'`: absolute magnitude of the coefficients.
                - `'complex'`: complex-valued coefficients.
        log_scale : bool, default=False
            Whether to plot the scalogram in decibel (dB) scale.
        cmap : str or Colormap or None, default=None
            The colormap to use for the scalogram.
        ax : Axes or None, default=None
            The Matplotlib axes to plot the scalogram. If None, a new figure
            will be created.

        Returns
        -------
        fig : MplFigure
            Matplotlib figure displaying the scalogram.
        """
        from .plotting import plot_tf_plane_mpl

        coeffs = self.transform(data, demean, tukey_alpha, mode, move_to_host=True)
        return plot_tf_plane_mpl(
            freqs=self._center_freqs_numpy,
            times=np.arange(coeffs.shape[-1]) * self.delta_t,
            xgram=coeffs,
            label=mode,
            log_scale=log_scale,
            cmap=cmap,
            ax=ax,
        )


class MorletWavelet(MorletWaveletGroup):
    """Single-scale complex Morlet wavelet."""

    def __init__(
        self,
        center_freq: float,
        shape_ratio: float,
        duration: float,
        sampling_freq: float,
    ) -> None:
        """Initialize the complex Morlet wavelet.

        Parameters
        ----------
        center_freq : float
            Center frequency of the wavelet.
        shape_ratio : float
            Shape ratio of the wavelet (a.k.a. number of cycles).
        duration : float
            Time duration of the wavelet.
        sampling_freq : float
            Sampling frequency of the wavelet (should be the same as the
            signals to be analyzed).

        Raises
        ------
        ValueError
            - If the center frequency is not positive or exceeds the Nyquist.
            - If the shape ratio is not positive.

        Notes
        -----
        - The unit of the `duration` and `sampling_freq` must be compatible
          with each other, since this is not checked internally:

          | `duration`   | `sampling_freq` |
          |--------------|-----------------|
          | seconds      | Hz              |
          | milliseconds | kHz             |
          | microseconds | MHz             |
        """
        super().__init__(
            center_freqs=[center_freq],
            shape_ratios=[shape_ratio],
            duration=duration,
            sampling_freq=sampling_freq,
            array_engine="numpy",
        )
        self.center_freq = center_freq
        self.shape_ratio = shape_ratio

    @property
    def time_width(self) -> float:
        """Time width of the wavelet.

        Returns
        -------
        time_width : float
            Time width of the wavelet. It is in the same units as the
            `duration`.
        """
        return self.time_widths.item()

    @property
    def freq_width(self) -> float:
        """Frequency width (bandwidth) of the wavelet.

        Returns
        -------
        freq_width : float
            Frequency width of the wavelet. It is in the same units as the
            `sampling_freq`.
        """
        return self.freq_widths.item()

    @property
    def waveform(self) -> NDArray:
        """Return the values of the wavelet in the time domain.

        Returns
        -------
        waveform : complex ndarray of shape (n_times,)
            Wavelet in the time domain.
        """
        return self.waveforms.squeeze(axis=0)

    @property
    def spectral_max_amp(self) -> float:
        """Maximum amplitude of the Fourier spectrum of the wavelet."""
        return self.spectral_max_amps.item()

    def transform(
        self,
        data: ArrayLike,
        demean: bool = True,
        tukey_alpha: float | None = 0.05,
        mode: Literal["power", "magnitude", "complex"] = "power",
        move_to_host: bool = False,
    ) -> NDArray:
        """Compute the wavelet transform of the input signal.

        Parameters
        ----------
        data : ndarray of shape (..., n_times)
            Input signal(s) to be analyzed.
        demean : bool, default=True
            Whether to demean the input signal(s) before computing the wavelet
            transform.
        tukey_alpha : float or None, default=0.05
            Alpha parameter for the Tukey window. If None, no windowing is
            applied.
        mode : {'power', 'magnitude', 'complex'}, default='power'
            Specifies the type of the returned values:
                - `'power'`: squared magnitude of the coefficients.
                - `'magnitude'`: magnitude of the coefficients.
                - `'complex'`: complex-valued coefficients.
        move_to_host : bool, default=False
            Whether to detach the arrays from the device and move them to the
            host memory (applicable only if the array engine is CuPy).

        Returns
        -------
        coeffs : ndarray of shape (..., n_times)
            Wavelet-transform coefficients, with the same shape as `data`.
        """
        x_trans = super().transform(data, demean, tukey_alpha, mode, move_to_host)
        axis = x_trans.ndim - 2
        return x_trans.squeeze(axis=axis)

    def magnitude_response(self, normalize: bool = True) -> tuple[NDArray, NDArray]:
        """Return the frequency response of the wavelet.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to return the normalized response.

        Returns
        -------
        freqs : ndarray of shape (n_freqs,)
            Frequency points.
        resp : ndarray of shape (n_freqs,)
            Frequency response of the wavelet.
        """
        freqs, resps = self.magnitude_responses(normalize)
        return freqs, resps.squeeze(axis=0)

    def __repr__(self) -> str:
        return (
            f"ComplexMorletWavelet(Fc={self.center_freq}, K={self.shape_ratio},"
            f" Fs={self.sampling_freq:.6f}, T={self.duration})"
        )


class MorletFilterBank(MorletWaveletGroup):
    """Complex Morlet-wavelet filter bank with constant-Q properties."""

    def __init__(
        self,
        n_octaves: int,
        n_intervals: int,
        shape_ratio: float,
        duration: float,
        sampling_freq: float,
        array_engine: Literal["numpy", "cupy"] = "numpy",
    ) -> None:
        """Initialize the complex Morlet-wavelet filter bank.

        Parameters
        ----------
        n_octaves : int
            Number of octaves.
        n_intervals : int
            Number of intervals per octave.
        shape_ratio : float
            Shape ratio of the wavelet (a.k.a. number of cycles).
        duration : float
            Time duration of the wavelets.
        sampling_freq : float
            Sampling frequency of the wavelets (should be the same as the
            signals to be analyzed).
        array_engine : {'numpy', 'cupy'}, default='numpy'
            The array module to use for computations.

        Raises
        ------
        ValueError
            - If the center frequencies are not positive or exceed the Nyquist.
            - If the shape ratios are not positive or have an incompatible
              shape with the center frequencies.

        Notes
        -----
        - The unit of the `duration` and `sampling_freq` must be compatible
          with each other, since this is not checked internally:

          | `duration`   | `sampling_freq` |
          |--------------|-----------------|
          | seconds      | Hz              |
          | milliseconds | kHz             |
          | microseconds | MHz             |
        """
        center_freqs = compute_morlet_center_freqs(
            n_octaves, n_intervals, shape_ratio, sampling_freq
        )
        super().__init__(
            center_freqs=center_freqs,
            shape_ratios=[shape_ratio],
            duration=duration,
            sampling_freq=sampling_freq,
            array_engine=array_engine,
        )
        self.n_octaves = n_octaves
        self.n_intervals = n_intervals
        self.shape_ratio = shape_ratio

    @property
    def omega0(self) -> float:
        """Angular frequency of the mother wavelet (Scipy's `omega0`)."""
        return (self.shape_ratio * PI) / math.sqrt(2.0 * Ln2)

    @property
    def scales(self) -> NDArray:
        """Scales of the wavelets."""
        return (self.omega0 * self.sampling_freq) / (2.0 * PI * self.center_freqs)

    def __repr__(self) -> str:
        return (
            f"ComplexMorletFilterBank(J={self.n_octaves}, Q={self.n_intervals},"
            f" K={self.shape_ratio}, Fs={self.sampling_freq:.6f}, T={self.duration})"
        )

    def plot_responses_plotly(
        self, normalize: bool = True, show_octaves: bool = True
    ) -> PlotlyFigure:
        """Plot the frequency responses of the wavelets using Plotly.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to plot the normalized responses.
        show_octaves : bool, default=True
            Whether to add vertical lines at the octave frequencies.

        Returns
        -------
        fig: PlotlyFigure
        """
        fig = super().plot_responses_plotly(normalize=normalize)

        if show_octaves:
            for j in range(self.n_octaves + 1):
                fig.add_vline(
                    self.nyquist_freq / 2**j,
                    line={"dash": "dash", "width": 1.5, "color": "dimgray"},
                )

        return fig

    def plot_responses_mpl(
        self, normalize: bool = True, show_octaves: bool = True
    ) -> MplFigure:
        """Plot the frequency responses of the wavelets using Matplotlib.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to plot the normalized responses.
        show_octaves : bool, default=True
            Whether to add vertical lines at the octave frequencies.
        """
        fig = super().plot_responses_mpl(normalize=normalize)

        if show_octaves:
            ax = fig.get_axes()[0]
            for j in range(self.n_octaves + 1):
                ax.axvline(self.nyquist_freq / 2**j, ls="--", lw=1.5, c="dimgray")

        return fig


def compute_morlet_center_freqs(
    n_octaves: int, n_intervals: int, shape_ratio: float, sampling_freq: float
) -> NDArray:
    """Compute the center frequencies of a complex Morlet-wavelet filter bank.

    Parameters
    ----------
    n_octaves : int
        Number of octaves.
    n_intervals : int
        Number of intervals per octave.
    shape_ratio : float
        Shape ratio of the wavelet (a.k.a. number of cycles).
    sampling_freq : float
        Sampling frequency of the wavelet.

    Returns
    -------
    center_freqs : ndarray of shape (n_center_freqs,)
        Center frequencies of the wavelets.
    """
    if n_octaves <= 0 or n_intervals <= 0:
        raise ValueError("Number of octaves and intervals must be positive.")

    if shape_ratio <= 0:
        raise ValueError("Shape ratio must be positive.")

    if sampling_freq <= 0:
        raise ValueError("Sampling frequency must be positive.")

    n_cf = n_octaves * n_intervals + 1
    ratios = np.linspace(-(n_octaves + 1), -1, n_cf)
    center_freqs = np.exp2(ratios) * sampling_freq
    freq_widths = (4.0 * Ln2 * center_freqs) / (PI * shape_ratio)
    mask = (center_freqs + 0.5 * freq_widths) < (0.5 * sampling_freq)
    return center_freqs[mask]


def _check_cwt_mode(mode: Literal["power", "magnitude", "complex"]) -> None:
    """Check whether the CWT output mode is valid."""
    if mode not in (valid_modes := {"power", "magnitude", "complex"}):
        raise ValueError(f"Invalid mode: '{mode}', must be one of {valid_modes}.")
