from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import torch

from .utils.array_utils_torch import tukey_window
from .utils.fft_utils_torch import _cwt_via_fft

if TYPE_CHECKING:
    from matplotlib.axes import Axes as MplAxes
    from matplotlib.colors import Colormap
    from plotly.graph_objects import Figure as PlotlyFigure

Ln2 = math.log(2.0)
PI = math.pi


class MorletWaveletGroup:
    """Base class for single and multi-scale complex Morlet wavelets (PyTorch version)."""

    def __init__(
        self,
        center_freqs: float | Sequence[float] | torch.Tensor | np.ndarray,
        shape_ratios: float | Sequence[float] | torch.Tensor | np.ndarray,
        duration: float,
        sampling_freq: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
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
        device : torch.device or str or None, default=None
            Device on which to create tensors ('cpu', 'cuda', etc.).
            If None, uses the default device.
        dtype : torch.dtype, default=torch.float64
            Data type for floating point tensors.

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
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        
        # Convert inputs to tensors
        if isinstance(center_freqs, (int, float)):
            center_freqs = [center_freqs]
        if isinstance(shape_ratios, (int, float)):
            shape_ratios = [shape_ratios]
            
        self._center_freqs = torch.atleast_1d(
            torch.as_tensor(center_freqs, dtype=dtype, device=self.device)
        )
        self._shape_ratios = torch.atleast_1d(
            torch.as_tensor(shape_ratios, dtype=dtype, device=self.device)
        )
        
        self.duration = duration
        self.sampling_freq = sampling_freq
        self._check_center_freqs()
        self._check_shape_ratios()

    def _check_center_freqs(self) -> None:
        """Check the center frequencies of the wavelets."""
        if self._center_freqs.numel() == 0:
            raise ValueError("Center frequencies must not be empty.")

        if self._center_freqs.ndim != 1:
            raise ValueError("Center frequencies must be a 1D array-like object.")

        if not torch.all(self._center_freqs > 0.0):
            raise ValueError("Center frequencies must be positive.")

        if not torch.all(self._center_freqs < self.nyquist_freq):
            raise ValueError("Center frequencies must be less than the Nyquist.")

    def _check_shape_ratios(self) -> None:
        """Check the shape ratios of the wavelets."""
        if not torch.all(self._shape_ratios > 0.0):
            raise ValueError("Shape ratios must be positive.")

        if (
            self._shape_ratios.numel() != 1
            and self._shape_ratios.shape != self._center_freqs.shape
        ):
            raise ValueError(
                "Shape ratios must be either a scalar or a 1D array-like "
                "object with the same length as the center frequencies."
            )

    def __len__(self) -> int:
        return self._center_freqs.numel()

    @property
    def center_freqs(self) -> torch.Tensor:
        """Center frequencies of the wavelets."""
        return self._center_freqs

    @property
    def shape_ratios(self) -> torch.Tensor:
        """Shape ratios of the wavelets."""
        if self._shape_ratios.numel() == 1:
            return self._shape_ratios.expand(len(self))
        return self._shape_ratios

    @property
    def nyquist_freq(self) -> float:
        """Nyquist frequency of the wavelets."""
        return 0.5 * self.sampling_freq

    @property
    def delta_t(self) -> float:
        """Sampling interval of the wavelets."""
        return 1.0 / self.sampling_freq

    @property
    def n_t(self) -> int:
        """Number of time samples of the wavelets."""
        return int(round(self.duration * self.sampling_freq)) + 1

    @property
    def times(self) -> torch.Tensor:
        """Time samples of the wavelets."""
        return torch.arange(self.n_t, dtype=self.dtype, device=self.device) * self.delta_t - 0.5 * self.duration

    @property
    def time_widths(self) -> torch.Tensor:
        """Time widths of the wavelets.

        Returns
        -------
        time_widths : torch.Tensor of shape (n_center_freqs,)
            Time widths of the wavelets. They are in the same units as the
            `duration`.
        """
        return self.shape_ratios / self.center_freqs

    @property
    def freq_widths(self) -> torch.Tensor:
        """Frequency widths (bandwidths) of the wavelets.

        Returns
        -------
        freq_widths : torch.Tensor of shape (n_center_freqs,)
            Frequency widths of the wavelets. They are in the same units as the
            `sampling_freq`.
        """
        return (4.0 * Ln2) / (PI * self.time_widths)

    @property
    def omega0s(self) -> torch.Tensor:
        """Angular frequencies of the wavelets (Scipy's `omega0`)."""
        return (self.shape_ratios * PI) / math.sqrt(2.0 * Ln2)

    @property
    def scales(self) -> torch.Tensor:
        """Scales of the wavelets."""
        return (self.omega0s * self.sampling_freq) / (2.0 * PI * self.center_freqs)

    @property
    def waveforms(self) -> torch.Tensor:
        """Return the values of the wavelets in the time domain.

        Returns
        -------
        waveforms : complex torch.Tensor of shape (n_center_freqs, n_times)
            Wavelets in the time domain.
        """
        # Compute Gaussian envelope
        time_ratio = self.times / self.time_widths[:, None]
        gaussian = torch.exp(-4.0 * Ln2 * time_ratio ** 2)
        
        # Compute oscillation
        oscillation = torch.exp(1j * 2.0 * PI * self.center_freqs[:, None] * self.times)
        
        return gaussian * oscillation

    @property
    def spectral_max_amps(self) -> torch.Tensor:
        """Maximum amplitudes of the Fourier spectra of the wavelets."""
        return 0.5 * math.sqrt(PI / Ln2) * self.time_widths

    def transform(
        self,
        data: torch.Tensor | np.ndarray,
        demean: bool = True,
        tukey_alpha: float | None = 0.1,
        mode: Literal["power", "magnitude", "complex"] = "power",
    ) -> torch.Tensor:
        """Compute the wavelet transform of the input signal(s).

        Parameters
        ----------
        data : torch.Tensor or ndarray of shape (..., n_times)
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

        Returns
        -------
        coeffs : torch.Tensor of shape (..., n_center_freqs, n_times)
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

        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(device=self.device, dtype=self.dtype)
        else:
            data = data.to(device=self.device, dtype=self.dtype)

        if demean:
            data = data - data.mean(dim=-1, keepdim=True)

        if tukey_alpha is not None:
            window = tukey_window(data.shape[-1], tukey_alpha, device=self.device)
            data = data * window

        wt_coeffs = _cwt_via_fft(data, self.waveforms, hermitian=True)
        wt_coeffs = wt_coeffs / torch.sqrt(self.scales[:, None])  # Normalize by the scales

        if mode == "power":
            wt_coeffs = torch.square(torch.abs(wt_coeffs))
        elif mode == "magnitude":
            wt_coeffs = torch.abs(wt_coeffs)
        # If 'mode=complex', do nothing. The coefficients are already complex.

        return wt_coeffs

    def magnitude_responses(
        self,
        normalize: bool = True,
        n_fft: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the frequency responses of the wavelets.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to return the normalized responses.
        n_fft : int or None, default=None
            Number of FFT points to use for computing the frequency responses.
            If None, the next power of two greater than or equal to `n_t`
            will be used.

        Returns
        -------
        freqs : torch.Tensor of shape (n_freqs,)
            Frequency points.
        resps : torch.Tensor of shape (n_center_freqs, n_freqs)
            Frequency responses of the wavelets.
        """
        if n_fft is None:
            n_fft = int(2 ** math.ceil(math.log2(self.n_t)))

        n_fft = max(n_fft, self.n_t)

        # Compute frequency points
        rfreqs = torch.fft.rfftfreq(n=n_fft, d=self.delta_t, device=self.device, dtype=self.dtype)
        
        # Compute responses
        phase_diffs = 2.0 * PI * (rfreqs - self.center_freqs[:, None])
        resps = torch.exp(
            -1.0 * torch.square(self.time_widths[:, None] * phase_diffs) / (16.0 * Ln2)
        )

        if not normalize:
            resps = resps * self.spectral_max_amps[:, None]

        return rfreqs, resps

    def plot_responses(
        self,
        ax: MplAxes,
        normalize: bool = True,
        n_fft: int | None = None,
        auto_xlabel: bool = True,
        auto_ylabel: bool = True,
        auto_title: bool = True,
    ) -> MplAxes:
        """Plot the frequency responses of the wavelets using Matplotlib.

        Parameters
        ----------
        ax : Axes
            The Matplotlib axes to plot the frequency responses.
        normalize : bool, default=True
            Whether to plot the normalized responses.
        n_fft : int or None, default=None
            Number of FFT points to use for computing the frequency responses.
            If None, the next power of two greater than or equal to `n_t`
            will be used.
        auto_xlabel : bool, default=True
            Whether to automatically set the x-axis label.
        auto_ylabel : bool, default=True
            Whether to automatically set the y-axis label.
        auto_title : bool, default=True
            Whether to automatically set the title.

        Returns
        -------
        ax : Axes
            Matplotlib axes displaying the frequency responses.
        """
        freqs, resps = self.magnitude_responses(normalize=normalize, n_fft=n_fft)
        
        # Convert to numpy for plotting
        freqs_np = freqs.cpu().numpy()
        resps_np = resps.cpu().numpy()

        for resp in resps_np:
            ax.plot(freqs_np, resp)

        if auto_xlabel:
            ax.set_xlabel("Frequency [Hz]")
        if auto_ylabel:
            ax.set_ylabel("Magnitude, normalized" if normalize else "Magnitude")
        if auto_title:
            ax.set_title("Wavelets Frequency Responses")
        return ax

    def plot_responses_plotly(
        self,
        fig: PlotlyFigure,
        normalize: bool = True,
        n_fft: int | None = None,
        auto_xlabel: bool = True,
        auto_ylabel: bool = True,
        auto_title: bool = True,
    ) -> PlotlyFigure:
        """Plot the frequency responses of the wavelets using Plotly.

        Parameters
        ----------
        fig : PlotlyFigure
            The Plotly figure to plot the frequency responses.
        normalize : bool, default=True
            Whether to plot the normalized responses.
        n_fft : int or None, default=None
            Number of FFT points to use for computing the frequency responses.
            If None, the next power of two greater than or equal to `n_t`
            will be used.
        auto_xlabel : bool, default=True
            Whether to automatically set the x-axis label.
        auto_ylabel : bool, default=True
            Whether to automatically set the y-axis label.
        auto_title : bool, default=True
            Whether to automatically set the title.

        Returns
        -------
        fig: PlotlyFigure
            Plotly figure displaying the frequency responses.
        """
        from plotly import graph_objects as go

        freqs, resps = self.magnitude_responses(normalize=normalize, n_fft=n_fft)
        
        # Convert to numpy for plotting
        freqs_np = freqs.cpu().numpy()
        resps_np = resps.cpu().numpy()

        for resp in resps_np:
            fig.add_trace(go.Scatter(x=freqs_np, y=resp, showlegend=False))

        if auto_xlabel:
            fig.update_xaxes(title_text="Frequency [Hz]")
        if auto_ylabel:
            fig.update_yaxes(
                title_text="Magnitude, normalized" if normalize else "Magnitude"
            )
        if auto_title:
            fig.update_layout(title="Wavelets Frequency Responses")
        return fig


class MorletWavelet(MorletWaveletGroup):
    """Single-scale complex Morlet wavelet (PyTorch version)."""

    def __init__(
        self,
        center_freq: float,
        shape_ratio: float,
        duration: float,
        sampling_freq: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
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
        device : torch.device or str or None, default=None
            Device on which to create tensors ('cpu', 'cuda', etc.).
        dtype : torch.dtype, default=torch.float64
            Data type for floating point tensors.

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
            device=device,
            dtype=dtype,
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
    def waveform(self) -> torch.Tensor:
        """Return the values of the wavelet in the time domain.

        Returns
        -------
        waveform : complex torch.Tensor of shape (n_times,)
            Wavelet in the time domain.
        """
        return self.waveforms.squeeze(dim=0)

    @property
    def spectral_max_amp(self) -> float:
        """Maximum amplitude of the Fourier spectrum of the wavelet."""
        return self.spectral_max_amps.item()

    def transform(
        self,
        data: torch.Tensor | np.ndarray,
        demean: bool = True,
        tukey_alpha: float | None = 0.05,
        mode: Literal["power", "magnitude", "complex"] = "power",
    ) -> torch.Tensor:
        """Compute the wavelet transform of the input signal.

        Parameters
        ----------
        data : torch.Tensor or ndarray of shape (..., n_times)
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

        Returns
        -------
        coeffs : torch.Tensor of shape (..., n_times)
            Wavelet-transform coefficients, with the same shape as `data`.
        """
        x_trans = super().transform(data, demean, tukey_alpha, mode)
        axis = x_trans.ndim - 2
        return x_trans.squeeze(dim=axis)

    def magnitude_response(
        self, normalize: bool = True, n_fft: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the frequency response of the wavelet.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to return the normalized response.
        n_fft : int or None, default=None
            Number of FFT points to use for computing the frequency responses.
            If None, the next power of two greater than or equal to `n_t`
            will be used.

        Returns
        -------
        freqs : torch.Tensor of shape (n_freqs,)
            Frequency points.
        resp : torch.Tensor of shape (n_freqs,)
            Frequency response of the wavelet.
        """
        freqs, resps = self.magnitude_responses(normalize, n_fft=n_fft)
        return freqs, resps.squeeze(dim=0)

    def __repr__(self) -> str:
        return (
            f"ComplexMorletWavelet(Fc={self.center_freq}, K={self.shape_ratio},"
            f" Fs={self.sampling_freq:.6f}, T={self.duration}, device={self.device})"
        )


class MorletFilterBank(MorletWaveletGroup):
    """Complex Morlet-wavelet filter bank with constant-Q properties (PyTorch version)."""

    def __init__(
        self,
        n_octaves: int,
        n_intervals: int,
        shape_ratio: float,
        duration: float,
        sampling_freq: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
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
        device : torch.device or str or None, default=None
            Device on which to create tensors ('cpu', 'cuda', etc.).
        dtype : torch.dtype, default=torch.float64
            Data type for floating point tensors.

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
            device=device,
            dtype=dtype,
        )
        self.n_octaves = n_octaves
        self.n_intervals = n_intervals
        self.shape_ratio = shape_ratio

    @property
    def omega0(self) -> float:
        """Angular frequency of the mother wavelet (Scipy's `omega0`)."""
        return (self.shape_ratio * PI) / math.sqrt(2.0 * Ln2)

    @property
    def scales(self) -> torch.Tensor:
        """Scales of the wavelets."""
        return (self.omega0 * self.sampling_freq) / (2.0 * PI * self.center_freqs)

    def __repr__(self) -> str:
        return (
            f"ComplexMorletFilterBank(J={self.n_octaves}, Q={self.n_intervals},"
            f" K={self.shape_ratio}, Fs={self.sampling_freq:.6f}, T={self.duration},"
            f" device={self.device})"
        )

    def plot_responses(
        self,
        ax: MplAxes,
        normalize: bool = True,
        n_fft: int | None = None,
        auto_xlabel: bool = True,
        auto_ylabel: bool = True,
        auto_title: bool = True,
        show_octaves: bool = False,
    ) -> MplAxes:
        """Plot the frequency responses of the wavelets using Matplotlib.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to plot the normalized responses.
        n_fft : int or None, default=None
            Number of FFT points to use for computing the frequency responses.
            If None, the next power of two greater than or equal to `n_t`
            will be used.
        auto_xlabel : bool, default=True
            Whether to automatically set the x-axis label.
        auto_ylabel : bool, default=True
            Whether to automatically set the y-axis label.
        auto_title : bool, default=True
            Whether to automatically set the title.
        show_octaves : bool, default=True
            Whether to add vertical lines at the octave frequencies.

        Returns
        -------
        ax : Axes
            Matplotlib axes displaying the frequency responses.
        """
        ax = super().plot_responses(
            ax=ax,
            normalize=normalize,
            n_fft=n_fft,
            auto_xlabel=auto_xlabel,
            auto_ylabel=auto_ylabel,
            auto_title=auto_title,
        )

        if show_octaves:
            for j in range(self.n_octaves + 1):
                ax.axvline(self.nyquist_freq / 2**j, ls="--", lw=1.0, c="dimgray")

        return ax

    def plot_responses_plotly(
        self,
        fig: PlotlyFigure,
        normalize: bool = True,
        auto_xlabel: bool = True,
        auto_ylabel: bool = True,
        auto_title: bool = True,
        show_octaves: bool = False,
    ) -> PlotlyFigure:
        """Plot the frequency responses of the wavelets using Plotly.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to plot the normalized responses.
        auto_xlabel : bool, default=True
            Whether to automatically set the x-axis label.
        auto_ylabel : bool, default=True
            Whether to automatically set the y-axis label.
        auto_title : bool, default=True
            Whether to automatically set the title.
        show_octaves : bool, default=False
            Whether to add vertical lines at the octave frequencies.

        Returns
        -------
        fig: PlotlyFigure
        """
        fig = super().plot_responses_plotly(
            fig=fig,
            normalize=normalize,
            auto_xlabel=auto_xlabel,
            auto_ylabel=auto_ylabel,
            auto_title=auto_title,
        )

        if show_octaves:
            for j in range(self.n_octaves + 1):
                fig.add_vline(
                    self.nyquist_freq / 2**j,
                    line={"dash": "dash", "width": 1.0, "color": "dimgray"},
                )

        return fig

    def plot_scalogram(
        self,
        ax: MplAxes,
        data: torch.Tensor | np.ndarray | None = None,
        scalogram: torch.Tensor | np.ndarray | None = None,
        demean: bool = True,
        tukey_alpha: float | None = 0.1,
        mode: Literal["power", "magnitude"] = "power",
        log_scale: bool = False,
        cmap: str | Colormap | None = None,
    ) -> MplAxes:
        """Plot the scalogram of the input signal(s).

        Parameters
        ----------
        data : torch.Tensor or ndarray of shape (..., n_times)
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
        ax : MplAxes
            Matplotlib axes displaying the scalogram.
        """
        from ._plotting import plot_tf_plane

        match (data, scalogram):
            case (None, None):
                raise ValueError("Either `data` or `scalogram` must be provided.")
            case (d, s) if d is not None and s is not None:
                raise ValueError(
                    "Only one of `data` or `scalogram` should be provided."
                )
            case (None, s) if s is not None:
                # Convert to numpy if needed
                if isinstance(s, torch.Tensor):
                    coeffs = s.cpu().numpy()
                else:
                    coeffs = s
                if coeffs.ndim != 2:
                    raise ValueError(
                        f"`scalogram` must be a 2D array, but got an array with "
                        f"shape {coeffs.shape}."
                    )
            case (d, None) if d is not None:
                coeffs_tensor = self.transform(
                    d, demean, tukey_alpha, mode
                )
                coeffs = coeffs_tensor.cpu().numpy()
            case _:  # This should never happen
                raise RuntimeError("Unexpected error in input arguments.")

        return plot_tf_plane(
            ax=ax,
            freqs=self._center_freqs.cpu().numpy(),
            times=np.arange(coeffs.shape[-1]) * self.delta_t,
            xgram=coeffs,
            label=mode,
            log_scale=log_scale,
            cmap=cmap,
        )


def compute_morlet_center_freqs(
    n_octaves: int, n_intervals: int, shape_ratio: float, sampling_freq: float
) -> np.ndarray:
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
