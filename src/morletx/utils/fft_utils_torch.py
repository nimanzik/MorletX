from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .array_utils_torch import get_centered_array

if TYPE_CHECKING:
    pass


def next_fast_len(target: int, real: bool = True) -> int:
    """Find the next fast size for FFT.

    Parameters
    ----------
    target : int
        Target length for the FFT.
    real : bool, default=True
        Whether the FFT is for real-valued input.

    Returns
    -------
    fast_len : int
        The smallest fast length greater than or equal to target.
    """
    # For PyTorch, powers of 2 are generally fast
    # This is a simple implementation; scipy's version is more sophisticated
    import math
    return 2 ** math.ceil(math.log2(target))


def _cwt_via_fft(
    data: torch.Tensor,
    waveforms: torch.Tensor,
    hermitian: bool = False,
) -> torch.Tensor:
    """Compute the CWT using the FFT.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_times)
        Input data to be analyzed.
    waveforms : torch.Tensor of shape (n_wavelets, n_times)
        Waveforms of the wavelet group.
    hermitian : bool, default=False
        Whether the wavelets are Hermitian. The safest option is to set this
        to False (which does not affect the results). If the wavelets are not
        Hermitian and this is set to True, the results will be incorrect.

    Returns
    -------
    coeffs : torch.Tensor of shape (..., n_wavelets, n_times)
        Wavelet-transform coefficients.

    Warning
    -------
    This function is not intended to be used directly. Use the `transform`
    method of the wavelet-group class instead.
    """
    complex_result = data.is_complex() or waveforms.is_complex()

    # xcorr -> 'full' convolution
    n_conv = data.shape[-1] + waveforms.shape[-1] - 1
    n_fft = next_fast_len(n_conv, real=not complex_result)

    if complex_result:
        fft_, ifft_ = torch.fft.fft, torch.fft.ifft
    else:
        fft_, ifft_ = torch.fft.rfft, torch.fft.irfft

    if hermitian:
        kernels = fft_(waveforms, n=n_fft)
    else:
        kernels = fft_(torch.conj(torch.flip(waveforms, dims=[-1])), n=n_fft)

    # Compute FFT of data and expand dimensions for broadcasting
    data_fft = fft_(data, n=n_fft)
    # Add dimension for wavelets: (..., 1, n_fft)
    data_fft = data_fft.unsqueeze(-2)
    
    # Multiply and inverse FFT
    coeffs = ifft_(kernels * data_fft, n=n_fft)[..., :n_conv]

    # Center with respect to the 'full' convolution
    final_shape = coeffs.shape[:-1] + (data.shape[-1],)
    return get_centered_array(coeffs, final_shape)
