from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from scipy.fft import next_fast_len

from .array_utils import get_array_module, get_centered_array

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _cwt_via_fft(
    data: NDArray,
    waveforms: NDArray,
    hermitian: bool = False,
    array_engine: Literal["numpy", "cupy"] = "numpy",
) -> NDArray:
    """Compute the CWT using the FFT.

    Parameters
    ----------
    data : ndarray of shape (..., n_times)
        Input data to be analyzed.
    waveforms : ndarray of shape (n_wavelets, n_times)
        Waveforms of the wavelet group.
    hermitian : bool, default=False
        Whether the wavelets are Hermitian. The safest option is to set this
        to False (which does not affect the results). If the wavelets are not
        Hermitian and this is set to True, the results will be incorrect.
    array_engine : {'numpy', 'cupy'}, default='numpy'
        The array module to use for computations.

    Returns
    -------
    coeffs : ndarray of shape (..., n_wavelets, n_times)
        Wavelet-transform coefficients.

    Warning
    -------
    This function is not intended to be used directly. Use the `transform`
    method of the wavelet-group class instead.
    """
    xp = get_array_module(array_engine)

    complex_result = data.dtype.kind == "c" or waveforms.dtype.kind == "c"

    # xcorr -> 'full' convolution
    n_conv = data.shape[-1] + waveforms.shape[-1] - 1
    n_fft = next_fast_len(n_conv, real=not complex_result)

    if complex_result:
        fft_, ifft_ = xp.fft.fft, xp.fft.ifft
    else:
        fft_, ifft_ = xp.fft.rfft, xp.fft.irfft

    if hermitian:
        kernels = fft_(waveforms, n=n_fft)
    else:
        kernels = fft_(xp.conj(waveforms[..., ::-1]), n=n_fft)

    coeffs = ifft_(kernels * fft_(data, n=n_fft)[..., None, :], n=n_fft)[..., :n_conv]

    # Center with respect to the 'full' convolution
    final_shape = coeffs.shape[:-1] + (data.shape[-1],)
    return get_centered_array(coeffs, final_shape)
