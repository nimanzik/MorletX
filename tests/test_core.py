"""Unit tests for morletx core functionality."""

import numpy as np
import pytest

from morletx import MorletWavelet, MorletWaveletGroup, compute_morlet_center_freqs


class TestMorletWavelet:
    """Test cases for MorletWavelet class."""

    def test_initialization(self):
        """Test MorletWavelet initialization."""
        wavelet = MorletWavelet(
            center_freq=10.0,
            shape_ratio=5.0,
            duration=1.0,
            sampling_freq=100.0
        )

        assert wavelet.center_freq == 10.0
        assert wavelet.shape_ratio == 5.0
        assert wavelet.duration == 1.0
        assert wavelet.sampling_freq == 100.0

    def test_properties(self):
        """Test MorletWavelet properties."""
        wavelet = MorletWavelet(
            center_freq=10.0,
            shape_ratio=5.0,
            duration=1.0,
            sampling_freq=100.0
        )

        # Test that properties return expected types and shapes
        assert isinstance(wavelet.center_freqs, np.ndarray)
        assert isinstance(wavelet.shape_ratios, np.ndarray)
        assert isinstance(wavelet.waveforms, np.ndarray)
        assert len(wavelet.center_freqs) == 1
        assert len(wavelet.shape_ratios) == 1

    def test_transform_basic(self):
        """Test basic wavelet transform functionality."""
        wavelet = MorletWavelet(
            center_freq=10.0,
            shape_ratio=5.0,
            duration=1.0,
            sampling_freq=100.0
        )

        # Create a simple test signal
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave

        # Transform should work without errors
        result = wavelet.transform(signal, mode="complex")

        assert isinstance(result, np.ndarray)
        assert result.shape == signal.shape
        assert result.dtype == np.complex128

    def test_transform_modes(self):
        """Test different transform modes."""
        wavelet = MorletWavelet(
            center_freq=10.0,
            shape_ratio=5.0,
            duration=1.0,
            sampling_freq=100.0
        )

        # Create test signal
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 10 * t)

        # Test all modes
        complex_result = wavelet.transform(signal, mode="complex")
        magnitude_result = wavelet.transform(signal, mode="magnitude")
        power_result = wavelet.transform(signal, mode="power")

        assert complex_result.dtype == np.complex128
        assert magnitude_result.dtype == np.float64
        assert power_result.dtype == np.float64

        # Power should be magnitude squared
        np.testing.assert_allclose(
            power_result,
            magnitude_result**2,
            rtol=1e-10
        )


class TestMorletWaveletGroup:
    """Test cases for MorletWaveletGroup class."""

    def test_initialization(self):
        """Test MorletWaveletGroup initialization."""
        group = MorletWaveletGroup(
            center_freqs=[5.0, 10.0, 20.0],
            shape_ratios=[3.0, 5.0, 7.0],
            duration=1.0,
            sampling_freq=100.0
        )

        assert len(group.center_freqs) == 3
        assert len(group.shape_ratios) == 3
        np.testing.assert_array_equal(group.center_freqs, [5.0, 10.0, 20.0])
        np.testing.assert_array_equal(group.shape_ratios, [3.0, 5.0, 7.0])

    def test_transform_multiscale(self):
        """Test multiscale wavelet transform."""
        group = MorletWaveletGroup(
            center_freqs=[5.0, 10.0, 20.0],
            shape_ratios=[5.0, 5.0, 5.0],
            duration=1.0,
            sampling_freq=100.0
        )

        # Create test signal with multiple frequencies
        t = np.linspace(0, 1, 100)
        signal = (np.sin(2 * np.pi * 5 * t) +
                 np.sin(2 * np.pi * 10 * t) +
                 np.sin(2 * np.pi * 20 * t))

        result = group.transform(signal, mode="power")

        # Result should have shape (n_wavelets, n_times)
        assert result.shape == (3, 100)
        assert result.dtype == np.float64


class TestComputeMorletCenterFreqs:
    """Test cases for compute_morlet_center_freqs function."""

    def test_basic_computation(self):
        """Test basic center frequency computation."""
        center_freqs = compute_morlet_center_freqs(
            n_octaves=3,
            n_intervals=12,
            shape_ratio=5.0,
            sampling_freq=1000.0
        )

        assert isinstance(center_freqs, np.ndarray)
        assert len(center_freqs) > 0
        assert all(center_freqs > 0)
        assert all(center_freqs < 500.0)  # Below Nyquist frequency

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(ValueError):
            compute_morlet_center_freqs(
                n_octaves=0,  # Invalid
                n_intervals=12,
                shape_ratio=5.0,
                sampling_freq=1000.0
            )

        with pytest.raises(ValueError):
            compute_morlet_center_freqs(
                n_octaves=3,
                n_intervals=0,  # Invalid
                shape_ratio=5.0,
                sampling_freq=1000.0
            )

        with pytest.raises(ValueError):
            compute_morlet_center_freqs(
                n_octaves=3,
                n_intervals=12,
                shape_ratio=0.0,  # Invalid
                sampling_freq=1000.0
            )

    def test_frequency_ordering(self):
        """Test that center frequencies are properly ordered."""
        center_freqs = compute_morlet_center_freqs(
            n_octaves=2,
            n_intervals=4,
            shape_ratio=5.0,
            sampling_freq=1000.0
        )

        # Frequencies should be in ascending order
        assert np.all(np.diff(center_freqs) > 0)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_sample_signal(self):
        """Test behavior with single sample signal."""
        wavelet = MorletWavelet(
            center_freq=10.0,
            shape_ratio=5.0,
            duration=0.01,  # Very short duration
            sampling_freq=100.0
        )

        signal = np.array([1.0])

        # Should handle gracefully
        result = wavelet.transform(signal, mode="complex")
        assert len(result) == 1

    def test_zero_signal(self):
        """Test behavior with zero signal."""
        wavelet = MorletWavelet(
            center_freq=10.0,
            shape_ratio=5.0,
            duration=1.0,
            sampling_freq=100.0
        )

        signal = np.zeros(100)
        result = wavelet.transform(signal, mode="power")

        # Result should be close to zero
        assert np.allclose(result, 0, atol=1e-10)
