"""Example demonstrating the PyTorch implementation of MorletX.

This script shows how to use the PyTorch version for GPU-accelerated
wavelet transforms.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from morletx._core_torch import MorletFilterBank, MorletWavelet, MorletWaveletGroup


def example_single_wavelet():
    """Example using a single Morlet wavelet."""
    print("=" * 60)
    print("Example 1: Single Morlet Wavelet")
    print("=" * 60)

    # Create wavelet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    wavelet = MorletWavelet(
        center_freq=10.0,
        shape_ratio=5.0,
        duration=2.0,
        sampling_freq=100.0,
        device=device,
    )

    print(f"Wavelet: {wavelet}")
    print(f"Time width: {wavelet.time_width:.4f} s")
    print(f"Frequency width: {wavelet.freq_width:.4f} Hz")

    # Generate test signal: 10 Hz sine wave with noise
    t = torch.linspace(0, 2, 200, device=device)
    signal = torch.sin(2 * torch.pi * 10 * t) + 0.1 * torch.randn(200, device=device)

    # Compute transform
    coeffs = wavelet.transform(signal, mode="power")
    print(f"Transform coefficients shape: {coeffs.shape}")
    print(f"Mean power: {coeffs.mean().item():.4f}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # Signal
    axes[0].plot(t.cpu().numpy(), signal.cpu().numpy())
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Input Signal (10 Hz sine wave + noise)")
    axes[0].grid(True)

    # Power
    axes[1].plot(t.cpu().numpy(), coeffs.cpu().numpy())
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Power")
    axes[1].set_title("Wavelet Transform Power")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("pytorch_single_wavelet.png", dpi=150)
    print("Saved plot to: pytorch_single_wavelet.png\n")


def example_wavelet_group():
    """Example using multiple wavelets."""
    print("=" * 60)
    print("Example 2: Wavelet Group (Multiple Frequencies)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create wavelet group with multiple frequencies
    center_freqs = [5.0, 10.0, 20.0, 40.0]
    wavelet_group = MorletWaveletGroup(
        center_freqs=center_freqs,
        shape_ratios=5.0,
        duration=2.0,
        sampling_freq=100.0,
        device=device,
    )

    print(f"Number of wavelets: {len(wavelet_group)}")
    print(f"Center frequencies: {wavelet_group.center_freqs.cpu().numpy()}")

    # Generate test signal: mixture of frequencies
    t = torch.linspace(0, 2, 200, device=device)
    signal = (
        torch.sin(2 * torch.pi * 5 * t)
        + 0.5 * torch.sin(2 * torch.pi * 10 * t)
        + 0.3 * torch.sin(2 * torch.pi * 20 * t)
        + 0.1 * torch.randn(200, device=device)
    )

    # Compute transform
    coeffs = wavelet_group.transform(signal, mode="power")
    print(f"Transform coefficients shape: {coeffs.shape}")

    # Plot frequency responses
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    wavelet_group.plot_responses(axes[0], normalize=True)

    # Plot power for each frequency
    t_np = t.cpu().numpy()
    coeffs_np = coeffs.cpu().numpy()

    for i, freq in enumerate(center_freqs):
        axes[1].plot(t_np, coeffs_np[i], label=f"{freq} Hz")

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Power")
    axes[1].set_title("Wavelet Transform Power at Different Frequencies")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("pytorch_wavelet_group.png", dpi=150)
    print("Saved plot to: pytorch_wavelet_group.png\n")


def example_filter_bank():
    """Example using a Morlet filter bank."""
    print("=" * 60)
    print("Example 3: Morlet Filter Bank (Constant-Q)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create filter bank
    filter_bank = MorletFilterBank(
        n_octaves=4,
        n_intervals=8,
        shape_ratio=5.0,
        duration=2.0,
        sampling_freq=100.0,
        device=device,
    )

    print(f"Number of wavelets: {len(filter_bank)}")
    print(f"Frequency range: {filter_bank.center_freqs.min().item():.2f} - "
          f"{filter_bank.center_freqs.max().item():.2f} Hz")

    # Generate chirp signal (frequency sweep)
    t = torch.linspace(0, 2, 200, device=device)
    # Frequency increases linearly from 5 to 40 Hz
    instantaneous_freq = 5 + (40 - 5) * t / 2
    phase = 2 * torch.pi * (5 * t + (40 - 5) * t**2 / 4)
    signal = torch.sin(phase) + 0.1 * torch.randn(200, device=device)

    # Compute transform
    coeffs = filter_bank.transform(signal, mode="power")
    print(f"Transform coefficients shape: {coeffs.shape}")

    # Plot scalogram
    fig, ax = plt.subplots(figsize=(12, 6))
    filter_bank.plot_scalogram(
        ax, data=signal, mode="power", log_scale=True, demean=True
    )
    ax.set_title("Scalogram of Chirp Signal (5-40 Hz sweep)")

    plt.tight_layout()
    plt.savefig("pytorch_filter_bank.png", dpi=150)
    print("Saved plot to: pytorch_filter_bank.png\n")


def example_batch_processing():
    """Example of batch processing multiple signals."""
    print("=" * 60)
    print("Example 4: Batch Processing")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create wavelet
    wavelet = MorletWavelet(
        center_freq=10.0,
        shape_ratio=5.0,
        duration=1.0,
        sampling_freq=100.0,
        device=device,
    )

    # Generate batch of signals
    batch_size = 16
    signal_length = 100

    print(f"Processing batch of {batch_size} signals...")

    # Each signal is a 10 Hz sine wave with different noise
    t = torch.linspace(0, 1, signal_length, device=device)
    signals = torch.sin(2 * torch.pi * 10 * t).unsqueeze(0).expand(batch_size, -1)
    signals = signals + 0.1 * torch.randn(batch_size, signal_length, device=device)

    # Process all signals at once
    coeffs = wavelet.transform(signals, mode="power")
    print(f"Input shape: {signals.shape}")
    print(f"Output shape: {coeffs.shape}")
    print(f"Mean power across batch: {coeffs.mean().item():.4f}")
    print(f"Std power across batch: {coeffs.std().item():.4f}\n")


def example_numpy_compatibility():
    """Example showing NumPy compatibility."""
    print("=" * 60)
    print("Example 5: NumPy Compatibility")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    wavelet = MorletWavelet(
        center_freq=10.0,
        shape_ratio=5.0,
        duration=1.0,
        sampling_freq=100.0,
        device=device,
    )

    # Input as NumPy array
    np.random.seed(42)
    signal_np = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100)) + 0.1 * np.random.randn(
        100
    )

    print("Processing NumPy array...")
    coeffs = wavelet.transform(signal_np, mode="power")
    print(f"Input type: {type(signal_np)}")
    print(f"Output type: {type(coeffs)}")
    print(f"Output device: {coeffs.device}")

    # Convert back to NumPy
    coeffs_np = coeffs.cpu().numpy()
    print(f"Converted to NumPy: {type(coeffs_np)}")
    print(f"Shape: {coeffs_np.shape}\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PyTorch MorletX Examples")
    print("=" * 60 + "\n")

    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("✗ CUDA not available, using CPU")

    print()

    example_single_wavelet()
    example_wavelet_group()
    example_filter_bank()
    example_batch_processing()
    example_numpy_compatibility()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
