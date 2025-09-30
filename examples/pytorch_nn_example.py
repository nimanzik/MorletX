"""Example demonstrating the nn.Module version with learnable parameters.

This script shows how to:
1. Use the nn.Module version for easy device management
2. Optimize wavelet parameters via backpropagation
3. Integrate wavelets into neural network pipelines
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from morletx._core_torch_nn import MorletFilterBank, MorletWavelet, MorletWaveletGroup


def example_basic_nn_module():
    """Example showing basic nn.Module features."""
    print("=" * 60)
    print("Example 1: Basic nn.Module Features")
    print("=" * 60)

    # Create wavelet as nn.Module
    wavelet = MorletWavelet(
        center_freq=10.0,
        shape_ratio=5.0,
        duration=1.0,
        sampling_freq=100.0,
        device='cpu',
    )

    print(f"Wavelet: {wavelet}")
    print(f"Is nn.Module: {isinstance(wavelet, nn.Module)}")
    print(f"Number of parameters: {sum(p.numel() for p in wavelet.parameters())}")
    print(f"Number of buffers: {len(list(wavelet.buffers()))}")

    # Easy device management
    if torch.cuda.is_available():
        wavelet = wavelet.cuda()
        print(f"Moved to CUDA: {next(wavelet.buffers()).device}")
        wavelet = wavelet.cpu()
        print(f"Moved back to CPU: {next(wavelet.buffers()).device}")

    # Use forward() method
    signal = torch.randn(100)
    coeffs = wavelet(signal)  # Calls forward()
    print(f"Output shape: {coeffs.shape}")

    # Save and load
    torch.save(wavelet.state_dict(), 'wavelet.pth')
    wavelet_loaded = MorletWavelet(10.0, 5.0, 1.0, 100.0)
    wavelet_loaded.load_state_dict(torch.load('wavelet.pth'))
    print("✓ Saved and loaded state dict\n")


def example_learnable_parameters():
    """Example with learnable wavelet parameters."""
    print("=" * 60)
    print("Example 2: Learnable Parameters")
    print("=" * 60)

    # Create wavelet with learnable center frequency
    wavelet = MorletWavelet(
        center_freq=15.0,  # Initial guess
        shape_ratio=5.0,
        duration=1.0,
        sampling_freq=100.0,
        device='cpu',
        learnable_center_freq=True,  # Make it learnable!
    )

    print(f"Wavelet: {wavelet}")
    print(f"Learnable parameters:")
    for name, param in wavelet.named_parameters():
        print(f"  {name}: {param.data.item():.4f} (requires_grad={param.requires_grad})")

    # Generate target signal at 10 Hz
    t = torch.linspace(0, 1, 100)
    target_signal = torch.sin(2 * torch.pi * 10 * t)

    # Optimize to find the true frequency
    optimizer = optim.Adam(wavelet.parameters(), lr=0.1)
    losses = []

    print("\nOptimizing center frequency...")
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Compute wavelet transform
        coeffs = wavelet(target_signal, mode='complex')
        
        # Loss: maximize correlation with target
        # (minimize negative correlation)
        correlation = torch.abs(torch.sum(coeffs * torch.conj(torch.from_numpy(target_signal.numpy()))))
        loss = -correlation
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            current_freq = wavelet._center_freqs.item()
            print(f"Epoch {epoch+1:3d}: Fc = {current_freq:.4f} Hz, Loss = {loss.item():.4f}")

    final_freq = wavelet._center_freqs.item()
    print(f"\nOptimization complete!")
    print(f"Initial frequency: 15.0 Hz")
    print(f"Final frequency: {final_freq:.4f} Hz")
    print(f"True frequency: 10.0 Hz")
    print(f"Error: {abs(final_freq - 10.0):.4f} Hz\n")

    # Plot optimization progress
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Optimization Progress')
    axes[0].grid(True)
    
    axes[1].plot(t.numpy(), target_signal.numpy(), label='Target Signal (10 Hz)')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Target Signal')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('pytorch_nn_learnable.png', dpi=150)
    print("Saved plot to: pytorch_nn_learnable.png\n")


def example_learnable_filter_bank():
    """Example with learnable filter bank."""
    print("=" * 60)
    print("Example 3: Learnable Filter Bank")
    print("=" * 60)

    # Create filter bank with learnable shape ratio
    filter_bank = MorletFilterBank(
        n_octaves=3,
        n_intervals=4,
        shape_ratio=5.0,  # Initial value
        duration=1.0,
        sampling_freq=100.0,
        device='cpu',
        learnable_shape_ratio=True,  # Make shape ratio learnable
    )

    print(f"Filter bank: {filter_bank}")
    print(f"Number of wavelets: {len(filter_bank)}")
    print(f"Learnable parameters:")
    for name, param in filter_bank.named_parameters():
        print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

    # Generate multi-frequency signal
    t = torch.linspace(0, 1, 100)
    signal = (
        torch.sin(2 * torch.pi * 5 * t) +
        torch.sin(2 * torch.pi * 10 * t) +
        torch.sin(2 * torch.pi * 20 * t)
    )

    # Compute transform
    coeffs = filter_bank(signal, mode='power')
    print(f"\nTransform output shape: {coeffs.shape}")
    print(f"Mean power: {coeffs.mean().item():.4f}\n")


def example_wavelet_layer_in_network():
    """Example integrating wavelet transform as a layer in a neural network."""
    print("=" * 60)
    print("Example 4: Wavelet Layer in Neural Network")
    print("=" * 60)

    class WaveletClassifier(nn.Module):
        """Simple classifier using wavelet features."""
        
        def __init__(self, n_freqs: int, n_times: int, n_classes: int):
            super().__init__()
            
            # Wavelet transform layer (fixed parameters)
            self.wavelet_layer = MorletWaveletGroup(
                center_freqs=[5.0, 10.0, 15.0, 20.0],
                shape_ratios=5.0,
                duration=1.0,
                sampling_freq=100.0,
                device='cpu',
            )
            
            # Classification layers
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_freqs * n_times, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, n_classes),
            )
        
        def forward(self, x):
            # x: (batch, time)
            # Compute wavelet transform
            wt = self.wavelet_layer(x, mode='power')  # (batch, freqs, time)
            # Classify based on wavelet features
            return self.classifier(wt)

    # Create model
    model = WaveletClassifier(n_freqs=4, n_times=100, n_classes=3)
    print(f"Model: {model}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 8
    dummy_input = torch.randn(batch_size, 100)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output[0].detach().numpy()}\n")


def example_end_to_end_optimization():
    """Example of end-to-end optimization with learnable wavelets."""
    print("=" * 60)
    print("Example 5: End-to-End Optimization")
    print("=" * 60)

    class AdaptiveWaveletClassifier(nn.Module):
        """Classifier with learnable wavelet parameters."""
        
        def __init__(self):
            super().__init__()
            
            # Learnable wavelet layer
            self.wavelet_layer = MorletWaveletGroup(
                center_freqs=[5.0, 10.0, 15.0, 20.0],
                shape_ratios=5.0,
                duration=1.0,
                sampling_freq=100.0,
                device='cpu',
                learnable_center_freqs=True,  # Learn optimal frequencies!
            )
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(4 * 100, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )
        
        def forward(self, x):
            wt = self.wavelet_layer(x, mode='power')
            return self.classifier(wt)

    # Generate synthetic dataset
    # Class 0: 8 Hz signal, Class 1: 12 Hz signal
    def generate_data(n_samples, freq, label):
        t = torch.linspace(0, 1, 100)
        signals = []
        for _ in range(n_samples):
            signal = torch.sin(2 * torch.pi * freq * t) + 0.1 * torch.randn(100)
            signals.append(signal)
        return torch.stack(signals), torch.full((n_samples,), label, dtype=torch.long)

    X_train_0, y_train_0 = generate_data(50, freq=8.0, label=0)
    X_train_1, y_train_1 = generate_data(50, freq=12.0, label=1)
    X_train = torch.cat([X_train_0, X_train_1])
    y_train = torch.cat([y_train_0, y_train_1])

    # Shuffle
    perm = torch.randperm(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    # Create model and optimizer
    model = AdaptiveWaveletClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("Training classifier with learnable wavelets...")
    print(f"Initial center frequencies: {model.wavelet_layer.center_freqs.detach().numpy()}")

    # Training loop
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                acc = (preds == y_train).float().mean()
            print(f"Epoch {epoch+1:2d}: Loss = {loss.item():.4f}, Acc = {acc.item():.4f}")

    print(f"\nFinal center frequencies: {model.wavelet_layer.center_freqs.detach().numpy()}")
    print("Note: Frequencies should adapt toward 8 Hz and 12 Hz (the signal frequencies)\n")


def example_device_management():
    """Example showing easy device management with nn.Module."""
    print("=" * 60)
    print("Example 6: Device Management")
    print("=" * 60)

    wavelet = MorletWavelet(10.0, 5.0, 1.0, 100.0, device='cpu')
    print(f"Initial device: {wavelet.device}")

    # Move to CUDA if available
    if torch.cuda.is_available():
        wavelet = wavelet.cuda()
        print(f"After .cuda(): {wavelet.device}")
        
        # Process GPU data
        signal_gpu = torch.randn(100, device='cuda')
        coeffs_gpu = wavelet(signal_gpu)
        print(f"Output device: {coeffs_gpu.device}")
        
        # Move back to CPU
        wavelet = wavelet.cpu()
        print(f"After .cpu(): {wavelet.device}")
    else:
        print("CUDA not available, skipping GPU test")

    # .to() method
    wavelet = wavelet.to('cpu')
    print(f"After .to('cpu'): {wavelet.device}")
    
    # Change dtype
    wavelet = wavelet.to(dtype=torch.float32)
    print(f"After .to(float32): {wavelet.dtype}\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PyTorch nn.Module MorletX Examples")
    print("=" * 60 + "\n")

    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("✗ CUDA not available, using CPU")

    print()

    example_basic_nn_module()
    example_learnable_parameters()
    example_learnable_filter_bank()
    example_wavelet_layer_in_network()
    example_end_to_end_optimization()
    example_device_management()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
