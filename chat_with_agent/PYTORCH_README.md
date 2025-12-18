# PyTorch Implementation of MorletX

This document describes the PyTorch port of MorletX, which provides GPU-accelerated wavelet transforms using PyTorch instead of NumPy/CuPy.

## New Files

The PyTorch implementation consists of **two versions** with shared utilities:

### Pure Signal Processing Version

1. **`src/morletx/_core_torch.py`** - Lightweight PyTorch implementation
   - No `nn.Module` inheritance
   - Pure signal processing tools
   - Minimal overhead
   - Best for: Analysis, feature extraction, non-ML applications

### Neural Network Integration Version

2. **`src/morletx/_core_torch_nn.py`** - `nn.Module` implementation
   - Inherits from `torch.nn.Module`
   - **Optional learnable parameters** (center frequencies, shape ratios)
   - Easy device management (`.cuda()`, `.to()`, etc.)
   - State dict support for saving/loading
   - Best for: End-to-end learning, neural network integration, optimization

### Shared Utilities

3. **`src/morletx/utils/array_utils_torch.py`** - PyTorch utility functions
4. **`src/morletx/utils/fft_utils_torch.py`** - PyTorch FFT utilities

## Which Version Should I Use?

| Use Case | Recommended Version |
|----------|---------------------|
| Signal analysis, feature extraction | `_core_torch.py` |
| Fixed wavelets in preprocessing | `_core_torch.py` |
| Minimal memory/overhead | `_core_torch.py` |
| **Learnable wavelet parameters** | `_core_torch_nn.py` |
| **Part of neural network** | `_core_torch_nn.py` |
| **End-to-end optimization** | `_core_torch_nn.py` |
| Easy model serialization | `_core_torch_nn.py` |

## Key Differences from NumPy/CuPy Version

### 1. Device Management
Instead of `array_engine: Literal["numpy", "cupy"]`, the PyTorch version uses:
- `device: torch.device | str | None` - Specify 'cpu', 'cuda', 'cuda:0', etc.
- `dtype: torch.dtype` - Specify data type (default: `torch.float64`)

### 2. Automatic GPU Support
```python
# NumPy/CuPy version (old)
wavelet = MorletWavelet(center_freq=10, shape_ratio=5, duration=1.0, 
                        sampling_freq=100, array_engine="cupy")

# PyTorch version (new)
wavelet = MorletWavelet(center_freq=10, shape_ratio=5, duration=1.0, 
                        sampling_freq=100, device="cuda")
```

### 3. No `detach_from_device` Parameter
The PyTorch version doesn't need this parameter. Instead:
- Results are PyTorch tensors on the specified device
- Use `.cpu()` to move to CPU
- Use `.numpy()` to convert to NumPy (automatically moves to CPU first)

### 4. Input Flexibility
The PyTorch version accepts both NumPy arrays and PyTorch tensors:
```python
import numpy as np
import torch

# Works with NumPy arrays
data_np = np.random.randn(1000)
result = wavelet.transform(data_np)

# Works with PyTorch tensors
data_torch = torch.randn(1000, device='cuda')
result = wavelet.transform(data_torch)
```

## Usage Examples

### Basic Usage

```python
from morletx._core_torch import MorletWavelet, MorletWaveletGroup, MorletFilterBank
import torch

# Single wavelet on CPU
wavelet = MorletWavelet(
    center_freq=10.0,
    shape_ratio=5.0,
    duration=1.0,
    sampling_freq=100.0,
    device='cpu'
)

# Generate test signal
signal = torch.randn(1000)

# Compute transform
coeffs = wavelet.transform(signal, mode='power')
print(coeffs.shape)  # (1000,)
```

### GPU Acceleration

```python
# Create wavelet on GPU
wavelet_gpu = MorletWavelet(
    center_freq=10.0,
    shape_ratio=5.0,
    duration=1.0,
    sampling_freq=100.0,
    device='cuda'  # or 'cuda:0' for specific GPU
)

# GPU signal
signal_gpu = torch.randn(1000, device='cuda')

# Fast GPU computation
coeffs_gpu = wavelet_gpu.transform(signal_gpu, mode='power')

# Move to CPU if needed
coeffs_cpu = coeffs_gpu.cpu()
coeffs_numpy = coeffs_gpu.cpu().numpy()
```

### Multi-scale Wavelets

```python
# Multiple center frequencies
center_freqs = [5.0, 10.0, 20.0, 40.0]
shape_ratios = 5.0

wavelet_group = MorletWaveletGroup(
    center_freqs=center_freqs,
    shape_ratios=shape_ratios,
    duration=1.0,
    sampling_freq=100.0,
    device='cuda'
)

signal = torch.randn(1000, device='cuda')
coeffs = wavelet_group.transform(signal, mode='power')
print(coeffs.shape)  # (4, 1000) - 4 frequencies, 1000 time points
```

### Filter Bank

```python
# Constant-Q filter bank
filter_bank = MorletFilterBank(
    n_octaves=5,
    n_intervals=12,
    shape_ratio=5.0,
    duration=1.0,
    sampling_freq=100.0,
    device='cuda'
)

signal = torch.randn(1000, device='cuda')
coeffs = filter_bank.transform(signal, mode='power')
print(coeffs.shape)  # (n_wavelets, 1000)
```

### Batch Processing

```python
# Process multiple signals at once
batch_size = 32
signal_length = 1000

signals = torch.randn(batch_size, signal_length, device='cuda')
coeffs = wavelet_group.transform(signals, mode='power')
print(coeffs.shape)  # (32, 4, 1000) - batch, frequencies, time
```

### Plotting

```python
import matplotlib.pyplot as plt

# Frequency responses
fig, ax = plt.subplots()
wavelet_group.plot_responses(ax, normalize=True)
plt.show()

# Scalogram
fig, ax = plt.subplots()
filter_bank.plot_scalogram(ax, data=signal, mode='power', log_scale=True)
plt.show()
```

## Performance Considerations

### Memory Management
- PyTorch manages GPU memory automatically
- For large batches, consider using `torch.cuda.empty_cache()` between operations
- Use `dtype=torch.float32` for lower memory usage (vs default `torch.float64`)

### Mixed Precision
```python
# Use float32 for faster computation with less precision
wavelet = MorletWavelet(
    center_freq=10.0,
    shape_ratio=5.0,
    duration=1.0,
    sampling_freq=100.0,
    device='cuda',
    dtype=torch.float32  # Half the memory of float64
)
```

### Data Transfer Overhead
- Minimize CPU ↔ GPU transfers
- Keep data on GPU for multiple operations
- Only transfer final results back to CPU

```python
# Good: Keep everything on GPU
signal_gpu = torch.randn(1000, device='cuda')
coeffs1 = wavelet1.transform(signal_gpu)
coeffs2 = wavelet2.transform(signal_gpu)
result = coeffs1 + coeffs2  # Still on GPU
final = result.cpu().numpy()  # Single transfer at end

# Bad: Multiple transfers
signal_gpu = torch.randn(1000, device='cuda')
coeffs1 = wavelet1.transform(signal_gpu).cpu()  # Transfer
coeffs2 = wavelet2.transform(signal_gpu).cpu()  # Transfer
result = coeffs1 + coeffs2
```

## Compatibility Notes

### Scipy Dependencies
The PyTorch version implements its own `tukey_window` function, removing the dependency on `scipy.signal.windows.tukey` for the core computation. However, scipy is still used for `next_fast_len` in FFT operations.

### Type Hints
- Input types: `torch.Tensor | np.ndarray`
- Output types: `torch.Tensor`
- The API accepts both but always returns PyTorch tensors

### Plotting
Plotting methods automatically convert tensors to NumPy arrays for matplotlib/plotly compatibility.

## Migration Guide

### From NumPy/CuPy to PyTorch

```python
# OLD (NumPy/CuPy)
from morletx._core import MorletWavelet
import numpy as np

wavelet = MorletWavelet(
    center_freq=10.0,
    shape_ratio=5.0,
    duration=1.0,
    sampling_freq=100.0,
    array_engine="cupy"  # or "numpy"
)
signal = np.random.randn(1000)
coeffs = wavelet.transform(signal, detach_from_device=True)

# NEW (PyTorch)
from morletx._core_torch import MorletWavelet
import torch

wavelet = MorletWavelet(
    center_freq=10.0,
    shape_ratio=5.0,
    duration=1.0,
    sampling_freq=100.0,
    device="cuda"  # or "cpu"
)
signal = torch.randn(1000, device="cuda")
coeffs = wavelet.transform(signal)
# Use .cpu().numpy() if you need NumPy array
```

## Using the nn.Module Version

The `_core_torch_nn.py` version provides the same API but with `nn.Module` benefits:

### Basic Usage

```python
from morletx._core_torch_nn import MorletWavelet
import torch

# Create as nn.Module
wavelet = MorletWavelet(
    center_freq=10.0,
    shape_ratio=5.0,
    duration=1.0,
    sampling_freq=100.0,
    device='cuda'
)

# Use forward() or transform()
signal = torch.randn(1000, device='cuda')
coeffs = wavelet(signal)  # Calls forward()
# or
coeffs = wavelet.transform(signal)  # Same result

# Easy device management
wavelet = wavelet.cuda()  # Move to GPU
wavelet = wavelet.cpu()   # Move to CPU
wavelet = wavelet.to('cuda:1')  # Specific GPU

# Save and load
torch.save(wavelet.state_dict(), 'wavelet.pth')
wavelet.load_state_dict(torch.load('wavelet.pth'))
```

### Learnable Parameters

The key feature of the nn.Module version is **learnable parameters**:

```python
from morletx._core_torch_nn import MorletWavelet
import torch.optim as optim

# Create wavelet with learnable center frequency
wavelet = MorletWavelet(
    center_freq=15.0,  # Initial guess
    shape_ratio=5.0,
    duration=1.0,
    sampling_freq=100.0,
    learnable_center_freq=True,  # Make it learnable!
)

# Check parameters
print(list(wavelet.parameters()))  # Shows learnable parameters
print(wavelet._center_freqs.requires_grad)  # True

# Optimize via backpropagation
optimizer = optim.Adam(wavelet.parameters(), lr=0.1)

for epoch in range(100):
    optimizer.zero_grad()
    coeffs = wavelet(signal, mode='complex')
    loss = your_loss_function(coeffs)
    loss.backward()
    optimizer.step()
    
print(f"Optimized frequency: {wavelet._center_freqs.item()}")
```

### Integration with Neural Networks

```python
import torch.nn as nn
from morletx._core_torch_nn import MorletWaveletGroup

class WaveletFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Wavelet layer (fixed or learnable)
        self.wavelet_layer = MorletWaveletGroup(
            center_freqs=[5.0, 10.0, 15.0, 20.0],
            shape_ratios=5.0,
            duration=1.0,
            sampling_freq=100.0,
            learnable_center_freqs=True,  # Learn optimal frequencies
        )
        
        # Downstream layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 100, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
    
    def forward(self, x):
        # x: (batch, time)
        wt = self.wavelet_layer(x, mode='power')  # (batch, freqs, time)
        return self.classifier(wt)

# Use like any PyTorch model
model = WaveletFeatureExtractor().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for batch in dataloader:
    signals, labels = batch
    outputs = model(signals)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

### When to Use Learnable Parameters

**Use learnable parameters when:**
- You want to **discover optimal frequencies** from data
- Building **end-to-end learnable systems**
- The optimal wavelet parameters are **unknown a priori**
- You have **sufficient training data** to learn them

**Keep parameters fixed when:**
- You know the **exact frequencies** to analyze
- Doing **pure signal analysis** (not learning)
- Want **interpretable, predefined** wavelets
- Have **limited training data**

## Advantages of PyTorch Version

1. **Unified API**: Single interface for CPU and GPU, no need to switch between NumPy and CuPy
2. **Automatic Differentiation**: Can compute gradients if needed (for optimization, learning, etc.)
3. **Better GPU Support**: More mature CUDA support and optimizations
4. **Ecosystem**: Easy integration with PyTorch models and pipelines
5. **Mixed Precision**: Native support for float16/bfloat16 for faster computation
6. **Distributed Computing**: Built-in support for multi-GPU and distributed training

## Testing

You can verify the implementation produces similar results:

```python
import numpy as np
import torch
from morletx._core import MorletWavelet as MorletWaveletNumPy
from morletx._core_torch import MorletWavelet as MorletWaveletTorch

# Create wavelets
wv_np = MorletWaveletNumPy(10.0, 5.0, 1.0, 100.0)
wv_torch = MorletWaveletTorch(10.0, 5.0, 1.0, 100.0, device='cpu')

# Same signal
np.random.seed(42)
signal_np = np.random.randn(1000)
signal_torch = torch.from_numpy(signal_np)

# Compare results
coeffs_np = wv_np.transform(signal_np, mode='power')
coeffs_torch = wv_torch.transform(signal_torch, mode='power').numpy()

print(f"Max difference: {np.abs(coeffs_np - coeffs_torch).max()}")
# Should be very small (< 1e-10 for float64)
```

## Requirements

The PyTorch implementation requires:
- `torch >= 1.10.0` (for complex tensor support)
- `numpy` (for some utility functions and plotting)
- `matplotlib` (optional, for plotting)
- `plotly` (optional, for interactive plotting)

Install PyTorch following the official instructions: https://pytorch.org/get-started/locally/
