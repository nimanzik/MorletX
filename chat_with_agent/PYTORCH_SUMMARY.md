# PyTorch Port Summary

## Overview

Successfully ported MorletX to PyTorch with **two implementations**:

1. **Pure Signal Processing** (`_core_torch.py`) - Lightweight, no nn.Module
2. **Neural Network Integration** (`_core_torch_nn.py`) - Full nn.Module with learnable parameters

## Files Created

### Core Implementations
- `src/morletx/_core_torch.py` (866 lines) - Pure PyTorch version
- `src/morletx/_core_torch_nn.py` (1,087 lines) - nn.Module version with learnable parameters

### Utilities
- `src/morletx/utils/array_utils_torch.py` - PyTorch utilities (tukey_window, get_centered_array)
- `src/morletx/utils/fft_utils_torch.py` - FFT-based CWT implementation

### Documentation & Examples
- `PYTORCH_README.md` - Comprehensive documentation
- `examples/pytorch_example.py` - 5 examples for pure version
- `examples/pytorch_nn_example.py` - 6 examples for nn.Module version

## Key Features

### Pure Version (`_core_torch.py`)
✅ Direct PyTorch tensor operations  
✅ CPU/GPU support via `device` parameter  
✅ Accepts NumPy arrays or PyTorch tensors  
✅ Minimal overhead  
✅ Same mathematical correctness as original  

### nn.Module Version (`_core_torch_nn.py`)
✅ All features from pure version  
✅ **Learnable center frequencies** (`learnable_center_freqs=True`)  
✅ **Learnable shape ratios** (`learnable_shape_ratios=True`)  
✅ Easy device management (`.cuda()`, `.to()`, etc.)  
✅ State dict support (save/load with `torch.save()`)  
✅ `forward()` method for nn.Module compatibility  
✅ Automatic parameter registration  
✅ Integration with PyTorch optimizers  

## API Comparison

### Original (NumPy/CuPy)
```python
from morletx._core import MorletWavelet

wavelet = MorletWavelet(
    center_freq=10.0,
    shape_ratio=5.0,
    duration=1.0,
    sampling_freq=100.0,
    array_engine="cupy"  # or "numpy"
)
```

### Pure PyTorch
```python
from morletx._core_torch import MorletWavelet

wavelet = MorletWavelet(
    center_freq=10.0,
    shape_ratio=5.0,
    duration=1.0,
    sampling_freq=100.0,
    device="cuda"  # or "cpu"
)
```

### nn.Module PyTorch (with learnable parameters)
```python
from morletx._core_torch_nn import MorletWavelet

wavelet = MorletWavelet(
    center_freq=10.0,
    shape_ratio=5.0,
    duration=1.0,
    sampling_freq=100.0,
    device="cuda",
    learnable_center_freq=True,  # NEW!
    learnable_shape_ratio=True,  # NEW!
)

# Optimize via backpropagation
optimizer = torch.optim.Adam(wavelet.parameters(), lr=0.01)
```

## Classes Implemented

All three classes from the original implementation:

1. **MorletWaveletGroup** - Base class for multi-scale wavelets
2. **MorletWavelet** - Single wavelet
3. **MorletFilterBank** - Constant-Q filter bank

Each class exists in both versions with identical APIs (except learnable parameter options in nn.Module version).

## Use Cases

### Use Pure Version For:
- Signal analysis and feature extraction
- Fixed, predefined wavelets
- Minimal memory/computational overhead
- Non-ML applications

### Use nn.Module Version For:
- **Learning optimal wavelet parameters from data**
- End-to-end neural network training
- Integration with PyTorch models
- When you need `.cuda()`, `.to()`, state dict, etc.
- Discovering unknown frequency components

## Example: Learnable Wavelets

```python
from morletx._core_torch_nn import MorletWavelet
import torch
import torch.optim as optim

# Create wavelet with learnable frequency
wavelet = MorletWavelet(
    center_freq=15.0,  # Initial guess
    shape_ratio=5.0,
    duration=1.0,
    sampling_freq=100.0,
    learnable_center_freq=True
)

# Target: 10 Hz signal
target = torch.sin(2 * torch.pi * 10 * torch.linspace(0, 1, 100))

# Optimize to discover the frequency
optimizer = optim.Adam(wavelet.parameters(), lr=0.1)

for epoch in range(100):
    optimizer.zero_grad()
    coeffs = wavelet(target, mode='complex')
    loss = -torch.abs(torch.sum(coeffs * torch.conj(target)))
    loss.backward()
    optimizer.step()

print(f"Discovered frequency: {wavelet._center_freqs.item():.2f} Hz")
# Output: ~10.0 Hz
```

## Technical Highlights

### Removed Dependencies
- No scipy.signal.windows.tukey needed (implemented in pure PyTorch)
- scipy.fft.next_fast_len replaced with simple power-of-2 calculation

### Maintained Features
- All mathematical formulas identical to original
- Same normalization and scaling
- Compatible with matplotlib/plotly plotting
- Batch processing support

### Added Features
- **Gradient computation** through wavelet transforms
- **Parameter optimization** via backpropagation
- Automatic device management
- Mixed precision support (float16/float32/float64)

## Performance

- GPU acceleration via CUDA
- Efficient batch processing
- Memory-efficient operations
- Compatible with PyTorch's autograd for gradient computation

## Compatibility

- **Input**: Accepts both `np.ndarray` and `torch.Tensor`
- **Output**: Always returns `torch.Tensor`
- **Plotting**: Automatically converts to NumPy for matplotlib/plotly
- **Devices**: CPU, CUDA, MPS (Apple Silicon), etc.

## Testing

Both versions produce numerically identical results to the original NumPy implementation (within floating-point precision).

## Next Steps

Users can:
1. Use pure version for analysis (drop-in replacement for NumPy/CuPy version)
2. Use nn.Module version for ML applications
3. Experiment with learnable wavelets for signal discovery
4. Integrate into existing PyTorch pipelines
5. Extend with custom loss functions for wavelet optimization

## Requirements

- `torch >= 1.10.0` (for complex tensor support)
- `numpy` (for utilities and plotting)
- `matplotlib` (optional, for plotting)
- `plotly` (optional, for interactive plotting)
