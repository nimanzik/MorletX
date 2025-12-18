# MorletX Implementation Comparison

## Quick Reference Table

| Feature | NumPy/CuPy Original | PyTorch Pure | PyTorch nn.Module |
|---------|---------------------|--------------|-------------------|
| **File** | `_core.py` | `_core_torch.py` | `_core_torch_nn.py` |
| **GPU Support** | Via CuPy | Via PyTorch | Via PyTorch |
| **Device Management** | `array_engine` param | `device` param | `.cuda()`, `.to()` methods |
| **Learnable Parameters** | ❌ No | ❌ No | ✅ **Yes** |
| **Gradient Computation** | ❌ No | ⚠️ Manual | ✅ Automatic |
| **nn.Module Integration** | ❌ No | ❌ No | ✅ Yes |
| **State Dict Save/Load** | ❌ No | ❌ No | ✅ Yes |
| **Memory Overhead** | Minimal | Minimal | Small (nn.Module) |
| **Input Types** | NumPy/CuPy | NumPy/Torch | NumPy/Torch |
| **Output Type** | NumPy/CuPy | Torch | Torch |
| **Backpropagation** | ❌ No | ⚠️ Manual | ✅ Automatic |
| **Optimizer Compatible** | ❌ No | ❌ No | ✅ Yes |
| **Best For** | NumPy workflows | PyTorch analysis | ML/DL pipelines |

## Detailed Comparison

### 1. Original NumPy/CuPy Version

**Pros:**
- Mature, well-tested implementation
- Works with existing NumPy code
- Optional GPU via CuPy
- No PyTorch dependency

**Cons:**
- Separate NumPy/CuPy handling
- No gradient computation
- No learnable parameters
- Manual device management

**Use When:**
- Working in NumPy ecosystem
- Don't need PyTorch features
- Legacy code compatibility

### 2. PyTorch Pure Version

**Pros:**
- Unified CPU/GPU interface
- Native PyTorch tensors
- Minimal overhead
- Gradients available (manual)
- Drop-in PyTorch replacement

**Cons:**
- Not an nn.Module
- No automatic parameter management
- Manual gradient handling
- No state dict support

**Use When:**
- Pure signal processing in PyTorch
- Don't need learnable parameters
- Want minimal overhead
- Analysis/feature extraction

### 3. PyTorch nn.Module Version

**Pros:**
- **Learnable wavelet parameters**
- Full nn.Module integration
- Automatic gradient computation
- Easy device management
- State dict save/load
- Optimizer compatible
- Can be part of larger models

**Cons:**
- Slightly more memory overhead
- Requires understanding of nn.Module
- Overkill for pure analysis

**Use When:**
- **Want to learn optimal wavelets**
- Building end-to-end systems
- Integrating with neural networks
- Need model serialization
- Optimizing wavelet parameters

## Code Examples

### Device Management

```python
# Original
wavelet = MorletWavelet(..., array_engine="cupy")

# Pure PyTorch
wavelet = MorletWavelet(..., device="cuda")

# nn.Module PyTorch
wavelet = MorletWavelet(..., device="cuda")
wavelet = wavelet.cuda()  # Also works
wavelet = wavelet.to('cuda:1')  # Specific GPU
```

### Learnable Parameters

```python
# Original - NOT POSSIBLE
# Pure PyTorch - NOT POSSIBLE

# nn.Module PyTorch - ONLY VERSION WITH THIS FEATURE
wavelet = MorletWavelet(
    center_freq=10.0,
    shape_ratio=5.0,
    duration=1.0,
    sampling_freq=100.0,
    learnable_center_freq=True,  # ✅ Learn this!
    learnable_shape_ratio=True,  # ✅ Learn this!
)

optimizer = torch.optim.Adam(wavelet.parameters())
# Now you can optimize wavelet parameters!
```

### Model Integration

```python
# Original - NOT POSSIBLE
# Pure PyTorch - MANUAL INTEGRATION

# nn.Module PyTorch - SEAMLESS
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavelet = MorletWavelet(...)  # Just add it!
        self.classifier = nn.Linear(100, 10)
    
    def forward(self, x):
        features = self.wavelet(x)
        return self.classifier(features)

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())  # Includes wavelet params!
```

### Save/Load

```python
# Original
import pickle
with open('wavelet.pkl', 'wb') as f:
    pickle.dump(wavelet, f)

# Pure PyTorch
# No built-in support, need custom serialization

# nn.Module PyTorch
torch.save(wavelet.state_dict(), 'wavelet.pth')
wavelet.load_state_dict(torch.load('wavelet.pth'))
```

## Performance Comparison

| Aspect | Original | Pure PyTorch | nn.Module PyTorch |
|--------|----------|--------------|-------------------|
| CPU Speed | Baseline | ~Same | ~Same |
| GPU Speed | Fast (CuPy) | Fast | Fast |
| Memory | Baseline | ~Same | +5-10% overhead |
| Batch Processing | Good | Good | Good |
| Gradient Computation | N/A | Available | Optimized |

## Migration Path

### From NumPy/CuPy to Pure PyTorch

```python
# Before
from morletx._core import MorletWavelet
wavelet = MorletWavelet(..., array_engine="numpy")
coeffs = wavelet.transform(signal)

# After
from morletx._core_torch import MorletWavelet
wavelet = MorletWavelet(..., device="cpu")
coeffs = wavelet.transform(signal)
```

### From Pure PyTorch to nn.Module

```python
# Before
from morletx._core_torch import MorletWavelet
wavelet = MorletWavelet(..., device="cuda")

# After
from morletx._core_torch_nn import MorletWavelet
wavelet = MorletWavelet(..., device="cuda")
# API is identical, but now you have nn.Module features!
```

## Recommendation Matrix

| Your Situation | Recommended Version |
|----------------|---------------------|
| NumPy-based analysis | Original |
| PyTorch analysis, fixed wavelets | Pure PyTorch |
| Learning optimal frequencies | **nn.Module PyTorch** |
| Part of neural network | **nn.Module PyTorch** |
| End-to-end training | **nn.Module PyTorch** |
| Need model checkpointing | **nn.Module PyTorch** |
| Minimal dependencies | Original |
| Maximum flexibility | **nn.Module PyTorch** |

## Summary

- **Original**: Best for NumPy workflows
- **Pure PyTorch**: Best for PyTorch analysis with fixed wavelets
- **nn.Module PyTorch**: **Best for ML/DL applications and learnable wavelets**

The nn.Module version is the most powerful and flexible, with the only downside being slightly more complexity if you just need simple signal analysis.
