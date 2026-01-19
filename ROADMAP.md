# LightGrad: High-Performance Autograd Engine

## Phase 1: The Tensor Core

- [X] **N-Dimensional Tensor Class**: Build a wrapper around NumPy (initially) that supports `.data` and `.grad`.
- [ ] **Topological Sort**: Implement a robust DAG walker for the `backward()` call to handle complex graphs without recursion limits.
- [X] **Broadcasting Engine**: Manually implement broadcasting logic for element-wise ops (Addition, Sub, Mul).
- [X] **Matrix Multiplication (MatMul)**: Implement the derivative for $Y = A \times B$ (this is the core of all NNs).
- [ ] **Fundamental Ops**: Implement `Sum`, `Mean`, `Max`, and `Transpose` with their respective gradient formulas.

## Phase 2: The Deep Learning API

- [ ] **The `nn.Module` Abstraction**: Create a base class for Layers (`Linear`, `Conv2d`, `LayerNorm`) that tracks parameters.
- [ ] **Weight Initialization**: Implement Xavier/Glorot and Kaiming/He initialization (critical for deep net stability).
- [ ] **Modern Optimizers**: 
    - [ ] Standard SGD with Momentum.
    - [ ] **AdamW**: The industry standard for Transformers (includes weight decay).
- [ ] **Activation Suite**: Implement ReLU, GELU (for Transformers), and Softmax.
- [ ] **Loss Functions**: Cross-Entropy with "Log-Sum-Exp" trick for numerical stability.

## Phase 3: Hardware & Systems Alpha

- [ ] **OpenAI Triton Integration**: 
    - [ ] Write a custom Triton Kernel for a "Fused MLP" (MatMul + ReLU + LayerNorm).
    - [ ] Benchmark your Triton Kernel vs. your naive NumPy implementation.
- [ ] **Mixed Precision (FP16/BF16)**: Implement a `GradScaler` to handle small gradients when training in half-precision.
- [ ] **Rust/C++ Extension**: 
    - [ ] Rewrite the "bottleneck" (usually the MatMul or Conv loop) in Rust or C++.
    - [ ] Use PyO3 (for Rust) or pybind11 (for C++) to link it back to your Python engine.
- [ ] **Memory Management**: Implement a simple "Grad Accumulation" feature to train large models on small GPUs.

## Phase 4: Validation & Benchmarking

- [ ] **The "Micro-GPT" Test**: Train a 3-layer Transformer on a tiny dataset (Shakespeare) using only your engine.
- [ ] **The "ResNet" Test**: Implement a basic ConvNet and train it on MNIST/CIFAR-10.
- [ ] **Profiling Artifacts**: Generate a PDF/README section showing **FLOPs** and **Memory Throughput** comparisons between your engine and PyTorch.
- [ ] **Graph Visualization**: Auto-generate a `.svg` of the computational graph for a single Transformer block using Graphviz.