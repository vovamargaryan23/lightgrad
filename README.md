# LightGrad

LightGrad is a from-scratch automatic differentiation engine designed to uncover the "black box" of modern deep learning frameworks. Instead of just calling a library, I built this to implement the underlying multivariate calculus and graph theory that makes neural networks possible.

## Core Engineering Features

* **N-Dimensional Autodiff:** Full support for multidimensional arrays, moving beyond simple scalar-based engines to handle real-world matrix operations.
* **The Broadcasting Puzzle:** Implemented a custom unbroadcasting engine that correctly accumulates gradients when shapes expand and contract during the forward pass.
* **Topological Execution:** Uses a directed acyclic graph (DAG) walker to ensure that every gradient is calculated only after its dependencies are ready.
* **Mathematical Atoms:** Includes a suite of optimized operations like MatMul (batch-safe), ReLU, Power, Exp, and Log.
* **Memory Gating:** Uses a `requires_grad` flag to optimize performance by skipping gradient calculations for fixed parameters.

## A Quick Look

The API is designed to feel familiar to anyone who has used PyTorch.

```python
from lightgrad.core import Tensor
import numpy as np

# Initialize parameters with gradient tracking
x = Tensor([[1.0, 2.0]], requires_grad=True)
w = Tensor([[3.0], [4.0]], requires_grad=True)

# Forward pass
y = x @ w

# Backward pass - the engine builds the graph and calculates gradients
y.backward()

print(f"Result: {y.data}")
print(f"W Gradient: {w.grad}")
```