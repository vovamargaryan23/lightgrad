import numpy as np


class Tensor:
    def __init__(self, data, _children=(), requires_grad = False):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._backward = lambda: None
        self.requires_grad = requires_grad

    def _unbroadcast(self, grad, shape):
        ...

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))
        out.requires_grad = self.requires_grad or other.requires_grad

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))
        out.requires_grad = self.requires_grad or other.requires_grad

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other))
        out.requires_grad = self.requires_grad or other.requires_grad

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(other.data @ self.data, (self, other))
        out.requires_grad = self.requires_grad or other.requires_grad

        def _backward():
            if self.requires_grad:
                self.grad += other.data.T @ out.grad
            if other.requires_grad:
                other.grad += out.grad @ self.data.T

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

# TODO: Add broadcasting
