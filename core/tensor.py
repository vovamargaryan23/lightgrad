import numpy as np


class Tensor:
    def __init__(self, data, _children=(), requires_grad = False, op = "", dtype = np.float32):
        self.dtype = dtype
        self.data = np.array(data, dtype=self.dtype)
        self.grad = np.zeros_like(self.data, dtype=self.dtype)
        self._prev = set(_children)
        self._backward = lambda: None
        self.requires_grad = requires_grad
        self._op = op

    @classmethod
    def _get_validated_other_object(cls, other, default_type=np.float32):
        if isinstance(other, cls):
            return other

        try:
            data = np.array(other, dtype=default_type)
            return cls(data)
        except Exception as e:
            raise TypeError(
                f"Cannot implicitly convert {type(other)} to Tensor. "
                f"Supported types: [Tensor, np.ndarray, int, float, list, tuple]. "
                f"Error: {e}"
            )

    @classmethod
    def _unbroadcast(cls, grad, shape):
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        for i, (g_dim, t_dim) in enumerate(zip(grad.shape, shape)):
            if t_dim == 1 and g_dim > 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __add__(self, other):
        other = self._get_validated_other_object(other)
        out = Tensor(self.data + other.data, (self, other), op="+")
        out.requires_grad = self.requires_grad or other.requires_grad

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._get_validated_other_object(other)
        out = Tensor(self.data * other.data, (self, other), op="*")
        out.requires_grad = self.requires_grad or other.requires_grad

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = self._get_validated_other_object(other)
        out = Tensor(self.data @ other.data, (self, other), op="@")
        out.requires_grad = self.requires_grad or other.requires_grad

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        other = self._get_validated_other_object(other)
        out = Tensor(other.data @ self.data, (self, other), op="@")
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

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Currently only int/float are supported!"
        out = Tensor(np.power(self.data, power), (self, ), op="^")
        out.requires_grad = self.requires_grad

        def _backward():
            if self.requires_grad:
                self.grad += power * np.power(self.data, (power - 1)) * out.grad

        out._backward = _backward

        return out

    def _get_current_topology(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        return topo

    def backward(self):
        topo = self._get_current_topology()

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        topo = self._get_current_topology()

        for v in reversed(topo):
            v.grad = np.zeros_like(v.data)

    def exp(self):
        out = Tensor(np.exp(self.data), (self, ), op="exp")
        out.requires_grad = self.requires_grad

        def _backward():
            self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self, ), op="ln")
        out.requires_grad = self.requires_grad

        def _backward():
            self.grad += out.grad * (1/self.data)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), op="ReLU")
        out.requires_grad = self.requires_grad

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad

        out._backward = _backward

        return out