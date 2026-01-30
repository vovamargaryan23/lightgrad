import numpy as np

from tensor.ops import (AddOp,
                        MultiplyOp,
                        MatMulOp,
                        PowerOp,
                        ExponentOp,
                        NaturalLogOp,
                        ReluOp,
                        SigmoidOp,
                        SumOp,
                        MeanOp,
                        TransposeOp,
                        ReshapeOp)


class Tensor:
    def __init__(self, data, _children=(), requires_grad = False, op = "", dtype = np.float32):
        self.dtype = dtype
        self.data = np.array(data, dtype=self.dtype)
        self.grad = np.zeros_like(self.data, dtype=self.dtype)
        self._prev = set(_children)
        self._backward = lambda: None
        self.grad_func = None
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

    def _get_current_topology(self):
        topo = []
        visited = set()

        stack = [(self, False)]

        while stack:
            v, expanded = stack.pop()

            if expanded:
                topo.append(v)
                continue

            if v in visited:
                continue

            visited.add(v)
            stack.append((v, True))

            for child in v._prev:
                if child not in visited:
                    stack.append((child, False))

        return topo

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __add__(self, other):
        other = self._get_validated_other_object(other)
        return AddOp.apply(self, other)

    def __mul__(self, other):
        other = self._get_validated_other_object(other)
        return MultiplyOp.apply(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        other = self._get_validated_other_object(other)
        return MatMulOp.apply(self, other)

    def __rmatmul__(self, other):
        other = self._get_validated_other_object(other)
        return MatMulOp.apply(other, self)

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
        return PowerOp.apply(self, power=power)

    def exp(self):
        return ExponentOp.apply(self)

    def log(self):
        return NaturalLogOp.apply(self)

    def relu(self):
        return ReluOp.apply(self)

    def sigmoid(self):
        return SigmoidOp.apply(self)

    def sum(self, axis=None, keepdims=False):
        return SumOp.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        return MeanOp.apply(self, axis=axis, keepdims=keepdims)

    def transpose(self, axes=None):
        return TransposeOp.apply(self, axes=axes)

    def reshape(self, shape):
        return ReshapeOp.apply(self, shape=shape)

    def backward(self, grad=None):
        topo = self._get_current_topology()

        if grad is None:
            self.grad = np.zeros_like(self.data, dtype=self.dtype)
            if self.data.size == 1:
                self.grad = np.array(1.0, dtype=self.dtype)
            else:
                raise RuntimeError("Grad must be specified for non-scalar tensors!")
        else:
            self.grad = np.array(grad, dtype=self.dtype)

        for v in reversed(topo):
            if v.grad_func is not None:
                grads = v.grad_func.backward(v.grad)

                if not isinstance(grads, tuple):
                    grads = (grads,)

                for parent, g in zip(v.grad_func.parents, grads):
                    if parent.requires_grad:
                        optimized_grad = self._unbroadcast(g, parent.data.shape)

                        parent.grad += optimized_grad

    def zero_grad(self):
        topo = self._get_current_topology()

        for v in reversed(topo):
            v.grad = np.zeros_like(v.data)
