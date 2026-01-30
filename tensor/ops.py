from abc import ABC, abstractmethod
import numpy as np


class OpBase(ABC):
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = []

    def save_tensors(self, *tensors):
        self.saved_tensors.extend(tensors)

    @abstractmethod
    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, *inputs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def op_repr(cls):
        raise NotImplementedError

    @classmethod
    def apply(cls, *tensors, **kwargs):
        ctx = cls(*tensors)

        raw_data = [t.data for t in tensors]
        ret_data = ctx.forward(*raw_data, **kwargs)

        # Imported here to avoid circular import error
        from tensor import Tensor
        result = Tensor(ret_data, _children=tensors, op=cls.op_repr())

        result.grad_func = ctx
        result.requires_grad = any(t.requires_grad for t in tensors)

        return result


class AddOp(OpBase):
    @classmethod
    def op_repr(cls):
        return "+"

    def forward(self, a, b, **kwargs):
        return a + b

    def backward(self, output_grad):
        op1 = output_grad
        op2 = output_grad

        return op1, op2


class MultiplyOp(OpBase):
    @classmethod
    def op_repr(cls):
        return "*"

    def forward(self, a, b, **kwargs):
        self.save_tensors(a, b)
        return a * b

    def backward(self, output_grad):
        a, b = self.saved_tensors

        op1 = output_grad * b
        op2 = output_grad * a

        return op1, op2


class MatMulOp(OpBase):
    @classmethod
    def op_repr(cls):
        return "@"

    def forward(self, a, b, **kwargs):
        self.save_tensors(a, b)
        return a @ b

    def backward(self, output_grad):
        a, b = self.saved_tensors

        swapped_b = b.swapaxes(-1, -2)
        swapped_a = a.swapaxes(-1, -2)

        op1 = output_grad @ swapped_b
        op2 = swapped_a @ output_grad

        return op1, op2


class PowerOp(OpBase):
    def __init__(self, *tensors):
        super().__init__(*tensors)
        self.power = None

    @classmethod
    def op_repr(cls):
        return "^"

    def forward(self, a, **kwargs):
        self.save_tensors(a)
        self.power = kwargs.get("power")
        return a ** self.power

    def backward(self, output_grad):
        a, = self.saved_tensors

        op1 = self.power * a**(self.power - 1) * output_grad

        return (op1,)


class ExponentOp(OpBase):
    @classmethod
    def op_repr(cls):
        return "exp"

    def forward(self, a, **kwargs):
        self.save_tensors(a)
        return np.exp(a)

    def backward(self, output_grad):
        a, = self.saved_tensors

        op1 = output_grad * np.exp(a)

        return (op1,)


class NaturalLogOp(OpBase):
    @classmethod
    def op_repr(cls):
        return "ln"

    def forward(self, a, **kwargs):
        self.save_tensors(a)
        return np.log(a)

    def backward(self, output_grad):
        a, = self.saved_tensors
        op1 = output_grad * (1/a)

        return (op1,)


class ReluOp(OpBase):
    @classmethod
    def op_repr(cls):
        return "ReLU"

    def forward(self, a, **kwargs):
        self.save_tensors(a)
        return np.maximum(0, a)

    def backward(self, output_grad):
        a, = self.saved_tensors

        op1 = output_grad * (a > 0)

        return (op1,)


class SumOp(OpBase):
    def __init__(self, *tensors):
        super().__init__(*tensors)
        self.keepdims = None
        self.axis = None

    @classmethod
    def op_repr(cls):
        return "Sum"

    def forward(self, a, **kwargs):
        self.axis = kwargs.get("axis")
        self.keepdims = kwargs.get("keepdims", False)

        self.save_tensors(a)
        return np.sum(a, axis=self.axis, keepdims=self.keepdims)

    def backward(self, output_grad):
        a, = self.saved_tensors

        if self.keepdims is False and self.axis is not None:
            output_grad = np.expand_dims(output_grad, axis=self.axis)

        op = output_grad * np.ones_like(a)

        return (op,)


class MeanOp(OpBase):
    def __init__(self, *tensors):
        super().__init__(*tensors)
        self.keepdims = None
        self.axis = None

    @classmethod
    def op_repr(cls):
        return "Mean"

    def forward(self, a, **kwargs):
        self.axis = kwargs.get("axis")
        self.keepdims = kwargs.get("keepdims", False)
        
        self.save_tensors(a)
        return np.mean(a, axis=self.axis, keepdims=self.keepdims)

    def backward(self, output_grad):
        a, = self.saved_tensors

        if self.keepdims is False and self.axis is not None:
            output_grad = np.expand_dims(output_grad, axis=self.axis)

        divisor = output_grad * np.ones_like(a)
        N = a.size / output_grad.size
        op = np.divide(divisor, N)

        return (op,)


class TransposeOp(OpBase):
    def __init__(self, *tensors):
        super().__init__(*tensors)
        self.axes = None

    @classmethod
    def op_repr(cls):
        return "^T"

    def forward(self, a, **kwargs):
        self.save_tensors(a)
        self.axes = kwargs.get("axes")

        return np.transpose(a, axes=self.axes)

    def backward(self, output_grad):
        if self.axes is None:
            return np.transpose(output_grad)

        inv_axes = np.argsort(self.axes)
        return (np.transpose(output_grad, axes=inv_axes),)


class ReshapeOp(OpBase):
    def __init__(self, *tensors):
        super().__init__(*tensors)
        self.shape = None

    @classmethod
    def op_repr(cls):
        return "Reshape"

    def forward(self, a, **kwargs):
        self.save_tensors(a)
        self.shape = kwargs.get("shape")

        return np.reshape(a, shape=self.shape)

    def backward(self, output_grad):
        a, = self.saved_tensors
        return (output_grad.reshape(a.shape),)
