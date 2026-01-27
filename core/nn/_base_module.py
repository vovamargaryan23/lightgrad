from abc import ABC, abstractmethod

from core.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module(ABC):
    def __init__(self):
        self._parameters = {}
        self._sub_modules = {}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._sub_modules[name] = value

        super().__setattr__(name, value)

    def parameters(self, recurse=True):
        if not recurse:
            return self._parameters

        params = []

        params.extend(self._parameters.values())

        for sub_module in self._sub_modules.values():
            params.extend(sub_module.parameters())

        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError
