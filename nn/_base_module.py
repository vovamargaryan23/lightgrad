from abc import ABC, abstractmethod

from nn import Parameter


class Module(ABC):
    def __init__(self):
        self._parameters = {}
        self._sub_modules = {}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._sub_modules[name] = value

    def parameters(self, recurse=True):
        for p in self._parameters.values():
            yield p

        if recurse:
            for sub_module in self._sub_modules.values():
                yield from sub_module.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError
