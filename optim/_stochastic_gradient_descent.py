from typing import Sequence

import numpy as np

from nn import Parameter
from optim import BaseOptimizer


class StochasticGradientDescent(BaseOptimizer):
    def __init__(self, parameters: Sequence[Parameter], lr: float, momentum: float | None = None):
        super().__init__(parameters, lr)
        if momentum is None:
            self._step_function = self._step_through_default_sgd
        else:
            if not 0.0 <= momentum < 1.0:
                raise ValueError("momentum must be in [0, 1)")
            self.momentum = momentum
            self._velocities = {p: np.zeros_like(p.data) for p in self.parameters}
            self._step_function = self._step_through_sgd_with_momentum

    def _step_through_default_sgd(self, p):
        grad = p.grad
        p.data -= self.learning_rate * grad

    def _step_through_sgd_with_momentum(self, p):
        v = self._velocities[p]

        v *= self.momentum
        v += p.grad

        p.data -= self.learning_rate * v

    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue

            self._step_function(p)
