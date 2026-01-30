from optim import BaseOptimizer


class StochasticGradientDescent(BaseOptimizer):
    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue

            grad = p.grad
            p.data -= self.learning_rate * grad
