import unittest

import numpy as np
from core.nn import Module, Parameter


class DummyModule(Module):
    def forward(self, x):
        return x


class NNModuleTest(unittest.TestCase):
    def test_module_registers_parameters(self):
        m = DummyModule()
        m.weight = Parameter(np.zeros((3, 3)))

        params = list(m.parameters())
        self.assertEqual(len(params), 1)
        self.assertEqual(m.weight, params[0])

    def test_module_parameters_recursive(self):
        parent = DummyModule()
        child = DummyModule()

        parent.child = child
        child.bias = Parameter(np.zeros(5))

        params = list(parent.parameters())

        self.assertEqual(len(params), 1)
        self.assertEqual(child.bias, params[0])

    def test_zero_grad_calls_all_parameters(self):
        m = DummyModule()
        p = Parameter(np.ones(3))
        m.p = p

        p.grad = np.ones(3)
        m.zero_grad()

        np.testing.assert_equal(p.grad, 0)