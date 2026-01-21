import unittest
import numpy as np
from core.tensor import Tensor


class TestTensorOps(unittest.TestCase):
    def test_add_broadcasting(self):
        """Tests matrix + vector broadcasting and gradient unbroadcasting."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([1, 1], requires_grad=True)
        c = a + b
        c.backward()

        expected_out = np.array([[2, 3], [4, 5]])
        np.testing.assert_allclose(c.data, expected_out)

        np.testing.assert_allclose(b.grad, np.array([2, 2]))

    def test_matmul_simple(self):
        """Tests simple 2D matrix multiplication gradients."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[5.0], [6.0]], requires_grad=True)
        c = a @ b  # Result: [[17], [39]]
        c.backward()

        expected_a_grad = np.array([[5.0, 6.0], [5.0, 6.0]])
        np.testing.assert_allclose(a.grad, expected_a_grad)

    def test_relu_mask(self):
        """Tests that gradients are zeroed for negative inputs."""
        a = Tensor([-2.0, 5.0], requires_grad=True)
        b = a.relu()
        b.backward()

        np.testing.assert_allclose(a.grad, np.array([0.0, 1.0]))

    def test_gradient_accumulation(self):
        """Tests the 'x + x' problem (Multivariable Chain Rule)."""
        a = Tensor([10.0], requires_grad=True)
        b = a + a + a
        b.backward()

        self.assertEqual(a.grad[0], 3.0)

    def test_complex_graph(self):
        """Tests a chain of multiple operations (Pow, Add, Mean)."""
        x = Tensor([2.0, 4.0], requires_grad=True)
        y = x ** 2
        z = y + 5
        loss = z.mean()
        loss.backward()

        self.assertEqual(loss.data, 12.5)

        np.testing.assert_allclose(x.grad, np.array([2.0, 4.0]))

    def test_reduction_axis(self):
        """Tests that sum(axis=...) handles shapes correctly."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = a.sum(axis=0)  # [4, 6]
        b.backward()

        np.testing.assert_allclose(a.grad, np.ones_like(a.data))


if __name__ == '__main__':
    unittest.main()