import unittest
import numpy as np

from core.tensor import Tensor

class TestTensorOps(unittest.TestCase):
    def assertArrayEqual(self, a, b):
        np.testing.assert_array_almost_equal(a, b)

    def test_get_validated_other_object_and_error(self):
        t = Tensor._get_validated_other_object(5)
        self.assertIsInstance(t, Tensor)
        self.assertArrayEqual(t.data, np.array(5, dtype=t.dtype))
        with self.assertRaises(TypeError):
            Tensor._get_validated_other_object(object())

    def test_add_and_backward(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        c = a + b
        self.assertArrayEqual(c.data, np.array([4.0, 6.0], dtype=c.dtype))
        s = c.sum()
        s.backward()
        self.assertArrayEqual(a.grad, np.array([1.0,1.0], dtype=a.dtype))
        self.assertArrayEqual(b.grad, np.array([1.0,1.0], dtype=b.dtype))

    def test_mul_and_unbroadcast_and_requires_grad(self):
        a = Tensor([[1.0],[2.0]], requires_grad=True)  # shape (2,1)
        b = Tensor([3.0,4.0], requires_grad=True)       # shape (2,)
        c = a * b
        expected = np.array([[3.0,4.0],[6.0,8.0]], dtype=a.dtype)
        self.assertArrayEqual(c.data, expected)
        s = c.sum()
        s.backward()
        self.assertArrayEqual(a.grad, np.array([[7.0],[7.0]], dtype=a.dtype))
        self.assertArrayEqual(b.grad, np.array([3.0,3.0], dtype=b.dtype))

    def test_neg_sub_rsub_and_radd(self):
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)
        r = -a + b
        s = r.sum()
        s.backward()
        self.assertArrayEqual(a.grad, np.array([-1.0], dtype=a.dtype))
        self.assertArrayEqual(b.grad, np.array([1.0], dtype=b.dtype))
        c = a + 5
        self.assertArrayEqual(c.data, np.array([7.0], dtype=c.dtype))
        d = 10 - a
        self.assertArrayEqual(d.data, np.array([8.0], dtype=d.dtype))

    def test_matmul_and_backward(self):
        a = Tensor(np.arange(6).reshape(2,3).astype(np.float32), requires_grad=True)
        b = Tensor(np.arange(12).reshape(3,4).astype(np.float32), requires_grad=True)
        c = a @ b
        s = c.sum()
        s.backward()
        expected_da = np.ones_like(c.data) @ b.data.swapaxes(-1,-2)
        expected_db = a.data.swapaxes(-1,-2) @ np.ones_like(c.data)
        self.assertArrayEqual(a.grad, expected_da)
        self.assertArrayEqual(b.grad, expected_db)

    def test_power_exponent_log_and_backwards(self):
        a = Tensor(2.0, requires_grad=True)
        p = a ** 3
        self.assertArrayEqual(p.data, np.array(8.0, dtype=p.dtype))
        s = p.sum()
        s.backward()
        self.assertArrayEqual(a.grad, np.array(12.0, dtype=a.dtype))
        a.zero_grad()
        e = a.exp()
        s2 = e.sum()
        s2.backward()
        expected = np.exp(2.0)
        self.assertArrayEqual(a.grad, np.array(expected, dtype=a.dtype))
        a.zero_grad()
        l = a.log()
        s3 = l.sum()
        s3.backward()
        expected_ln = 1.0 / 2.0
        self.assertArrayEqual(a.grad, np.array(expected_ln, dtype=a.dtype))
        with self.assertRaises(AssertionError):
            _ = a ** "bad"

    def test_relu(self):
        a = Tensor(np.array([-1.0, 0.0, 2.0]), requires_grad=True)
        r = a.relu()
        self.assertArrayEqual(r.data, np.array([0.0, 0.0, 2.0], dtype=r.dtype))
        s = r.sum()
        s.backward()
        self.assertArrayEqual(a.grad, np.array([0.0, 0.0, 1.0], dtype=a.dtype))

    def test_sum_mean_axis_keepdims_variants(self):
        a = Tensor(np.array([[1.0,2.0],[3.0,4.0]]), requires_grad=True)
        s = a.sum(axis=0, keepdims=False)
        grad = np.array([1.0,1.0], dtype=s.dtype)
        s.backward(grad)
        self.assertArrayEqual(a.grad, np.array([[1.0,1.0],[1.0,1.0]], dtype=a.dtype))
        a.zero_grad()
        s2 = a.sum(axis=1, keepdims=True)
        s2.backward(np.ones_like(s2.data))
        self.assertArrayEqual(a.grad, np.ones_like(a.data))
        a.zero_grad()
        m = a.mean(axis=0, keepdims=False)
        m.backward(np.array([1.0,1.0], dtype=m.dtype))
        N = a.data.size / m.data.size
        expected = np.ones_like(a.data) / N
        self.assertArrayEqual(a.grad, expected)
        a.zero_grad()
        m2 = a.mean()
        m2.backward()
        N2 = a.data.size / m2.data.size
        expected2 = np.ones_like(a.data) / N2
        self.assertArrayEqual(a.grad, expected2)

    def test_transpose_and_backward(self):
        a = Tensor(np.arange(6).reshape(2,3), requires_grad=True)
        t = a.transpose()
        grad = np.ones_like(t.data)
        t.backward(grad)
        self.assertArrayEqual(a.grad, np.transpose(grad))
        a.zero_grad()
        t2 = a.transpose(axes=(1,0))
        g2 = np.ones_like(t2.data)
        t2.backward(g2)
        inv = np.argsort((1,0))
        self.assertArrayEqual(a.grad, np.transpose(g2, axes=inv))

    def test_reshape_backward_and_zero_grad(self):
        a = Tensor(np.arange(6).reshape(2,3).astype(np.float32), requires_grad=True)
        r = a.reshape((6,))
        grad = np.arange(6).astype(np.float32)
        r.backward(grad)
        self.assertArrayEqual(a.grad, grad.reshape(a.data.shape))
        a.zero_grad()
        self.assertArrayEqual(a.grad, np.zeros_like(a.data))

    def test_repr_and_topology(self):
        a = Tensor([1.0])
        s = str(a)
        self.assertIn("Tensor(data=", s)
        topo = a._get_current_topology()
        self.assertIsInstance(topo, list)
        self.assertTrue(len(topo) >= 1)

    def test_requires_grad_flag_behavior(self):
        x = Tensor(2.0, requires_grad=True)
        y = Tensor(3.0, requires_grad=False)
        z = x * y
        self.assertTrue(z.requires_grad)
        s = z.sum()
        s.backward()
        self.assertArrayEqual(x.grad, np.array(3.0, dtype=x.dtype))
        self.assertArrayEqual(y.grad, np.zeros_like(y.data))

if __name__ == "__main__":
    unittest.main()
