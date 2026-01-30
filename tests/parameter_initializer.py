import unittest
import numpy as np

from nn import ParameterInitializer, InitType, Parameter


class ParameterInitializerTest(unittest.TestCase):
    def test_calculate_fans_1d(self):
        fan_in, fan_out = ParameterInitializer._calculate_fans((10,))

        self.assertEqual(fan_in, 10)
        self.assertEqual(fan_out, 10)

    def test_calculate_fans_2d(self):
        fan_in, fan_out = ParameterInitializer._calculate_fans((3,7))

        self.assertEqual(fan_in, 3)
        self.assertEqual(fan_out, 7)

    def test_calculate_fans_conv2d(self):
        fan_in, fan_out = ParameterInitializer._calculate_fans((16,3,5,5))

        self.assertEqual(fan_in, 3 * 5 * 5)
        self.assertEqual(fan_out, 16 * 5 * 5)

    def test_calculate_fans_invalid_dims(self):
        self.assertRaises(ValueError, ParameterInitializer._calculate_fans, ())

    def test_initialize_returns_parameter_with_correct_shape(self):
        dims = (4, 5)
        p = ParameterInitializer.initialize(InitType.NORMAL, dims)

        self.assertIsInstance(p, Parameter)
        self.assertEqual(p.data.shape, dims)
        self.assertTrue(p.requires_grad)

    def test_initialize_reproducible_with_seed(self):
        dims = (3,3)
        seed = 123

        p1 = ParameterInitializer.initialize(InitType.XAVIER_NORMAL, dims, random_seed=seed)
        p2 = ParameterInitializer.initialize(InitType.XAVIER_NORMAL, dims, random_seed=seed)

        np.testing.assert_allclose(p1.data, p2.data)

    def test_initialize_does_not_pollute_numpy_random_generator(self):
        default_seed = 999
        np.random.seed(default_seed)

        ParameterInitializer.initialize(
            InitType.KAIMING_UNIFORM, (10,), random_seed=42
        )

        after = np.random.random()

        np.random.seed(default_seed)
        expected_after = np.random.random()

        self.assertEqual(after, expected_after)

    def test_xavier_normal_statistics(self):
        dims = (1000, 1000)
        p = ParameterInitializer.initialize(
            InitType.XAVIER_NORMAL, dims, random_seed=0
        )

        fan_in, fan_out = dims
        expected_std = np.sqrt(2.0 / (fan_in + fan_out))


        self.assertLess(p.data.mean(), 1e-2)
        self.assertLess((p.data.std() - expected_std) / expected_std, 0.1)

    def test_xavier_uniform_bounds(self):
        dims = (128, 256)
        p = ParameterInitializer.initialize(
            InitType.XAVIER_UNIFORM, dims, random_seed=0
        )

        fan_in, fan_out = dims
        bound = np.sqrt(6.0 / (fan_in + fan_out))

        self.assertGreaterEqual(p.data.min(), -bound)
        self.assertLessEqual(p.data.max(), bound)

    def test_kaiming_normal_statistics(self):
        dims = (500, 500)
        p = ParameterInitializer.initialize(
            InitType.KAIMING_NORMAL, dims, random_seed=4
        )

        fan_in, fan_out = dims
        expected_std = np.sqrt(2.0 / fan_in)


        self.assertLess(p.data.mean(), 1e-2)
        self.assertLess((p.data.std() - expected_std) / expected_std, 0.1)

    def test_kaiming_uniform_bounds(self):
        dims = (128, 256)
        p = ParameterInitializer.initialize(
            InitType.KAIMING_UNIFORM, dims, random_seed=4
        )

        fan_in, fan_out = dims
        bound = np.sqrt(6.0 / fan_in)

        self.assertGreaterEqual(p.data.min(), -bound)
        self.assertLessEqual(p.data.max(), bound)

    def test_zeros_initializer(self):
        dims = (10, 10)
        p = ParameterInitializer.initialize(InitType.ZEROS, dims)

        self.assertEqual(np.count_nonzero(p.data), 0)

