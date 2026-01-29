import numpy as np

from enum import StrEnum
from core.nn import Parameter


class InitType(StrEnum):
    XAVIER_NORMAL = "xavier_normal"
    XAVIER_UNIFORM = "xavier_uniform"
    KAIMING_NORMAL = "kaiming_normal"
    KAIMING_UNIFORM = "kaiming_uniform"
    NORMAL = "normal"
    ZEROS = "zeros"


class ParameterInitializer:
    @staticmethod
    def _xavier_normal(dims, random_state):
        """
        initialize values with Xavier/Glorot normal initialization for given dimensions
        :param dims: tuple of ints representing each dimension
        :return: numpy.array
        """
        fan_in, fan_out = ParameterInitializer._calculate_fans(dims)
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return random_state.normal(0, std, size=dims)

    @staticmethod
    def _xavier_uniform(dims, random_state):
        """
        initialize values with Xavier/Glorot uniform initialization for given dimensions
        :param dims: tuple of ints representing each dimension
        :return: numpy.array
        """
        fan_in, fan_out = ParameterInitializer._calculate_fans(dims)

        bound = np.sqrt(6/(fan_in + fan_out))

        return random_state.uniform(-bound, bound, size=dims)

    @staticmethod
    def _calculate_fans(dims):
        if len(dims) < 1:
            raise ValueError(f"Invalid dimensions provided: {dims}")
        elif len(dims) == 1:
            return dims[0], dims[0]
        elif len(dims) == 2:
            #(in_channels, out_channels)
            return dims[0], dims[1]
        else:
            # the input and output positions are changed for N-D array
            # (out_channels, in_channels, ...)
            receptive_field_size = np.prod(dims[2:])
            fan_in = dims[1] * receptive_field_size
            fan_out = dims[0] * receptive_field_size
            return fan_in, fan_out

    @staticmethod
    def _kaiming_normal(dims, random_state):
        """
        initialize values with Kaiming/He normal initialization for given dimensions
        :param dims: tuple of ints representing each dimension
        :return: numpy.array
        """
        fan_in, _ = ParameterInitializer._calculate_fans(dims)
        std = np.sqrt(2.0 / fan_in)
        return random_state.normal(0, std, size=dims)

    @staticmethod
    def _kaiming_uniform(dims, random_state):
        """
        initialize values with Kaiming/He uniform initialization for given dimensions
        :param dims: tuple of ints representing each dimension
        :return: numpy.array
        """
        fan_in, _ = ParameterInitializer._calculate_fans(dims)

        bound = np.sqrt(6/fan_in)

        return random_state.uniform(-bound, bound, size=dims)

    @staticmethod
    def _normal(dims, random_state):
        """
        initialize values with normal distribution for given dimensions
        :param dims: tuple of ints representing each dimension
        :return: numpy.array
        """
        return random_state.normal(size=dims)

    @staticmethod
    def _zeros(dims, random_state):
        """
        initialize values with zero values for given dimensions
        :param dims: tuple of ints representing each dimension
        :return: numpy.array
        """
        return np.zeros(dims)

    _ENUM_TO_METHOD_MAPPING = {
        InitType.XAVIER_NORMAL: _xavier_normal,
        InitType.XAVIER_UNIFORM: _xavier_uniform,
        InitType.NORMAL: _normal,
        InitType.KAIMING_NORMAL: _kaiming_normal,
        InitType.KAIMING_UNIFORM: _kaiming_uniform,
        InitType.ZEROS: _zeros
    }

    @classmethod
    def initialize(cls, init_type: InitType, tensor_dims: tuple, random_seed: int = None):
        """
        Initialize Parameter instance with given characteristics
        :param init_type: enum of type InitType
        :param tensor_dims: tuple of ints representing each dimension
        :param random_seed: Optional if given sets the numpy random seed to the given value only for this context
        :return: Parameter with randomly initialized values of given initialization type
        """

        if random_seed is not None:
            state = np.random.RandomState(seed=random_seed)
        else:
            state = np.random

        method = cls._ENUM_TO_METHOD_MAPPING.get(init_type)
        if method is None:
            raise ValueError(f"Unknown initialization type: {init_type}")
        wrapped_parameter = Parameter(method(tensor_dims, random_state=state))

        return wrapped_parameter
