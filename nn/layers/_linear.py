from nn import Module, InitType, ParameterInitializer


class Linear(Module):
    def __init__(self, input_size, output_size, initialization_type: InitType = InitType.XAVIER_NORMAL):
        super(Linear, self).__init__()

        self._weights = ParameterInitializer.initialize(initialization_type, (input_size, output_size))
        self._biases = ParameterInitializer.initialize(InitType.ZEROS, (output_size,))

    def forward(self, x_param):
        return x_param @ self._weights + self._biases

    def parameters(self, recurse=True):
        return [self._weights, self._biases]
