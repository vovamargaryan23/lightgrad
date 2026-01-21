class OpBase:
    @staticmethod
    def forward(*inputs):
        raise NotImplementedError

    @staticmethod
    def backward(*inputs):
        raise NotImplementedError