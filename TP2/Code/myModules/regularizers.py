import numpy as np


class Regularizer:
    def __init__(self, lambd=1e-2):
        self.lambd = lambd

    def __call__(self):
        pass

    def gradient(self):
        pass


class L2(Regularizer):
    def __call__(self, W):
        return self.lambd * np.sum(W ** 2)

    def gradient(self, W):
        return 2 * self.lambd * W


class L1(Regularizer):
    def __call__(self, W):
        return self.lambd * np.sum(np.abs(W))

    def gradient(self, W):
        return self.lambd * np.sign(W)

