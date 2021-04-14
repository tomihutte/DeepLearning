import numpy as np


class Activation:
    def __init__(self):
        pass

    def __call__(self):
        pass

    def gradient(self):
        pass


class ReLU(Activation):
    def __call__(self, scores, val=0):
        return np.maximum(val, scores)

    def gradient(self, scores, val=0):
        return np.heaviside(scores, 0) + val * np.heaviside(-scores, 0)


class Sigmoid(Activation):
    def __call__(self, scores):
        return 1 / (1 + np.exp(-scores))

    def gradient(self, scores):
        sigm = 1 / (1 + np.exp(-scores))
        return (1 - sigm) * sigm


class Identity(Activation):
    def __call__(self, scores):
        return scores

    def gradient(self, scores):
        return 1


class Tanh(Activation):
    def __call__(self, scores):
        return np.tanh(scores)

    def gradient(self, scores):
        return 1 - np.tanh(scores) ** 2

