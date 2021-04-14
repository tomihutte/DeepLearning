import numpy as np


class Loss:
    def __init__(self):
        pass

    def __call__(self):
        pass

    def gradient(self):
        pass


class MSE(Loss):
    def __call__(self, scores, y_true):
        index = np.arange(y_true.shape[0])
        scores_aux = np.copy(scores)
        scores_aux[index, y_true] -= 1
        return np.mean(np.sum(scores_aux ** 2, axis=1))

    def gradient(self, scores, y_true):
        index = np.arange(y_true.shape[0])
        scores_aux = np.copy(scores)
        scores_aux[index, y_true] -= 1
        return 2 * scores_aux / y_true.shape[0]


class CCE(Loss):
    def __call__(self, scores, y_true):
        scores_aux = np.copy(scores)
        scores_aux -= scores_aux.max(axis=1)[:, np.newaxis]
        index = np.arange(y_true.shape[0])
        scores_true = scores_aux[index, y_true]
        scores_aux = np.exp(scores_aux)
        return (-scores_true + np.log(scores_aux.sum(axis=1))).mean()

    def gradient(self, scores, y_true):
        scores_aux = np.copy(scores)
        scores_aux = np.exp(scores_aux)
        scores_sum = scores_aux.sum(axis=1)
        scores_aux /= scores_sum[:, np.newaxis]
        index = np.arange(y_true.shape[0])
        scores_aux[index, y_true] -= 1
        return scores_aux / y_true.shape[0]


class MSE_XOR(Loss):
    def __call__(self, scores, y_true):
        return np.mean((scores - y_true) ** 2)

    def gradient(self, scores, y_true):
        return 2 * (scores - y_true) / len(y_true)

