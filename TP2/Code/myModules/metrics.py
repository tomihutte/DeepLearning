import numpy as np


def mse(scores, y_true):
    index = np.arange(y_true.shape[0])
    scores_aux = np.copy(scores)
    scores_aux[index, y_true] -= 1
    return np.mean(np.sum(scores_aux ** 2, axis=1))


def accuracy(scores, y_true):
    y_pred = np.argmax(scores, axis=1)
    return np.mean(y_pred == y_true)


def accuracy_xor(scores, y_true, val=0.9):
    scores_aux = np.copy(scores)
    scores_aux[scores > val] = 1
    scores_aux[scores < -val] = -1
    return np.mean(scores_aux == y_true)

