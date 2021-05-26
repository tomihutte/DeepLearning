from keras.datasets import cifar10, mnist
import numpy as np
import time

start = time.time()
n = 20  # cantidad de datos a testear entre 1-10.000
k = 4  # numero de vecinos a tomar


class KVecinosCercanos:
    def __init__(self):
        self.X = None
        self.Y = None

    def train(self, X, Y):
        self.im_shape = X.shape[1:]
        self.X = np.reshape(X, (X.shape[0], np.prod(self.im_shape)))
        self.Y = Y

    def predict(self, X, k):
        assert self.X is not None, "Hay que entrenar primero"
        Yp = np.zeros(X.shape[0], np.uint8)
        for idx in range(X.shape[0]):
            norm = np.linalg.norm(self.X - X[idx].ravel(), axis=-1)
            kmin = norm.argsort()[:k]
            Yp[idx] = np.bincount(self.Y[kmin]).argmax()
        return Yp


(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = y_train.ravel()
x_train = x_train.astype(np.int16)

D = KVecinosCercanos()
D.train(x_train, y_train)

y_pred = D.predict(x_test[:n], k)
l = np.mean(y_pred == y_test[:n].ravel())
print("El porcentaje de aciertos usando k={}, es {}%".format(k, l * 100))
print("Tardó {} s".format(time.time() - start))

