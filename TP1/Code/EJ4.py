import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import time

n = 2  # numero de dimensiones de las clases
p = 4  # numero de clases originales
m = 50  # numero de datos por clase
k = 1  # kNN
ncol = 1
varmed = 5
vardata = 0.5


def data2D_create(m, p):
    medias = np.random.multivariate_normal(
        mean=[0, 0], cov=np.identity(2) * varmed, size=p
    )  # se puede cambiar por valores que quieras
    varianzas = (
        np.identity(2) * vardata
    )  # medias y varianzas de cada grupo en todas las dim
    x = np.array(
        [
            np.random.multivariate_normal(mean=medias[i], cov=varianzas, size=m)
            for i in range(p)
        ]
    ).reshape(p * m, 2)
    y = np.full(shape=(m, p), fill_value=np.arange(0, p, 1)).T.ravel()
    # devuelve vector x dato y vector y clasificacion del dato
    return x, y


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

    def grafica(self, k, fs):
        sns.set(style="whitegrid")
        fig = plt.figure(figsize=(8, 6))
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        Z_test, x, y = test_data(int(np.max(abs(self.X)) + 1), 0.1)
        Z = kNN.predict(Z_test, k).reshape(x.shape[0], y.shape[0])
        plt.title(r"$K = {} - N = {}$".format(k, m), fontsize=fs + 4)
        plt.contourf(x, y, Z, alpha=0.5, cmap="jet", interpolate=False)
        # plt.contour(x, y, Z, colors="black", levels=p)
        plt.scatter(
            self.X[:, 0], self.X[:, 1], c=self.Y, cmap="jet", s=100, edgecolors="black"
        )
        plt.xlim(np.min(self.X[:, 0]), np.max(self.X[:, 0]))
        plt.ylim(np.min(self.X[:, 1]), np.max(self.X[:, 1]))
        plt.tight_layout()
        # plt.savefig("Ej4_k={}_M={}.pdf".format(k, m))
        return fig


def test_data(l, n):
    x = np.arange(-l, l + 1, n)
    y = np.arange(-l, l + 1, n)
    return (
        np.array([np.array([np.append(x, y) for x in x]) for y in y]).reshape(
            int(((2 * l + 1) / n) ** 2), 2
        ),
        x,
        y,
    )


kNN = KVecinosCercanos()
x_train, y_train = data2D_create(m, p)
kNN.train(x_train, y_train)
fig = kNN.grafica(k, 16)
plt.show(fig)
