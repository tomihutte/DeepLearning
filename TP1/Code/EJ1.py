import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipdb


n = 110  # numero de dimensiones max
m = 20  # paso en datos
k = 5  # n de pasos en datos + 2


def LS(X, y):
    # X es la matriz de los datos, y es el vector de resultados ( yj = a1 xj1+...+an xjn) )
    x = np.append(X, np.full(len(y), 1).reshape(len(y), 1), axis=1)  # agrego el bias
    # ipdb.set_trace()
    a = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)  # vector a+bias estimados
    return [np.dot(X, a[0:-1]) + a[-1], a]
    # devuelve el y aproximado y los coeficientes


def y_ruido(n, i, rangA=1, rangX=10, rangW=1, rangB=1):
    # crea i puntos de un hiperplano de dimensión n + ruido
    a = np.random.uniform(-rangA, rangA, n)  # coeficientes
    x = np.random.uniform(-rangX, rangX, n * i).reshape(i, n)  # valores de x
    b = np.full(i, np.random.uniform(-rangB, rangB, 1))  # vector de ordenadas al origen
    w = np.random.uniform(-rangW, rangW, i)  # ruido
    y = np.dot(x, a) + b + w
    return [y, a, x, b, w]


def errorf(n, i):
    if n > i + 1:
        return 0
    y = y_ruido(n, i)
    yaprox = LS(y[2], y[0])
    return 1 / n * np.linalg.norm(np.array(y[0]) - np.array(yaprox[0])) ** 2


error1 = np.vectorize(errorf)
N = np.arange(3, n, 1)
M = np.arange(m, m * k + 1, m)
N, M = np.meshgrid(N, M)
error = np.mean([error1(N, M) for i in range(500)], axis=0)  # error[i][n] i datos-n dim


sns.set(style="whitegrid")
fs = 25
lw = 6
x = np.arange(3, n, 1)
plt.figure(figsize=(10, 6))
plt.yscale("log")

for i in range(5):
    plt.plot(
        x[0 : len(np.unique(error[i], 0)[1::])],
        np.unique(error[i], 0)[1::],
        label="M = {}".format(m * (i + 1)),
        lw=lw,
    )

plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xlabel("Dimensión del problema", fontsize=fs)
plt.ylabel("Error cuadrático medio", fontsize=fs)
# plt.title(
#     "Error en función de la dimensión para diferentes cantida de datos M", fontsize=fs
# )
plt.legend(fontsize=fs, ncol=3, loc="upper center", framealpha=1)
plt.ylim(ymin=10 ** (-2))
plt.tight_layout()
# plt.savefig("Ej1.pdf")
plt.show()

