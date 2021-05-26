import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation


n = 3  # numero de dimensiones de las clases
p = 4  # numero de clases originales
m = 100  # numero de datos por clase
k = 4  # k means
cota = 0  # 10 ** -9  # cota para convergencia
varmed = 10  # varianza de las medias (elegidas aleatoriamente)
# si queres que esten separados los datos elegir mas grande

vardata = 0.5  # varianza de los datos
paso = 2  # donde graficar punto intermedio
ncol = 1
o, q = 0, 1  # dimensiones sobre las que observar


# se puede cambiar por valores que quieras
# medias y varianzas de cada grupo en todas las dim


def data_create(m, n, p):
    medias = np.random.multivariate_normal(
        mean=np.zeros(n), cov=np.identity(n) * varmed, size=p
    )
    varianzas = (
        np.identity(n) * vardata
    )  # medias y varianzas de cada grupo en todas las dim
    r = np.array(
        [
            np.random.multivariate_normal(mean=medias[i], cov=varianzas, size=m)
            for i in range(p)
        ]
    )
    # r[i,j,k] coordenada k del punto j del grupo i
    return r


def clasifica(centros, data):
    k = len(centros)
    l = np.linalg.norm(
        centros - d[:, np.newaxis], axis=2
    )  # vector con vectores con distancias a centro de masa
    # l[i,j] es la distancia del punto i al centro j
    i = np.argmin(l, axis=1)
    return np.array([data[i == j] for j in range(k)])


def centros_masa(clusters):
    return np.array([np.mean(d, axis=0) for d in clusters])


def steps(data, centros):
    cluster = clasifica(centros, data)
    return centros_masa(cluster), cluster


def grafica(clusters, centros, fs, w, h, o, q, titulo, savefig, legend, scentro):

    if legend:
        fig = plt.figure(figsize=(w + 2, h))
    else:
        fig = plt.figure(figsize=(w, h))
    sns.set(style="whitegrid")
    for i in range(len(clusters)):
        plt.scatter(
            clusters[i][:, o],
            clusters[i][:, q],
            s=s,
            cmap="jet",
            label="Clase {}".format(i),
        )
    for i in range(len(clusters)):
        plt.scatter(
            centros[i, o], centros[i, q], c="black", marker="X", s=s * 2 * scentro
        )
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xlabel("Coordenada {}".format(o), fontsize=fs)
    plt.ylabel("Coordenada {}".format(q), fontsize=fs)
    plt.title(titulo, fontsize=fs)
    if legend:
        plt.legend(fontsize=fs, ncol=ncol, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    # plt.savefig(savefig)
    return fig


r = data_create(m, n, p)

fs = 20
s = 50
w = 7.5
h = 6

plt.show(
    grafica(
        r,
        r[0, :p],
        fs,
        w,
        h,
        o,
        q,
        r"Clases originales - $m = {}$ - $n = {}$".format(m, n),
        "Ej2_clases_originales.pdf",
        1,
        0,
    )
)

d = r.reshape(m * p, n)  # borro la información del grupo, paso a una lista de puntos
np.random.shuffle(d)
centros = d[np.arange(0, k, 1)]
# centros [i,k] coordenada k del centro i, elijo aleatoriamente entre los puntos


ncentros, clusters = steps(d, centros)


plt.show(
    grafica(
        clusters,
        centros,
        fs,
        w,
        h,
        o,
        q,
        r"Instante inicial - $n = {}$".format(n),
        "Ej2_inicial.pdf",
        0,
        1,
    )
)

dif = 1
j = 0
while dif > cota:
    oldcentros = ncentros
    ncentros, clusters = steps(d, oldcentros)
    dif = np.linalg.norm(ncentros - oldcentros)
    j += 1
    grafica(
        clusters,
        ncentros,
        fs,
        w,
        h,
        o,
        q,
        r"Paso {} - $n={}$".format(j, n),
        "Ej2_{}.pdf".format(j),
        0,
        1,
    )


plt.show(
    grafica(
        clusters,
        ncentros,
        fs,
        w,
        h,
        o,
        q,
        r"Estacionario - $n={}$".format(n),
        "Ej2_final.pdf",
        1,
        1,
    )
)

