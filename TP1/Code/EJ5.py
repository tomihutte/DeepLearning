import numpy as np
from keras.datasets import cifar10, mnist
import matplotlib.pyplot as plt
import ipdb
import time
import seaborn as sns

sns.set(style="whitegrid")
start_time = time.time()

# np.random.seed(10)

reg = 0.1  # parametro lambda de regularizacion
delta = 1.0  # delta de SVM
epochs = 200
batch_size = 50
step_size = 1e-6  # learning rate
ndat = 2560  # datos usados para entrenar


class LinearClassifier:
    def __init__(self, dim, kclass, reg):
        self.d = dim
        self.k = kclass
        self.reg = reg
        self.W = np.random.uniform(
            low=-0.1, high=0.1, size=(kclass, dim + 1)
        )  # aleatorio al principio?

    def loss_gradient(self):
        pass

    def loss(self):
        pass

    def score(self, x):
        # ipdb.set_trace()
        return self.W.dot(
            np.vstack((x, np.ones(x.shape[1])))
        )  # score[i,j] es el peso de la categoria i para el vector j

    def predict(self, x):
        return np.argmax(self.score(x), axis=0)

    def accuracy(self, x, y):
        y_pred = self.predict(x)
        return np.sum(y_pred == y) / y.shape[0]

    def fit(self, x, y, batch_size, epochs, stepsize, x_test, y_test):
        self.batch_size = batch_size
        self.epochs = epochs
        self.stepsize = stepsize

        nbatch = int(x.shape[1] / self.batch_size)

        loss_vec = []
        acurracy_vec = []
        loss_test = []
        accuracy_test = []

        for epoch_idx in range(self.epochs):

            loss_vec.append(self.loss(x, y)[0])
            acurracy_vec.append(self.accuracy(x, y))

            loss_test.append(self.loss(x_test, y_test)[0])
            accuracy_test.append(self.accuracy(x_test, y_test))
            # ipdb.set_trace()
            print(
                "Ep: {:d} -- acc: {:.4f} -- Loss: {:.4f} -- acc_test: {:.4f} -- loss_test: {:.4f}".format(
                    epoch_idx,
                    acurracy_vec[epoch_idx],
                    loss_vec[epoch_idx],
                    accuracy_test[epoch_idx],
                    loss_test[epoch_idx],
                )
            )
            # stepsize = self.stepsize * (1 - (acurracy_vec[epoch_idx]) ** 2)
            for batch_idx in range(nbatch):
                x_train = x[
                    :, batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ]
                y_train = y[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ]
                dW = self.loss_gradient(x_train, y_train)

                self.W -= self.stepsize * dW

        loss_vec.append(self.loss(x, y)[0])
        acurracy_vec.append(self.accuracy(x, y))
        loss_test.append(self.loss(x_test, y_test)[0])
        accuracy_test.append(self.accuracy(x_test, y_test))

        self.loss_vec = np.array(loss_vec)
        self.acurracy_vec = np.array(acurracy_vec)
        self.accuracy_test = np.array(accuracy_test)
        self.loss_test = np.array(loss_test)

        return self.loss_vec, self.acurracy_vec


class SVM(LinearClassifier):
    def __init__(self, dim, kclass, reg, delta):
        super().__init__(dim, kclass, reg)
        self.delta = delta

    def loss(self, x, y):
        scores = self.score(x)  # sacamos los scores
        margins = np.maximum(
            0, scores - scores[y, np.arange(y.shape[0])] + self.delta
        )  # sacamos los Li
        margins[
            y, np.arange(y.shape[0])
        ] = 0  # borramos la contribucion de los scores correctos
        return np.mean(np.sum(margins, axis=0)), margins

    def loss_gradient(self, x_batch, y_batch):
        # x_batch es una matriz donde cada imagen es una columna, es de dimension
        # DxN donde D es la dimensión de la imagen y N la cantidad de imagenes
        # y_batch es un vector de largo N donde estan especificadas las clases de x_batch
        #
        # aca calculo el loss de x_batch/y_batch y también margins que es una matriz
        # de KxN, en margins[i,j] tengo el valor de costo asociado a la clase i predicho para
        # el vector j-esimo de x
        l, margins = self.loss(x_batch, y_batch)

        # agrego una fila de unos abajo de x_batch para contar el bias
        x_batch1 = np.vstack((x_batch, np.ones(x_batch.shape[1])))

        # En la derivada de la función loss solo importa si el margins es positivo o negativo
        # esto es como aplicarle una heaviside a margins
        margins[margins > 0] = 1

        # En el gradiente los valores de la columna j se suman y se guardan en margins[i,j]
        # donde y_batch[j]=i
        margins[y_batch, np.arange(y_batch.shape[0])] = -np.sum(margins, axis=0)
        # margins es de Kclases x Nimagenes+1, x_batch1 es de D_dimensionesimagen x Nimagenes+1, traspongo x_batch1
        # y hago producto interno

        gradient = np.dot(margins, x_batch1.T)

        dW = gradient / x_batch1.shape[1]
        # sumo gradiente de regularización 2
        dW += self.reg * self.W
        return dW


class SoftMax(LinearClassifier):
    def __init__(self, dim, kclass, reg):
        super().__init__(dim, kclass, reg)

    def loss(self, x, y):
        scores = self.score(x)
        scores -= np.max(scores, axis=0)
        exp_score = np.exp(scores)
        # ipdb.set_trace()
        return (
            np.mean(-scores[y, np.arange(y.size)] + np.log(np.sum(exp_score, axis=0))),
            exp_score,
        )

    def loss_gradient(self, x, y):
        loss, exp_scores = self.loss(x, y)
        gradient = 1.0 / np.sum(exp_scores, axis=0) * exp_scores
        gradient[y, np.arange(y.size)] -= 1
        x1 = np.vstack((x, np.ones(x.shape[1])))
        dW = np.dot(gradient, x1.T) / x1.shape[1]
        dW += self.reg * self.W
        return dW


def grafica(
    soft_vec, svm_vec, soft_test, svm_test, title, save, fs, lw, ylabel, yscale
):

    # plt.plot(soft_vec, label="SoftMax train", lw=lw, c="blue")
    plt.plot(soft_test, label="SoftMax test", lw=lw, c="blue", ls="-")
    # plt.plot(svm_vec, label="SVM train", lw=lw, c="red")
    plt.plot(svm_test, label="SVM test", lw=lw, c="red", ls="-")
    plt.title(title, fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel(ylabel, fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.yscale(yscale)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig(save)
    plt.show()


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
data_dim = np.prod(x_train.shape[1:])
ndat = np.minimum(ndat, x_train.shape[0])

y_train, y_test = y_train.ravel(), y_test.ravel()
x_train, x_test = (
    x_train.reshape(x_train.shape[0], data_dim).T.astype(np.float),
    x_test.reshape(x_test.shape[0], data_dim).T.astype(np.float),
)

SVMachine = SVM(data_dim, np.unique(y_train).shape[0], reg, delta)
SVMachine.fit(
    x_train[:, :ndat], y_train[:ndat], batch_size, epochs, step_size, x_test, y_test
)
Soft = SoftMax(data_dim, np.unique(y_train).shape[0], reg)
Soft.fit(
    x_train[:, :ndat], y_train[:ndat], batch_size, epochs, step_size, x_test, y_test
)


print(time.time() - start_time)

fs = 16
lw = 2

grafica(
    Soft.loss_vec / np.max(Soft.loss_vec),
    SVMachine.loss_vec / np.max(SVMachine.loss_vec),
    Soft.loss_test / np.max(Soft.loss_test),
    SVMachine.loss_test / np.max(SVMachine.loss_test),
    "",
    "EJ5_loss.pdf",
    fs,
    lw,
    "Función Loss normalizada",
    "linear",
)
grafica(
    Soft.acurracy_vec,
    SVMachine.acurracy_vec,
    Soft.accuracy_test,
    SVMachine.accuracy_test,
    "",
    "EJ5_acc.pdf",
    fs,
    lw,
    "Precisión",
    "linear",
)

