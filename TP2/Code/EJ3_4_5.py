import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10

n_clases = 10  # clasificaciones de los datos
n_neuronas = 100  # neuronas de la capa intermedia
epochs = 200  # epocas de entrenamiento
batch_size = 50  # tamaño del batch
lambd = 1e-2  # 1e-4  # factor de regularización
lr = 1e-5  # learning rate
weight = 1e-3  # pesos iniciales de las matrices


def L2(W):
    """Devuelve un numero que es np.sum(W**2)"""
    return np.sum(W ** 2)


def grad_L2(W):
    """Devuelve 2*W"""
    return 2 * W


def accuracy(scores, y_true):
    """Devuelve un numero"""
    y_pred = np.argmax(scores, axis=1)
    return np.mean(y_true == y_pred) * 100


def mse(scores, y_true):
    """Devuelve un numero"""
    # import ipdb

    # ipdb.set_trace(context=15)  # XXX BREAKPINT
    index = np.arange(y_true.shape[0])
    scores_aux = np.copy(scores)
    scores_aux[index, y_true] -= 1
    return np.mean(np.sum(scores_aux ** 2, axis=1))


def grad_mse(scores, y_true):
    """Devuelve una matriz de igual dimension a scores"""
    # import ipdb

    # ipdb.set_trace(context=15)  # XXX BREAKPINT
    index = np.arange(y_true.shape[0])
    scores_aux = np.copy(scores)
    scores_aux[index, y_true] -= 1
    return 2 * scores_aux / y_true.shape[0]


def soft_max(scores, y_true):
    """Devuelve un numero"""
    scores_aux = np.copy(scores)
    scores_aux -= scores_aux.max(axis=1)[:, np.newaxis]
    index = np.arange(y_true.shape[0])
    scores_true = scores_aux[index, y_true]
    scores_aux = np.exp(scores_aux)
    return (-scores_true + np.log(scores_aux.sum(axis=1))).mean()


def grad_soft_max(scores, y_true):
    """Devuelve una matriz"""
    scores_aux = np.copy(scores)
    scores_aux = np.exp(scores_aux)
    scores_sum = scores_aux.sum(axis=1)
    scores_aux /= scores_sum[:, np.newaxis]
    index = np.arange(y_true.shape[0])
    scores_aux[index, y_true] -= 1
    return scores_aux / y_true.shape[0]


def sigmoid(x):
    """ Devuelve algo de la misma dimensión que x """
    return 1 / (1 + np.exp(-x))


def grad_sigmoid(x):
    """"Devuelve algo de la misma dimensión que x"""
    sigm = sigmoid(x)
    return (1 - sigm) * sigm


def ReLU(x):
    return np.maximum(0, x)


def grad_ReLU(scores):
    return np.heaviside(scores, 0)


def identity(x):
    return x


def grad_identity(x):
    return 1


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocesado

x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:])).astype(np.float)
x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:])).astype(np.float)
y_test, y_train = y_test.ravel(), y_train.ravel()
x_test -= np.mean(x_train, axis=0)
x_train -= np.mean(x_train, axis=0)


# x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))
x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))


# Inicializo matrices

w1 = np.random.uniform(-1, 1, size=(x_train.shape[1] + 1, n_neuronas)) * weight
w2 = np.random.uniform(-1, 1, size=(n_neuronas + 1, n_clases)) * weight


# Fit
## Lo primero que hacemos es elegir que metrica y funciones de activación usar

metric = mse  # elegimos metrica -- soft_max | mse
grad_metric = grad_mse  # gradiente de métrica -- grad_soft_max | grad_mse
reg = L2  # elegimos regularización
grad_reg = grad_L2  # gradiente de regularizacion
activation1 = sigmoid  # funcion de activacion de la primer capa -- ReLU | sigmoid | identity
grad_activation1 = grad_sigmoid  # gradiente de activación de la primer capa -- grad_ReLU | grad_sigmoid | grad identity
activation2 = identity  # funcion de acitvacion de la segunda capa
grad_activation2 = grad_identity  # funcion gradientede la segunda capa

n_batchs = int(x_train.shape[0] / batch_size)

loss_train = np.array([])
acc_train = np.array([])
loss_test = np.array([])
acc_test = np.array([])
index = np.arange(x_train.shape[0])

for epoch_idx in range(epochs):

    np.random.shuffle(index)
    loss = 0
    acc = 0

    for batch in range(n_batchs):
        id_batch = index[batch * batch_size : (batch + 1) * batch_size]
        x_batch = x_train[id_batch]
        y_batch = y_train[id_batch]
        x_batch = np.hstack((x_batch, np.ones((x_batch.shape[0], 1))))

        # Forward
        # import ipdb

        # ipdb.set_trace(context=15)  # XXX BREAKPINT
        y1 = np.dot(x_batch, w1)  # x_batch.dot(w1)
        S1 = activation1(y1)
        S1_1 = np.hstack((S1, np.ones((S1.shape[0], 1))))
        y2 = np.dot(S1_1, w2)  # S1_1.dot(w2)
        S2 = activation2(y2)

        r = reg(w1) + reg(w2)

        loss += metric(S2, y_batch) + 0.5 * lambd * r
        acc += accuracy(S2, y_batch)

        # Backward
        grad = grad_metric(S2, y_batch)

        # Capa 2
        grad *= grad_activation2(y2)
        gradW2 = np.dot(S1_1.T, grad)  # S1_1.T.dot(grad)
        grad = np.dot(grad, w2.T)  # grad.dot(w2.T)[:, :-1]
        grad = grad[:, :-1]

        # Capa 1
        grad_activation = grad_activation1(y1)
        grad = grad * grad_activation

        gradW1 = np.dot(x_batch.T, grad)  # x_batch.T.dot(grad)

        # Learning
        w1 -= lr * (gradW1 + lambd * 0.5 * grad_reg(w1))
        w2 -= lr * (gradW2 + lambd * 0.5 * grad_reg(w2))

    # Loss y accuracy train
    loss_train = np.append(loss_train, loss / n_batchs)
    acc_train = np.append(acc_train, acc / n_batchs)

    # Loss y accuracy test
    y1 = np.dot(x_test, w1)  # x_test.dot(w1)
    S1 = activation1(y1)
    S1_1 = np.hstack((S1, np.ones((S1.shape[0], 1))))
    y2 = np.dot(S1_1, w2)  # S1_1.dot(w2)
    S2 = activation2(y2)

    r = reg(w1) + reg(w2)

    loss_test = np.append(loss_test, metric(S2, y_test) + 0.5 * lambd * r)

    acc_test = np.append(acc_test, accuracy(S2, y_test))

    # if not (epoch_idx % 10):

    print(
        "Ep: {:d} -- acc: {:.4f} -- Loss: {:.4f} -- acc_test: {:.4f} -- loss_test: {:.4f}".format(
            epoch_idx, acc_train[epoch_idx], loss_train[epoch_idx], acc_test[epoch_idx], loss_test[epoch_idx],
        )
    )

np.savetxt("EJ3_MSE_Sigmoid_Lineal_lr-6_lambda-1.txt", np.column_stack((loss_train, acc_train, loss_test, acc_test)), delimiter="\t")

