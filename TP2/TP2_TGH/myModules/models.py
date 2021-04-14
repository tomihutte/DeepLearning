import numpy as np
import matplotlib.pyplot as plt
import myModules.layers as layers
import myModules.losses as losses
import myModules.optimizers as optimizers
import myModules.metrics as metrics


class Network:
    def __init__(self, InputLayer):
        # mejor vacio?
        self.layers = np.array([InputLayer], dtype=object)

    def get_layer(self, index):
        return self.layers[index]

    def add_layer(self, Layer):
        Layer.set_input_shape(self.layers[-1].get_output_shape())
        self.layers = np.append(self.layers, Layer)

    def forward(self, X, j=None):
        if j == None:
            j = len(self.layers)

        S = np.copy(X)

        for i in range(j):
            if isinstance(self.layers[i], layers.ConcatLayer):
                S = self.layers[i](S, X)
            else:
                S = self.layers[i](S)
        return S

    def fit(
        self, x_train, y_train, epochs, loss=losses.MSE(), opt=optimizers.SGD(), accuracy=metrics.accuracy, x_test=None, y_test=None, print_epoch=1,
    ):
        self.loss = loss
        self.accuracy = accuracy
        self.opt = opt
        self.epochs = epochs

        loss_train = np.array([])
        acc_train = np.array([])

        loss_test = np.array([])
        acc_test = np.array([])

        for epoch_idx in range(epochs):
            self.opt(x_train, y_train, self)

            scores = self.forward(x_train)
            loss_train = np.append(loss_train, self.loss(scores, y_train))
            acc_train = np.append(acc_train, self.accuracy(scores, y_train))

            if isinstance(x_test, np.ndarray):
                scores = self.forward(x_test)
                loss_test = np.append(loss_test, self.loss(scores, y_test))
                acc_test = np.append(acc_test, self.accuracy(scores, y_test))
                if not (epoch_idx % print_epoch):
                    print(
                        "Ep: {:d} -- acc: {:.4f} -- Loss: {:.4f} -- acc_test: {:.4f} -- loss_test: {:.4f}".format(
                            epoch_idx, acc_train[epoch_idx], loss_train[epoch_idx], acc_test[epoch_idx], loss_test[epoch_idx],
                        )
                    )

            else:
                if not (epoch_idx % print_epoch):
                    print("Ep: {:d} -- acc: {:.4f} -- Loss: {:.4f} ".format(epoch_idx, acc_train[epoch_idx], loss_train[epoch_idx],))
        plt.ion()
        plt.figure()
        plt.plot(loss_train)

        plt.figure()
        plt.plot(acc_train)

        plt.show()

        return loss_train, acc_train, loss_test, acc_test

    def backward(
        self, X, Y, grad=None, j=None,
    ):
        if j is None:
            j = len(self.layers)

            grad = self.loss.gradient(self.forward(X), Y)

        for i in range(j - 1, -1, -1):

            if isinstance(self.layers[i], layers.WLayer):

                s_prev = self.forward(X, i)
                s_prev_1 = np.hstack((np.ones((s_prev.shape[0], 1)), s_prev))
                y_current = self.layers[i].dot(s_prev)

                grad *= self.layers[i].activation.gradient(y_current)

                gradW = np.dot(s_prev_1.T, grad)

                dW = self.opt.update_weights(gradW, self.layers[i])
                self.layers[i].update_weights(dW)

                grad = np.dot(grad, self.layers[i].get_weights().T)
                grad = grad[:, 1:]

            elif isinstance(self.layers[i], layers.ConcatLayer):

                grad2 = grad[:, self.layers[i].get_input1_shape() :]
                self.backward(X, Y, grad2, self.layers[i].index + 1)
                grad = grad[:, : self.layers[i].get_input1_shape()]

