import numpy as np


class Optimizer:
    def __init__(self, lr):
        self.lr = lr

    def __call__(self):
        pass

    def update_weights(self, W, gradW):
        pass


class SGD(Optimizer):
    def __init__(self, lr=1e-5, batch_size=False):
        super().__init__(lr)
        self.batch_size = batch_size

    def __call__(self, x_train, y_train, model):
        if not (self.batch_size):
            model.backward(x_train, y_train)
        else:
            index = np.arange(x_train.shape[0])
            np.random.shuffle(index)
            n_batch = int(x_train.shape[0] / self.batch_size)
            for batch in range(n_batch):
                x_batch = x_train[
                    batch * self.batch_size : (batch + 1) * self.batch_size
                ]
                y_batch = y_train[
                    batch * self.batch_size : (batch + 1) * self.batch_size
                ]
                model.backward(x_batch, y_batch)

    def update_weights(self, gradW, layer):
        # import ipdb

        # ipdb.set_trace(context=15)  # XXX BREAKPINT
        dW = self.lr * (gradW + layer.reg.gradient(layer.get_weights()))
        return dW
