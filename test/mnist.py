#!/usr/bin/env python
from turtle import forward
import numpy as np
from tqdm import trange
import sys

sys.path.append(r"C:\Users\xxngu\Documents\sourcecode\minorgrad")
np.random.seed(42)

from minorgrad.tensor import Tensor
from minorgrad.utils import fetch_mnist
import minorgrad.optim as optim

# load the mnist dataset
X_train, Y_train, X_test, Y_test = fetch_mnist()

# train a model
def layer_init(m, h):
    ret = np.random.uniform(-1.0, 1.0, size=(m, h)) / np.sqrt(m * h)
    return ret.astype(np.float32)


class TinyBobNet:
    def __init__(self):
        self.l1 = Tensor(layer_init(784, 128))
        self.l2 = Tensor(layer_init(128, 10))

    def forward(self, x):
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()


# optimizer
model = TinyBobNet()
optim = optim.SGD([model.l1, model.l2], lr=0.01)
# optim = optim.Adam([model.l1, model.l2], lr=0.001)

BATCH_SIZE = 128
NUM_ITERATIONS = 1000
losses, accuracies = [], []
for i in (t := trange(NUM_ITERATIONS)):
    samp = np.random.randint(0, X_train.shape[0], size=(BATCH_SIZE))

    x = Tensor(X_train[samp].reshape((-1, 28 * 28)))
    Y = Y_train[samp]
    y = np.zeros((len(samp), 10), np.float32)
    # Setting the correct class to -1.0.
    # This is a trick to make the loss function work.
    y[range(y.shape[0]), Y] = -1.0
    y = Tensor(y)

    # forward propagation
    outs = model.forward(x)

    # NLL loss function
    loss = outs.mul(y).mean()
    # backward propagation
    loss.backward()
    optim.step()

    cat = np.argmax(outs.data, axis=1)
    accuracy = (cat == Y).mean()

    # printing
    loss = loss.data
    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

# evaluate
def numpy_eval():
    Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28 * 28))))
    Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
    return (Y_test == Y_test_preds).mean()


accuracy = numpy_eval()
print("test set accuracy is %f" % accuracy)
assert accuracy > 0.95
