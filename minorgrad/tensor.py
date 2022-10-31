# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from functools import partialmethod
import numpy as np

# **** start with two base classes ****


class Tensor:
    def __init__(self, data):
        if type(data) != np.ndarray:
            print("error constructing tensor with %r" % data)
            assert False
        self.data = data
        self.grad = None

        # internal variables used for autograd graph construction
        self._ctx = None

    def __str__(self):
        return "Tensor %r with grad %r" % (self.data, self.grad)

    def backward(self, allow_fill=True):
        # print("running backward on", self)
        if self._ctx is None:
            return

        if self.grad is None and allow_fill:
            # fill in the first grad with one
            # this is "implicit gradient creation"
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)

        assert self.grad is not None

        grads = self._ctx.backward(self._ctx, self.grad)
        if len(self._ctx.parents) == 1:
            grads = [grads]
        for t, g in zip(self._ctx.parents, grads):
            if g.shape != t.data.shape:
                print(
                    "grad shape must match tensor shape in %r, %r != %r"
                    % (self._ctx, g.shape, t.data.shape)
                )
                assert False
            t.grad = g
            t.backward(False)

    def mean(self):
        div = Tensor(np.array([1 / self.data.size]))
        return self.sum().mul(div)


# An instantiation of the Function is the Context
class Function:
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = []

    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)

    def apply(x: Tensor, *y: Tensor, fxn_class) -> Tensor:
        ctx = fxn_class(x, *y)
        ret = Tensor(fxn_class.forward(ctx, x.data, *[t.data for t in y]))
        ret._ctx = ctx  # used by autograd engine

        return ret

    @staticmethod
    def forward(ctx, *args, **kwargs):
        r"""Performs the operation.

        This function is to be overridden by all subclasses.

        It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).

        The context can be used to store arbitrary data that can be then
        retrieved during the backward pass. Tensors should not be stored
        directly on `ctx` (though this is not currently enforced for
        backward compatibility). Instead, tensors should be saved either with
        :func:`ctx.save_for_backward` if they are intended to be used in
        ``backward``
        """
        raise NotImplementedError(
            f"You must implement the forward function for custom tensor.Function"
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        r"""Defines a formula for differentiating the operation with backward mode
        automatic differentiation (alias to the vjp function).

        This function is to be overridden by all subclasses.
        """
        raise NotImplementedError(
            f"You must implement the backward function for custom tensor.Function"
        )


def register(name, fxn_class):
    # a = Tensor(np.array([1, 2, 3]))
    # b = Tensor(np.array([4, 5, 6]))
    # a.add(b)
    setattr(Tensor, name, partialmethod(fxn_class.apply, fxn_class=fxn_class))


# **** implement a few functions/operators ****


class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        x, y = ctx.saved_tensors
        return y * grad_output, x * grad_output


register("mul", Mul)


class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x + y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


register("add", Add)


class Dot(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return input.dot(weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output.dot(weight.T)
        grad_weight = grad_output.T.dot(input).T
        return grad_input, grad_weight


register("dot", Dot)


class Sum(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.array([input.sum()])

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output * np.ones_like(input)


register("sum", Sum)


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, input):
        def logsumexp(x):
            # return np.log(np.exp(x).sum(axis=1))
            c = x.max(axis=1)
            return c + np.log(np.exp(x - c.reshape((-1, 1))).sum(axis=1))

        output = input - logsumexp(input).reshape((-1, 1))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        return grad_output - np.exp(output) * grad_output.sum(axis=1).reshape((-1, 1))


register("logsoftmax", LogSoftmax)
# ***** activation functions (unary) *****
class ReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.maximum(input, 0)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.copy()
        grad_input[input < 0] = 0
        return grad_input


register("relu", ReLU)


class LeakyReLU(Function):
    @staticmethod
    def forward(ctx, input, alpha=0.01):
        ctx.save_for_backward(input)
        return np.where(input > 0, input, input * alpha)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.copy()
        grad_input[input < 0] *= 0.01
        return grad_input


register("leaky_relu", LeakyReLU)
