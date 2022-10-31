import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params


class SGD(Optimizer):
    def __init__(self, params, lr=0.001):
        super(SGD, self).__init__(params)
        self.lr = lr

    def step(self):
        for t in self.params:
            t.data -= self.lr * t.grad


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super(Adam, self).__init__(params)
        self.lr = lr
        # Exponential decay rates for the moment estimates
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.timestep = 0

        # Initialize 1st moment vector
        self.m = [np.zeros_like(t.data) for t in self.params]
        # Initialize 2st moment vector
        self.v = [np.zeros_like(t.data) for t in self.params]

    def step(self):
        for i, t in enumerate(self.params):
            self.timestep += 1
            # Update biased first moment estimate
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * t.grad
            # Update biased second raw moment estimate
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.square(t.grad)
            # Compute bias-corrected first moment estimate
            mhat = self.m[i] / (1.0 - self.b1**self.timestep)
            # Compute bias-corrected second raw moment estimate
            vhat = self.v[i] / (1.0 - self.b2**self.timestep)
            # Update parameters
            t.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
