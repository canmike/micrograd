from typing import List
from mikegrad.engine import Value


class BaseOptimizer:
    def __init__(self):
        pass

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError


class SGD(BaseOptimizer):
    def __init__(
        self,
        params: List[Value],
        lr: float = 0.001,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        maximize: bool = False,
    ):
        super().__init__()
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize
        self.momentum_buffers = [None for _ in params]  # One buffer per parameter

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad = 0

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad

            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            if self.momentum != 0:
                if self.momentum_buffers[i] is None:
                    self.momentum_buffers[i] = grad
                else:
                    buf = self.momentum_buffers[i]
                    buf = self.momentum * buf + (1 - self.dampening) * grad
                    self.momentum_buffers[i] = buf

                if self.nesterov:
                    grad = grad + self.momentum * buf
                else:
                    grad = buf

            update = self.lr * grad
            if self.maximize:
                param.data += update
            else:
                param.data -= update

        return self.params
