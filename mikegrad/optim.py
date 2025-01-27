from typing import List
from mikegrad.engine import Value


class BaseOptimizer:
    def __init__(self):
        pass

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad = 0


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


class Adam(BaseOptimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        maximize=False,
    ):
        super().__init__()
        self.params = params
        self.lr = lr
        self.B1, self.B2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize

        self.m = [0 for _ in params]
        self.v = [0 for _ in params]
        self.vmax = [0 for _ in params]

        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad

            if self.maximize:
                grad = -grad

            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            self.m[i] = self.B1 * self.m[i] + (1 - self.B1) * grad
            self.v[i] = self.B2 * self.v[i] + (1 - self.B2) * (grad**2)

            m_hat = self.m[i] / (1 - (self.B1**self.t))
            v_hat = self.v[i] / (1 - (self.B2**self.t))

            if self.amsgrad:
                self.vmax[i] = max(self.vmax[i], v_hat)
                param.data -= self.lr * m_hat / (self.vmax[i] ** 0.5 + self.eps)
            else:
                param.data -= self.lr * m_hat / (v_hat**0.5 + self.eps)


class AdamW(BaseOptimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
        maximize=False,
    ):
        super().__init__()
        self.params = params
        self.lr = lr
        self.B1, self.B2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize

        self.m = [0 for _ in params]
        self.v = [0 for _ in params]
        self.vmax = [0 for _ in params]

        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad

            if self.maximize:
                grad = -grad

            param.data -= self.lr * self.weight_decay * param.data

            self.m[i] = self.B1 * self.m[i] + (1 - self.B1) * grad
            self.v[i] = self.B2 * self.v[i] + (1 - self.B2) * (grad**2)

            m_hat = self.m[i] / (1 - (self.B1**self.t))
            v_hat = self.v[i] / (1 - (self.B2**self.t))

            if self.amsgrad:
                self.vmax[i] = max(self.vmax[i], v_hat)
                param.data -= self.lr * m_hat / (self.vmax[i] ** 0.5 + self.eps)
            else:
                param.data -= self.lr * m_hat / (v_hat**0.5 + self.eps)
