import random
from mikegrad.engine import Value


class BaseModule:
    def __call__(self, *args):
        return self.forward(*args)


class Neuron(BaseModule):

    def __init__(self, nin):  # nin: number of inputs
        super().__init__()
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def forward(self, x):
        # w * x + b
        act = sum([wi * xi for wi, xi in zip(self.w, x)], self.b)
        out = act.sigmoid()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer(BaseModule):

    def __init__(self, nin, nout):
        super().__init__()
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def forward(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(BaseModule):
    def __init__(self, nin, nouts):
        super().__init__()
        sizes = [nin] + nouts  # [2] + [3, 4] = [2, 3, 4]
        self.layers = [
            Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))
        ]  # len(nouts) = len(sizes) - 1

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
