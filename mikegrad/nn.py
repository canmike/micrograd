import random
from mikegrad.engine import Value
import mikegrad as mg


class BaseModule:
    def __call__(self, *args):
        return self.forward(*args)


class Identity(BaseModule):
    def forward(self, x):
        return x


class Sigmoid(BaseModule):
    def forward(self, x):
        if type(x) == list:
            return [mg.sigmoid(xi) for xi in x]
        return mg.sigmoid(x)


class Tanh(BaseModule):
    def forward(self, x):
        if type(x) == list:
            return [mg.tanh(xi) for xi in x]
        return mg.tanh(x)


class ReLU(BaseModule):
    def forward(self, x):
        if type(x) == list:
            return [mg.relu(xi) for xi in x]
        return mg.relu(x)


class LeakyReLU(BaseModule):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        if type(x) == list:
            return [mg.leaky_relu(xi, self.negative_slope) for xi in x]
        return mg.leaky_relu(x, self.negative_slope)


class Softmax(BaseModule):
    def forward(self, x):
        exps = [mg.exp(xi) for xi in x]
        exp_sum = sum(exps)
        return [exp_i / exp_sum for exp_i in exps]


class Neuron(BaseModule):

    def __init__(self, nin, act=None):  # nin: number of inputs
        super().__init__()
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.act = act
        if act is None:
            self.act = Identity()

    def forward(self, x):
        # w * x + b
        out = sum([wi * xi for wi, xi in zip(self.w, x)], self.b)
        out = self.act(out)
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer(BaseModule):

    def __init__(self, nin, nout, act=None, layer_act=None):
        super().__init__()
        if act is None:
            act = Identity()
        self.layer_act = layer_act
        self.neurons = [Neuron(nin, act) for _ in range(nout)]

    def forward(self, x):
        outs = [n(x) for n in self.neurons]
        if self.layer_act is not None:
            outs = self.layer_act(outs)
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(BaseModule):
    def __init__(self, nin, nouts, act=None, out_act=None):
        super().__init__()
        if act is None:
            act = Sigmoid()
        if out_act is None:
            out_act = Identity()

        sizes = [nin] + nouts  # [2] + [3, 4] = [2, 3, 4]
        self.layers = [
            Layer(
                sizes[i],
                sizes[i + 1],
                act if i != len(nouts) - 1 else None,
                out_act if i == len(nouts) - 1 else None,
            )
            for i in range(len(nouts))
        ]  # len(nouts) = len(sizes) - 1

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
