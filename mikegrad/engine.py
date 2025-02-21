import math

EPS = 1e-8


class Value:

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def _clip(self, data):
        return max(EPS, min(1 - EPS, self.data))

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data**other.data, (self, other), f"**")

        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            if self.data > 0:  # log is only valid for positive bases
                other.grad += (self.data**other.data) * math.log(self.data) * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __radd__(self, other):  # other + self
        return self + other

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        x = self.data
        r = max(0, x)
        out = Value(r, (self,), "relu")

        def _backward():
            self.grad += (x > 0) * out.grad

        out._backward = _backward

        return out

    def leaky_relu(self, negative_slope=0.01):
        x = self.data
        r = max(0, x) + negative_slope * min(0, x)
        out = Value(r, (self,), "leaky_relu")

        def _backward():
            self.grad += (x > 0) * out.grad
            self.grad += (x <= 0) * negative_slope * out.grad

        out._backward = _backward

        return out

    def log(self):
        x = self.data
        out = Value(math.log(self._clip(x)), (self,), "log")

        def _backward():
            self.grad += (1 / self._clip(x)) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
