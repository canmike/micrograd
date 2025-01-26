from mikegrad.engine import Value
import math


def log(value: Value):
    x = value.data
    out = Value(math.log(x), (value,), "log")

    def _backward():
        value.grad += (1 / x) * out.grad

    out._backward = _backward

    return out
