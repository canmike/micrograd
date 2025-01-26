from mikegrad.engine import Value
from typing import List
import math


def from_list(values):
    """
    Generate a list of Value objects from a N dimension list of numbers.
    """
    if type(values) == list:
        return [from_list(value) for value in values]
    return Value(values)


def to_list(values):
    """
    Generate a N dimension list of numbers from a list of Value objects.
    """
    if type(values) == list:
        return [to_list(value) for value in values]
    return values.data


def one_hot_encode(values: List[Value], num_classes: int):
    one_hot = []
    for value in values:
        one_hot.append([Value(0) for _ in range(num_classes)])
        one_hot[-1][int(value.data)] = Value(1)
    return one_hot


def argmax(values):
    max_value = -math.inf
    max_index = -1
    for i, value in enumerate(values):
        if value.data > max_value:
            max_value = value.data
            max_index = i

    return max_index


def log(value: Value):
    return value.log()


def exp(value: Value):
    return value.exp()


def sigmoid(value: Value):
    return value.sigmoid()


def tanh(value: Value):
    return value.tanh()
