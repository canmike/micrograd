from mikegrad.engine import Value
from typing import List
import math
import numpy as np


def from_pandas(df):
    """
    Generate a list of Value objects from a pandas DataFrame.
    """
    return from_numpy(df.values)


def from_numpy(array):
    """
    Generate a list of Value objects from a numpy array.
    """
    return from_list(array.tolist())


def to_numpy(values):
    """
    Generate a numpy array from a list of Value objects.
    """
    return np.array(to_list(values))


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


def relu(value: Value):
    return value.relu()


def leaky_relu(value: Value, negative_slope=0.01):
    return value.leaky_relu(negative_slope)


def train(mlp, xs, ys, loss_fn, optimizer, nepochs=100, batch_size=-1, use_tqdm=True):
    if batch_size == -1:
        batch_size = len(xs)

    if use_tqdm:
        from tqdm import tqdm

        epoch_progress = tqdm(range(nepochs), desc="Epochs")

    for i in range(nepochs):
        epoch_loss = 0.0
        n = 0

        use_batch_progress = use_tqdm and batch_size != len(xs)
        if use_batch_progress:
            len_batches = math.ceil(len(xs) / batch_size)
            batch_progress = tqdm(range(len_batches), desc="Batches", leave=False)

        for j in range(0, len(xs), batch_size):
            x_batch = xs[j : j + batch_size]
            y_batch = ys[j : j + batch_size]
            optimizer.zero_grad()

            y_pred = [mlp(x) for x in x_batch]
            loss = loss_fn(y_pred, y_batch)
            epoch_loss += loss.data
            n += 1

            loss.backward()
            optimizer.step()
            if use_batch_progress:
                batch_progress.set_postfix(loss=f"{(loss.data):.6f}")
                batch_progress.update(1)

        if use_tqdm:
            epoch_progress.set_postfix(loss=f"{(epoch_loss/n):.6f}")
            epoch_progress.update(1)
        else:
            print(f"Epoch [{i+1}/{nepochs}] | Loss: {(epoch_loss/n):.6f}")


def test(mlp, xs, ys, loss_fn, use_tqdm=True):
    total_loss = 0.0
    n = 0
    y_preds = []

    if use_tqdm:
        from tqdm import tqdm

        progress = tqdm(range(len(xs)), desc="Testing")

    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]

        y_pred = mlp(x)
        y_preds.append(y_pred)
        loss = loss_fn([y_pred], [y])
        total_loss += loss.data
        n += 1

        if use_tqdm:
            progress.set_postfix(loss=f"{(loss.data):.6f}")
            progress.update(1)

    return total_loss / n, y_preds
