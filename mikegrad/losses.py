from mikegrad import Value
import mikegrad as mg
from mikegrad.nn import BaseModule
from typing import List


class BaseLoss(BaseModule):
    def __init__(self):
        pass


class MSELoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        if len(y_pred) != len(y_true):
            raise ValueError("y_pred and y_true must have the same length")

        loss = sum([(y_pred[i] - y_true[i]) ** 2 for i in range(len(y_pred))]) / len(
            y_pred
        )
        return loss


class CrossEntropyLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: List[List[Value]], y_true: List[List[Value]] | List[Value]
    ):
        if len(y_pred) != len(y_true):
            raise ValueError("y_pred and y_true must have the same size")
        if type(y_true[0]) != list:
            y_true = mg.one_hot_encode(y_true, len(y_pred[0]))

        loss = sum(
            [
                -sum(
                    [y_true[i][j] * mg.log(y_pred[i][j]) for j in range(len(y_pred[i]))]
                )
                for i in range(len(y_pred))
            ]
        ) / len(y_pred)

        return loss


class BCELoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: List[Value], y_true: List[Value]):
        if len(y_pred) != len(y_true):
            raise ValueError("y_pred and y_true must have the same length")

        loss = -sum(
            [
                y_true[i] * mg.log(y_pred[i]) + (1 - y_true[i]) * mg.log(1 - y_pred[i])
                for i in range(len(y_pred))
            ]
        ) / len(y_pred)

        return loss
