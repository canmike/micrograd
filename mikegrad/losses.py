from mikegrad import Value
import mikegrad as mg
from mikegrad.nn import BaseModule


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

    def forward(self, y_pred, y_true):
        if len(y_pred) != len(y_true):
            raise ValueError("y_pred and y_true must have the same length")

        loss = sum([-y_true[i] * mg.log(y_pred[i]) for i in range(len(y_pred))]) / len(
            y_pred
        )
        return loss
