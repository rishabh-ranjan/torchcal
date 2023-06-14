import warnings

import torch
import torch.nn.functional as F
import torchmin


class TempScaler:
    def __init__(self, num_classes=None, method="newton-exact", device="cpu"):
        self.method = method

        self.temp = torch.ones(1, device=device)

    @property
    def num_params(self):
        return 1

    def fit(self, yhat, y):
        def loss(temp):
            return F.cross_entropy(yhat / temp, y)

        res = torchmin.minimize(loss, self.temp, method=self.method)
        if not res.success:
            warnings.warn(
                f"{self.__class__}: {res.message} Not updating calibrator params."
            )
        else:
            self.temp = res.x

    def __call__(self, yhat):
        return yhat / self.temp


class NoBiasVectorScaler:
    def __init__(self, num_classes, method="bfgs", device="cpu"):
        self.num_classes = num_classes
        self.method = method

        self.temp = torch.ones(num_classes, device=device)

    @property
    def num_params(self):
        return self.num_classes

    def fit(self, yhat, y):
        def loss(temp):
            return F.cross_entropy(yhat / temp, y)

        res = torchmin.minimize(loss, self.temp, method=self.method)
        if not res.success:
            warnings.warn(
                f"{self.__class__}: {res.message} Not updating calibrator params."
            )
        else:
            self.temp = res.x

    def __call__(self, yhat):
        return yhat / self.temp


class BiasCorrectedTempScaler:
    def __init__(self, num_classes, method="bfgs", device="cpu"):
        self.num_classes = num_classes
        self.method = method

        self.temp = torch.ones(1, device=device)
        self.bias = torch.zeros(num_classes, device=device)

    @property
    def num_params(self):
        return 1 + self.num_classes

    def fit(self, yhat, y):
        def loss(temp_bias):
            temp, bias = temp_bias.split([1, self.num_classes])
            return F.cross_entropy(yhat / temp + bias, y)

        temp_bias = torch.cat([self.temp, self.bias])
        res = torchmin.minimize(loss, temp_bias, method=self.method)
        if not res.success:
            warnings.warn(
                f"{self.__class__}: {res.message} Not updating calibrator params."
            )
        else:
            self.temp, self.bias = res.x.split([1, self.num_classes])

    def __call__(self, yhat):
        return yhat / self.temp + self.bias


class VectorScaler:
    def __init__(self, num_classes, method="bfgs", device="cpu"):
        self.num_classes = num_classes
        self.method = method

        self.temp = torch.ones(num_classes, device=device)
        self.bias = torch.zeros(num_classes, device=device)

    @property
    def num_params(self):
        return 2 * self.num_classes

    def fit(self, yhat, y):
        def loss(temp_bias):
            temp, bias = temp_bias.split([self.num_classes, self.num_classes])
            return F.cross_entropy(yhat / temp + bias, y)

        temp_bias = torch.cat([self.temp, self.bias])
        res = torchmin.minimize(loss, temp_bias, method=self.method)
        if not res.success:
            warnings.warn(
                f"{self.__class__}: {res.message} Not updating calibrator params."
            )
        else:
            self.temp, self.bias = res.x.split([self.num_classes, self.num_classes])

    def __call__(self, yhat):
        return yhat / self.temp + self.bias


class NoBiasMatrixScaler:
    def __init__(self, num_classes, method="l-bfgs", device="cpu"):
        self.num_classes = num_classes
        self.method = method

        self.itemp = torch.eye(num_classes, num_classes, device=device)

    @property
    def num_params(self):
        return self.num_classes**2

    def fit(self, yhat, y):
        def loss(itemp):
            itemp = itemp.view(self.num_classes, self.num_classes)
            return F.cross_entropy(yhat @ itemp, y)

        res = torchmin.minimize(loss, self.itemp.view(-1), method=self.method)
        if not res.success:
            warnings.warn(
                f"{self.__class__}: {res.message} Not updating calibrator params."
            )
        else:
            self.itemp = res.x.view(self.num_classes, self.num_classes)

    def __call__(self, yhat):
        return yhat @ self.itemp


class MatrixScaler:
    def __init__(self, num_classes, method="l-bfgs", device="cpu"):
        self.num_classes = num_classes
        self.method = method

        self.itemp = torch.eye(num_classes, num_classes, device=device)
        self.bias = torch.zeros(num_classes, device=device)

    @property
    def num_params(self):
        return self.num_classes**2 + self.num_classes

    def fit(self, yhat, y):
        def loss(itemp_bias):
            itemp, bias = itemp_bias.split([self.num_classes**2, self.num_classes])
            itemp = itemp.view(self.num_classes, self.num_classes)
            return F.cross_entropy(yhat @ itemp + bias, y)

        itemp_bias = torch.cat([self.itemp.view(-1), self.bias])
        res = torchmin.minimize(loss, itemp_bias, method=self.method)
        if not res.success:
            warnings.warn(
                f"{self.__class__}: {res.message} Not updating calibrator params."
            )
        else:
            self.itemp, self.bias = res.x.split(
                [self.num_classes**2, self.num_classes]
            )
            self.itemp = self.itemp.view(self.num_classes, self.num_classes)

    def __call__(self, yhat):
        return yhat @ self.itemp + self.bias


CLS = {
    "temp_scaler": TempScaler,
    "no_bias_vector_scaler": NoBiasVectorScaler,
    "bias_corrected_temp_scaler": BiasCorrectedTempScaler,
    "vector_scaler": VectorScaler,
    "no_bias_matrix_scaler": NoBiasMatrixScaler,
    "matrix_scaler": MatrixScaler,
}


def calibrator(name, num_classes=None, method=None, device="cpu"):
    cal = CLS[name](num_classes, method, device)

    if method is None:
        if cal.num_params == 1:
            method = "newton-exact"
        elif cal.num_params <= 1000:
            method = "bfgs"
        else:
            method = "l-bfgs"

        cal.method = method

    return cal
