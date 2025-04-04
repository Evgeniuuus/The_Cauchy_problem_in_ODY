import numpy as np
from tqdm import tqdm


def f(x, y):
    return x * np.exp(-x ** 2) - 2 * x * y


def correct_f(x):
    return (1 / 2) * x ** 2 * np.exp(-x ** 2)


def check(a, b):
    res = a[0] - b[0]
    for i in range(len(a)):
        if a[i] - b[i] > res:
            res = a[i] - b[i]
    return res


# Эйлер
def Euler(x0, y0, h, xn):
    x = np.arange(x0, xn + h, h)
    y = np.zeros_like(x)
    y[0] = y0
    for j in range(len(x) - 1):
        y[j + 1] = y[j] + h * f(x[j], y[j])

    return x, y


# Модифицированный Эйлер
def modified_Euler(x0, y0, h, xn, epsilon):
    x = np.arange(x0, xn + h, h)
    y = np.zeros_like(x)
    y[0] = y0

    for j in range(len(x) - 1):
        y_k = y[j] + h * f(x[j], y[j])
        while True:
            y_k_1 = y[j] + (h / 2) * (f(x[j], y[j]) + f(x[j + 1], y_k))
            if abs(y_k_1 - y_k) < epsilon:
                break
            y_k = y_k_1
        y[j + 1] = y_k_1

    return x, y


def Runge_kutta_4(x0, y0, h, xn):
    x = np.arange(x0, xn + h, h)
    y = np.zeros_like(x)
    y[0] = y0
    for j in tqdm(range(len(x) - 1), ncols=40):
        k1 = h * f(x[j], y[j])
        k2 = h * f(x[j] + h / 2, y[j] + k1 / 2)
        k3 = h * f(x[j] + h / 2, y[j] + k2 / 2)
        k4 = h * f(x[j] + h, y[j] + k3)
        y[j + 1] = y[j] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x, y


def Adams(x0, y0, h, xn):
    x, y_rk = Runge_kutta_4(x0, y0, h, x0 + 3 * h)
    x = np.arange(x0, xn + h, h)
    y = np.zeros_like(x)
    y[:4] = y_rk[:4]

    for j in tqdm(range(3, len(x) - 1), ncols=40):
        y_predict = y[j] + h / 24 * (
                55 * f(x[j], y[j]) - 59 * f(x[j - 1], y[j - 1]) +
                37 * f(x[j - 2], y[j - 2])
                - 9 * f(x[j - 3], y[j - 3]))

        y[j + 1] = y[j] + h / 24 * (
                9 * f(x[j + 1], y_predict) + 19 * f(x[j], y[j]) -
                5 * f(x[j - 1], y[j - 1]) + f(x[j - 2], y[j - 2])
        )

    return x, y
