import numpy as np


def f(x, y):
    return x * np.exp(-x ** 2) - 2 * x * y


def correct_f(x):
    return (1 / 2) * x ** 2 * np.exp(-x ** 2)


# Эйлер
def Euler(x0, y0, h, xn):
    x = np.arange(x0, xn + h, h)
    y = np.zeros_like(x)
    y[0] = y0
    i = 0
    for i in range(len(x) - 1):
        y[i + 1] = y[i] + h * f(x[i], y[i])

    return x, y, i


# Модифицированный Эйлер
def modified_Euler(x0, y0, h, xn, epsilon):
    x = np.arange(x0, xn + h, h)
    y = np.zeros_like(x)
    y[0] = y0
    i = 0

    for j in range(len(x) - 1):
        y_k = y[j] + h * f(x[j], y[j])
        while True:
            y_k_1 = y[j] + (h / 2) * (f(x[j], y[j]) + f(x[j + 1], y_k))
            i += 1
            if abs(y_k_1 - y_k) < epsilon:
                break
            y_k = y_k_1
        y[j + 1] = y_k_1

    return x, y, i


def runge_kutta_4(x0, y0, h, xn):
    x = np.arange(x0, xn + h, h)
    y = np.zeros_like(x)
    y[0] = y0

    for i in range(len(x) - 1):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(x[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x, y


def adams_bashforth_4(x0, y0, h, xn):
    x, y_rk = runge_kutta_4(x0, y0, h, x0 + 3 * h)
    x = np.arange(x0, xn + h, h)
    y = np.zeros_like(x)
    y[:4] = y_rk[:4]

    for i in range(3, len(x) - 1):
        y[i + 1] = y[i] + h / 24 * (
                    55 * f(x[i], y[i]) - 59 * f(x[i - 1], y[i - 1]) + 37 * f(x[i - 2], y[i - 2]) - 9 * f(x[i - 3],
                                                                                                         y[i - 3]))

    return x, y