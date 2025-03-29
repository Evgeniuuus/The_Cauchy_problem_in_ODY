import matplotlib.pyplot as plt
from functions import *
from tabulate import tabulate

# Начальные условия
x0, y0, h, xn = 0, 0, 0.1, 1

epsilon = 0.001

x_table = np.arange(x0, xn + h, h)
y_table = np.zeros_like(x_table)
for i in range(len(x_table)):
    y_table[i] = correct_f(x_table[i])

x_e, y_e, i1 = Euler(x0, y0, h, xn)

x_e1, y_e1, i2 = modified_Euler(x0, y0, h, xn, epsilon)

# ------------------------------------------------------------------------

plt.subplot(131)
plt.title('Эйлер', fontsize=14, fontname='Times New Roman')
plt.plot(x_table, y_table, 'go')
# Дополнительное построение  (соединяем точки точных значений)
x = np.linspace(x0, xn, 1000)
y = correct_f(x)
plt.plot(x, y, 'g-', label='Точное решение')

plt.plot(x_e, y_e, 'bo-', label='Метод Эйлера')
plt.plot(x_e1, y_e1, 'ro-', label='Модифицированный метод Эйлера')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()

# ------------------------------------------------------------------------

x0, y0, h, xn = 0, 0, 0.1, 1
x_rk, y_rk, i3 = Runge_kutta_4(x0, y0, h, xn)

plt.subplot(132)
plt.title('Рунге-Кутта', fontsize=14, fontname='Times New Roman')
plt.plot(x_table, y_table, 'go')
# Дополнительное построение  (соединяем точки точных значений)
x = np.linspace(x0, xn, 1000)
y = correct_f(x)
plt.plot(x, y, 'g-', label='Точное решение')
plt.plot(x_rk, y_rk, 'mo-', label='Метод Рунге-Кутта (4-го порядка)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()

# ------------------------------------------------------------------------

x_adams, y_adams, i4 = Adams(x0, y0, h, xn)
plt.subplot(133)
plt.title('Адамс', fontsize=14, fontname='Times New Roman')
plt.plot(x_table, y_table, 'go')
# Дополнительное построение  (соединяем точки точных значений)
x = np.linspace(x0, xn, 1000)
y = correct_f(x)
plt.plot(x, y, 'g-', label='Точное решение')
plt.plot(x_adams, y_adams, 'co-', label='Метод Адамса (4-го порядка)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

# ------------------------------------------------------------------------

table_data = []
for i in range(len(x_table)):
    table_data.append([
        f"{x_table[i]:.1f}",
        f"{y_table[i]:.9f}",
        f"{y_e[i]:.9f}",
        f"{y_e1[i]:.9f}",
        f"{y_rk[i]:.9f}",
        f"{y_adams[i]:.9f}"
    ])

headers = ["x", "Точное значение", "Метод Эйлера", "Мод. метод Эйлера", "Метод Рунге-Кутта (4)", "Метод Адамса"]

print(tabulate(table_data, headers=headers, tablefmt="grid"))

footer = [
    ["Макс. отклонение", f"{check(y_table, y_e):.9f}",
                         f"{check(y_table, y_e1):.9f}",
                         f"{check(y_table, y_rk):.9f}",
                         f"{check(y_table, y_adams):.9f}"],
    ["Количество итераций", i1, i2, i3, i4]
]

print("\n" + tabulate(footer, headers=["",
                                       "Метод Эйлера",
                                       "Мод. метод Эйлера",
                                       "Метод Рунге-Кутта (4)",
                                       "Метод Адамса"], tablefmt="grid"))

print("\nepsilon = ", epsilon)
