import matplotlib.pyplot as plt
from functions import *

# Начальные условия
x0, y0, h, xn = 0, 0, 0.1, 1

epsilon = 0.001

# Точные значения
x_table = np.arange(x0, xn + h, h)
y_table = np.zeros_like(x_table)
for i in range(len(x_table)):
    y_table[i] = correct_f(x_table[i])
print('Таблица точных значений: \n', x_table, '\n', y_table, '\n')

# ----------------------------------------------Эйлер-------------------------------------------

# Обычный
x_e, y_e, i = Euler(x0, y0, h, xn)
print('Таблица значений в методе Эйлера: \n', x_e, '\n', y_e, '\n')

print(f'Число итераций: {i} \n')


# Модифицированный
(x_e1, y_e1, i) = modified_Euler(x0, y0, h, xn, epsilon)
print('Таблица значений в модифицированном методе Эйлера: \n', x_e1, '\n', y_e1, '\n')

print(f'Число итераций: {i} \n')

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

# ------------------------------------------Рунге-Кутта-----------------------------------------

x0, y0, h, xn = 0, 0, 0.1, 1
x_rk, y_rk = runge_kutta_4(x0, y0, h, xn)

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


# ---------------------------------------------Адамс----------------------------------------------

x_adams, y_adams = adams_bashforth_4(x0, y0, h, xn)

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