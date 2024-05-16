import numpy as np
import matplotlib.pyplot as plt

# параметры
a = 4
t1 = -2
t2 = 2
T = 20
dt = 0.01
b = 1
c = 2
d = 1
nu0 = 0.2  # частота среза для низких частот
# массив времени
t = np.arange(-T/2, T/2, dt)
g = np.zeros_like(t)
g[(t >= t1) & (t <= t2)] = a

# зашумленный сигнал
u = g + b * (np.random.rand(len(t)) - 0.5) + c * np.sin(d * t)

# Прямое преобразование Фурье
U = np.fft.fftshift(np.fft.fft(u))
V = 1 / dt
dv = 1 / T
v = np.arange(-V/2, V/2, dv)

# Фильтрация Фурье-образа сигнала
low_cutoff = 0.3
high_cutoff = 50
low_index = round((low_cutoff + V/2) / dv)
high_index = round((high_cutoff + V/2) / dv)
U_filtered = U.copy()
U_filtered[low_index:high_index] = 0
U_filtered[-high_index:-low_index] = 0

# обнуляем значения в окрестности нулевой частоты
U_filtered[np.abs(v) < nu0] = 0

# Обратное преобразование Фурье
u_filtered = np.fft.ifft(np.fft.ifftshift(U_filtered))

plt.plot(t, u, color='red')
plt.title('Исходный сигнал')
plt.xlim([-7, 7])
plt.show()

plt.figure()
plt.plot(t, u_filtered, 'blue')
plt.title('Фильтрованный сигнал')
plt.xlim([-7, 7])
plt.show()

plt.figure()
plt.plot(v, np.abs(U), color='green')
plt.title('Модуль исходного Фурье-образа')
plt.xlim([-7, 7])
plt.show()

plt.figure()
plt.plot(v, np.abs(U_filtered), color='purple')
plt.title('Модуль фильтрованного Фурье-образа')
plt.xlim([-7, 7])
plt.show()
