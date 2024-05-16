import numpy as np
import matplotlib.pyplot as plt

# параметры
a = 4
t1 = -2
t2 = 2
T = 20
dt = 0.01
b = 0.3
nu0 = 10

# массив времени
t = np.arange(-T/2, T/2, dt)
g = np.zeros_like(t)
g[(t >= t1) & (t <= t2)] = a

# зашумленный сигнал
u = g + b * (np.random.rand(len(t)) - 0.5)

# жесткая фильтрация
U = np.fft.fftshift(np.fft.fft(u))  # Фурье-образ сигнала u
f = np.linspace(-1/(4*dt), 1/(4 * dt), len(t))  # массив частот
U[np.abs(f) > nu0] = 0  # обнуляем значения на некоторых диапазонах частот
u_filtered = np.fft.ifft(np.fft.ifftshift(U))  # восстанавливаем сигнал с помощью обратного преобразования

# графики
plt.figure()
plt.plot(t, u, color='red')
plt.title('Исходный сигнал')
plt.xlim([-10, 10])
plt.show()

plt.figure()

plt.plot(t, u_filtered, color='blue')
plt.title('Фильтрованный сигнал')
plt.xlim([-10, 10])
plt.show()
