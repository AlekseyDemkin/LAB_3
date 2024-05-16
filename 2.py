import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd

# Загрузка аудиофайла
Fs, y = wavfile.read('MUHA.wav')
y = y.astype(float)

# Параметры времени
dt = 1.0/Fs
T = len(y)*dt
t = np.arange(0, T, dt)

# Преобразование Фурье
Y = np.fft.fft(y)

# частоты
n = len(Y)
f = np.arange(0, n)*(Fs/n)

# Разработка фильтра
low_freq = 300
high_freq = 3400
samples_per_freq = f[1] - f[0]
low_index = round(low_freq / samples_per_freq)
high_index = round(high_freq / samples_per_freq)

# Создание фильтра
filter = np.zeros(n)
filter[low_index : high_index] = 1
filter[n - high_index : n - low_index] = 1

# Применение фильтра
Y_filtered = Y * filter

# Обратное преобразование Фурье
y_filtered = np.fft.ifft(Y_filtered)

# Воспроизведение отфильтрованного звука
sd.play(np.real(y_filtered), Fs)

plt.figure()
plt.plot(t, y, 'g')
plt.title('Исходный сигнал')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.xlim([0, 7])
plt.show()

plt.figure()
plt.plot(t, np.real(y_filtered), 'g')
plt.title('Отфильтрованный сигнал')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.xlim([0, 7])
plt.show()

plt.figure()
plt.plot(f[:n//2], np.abs(Y[:n//2]), 'g')
plt.title('Фурье-образ исходного сигнала')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.xlim([0, Fs/2])
plt.show()

plt.figure()
plt.plot(f[:n//2], np.abs(Y_filtered[:n//2]), 'g')
plt.title('Фурье-образ отфильтрованного сигнала')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.xlim([0, Fs/2])
plt.show()

