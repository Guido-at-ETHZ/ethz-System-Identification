import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parameters
N = 1024  # Number of points
fs = 1000  # Sampling frequency

# Time vector
t = np.arange(N) / fs

# Create different levels of Gaussian window smoothing
alpha_values = [0.1, 0.2, 0.5, 1.0]  # Increasing smoothness
colors = ['b', 'g', 'r', 'purple']

# Plot time domain
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
for alpha, color in zip(alpha_values, colors):
    w = signal.windows.gaussian(N, alpha * N/2)
    plt.plot(t, w, color=color, label=f'α = {alpha}')

plt.title('Gaussian Windows - Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Plot frequency domain
plt.subplot(2, 1, 2)
for alpha, color in zip(alpha_values, colors):
    w = signal.windows.gaussian(N, alpha * N/2)
    W = np.fft.fft(w)
    f = np.fft.fftfreq(N, 1/fs)
    plt.plot(f[:N//2], 20 * np.log10(np.abs(W[:N//2])/np.max(np.abs(W))),
             color=color, label=f'α = {alpha}')

plt.title('Gaussian Windows - Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.ylim([-100, 5])
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()