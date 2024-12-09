import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 0.1  # Sampling period (seconds)
N = 100  # Number of samples
N_improved = 200  # Number of samples for improved resolution
t = np.arange(N) * T  # Time vector
t_improved = np.arange(N_improved) * T  # Time vector for improved resolution

# Calculate frequency parameters
fs = 1 / T  # Sampling frequency
df = fs / N  # Frequency resolution
df_improved = fs / N_improved  # Improved frequency resolution
freq = np.fft.fftfreq(N, T)  # Frequency vector
freq_improved = np.fft.fftfreq(N_improved, T)  # Improved frequency vector

# Create example signals
# Original signal with potential aliasing
f1 = 2  # 2 Hz component
f2 = 4.9  # Close to Nyquist frequency (5 Hz)
signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
signal_improved = np.sin(2 * np.pi * f1 * t_improved) + 0.5 * np.sin(2 * np.pi * f2 * t_improved)

# Calculate FFT
fft_result = np.fft.fft(signal)
fft_magnitude = np.abs(fft_result)
fft_result_improved = np.fft.fft(signal_improved)
fft_magnitude_improved = np.abs(fft_result_improved)


# Plotting functions
def plot_signal_and_spectrum(t, signal, freq, magnitude, title):
    plt.figure(figsize=(12, 6))

    # Time domain plot
    plt.subplot(1, 2, 1)
    plt.plot(t, signal)
    plt.title(f'Time Domain - {title}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Frequency domain plot
    plt.subplot(1, 2, 2)
    # Plot only positive frequencies
    pos_freq_mask = freq >= 0
    plt.plot(freq[pos_freq_mask], magnitude[pos_freq_mask])
    plt.title(f'Frequency Domain - {title}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    plt.tight_layout()


# Plot original resolution
plt.figure(1)
plot_signal_and_spectrum(t, signal, freq, fft_magnitude, 'Original Resolution')

# Plot improved resolution
plt.figure(2)
plot_signal_and_spectrum(t_improved, signal_improved, freq_improved,
                         fft_magnitude_improved, 'Improved Resolution')

# Print results
print(f"a) Original frequency spacing (df): {df:.3f} Hz")
print(f"b) Improved frequency spacing (df_improved): {df_improved:.3f} Hz")
print(f"\nc) Trade-off demonstration:")
print(f"   Original experiment time: {N * T:.1f} seconds")
print(f"   Improved experiment time: {N_improved * T:.1f} seconds")
print(f"\nd) Nyquist frequency: {fs / 2:.1f} Hz")
print(f"   Highest frequency in signal: {f2:.1f} Hz")

plt.show()