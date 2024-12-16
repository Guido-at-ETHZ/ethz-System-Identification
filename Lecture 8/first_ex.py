import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Generate system with long impulse response
N = 256  # measurement period
tau_max = 300  # impulse response length (longer than measurement period)
num_periods = 4  # number of periods to simulate
noise_level = 0.1

# Create slowly decaying impulse response
g_true = 0.5 * np.exp(-0.01 * np.arange(tau_max))

# Generate PRBS input for multiple periods
u = np.random.choice([-1, 1], N * num_periods)

# Generate output
y = np.zeros_like(u)
for k in range(len(y)):
    # Implement periodic convolution
    for i in range(tau_max):
        if k - i >= 0:
            y[k] += g_true[i] * u[k - i]
        else:
            # Periodic extension of input
            y[k] += g_true[i] * u[k - i + N]

# Add measurement noise
y = y + noise_level * np.random.randn(len(y))


# Function to estimate frequency response using DFT ratio
def estimate_freq_response(u, y, N):
    U = np.fft.fft(u[:N])
    Y = np.fft.fft(y[:N])
    # Avoid division by zero
    mask = np.abs(U) > 1e-10
    G = np.zeros_like(U, dtype=complex)
    G[mask] = Y[mask] / U[mask]
    freq = np.fft.fftfreq(N)
    return freq, G


# Function to estimate using averaged periodogram
def estimate_freq_response_averaged(u, y, N, num_periods):
    U_avg = np.zeros(N, dtype=complex)
    Y_avg = np.zeros(N, dtype=complex)

    for i in range(num_periods):
        start_idx = i * N
        U_period = np.fft.fft(u[start_idx:start_idx + N])
        Y_period = np.fft.fft(y[start_idx:start_idx + N])
        U_avg += U_period
        Y_avg += Y_period

    U_avg /= num_periods
    Y_avg /= num_periods

    # Compute frequency response
    mask = np.abs(U_avg) > 1e-10
    G = np.zeros_like(U_avg, dtype=complex)
    G[mask] = Y_avg[mask] / U_avg[mask]
    freq = np.fft.fftfreq(N)
    return freq, G


# Calculate true frequency response for comparison
freq_true = np.fft.fftfreq(N)
G_true = np.zeros_like(freq_true, dtype=complex)
for k, f in enumerate(freq_true):
    G_true[k] = np.sum(g_true * np.exp(-2j * np.pi * f * np.arange(tau_max)))

# Estimate using single period
freq_single, G_single = estimate_freq_response(u, y, N)

# Estimate using averaged periodogram
freq_avg, G_avg = estimate_freq_response_averaged(u, y, N, num_periods)

# Plot results
plt.figure(figsize=(15, 10))

# Plot 1: Time domain signals
plt.subplot(3, 1, 1)
t = np.arange(len(u))
plt.plot(t, u, 'b-', label='Input', alpha=0.5)
plt.plot(t, y, 'r-', label='Output', alpha=0.5)
plt.grid(True)
plt.legend()
plt.title('Time Domain Signals')
plt.xlabel('Sample')

# Plot 2: Magnitude response
plt.subplot(3, 1, 2)
plt.semilogx(np.abs(freq_true[1:N // 2]), 20 * np.log10(np.abs(G_true[1:N // 2])), 'k-', label='True')
plt.semilogx(np.abs(freq_single[1:N // 2]), 20 * np.log10(np.abs(G_single[1:N // 2])), 'b--', label='Single Period')
plt.semilogx(np.abs(freq_avg[1:N // 2]), 20 * np.log10(np.abs(G_avg[1:N // 2])), 'r--', label='Averaged')
plt.grid(True)
plt.legend()
plt.title('Magnitude Response')
plt.ylabel('Magnitude (dB)')

# Plot 3: Phase response
plt.subplot(3, 1, 3)
plt.semilogx(np.abs(freq_true[1:N // 2]), np.angle(G_true[1:N // 2]) * 180 / np.pi, 'k-', label='True')
plt.semilogx(np.abs(freq_single[1:N // 2]), np.angle(G_single[1:N // 2]) * 180 / np.pi, 'b--', label='Single Period')
plt.semilogx(np.abs(freq_avg[1:N // 2]), np.angle(G_avg[1:N // 2]) * 180 / np.pi, 'r--', label='Averaged')
plt.grid(True)
plt.legend()
plt.title('Phase Response')
plt.ylabel('Phase (degrees)')
plt.xlabel('Frequency (normalized)')

plt.tight_layout()
plt.show()


# Calculate and print error metrics
def calculate_errors(G_est, G_true):
    error = np.abs(G_est - G_true)
    mae = np.mean(error)
    rmse = np.sqrt(np.mean(error ** 2))
    return mae, rmse


mae_single, rmse_single = calculate_errors(G_single[1:N // 2], G_true[1:N // 2])
mae_avg, rmse_avg = calculate_errors(G_avg[1:N // 2], G_true[1:N // 2])

print("\nError Metrics:")
print(f"Single Period - MAE: {mae_single:.4f}, RMSE: {rmse_single:.4f}")
print(f"Averaged     - MAE: {mae_avg:.4f}, RMSE: {rmse_avg:.4f}")