import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class SecondOrderSystem:
    def __init__(self, w0=0.2 * 2 * np.pi, zeta=0.1, dt=1.0):
        self.w0 = w0
        self.zeta = zeta
        self.dt = dt
        self.state = np.zeros(2)

    def simulate(self, u, noise_level=0.1):
        """Simulates the system response to input u"""
        x = np.zeros((2, len(u)))
        y = np.zeros(len(u))

        for k in range(1, len(u)):
            # Update states
            x[0, k] = x[0, k - 1] + self.dt * x[1, k - 1]
            x[1, k] = (x[1, k - 1] - self.dt * (self.w0 ** 2 * x[0, k - 1] +
                                                2 * self.zeta * self.w0 * x[1, k - 1]) + self.dt * u[k - 1])

            # Output with noise
            y[k] = x[0, k] + noise_level * np.random.randn()

        return y


def generate_sine_sweep(N, f_start, f_end):
    """Generate a logarithmic sine sweep"""
    t = np.arange(N)
    phase = np.exp(np.log(f_end / f_start) * t / N) * f_start
    return np.sin(2 * np.pi * np.cumsum(phase) / N)


def generate_multisine(N, freqs, amplitudes=None):
    """Generate a multi-sine signal"""
    if amplitudes is None:
        amplitudes = np.ones_like(freqs)
    t = np.arange(N)
    signal = np.zeros(N)
    for f, a in zip(freqs, amplitudes):
        signal += a * np.sin(2 * np.pi * f * t / N)
    return signal


def generate_prbs(N, min_switch=5):
    """Generate a PRBS signal"""
    signal = np.ones(N)
    t = 0
    while t < N:
        if t % min_switch == 0:
            signal[t:t + min_switch] *= np.random.choice([-1, 1])
        t += 1
    return signal


# Create test signals
N = 1024
noise_level = 0.1
system = SecondOrderSystem()

# Generate different input signals
t = np.arange(N)

# 1. Sine sweep
u_sweep = generate_sine_sweep(N, 0.01, 0.4)
y_sweep = system.simulate(u_sweep, noise_level)

# 2. Multi-sine
freqs = np.linspace(0.05, 0.35, 10)
u_multi = generate_multisine(N, freqs)
y_multi = system.simulate(u_multi, noise_level)

# 3. PRBS
u_prbs = generate_prbs(N)
y_prbs = system.simulate(u_prbs, noise_level)


def estimate_frequency_response(u, y, window=None):
    """Estimate frequency response using Welch's method"""
    if window is None:
        window = signal.windows.hann(N)

    freq, Pyu = signal.csd(y, u, fs=1.0, window=window, nperseg=N // 2, noverlap=N // 4)
    freq, Puu = signal.welch(u, fs=1.0, window=window, nperseg=N // 2, noverlap=N // 4)

    H = Pyu / Puu
    return freq, H


# Calculate true frequency response for comparison
w_true = np.linspace(0, np.pi, N // 2)
s = 1j * w_true
w0, zeta = system.w0, system.zeta
H_true = 1 / (-w_true ** 2 + 2j * zeta * w0 * w_true + w0 ** 2)

# Estimate frequency responses
window = signal.windows.hann(N // 2)
freq_sweep, H_sweep = estimate_frequency_response(u_sweep, y_sweep, window)
freq_multi, H_multi = estimate_frequency_response(u_multi, y_multi, window)
freq_prbs, H_prbs = estimate_frequency_response(u_prbs, y_prbs, window)

# Plotting
plt.figure(figsize=(15, 12))

# Time domain signals
plt.subplot(421)
plt.plot(t[:100], u_sweep[:100], 'b-', label='Input')
plt.plot(t[:100], y_sweep[:100], 'r-', label='Output')
plt.title('Sine Sweep')
plt.legend()

plt.subplot(423)
plt.plot(t[:100], u_multi[:100], 'b-', label='Input')
plt.plot(t[:100], y_multi[:100], 'r-', label='Output')
plt.title('Multi-sine')
plt.legend()

plt.subplot(425)
plt.plot(t[:100], u_prbs[:100], 'b-', label='Input')
plt.plot(t[:100], y_prbs[:100], 'r-', label='Output')
plt.title('PRBS')
plt.legend()

# Frequency responses - Magnitude
plt.subplot(422)
plt.semilogx(w_true, 20 * np.log10(np.abs(H_true)), 'k-', label='True')
plt.semilogx(freq_sweep, 20 * np.log10(np.abs(H_sweep)), 'r--', label='Estimated')
plt.title('Frequency Response (Sweep)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True)

plt.subplot(424)
plt.semilogx(w_true, 20 * np.log10(np.abs(H_true)), 'k-', label='True')
plt.semilogx(freq_multi, 20 * np.log10(np.abs(H_multi)), 'r--', label='Estimated')
plt.title('Frequency Response (Multi-sine)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True)

plt.subplot(426)
plt.semilogx(w_true, 20 * np.log10(np.abs(H_true)), 'k-', label='True')
plt.semilogx(freq_prbs, 20 * np.log10(np.abs(H_prbs)), 'r--', label='Estimated')
plt.title('Frequency Response (PRBS)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True)

# Phase responses
plt.subplot(428)
plt.semilogx(w_true, np.angle(H_true) * 180 / np.pi, 'k-', label='True')
plt.semilogx(freq_prbs, np.angle(H_prbs) * 180 / np.pi, 'r--', label='PRBS')
plt.semilogx(freq_multi, np.angle(H_multi) * 180 / np.pi, 'g--', label='Multi-sine')
plt.semilogx(freq_sweep, np.angle(H_sweep) * 180 / np.pi, 'b--', label='Sweep')
plt.title('Phase Response Comparison')
plt.ylabel('Phase (degrees)')
plt.xlabel('Frequency (rad/sample)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# Calculate and print error metrics
def calculate_error_metrics(H_est, freq_est, H_true, w_true):
    # Interpolate true response to estimated frequencies
    H_true_interp = np.interp(freq_est, w_true, np.abs(H_true))

    # Calculate errors
    rel_error = np.abs(np.abs(H_est) - H_true_interp) / H_true_interp
    mae = np.mean(np.abs(rel_error))
    rmse = np.sqrt(np.mean(rel_error ** 2))

    return mae, rmse


print("\nError Metrics:")
mae_sweep, rmse_sweep = calculate_error_metrics(H_sweep, freq_sweep, H_true, w_true)
mae_multi, rmse_multi = calculate_error_metrics(H_multi, freq_multi, H_true, w_true)
mae_prbs, rmse_prbs = calculate_error_metrics(H_prbs, freq_prbs, H_true, w_true)

print(f"Sine Sweep - MAE: {mae_sweep:.4f}, RMSE: {rmse_sweep:.4f}")
print(f"Multi-sine - MAE: {mae_multi:.4f}, RMSE: {rmse_multi:.4f}")
print(f"PRBS      - MAE: {mae_prbs:.4f}, RMSE: {rmse_prbs:.4f}")