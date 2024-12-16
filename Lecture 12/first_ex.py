import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def generate_prbs(length, prob_switch=0.2):
    """
    Generate a Pseudo-Random Binary Sequence for system excitation.
    """
    signal = np.zeros(length)
    current_value = 1.0

    for i in range(length):
        if np.random.random() < prob_switch:
            current_value *= -1
        signal[i] = current_value

    return signal


def simulate_closed_loop(G, C, r, noise_level=0.1):
    """
    Simulate closed-loop system with controller and reference.
    """
    T = length = len(r)
    t = np.linspace(0, T, length)

    y = np.zeros(length)
    u = np.zeros(length)
    e = np.zeros(length)

    for k in range(1, length):
        e[k] = r[k] - y[k - 1]
        u[k] = u[k - 1] + C[0] * e[k] + C[1] * (e[k] - e[k - 1])

        if k > 1:
            y[k] = -G[1] * y[k - 1] - G[2] * y[k - 2] + G[0] * u[k - 1]

        y[k] += np.random.normal(0, noise_level)

    return t, y, u


def estimate_frequency_response(y, u, Fs):
    """
    Estimate frequency response using spectral analysis with proper handling of numerical issues.
    """
    # Use smaller segments to avoid memory issues
    nperseg = min(len(y) // 4, 256)  # Use shorter segments

    # Add small regularization to avoid division by zero
    eps = 1e-10

    # Compute power spectral densities with overlap
    f, Pyu = signal.csd(y, u, fs=Fs, nperseg=nperseg, noverlap=nperseg // 2)
    f, Puu = signal.welch(u, fs=Fs, nperseg=nperseg, noverlap=nperseg // 2)

    # Add regularization and compute transfer function
    G_hat = Pyu / (Puu + eps)

    # Remove any remaining invalid values
    G_hat = np.nan_to_num(G_hat, nan=0.0, posinf=0.0, neginf=0.0)

    return f, G_hat


def plot_results(t, y, u, r, f, G_hat):
    """
    Plot time and frequency domain results.
    """
    plt.figure(figsize=(12, 10))

    # Time domain plots
    plt.subplot(411)
    plt.plot(t, r, 'b-', label='Reference')
    plt.grid(True)
    plt.legend()
    plt.title('Reference Signal')

    plt.subplot(412)
    plt.plot(t, y, 'r-', label='Output')
    plt.grid(True)
    plt.legend()
    plt.title('System Output')

    plt.subplot(413)
    plt.plot(t, u, 'g-', label='Control')
    plt.grid(True)
    plt.legend()
    plt.title('Control Signal')

    # Frequency domain plot
    plt.subplot(414)
    plt.semilogx(f, 20 * np.log10(np.abs(G_hat)))
    plt.grid(True)
    plt.title('Estimated Frequency Response')
    plt.ylabel('Magnitude (dB)')
    plt.xlabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()


def main():
    # Simulation parameters
    T = 1000
    r = generate_prbs(T, prob_switch=0.1)  # Reduced switching probability

    # System and controller parameters
    G = [1.0, -0.8, 0.15]  # System coefficients
    C = [0.5, 0.1]  # Reduced controller gains

    # Simulate system
    t, y, u = simulate_closed_loop(G, C, r, noise_level=0.05)  # Reduced noise level

    # Estimate frequency response
    Fs = 1 / (t[1] - t[0])
    f, G_hat = estimate_frequency_response(y, u, Fs)

    # Plot results
    plot_results(t, y, u, r, f, G_hat)


if __name__ == "__main__":
    main()