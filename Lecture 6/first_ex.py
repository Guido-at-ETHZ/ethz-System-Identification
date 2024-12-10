import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def generate_motor_data(Ts=0.01, T=20):
    """Generate simulated DC motor data
    Args:
        Ts: sampling time in seconds
        T: experiment duration in seconds
    Returns:
        t: time vector
        u: input signal
        y: noise-free output
        y_noisy: noisy output
    """
    # Time vector
    t = np.arange(0, T, Ts)

    # Generate multi-sine input
    f1, f2, f3 = 0.5, 2, 5  # Hz
    u = (np.sin(2 * np.pi * f1 * t) +
         0.5 * np.sin(2 * np.pi * f2 * t) +
         0.25 * np.sin(2 * np.pi * f3 * t))

    # DC motor parameters
    K = 1.0  # Motor gain
    J = 0.01  # Inertia
    b = 0.1  # Damping

    # Create transfer functions
    num = [K]
    den = [J, b, K]

    # Create discrete transfer function
    system = signal.TransferFunction(num, den)
    dt = signal.cont2discrete((system.num, system.den), Ts, method='zoh')

    # Get numerator and denominator of discrete system
    num_d = dt[0][0]
    den_d = dt[1]

    # Simulate system
    y = signal.lfilter(num_d, den_d, u)

    # Add noise
    noise_level = 0.1
    y_noisy = y + noise_level * np.random.randn(*y.shape)

    return t, u, y, y_noisy


def calculate_etfe(y, u, Ts):
    """Calculate Empirical Transfer Function Estimate"""
    N = len(y)

    # Calculate FFT
    Y = np.fft.fft(y)
    U = np.fft.fft(u)

    # Positive frequencies up to Nyquist
    nfreq = N // 2
    omega = 2 * np.pi / (N * Ts) * np.arange(nfreq)

    # Calculate ETFE
    G_hat = Y[:nfreq] / U[:nfreq]

    return G_hat, omega


def smooth_etfe(G_hat, gamma):
    """Apply window smoothing to ETFE"""
    N = len(G_hat)
    G_smooth = np.zeros_like(G_hat, dtype=complex)

    for k in range(N):
        # Window indices
        start = max(0, k - gamma)
        end = min(N, k + gamma + 1)
        indices = slice(start, end)

        # Create Hann window
        window = np.hanning(end - start)
        window = window / np.sum(window)

        # Apply window
        G_smooth[k] = np.sum(G_hat[indices] * window)

    return G_smooth


def plot_results(omega, G_hat, G_smooth, G_true):
    """Plot frequency response estimates"""
    plt.figure(figsize=(10, 8))

    # Magnitude plot
    plt.subplot(211)
    plt.semilogx(omega, 20 * np.log10(np.abs(G_hat)), 'b.', label='Raw ETFE')
    plt.semilogx(omega, 20 * np.log10(np.abs(G_smooth)), 'r-', label='Smoothed')
    plt.semilogx(omega, 20 * np.log10(np.abs(G_true)), 'k--', label='True')
    plt.grid(True)
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.title('Frequency Response Estimation')

    # Phase plot
    plt.subplot(212)
    plt.semilogx(omega, np.angle(G_hat, deg=True), 'b.')
    plt.semilogx(omega, np.angle(G_smooth, deg=True), 'r-')
    plt.semilogx(omega, np.angle(G_true, deg=True), 'k--')
    plt.grid(True)
    plt.ylabel('Phase (deg)')
    plt.xlabel('Frequency (rad/s)')

    plt.tight_layout()
    plt.show()


def main():
    # Generate data
    t, u, y, y_noisy = generate_motor_data()

    # Calculate ETFE
    G_hat, omega = calculate_etfe(y_noisy, u, Ts=0.01)

    # Smooth ETFE
    gamma = 50
    G_smooth = smooth_etfe(G_hat, gamma)

    # Get true frequency response
    K, J, b = 1.0, 0.01, 0.1
    sys = signal.TransferFunction([K], [J, b, K])
    w, mag, phase = signal.bode(sys, omega)
    G_true = mag * np.exp(1j * phase * np.pi / 180)

    # Plot results
    plot_results(omega, G_hat, G_smooth, G_true)


if __name__ == '__main__':
    main()