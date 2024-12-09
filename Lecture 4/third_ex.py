import numpy as np
import matplotlib.pyplot as plt


def calculate_autocorrelation(x, lags):
    """
    Calculate autocorrelation for given lags
    """
    N = len(x)
    x_periodic = np.tile(x, 3)  # Create periodic extension
    R = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        # Calculate autocorrelation for each lag
        R[i] = np.sum(x * np.roll(x, lag)) / N

    return R


def calculate_psd(x):
    """
    Calculate Power Spectral Density using DFT
    """
    N = len(x)
    X = np.fft.fft(x)
    psd = (np.abs(X) ** 2) / N
    freqs = np.fft.fftfreq(N) * 2 * np.pi

    # Sort frequencies and PSD
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    psd = psd[idx]

    return freqs, psd


def exercise3():
    """
    Solution to Exercise 3: Autocorrelation and Power Spectral Density
    """
    # Given signal
    x = np.array([1, 0, -1, 0, 1, 0, -1, 0])
    M = len(x)

    # a) Calculate autocorrelation for τ = 0,1,2,3
    lags = np.arange(4)
    R = calculate_autocorrelation(x, lags)

    print("a) Autocorrelation values:")
    for lag, r in zip(lags, R):
        print(f"   R_x({lag}) = {r:.4f}")

    # b) Verify R_x(0) ≥ |R_x(τ)|
    print("\nb) Verification of R_x(0) ≥ |R_x(τ)|:")
    R_0 = R[0]
    for lag, r in zip(lags[1:], R[1:]):
        print(f"   |R_x({lag})| = {abs(r):.4f} ≤ R_x(0) = {R_0:.4f}: {abs(r) <= R_0}")

    # c) Calculate PSD
    freqs, psd = calculate_psd(x)

    # Find values at ω = 0 and ω = π/4
    idx_0 = np.argmin(np.abs(freqs - 0))
    idx_pi4 = np.argmin(np.abs(freqs - np.pi / 4))

    print("\nc) Power Spectral Density:")
    print(f"   φ_x(e^(j0)) = {psd[idx_0]:.4f}")
    print(f"   φ_x(e^(jπ/4)) = {psd[idx_pi4]:.4f}")

    # Visualizations
    plt.figure(figsize=(12, 8))

    # Plot original signal
    plt.subplot(3, 1, 1)
    plt.stem(range(M), x)
    plt.title('Original Signal x[k]')
    plt.grid(True)

    # Plot autocorrelation
    plt.subplot(3, 1, 2)
    plt.stem(lags, R)
    plt.title('Autocorrelation R_x[τ]')
    plt.grid(True)

    # Plot PSD
    plt.subplot(3, 1, 3)
    plt.plot(freqs, psd)
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (rad/sample)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Additional verification plots
    plt.figure(figsize=(10, 4))
    plt.stem(range(M), x)
    plt.stem(range(M, 2 * M), x, linefmt='r-', markerfmt='ro')
    plt.stem(range(2 * M, 3 * M), x, linefmt='g-', markerfmt='go')
    plt.title('Periodic Extension of Signal')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("Exercise 3 Solution:")
    print("-" * 50)
    exercise3()