import numpy as np
import matplotlib.pyplot as plt


def calculate_dft_manually(x, k):
    """
    Calculate DFT manually for pedagogical purposes
    X[k] = Σ x[n]e^(-j2πnk/N)
    """
    N = len(x)
    X_k = 0
    for n in range(N):
        X_k += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X_k


def exercise5_solution():
    """Complete solution to Exercise 5"""
    # Given sequence
    x = np.array([1, 2, 1, 0, -1, 0, 1, 0])
    N = len(x)

    # a) Calculate DFT for n = 0,1,2,3
    # Using both manual calculation and numpy.fft for verification
    X_manual = np.zeros(4, dtype=complex)
    for k in range(4):
        X_manual[k] = calculate_dft_manually(x, k)

    # Using numpy.fft for full DFT
    X_full = np.fft.fft(x)

    print("Exercise 5 Solution:")
    print("-" * 50)
    print("\na) DFT values for n = 0,1,2,3:")

    for k in range(4):
        print(f"\nX[{k}]:")
        print(f"   Magnitude: {abs(X_manual[k]):.4f}")
        print(f"   Phase: {np.angle(X_manual[k], deg=True):.2f} degrees")
        print(f"   Real part: {X_manual[k].real:.4f}")
        print(f"   Imaginary part: {X_manual[k].imag:.4f}")

    # b) Physical meaning of X[0]
    dc_component = X_manual[0] / N
    mean_value = np.mean(x)
    print("\nb) Physical meaning of X[0]:")
    print(f"   X[0] = {X_manual[0]:.4f} = N * {dc_component:.4f}")
    print(f"   Mean of signal = {mean_value:.4f}")
    print("   X[0] represents the DC component (average) of the signal scaled by N")

    # c) Symmetry relationship
    print("\nc) Relationship between X[n] and X[N-n]:")
    print("   For real signals, X[n] and X[N-n] are complex conjugates:")
    for k in range(1, N // 2):
        print(f"\n   X[{k}] and X[{N - k}]:")
        print(f"   X[{k}] = {X_full[k]:.4f}")
        print(f"   X[{N - k}] = {X_full[N - k]:.4f}")

    # Visualizations
    plt.figure(figsize=(12, 10))

    # Original signal
    plt.subplot(3, 1, 1)
    plt.stem(range(N), x)
    plt.title('Original Signal x[n]')
    plt.grid(True)

    # DFT magnitude
    plt.subplot(3, 1, 2)
    plt.stem(range(N), np.abs(X_full))
    plt.title('DFT Magnitude |X[k]|')
    plt.grid(True)

    # DFT phase
    plt.subplot(3, 1, 3)
    plt.stem(range(N), np.angle(X_full, deg=True))
    plt.title('DFT Phase ∠X[k] (degrees)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Additional visualization for symmetry
    plt.figure(figsize=(10, 6))
    plt.plot(range(N), np.real(X_full), 'b-', label='Real part')
    plt.plot(range(N), np.imag(X_full), 'r--', label='Imaginary part')
    plt.title('DFT Symmetry: Real and Imaginary Parts')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    exercise5_solution()