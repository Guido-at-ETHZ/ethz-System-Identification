import numpy as np
import matplotlib.pyplot as plt


def exercise1():
    """
    Solution to Exercise 1: Sampling and Frequency Analysis
    """
    # Given parameters
    T = 0.1  # sampling period in seconds
    fs = 1 / T  # sampling frequency in Hz
    f_signal = 15  # highest frequency component in Hz
    N = 100  # number of samples

    # a) Check for aliasing
    f_nyquist = fs / 2
    print(f"a) Nyquist frequency: {f_nyquist} Hz")
    print(f"   Signal frequency: {f_signal} Hz")
    print(f"   Aliasing will{' not' if f_nyquist > f_signal else ''} occur")

    # b) Frequency resolution
    df = fs / N
    print(f"\nb) Frequency resolution: {df} Hz")

    # c) Nyquist frequency in Hz and rad/sec
    nyquist_rad = np.pi / T
    print(f"\nc) Nyquist frequency:")
    print(f"   {f_nyquist} Hz")
    print(f"   {nyquist_rad} rad/sec")

    # d) Visualization of sampling period doubling effect
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * f_signal * t)

    # Original sampling
    t_sampled = np.arange(0, 1, T)
    signal_sampled = np.sin(2 * np.pi * f_signal * t_sampled)

    # Double sampling period
    t_sampled_2T = np.arange(0, 1, 2 * T)
    signal_sampled_2T = np.sin(2 * np.pi * f_signal * t_sampled_2T)

    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, 'b-', label='Original Signal')
    plt.plot(t_sampled, signal_sampled, 'ro', label=f'Sampled (T={T}s)')
    plt.plot(t_sampled_2T, signal_sampled_2T, 'go', label=f'Sampled (T={2 * T}s)')
    plt.legend()
    plt.title('Effect of Doubling Sampling Period')
    plt.grid(True)
    plt.show()


def exercise2():
    """
    Solution to Exercise 2: Periodic Signals and DFT
    """
    # Given parameters
    M = 16  # period
    T = 0.2  # sampling period in seconds

    # a) Number of unique frequencies
    n_freqs = M // 2 + 1
    print(f"a) Number of unique frequencies: {n_freqs}")

    # b) Fundamental frequency
    omega_1 = 2 * np.pi / M
    print(f"\nb) Fundamental frequency: {omega_1} rad/sample")

    # c) All frequencies in rad/sample
    frequencies_rad = np.array([n * omega_1 for n in range(n_freqs)])
    print("\nc) Frequencies in rad/sample:")
    for i, freq in enumerate(frequencies_rad):
        print(f"   ω{i} = {freq:.4f}")

    # d) Frequencies in Hz
    frequencies_hz = frequencies_rad / (2 * np.pi * T)
    print("\nd) Frequencies in Hz:")
    for i, freq in enumerate(frequencies_hz):
        print(f"   f{i} = {freq:.4f} Hz")

    # Visualization
    plt.figure(figsize=(10, 4))
    plt.stem(frequencies_rad, np.ones_like(frequencies_rad))
    plt.title('DFT Frequency Points')
    plt.xlabel('Frequency (rad/sample)')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("Exercise 1 Solution:")
    print("-" * 50)
    exercise1()

    print("\nExercise 2 Solution:")
    print("-" * 50)
    exercise2()