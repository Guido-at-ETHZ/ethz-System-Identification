import numpy as np
import matplotlib.pyplot as plt


def continuous_signal(t):
    """Generate the continuous signal x(t) = sin(2πt) + 0.5sin(6πt)"""
    return np.sin(2 * np.pi * t) + 0.5 * np.sin(6 * np.pi * t)


def zoh_reconstruction(t_cont, t_samples, x_samples):
    """Reconstruct signal using Zero-Order Hold"""
    x_recon = np.zeros_like(t_cont)
    for i in range(len(t_cont)):
        # Find the last sample before current time
        idx = np.where(t_samples <= t_cont[i])[0]
        if len(idx) > 0:
            x_recon[i] = x_samples[idx[-1]]
    return x_recon


def exercise4_solution():
    """Complete solution to Exercise 4"""
    # Parameters
    T = 0.25  # sampling period
    t_end = 1.0  # end time for visualization

    # a) Create signals and plot
    # Fine time grid for continuous signal
    t_cont = np.linspace(0, t_end, 1000)
    x_cont = continuous_signal(t_cont)

    # Sampling times
    t_samples = np.arange(0, t_end + T, T)
    x_samples = continuous_signal(t_samples)

    # ZOH reconstruction
    x_recon = zoh_reconstruction(t_cont, t_samples, x_samples)

    # Plotting
    plt.figure(figsize=(12, 8))

    # Original and reconstructed signals
    plt.subplot(2, 1, 1)
    plt.plot(t_cont, x_cont, 'b-', label='Original x(t)')
    plt.plot(t_cont, x_recon, 'r-', label='ZOH Reconstruction')
    plt.plot(t_samples, x_samples, 'ko', label='Samples')
    plt.grid(True)
    plt.legend()
    plt.title('Signal Reconstruction with ZOH')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Frequency content
    freq = np.fft.fftfreq(len(t_cont), t_cont[1] - t_cont[0])
    X_cont = np.fft.fft(x_cont)

    plt.subplot(2, 1, 2)
    plt.plot(freq[:len(freq) // 2], np.abs(X_cont)[:len(freq) // 2])
    plt.axvline(1 / (2 * T), color='r', linestyle='--', label=f'Nyquist Frequency ({1 / (2 * T)} Hz)')
    plt.grid(True)
    plt.legend()
    plt.title('Frequency Content of Original Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.show()

    # Print analysis
    print("Exercise 4 Solution:")
    print("-" * 50)

    # b) Information loss analysis
    print("\nb) Information Loss Analysis:")
    f1 = 1  # First frequency component (Hz)
    f2 = 3  # Second frequency component (Hz)
    fs = 1 / T  # Sampling frequency (Hz)
    f_nyquist = fs / 2  # Nyquist frequency
    print(f"   - Signal contains frequencies: {f1} Hz and {f2} Hz")
    print(f"   - Sampling frequency: {fs} Hz")
    print(f"   - Nyquist frequency: {f_nyquist} Hz")
    if f_nyquist < max(f1, f2):
        print("   - Information is lost due to aliasing (sampling frequency too low)")
    else:
        print("   - No information loss due to aliasing")

    # c) Maximum sampling period
    f_max = max(f1, f2)  # Maximum frequency in signal
    T_max = 1 / (2 * f_max)
    print(f"\nc) Maximum allowed sampling period:")
    print(f"   T_max = {T_max:.3f} seconds")
    print(f"   This ensures sampling frequency is at least {2 * f_max} Hz")


if __name__ == "__main__":
    exercise4_solution()