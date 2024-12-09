import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def calculate_frequency_hz(omega, T):
    """Convert frequency from rad/sample to Hz"""
    return (omega / (2 * np.pi)) / T


def noise_effect_simulation(omega, T, H_z, num_samples=1000):
    """Simulate effect of noise on frequency response estimation"""
    # Generate sinusoidal input
    t = np.arange(num_samples) * T
    u = np.sin(omega * t / T)

    # Generate noise through H(z)
    e = np.random.randn(num_samples)
    v = signal.lfilter([1], [1, -0.5], e)

    # Calculate frequency response with and without noise
    f = omega / (2 * np.pi * T)  # frequency in Hz
    return u, v, f


def exercise6_solution():
    """Complete solution to Exercise 6"""
    # Given parameters
    T = 0.1  # sampling period (seconds)
    omega = np.pi / 4  # frequency (rad/sample)

    print("Exercise 6 Solution:")
    print("-" * 50)

    # a) Convert frequency to Hz
    freq_hz = calculate_frequency_hz(omega, T)
    print("\na) Frequency conversion:")
    print(f"   ω = π/4 rad/sample = {freq_hz:.2f} Hz")

    # b) Block diagram explanation
    print("\nb) Block diagram components:")
    print("   1. Input u(t) → ZOH → G(s) → Sampler → y(k)")
    print("   2. Noise e(k) → H(z) = 1/(1-0.5z^(-1)) → v(k)")
    print("   3. Output: y(k) = y_s(k) + v(k)")

    # c) Noise effect analysis
    print("\nc) Effect of noise on frequency response estimates:")
    print("   1. The noise model H(z) = 1/(1-0.5z^(-1)) is a first-order IIR filter")
    print("   2. This filter amplifies noise more at low frequencies")
    print("   3. Effect on estimates:")
    print("      - Low frequencies: Higher uncertainty")
    print("      - High frequencies: Lower uncertainty")
    print("      - Frequency response estimates will be biased")

    # d) Methods to minimize noise effect
    print("\nd) Methods to minimize noise effect:")
    print("   1. Averaging multiple experiments")
    print("      - Reduces variance of estimates")
    print("      - Time-consuming but effective")
    print("\n   2. Using frequency-domain smoothing")
    print("      - Reduces noise effects in spectral estimates")
    print("      - Trade-off between bias and variance")

    # Visualizations
    # 1. Frequency response of noise model
    w, h = signal.freqz([1], [1, -0.5], worN=1000)
    freq = w / (2 * np.pi)

    plt.figure(figsize=(12, 8))

    # Magnitude response of noise filter
    plt.subplot(2, 1, 1)
    plt.semilogx(freq, 20 * np.log10(np.abs(h)))
    plt.grid(True)
    plt.title('Frequency Response of Noise Model H(z)')
    plt.ylabel('Magnitude (dB)')

    # Simulate noise effect
    u, v, f = noise_effect_simulation(omega, T, h)

    # Time domain signals
    plt.subplot(2, 1, 2)
    plt.plot(u[:100], label='Input signal')
    plt.plot(v[:100], label='Filtered noise')
    plt.grid(True)
    plt.legend()
    plt.title('Time Domain Signals')
    plt.ylabel('Amplitude')
    plt.xlabel('Sample')

    plt.tight_layout()
    plt.show()

    # Additional visualization: Block diagram
    from matplotlib.patches import Rectangle, Arrow

    plt.figure(figsize=(12, 6))
    plt.subplot(111, aspect='equal')

    # Draw blocks
    blocks = {
        'ZOH': [0.2, 0.4, 0.2, 0.2],
        'G(s)': [0.5, 0.4, 0.2, 0.2],
        'Sampler': [0.8, 0.4, 0.1, 0.2],
        'H(z)': [0.5, 0.1, 0.2, 0.2]
    }

    for name, dims in blocks.items():
        plt.gca().add_patch(Rectangle((dims[0], dims[1]), dims[2], dims[3],
                                      fill=False))
        plt.text(dims[0] + dims[2] / 2, dims[1] + dims[3] / 2, name,
                 horizontalalignment='center', verticalalignment='center')

    # Draw arrows
    arrows = [
        [0.1, 0.5, 0.2, 0.5],  # Input to ZOH
        [0.4, 0.5, 0.5, 0.5],  # ZOH to G(s)
        [0.7, 0.5, 0.8, 0.5],  # G(s) to Sampler
        [0.9, 0.5, 1.0, 0.5],  # Sampler to output
        [0.4, 0.2, 0.5, 0.2],  # Noise to H(z)
        [0.7, 0.2, 0.85, 0.4]  # H(z) to sum
    ]

    for arrow in arrows:
        plt.arrow(arrow[0], arrow[1], arrow[2] - arrow[0], arrow[3] - arrow[1],
                  head_width=0.02, head_length=0.02, fc='k', ec='k')

    # Labels
    plt.text(0.05, 0.5, 'u(t)', verticalalignment='center')
    plt.text(0.95, 0.5, 'y(k)', verticalalignment='center')
    plt.text(0.35, 0.2, 'e(k)', verticalalignment='center')

    plt.title('Block Diagram of Sampled-Data System')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    exercise6_solution()