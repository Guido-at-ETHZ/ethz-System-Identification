import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parameters
N = 1024  # Number of points
fs = 1000  # Sampling frequency
f1 = 50  # First frequency
f2 = 55  # Second frequency (close to f1 to show resolution)
t = np.arange(N) / fs  # Time vector

# Create test signal
x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)


def custom_window(window_type, N, param=None):
    """Create custom windows with adjustable parameters."""
    if window_type == 'rectangular':
        return np.ones(N)
    elif window_type == 'hann':
        return signal.windows.hann(N)
    elif window_type == 'hamming':
        return signal.windows.hamming(N)
    elif window_type == 'gaussian':
        return signal.windows.gaussian(N, param * N / 2)
    elif window_type == 'tukey':
        return signal.windows.tukey(N, param)
    elif window_type == 'kaiser':
        return signal.windows.kaiser(N, param)
    else:
        raise ValueError(f"Unknown window type: {window_type}")


def analyze_window(window_type, N, param=None):
    """Analyze window characteristics."""
    # Create window
    w = custom_window(window_type, N, param)

    # Apply window to signal
    x_windowed = x * w

    # Compute FFTs
    X = np.fft.fft(x_windowed)
    W = np.fft.fft(w)

    # Frequency vector for plotting
    f = np.fft.fftfreq(N, 1 / fs)

    # Compute window characteristics
    mainlobe_width = 2 * fs / N * np.argmin(abs(W[:N // 2]) < max(abs(W)) / np.sqrt(2))
    sidelobe_level = 20 * np.log10(np.max(abs(W[N // 10:])) / np.max(abs(W)))

    return w, X, W, f, mainlobe_width, sidelobe_level


# Window types to analyze
window_types = [
    ('rectangular', None),
    ('hann', None),
    ('hamming', None),
    ('gaussian', 0.3),
    ('tukey', 0.5),
    ('kaiser', 3)
]

# Create subplots
fig1, axes1 = plt.subplots(3, 2, figsize=(15, 12))
fig1.suptitle('Window Functions - Time Domain')
fig2, axes2 = plt.subplots(3, 2, figsize=(15, 12))
fig2.suptitle('Window Functions - Frequency Domain')

# Analyze each window
for idx, (wtype, param) in enumerate(window_types):
    # Get subplot indices
    row, col = idx // 2, idx % 2

    # Analyze window
    w, X, W, f, mainlobe, sidelobe = analyze_window(wtype, N, param)

    # Plot time domain
    axes1[row, col].plot(t, w)
    axes1[row, col].set_title(f'{wtype.capitalize()} Window')
    axes1[row, col].set_xlabel('Time (s)')
    axes1[row, col].set_ylabel('Amplitude')
    axes1[row, col].grid(True)

    # Plot frequency domain
    axes2[row, col].plot(f[:N // 2], 20 * np.log10(np.abs(W[:N // 2]) / np.max(np.abs(W))))
    axes2[row, col].set_title(f'{wtype.capitalize()} Window Spectrum')
    axes2[row, col].set_xlabel('Frequency (Hz)')
    axes2[row, col].set_ylabel('Magnitude (dB)')
    axes2[row, col].set_ylim([-100, 5])
    axes2[row, col].grid(True)

    # Print characteristics
    print(f"\nWindow: {wtype}")
    print(f"Mainlobe width: {mainlobe:.2f} Hz")
    print(f"First sidelobe level: {sidelobe:.2f} dB")


# Parameter variation analysis
def analyze_window_parameters(window_type, param_range):
    fig, ax = plt.subplots(figsize=(10, 6))
    for param in param_range:
        w, _, W, f, _, _ = analyze_window(window_type, N, param)
        ax.plot(f[:N // 2], 20 * np.log10(np.abs(W[:N // 2]) / np.max(np.abs(W))),
                label=f'param={param:.2f}')

    ax.set_title(f'{window_type.capitalize()} Window - Parameter Variation')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_ylim([-100, 5])
    ax.grid(True)
    ax.legend()


# Analyze parameter variation for Gaussian and Kaiser windows
analyze_window_parameters('gaussian', np.arange(0.1, 0.6, 0.1))
analyze_window_parameters('kaiser', [0.5, 1, 2, 4, 8])

plt.show()