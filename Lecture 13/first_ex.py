import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def generate_test_system():
    """Generate a test state space system."""
    # Define a simple 2nd order system
    A = np.array([[0.8, 0.2],
                  [-0.2, 0.7]])
    B = np.array([[1.0],
                  [0.5]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])

    return A, B, C, D


def generate_frequency_data(A, B, C, D, frequencies, noise_level=0.0):
    """Generate frequency response data for the system."""
    # Create state space system
    sys = signal.StateSpace(A, B, C, D, dt=1.0)

    # Get frequency response
    w, mag, phase = signal.dbode(sys, w=frequencies)

    # Convert to complex frequency response
    G = mag * np.exp(1j * phase)

    # Add noise if specified
    if noise_level > 0:
        noise = noise_level * (np.random.randn(len(frequencies)) +
                               1j * np.random.randn(len(frequencies)))
        G = G + noise

    return G


def construct_hankel_matrix(impulse_response, num_rows, num_cols):
    """Construct Hankel matrix from impulse response data."""
    hankel = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            if i + j < len(impulse_response):
                hankel[i, j] = impulse_response[i + j]
    return hankel


def subspace_id_simple(G, frequencies, order, num_block_rows=10):
    """Simple implementation of frequency-domain subspace identification."""
    # Step 1: Get time domain response via IFFT
    h = np.real(np.fft.ifft(G))

    # Step 2: Construct Hankel matrix
    H = construct_hankel_matrix(h, num_block_rows, len(h) - num_block_rows)

    # Step 3: SVD of Hankel matrix
    U, S, Vh = np.linalg.svd(H)

    # Step 4: Truncate to desired order
    U1 = U[:, :order]
    S1 = np.diag(S[:order])
    V1 = Vh[:order, :].T

    # Step 5: Estimate C and A matrices
    C_est = U1[0:1, :]  # First block row
    U1_shifted = U1[1:, :]  # Remove first row
    U1_unshifted = U1[:-1, :]  # Remove last row
    A_est = np.linalg.lstsq(U1_unshifted, U1_shifted, rcond=None)[0]

    return A_est, C_est, S


def plot_results(frequencies, G_true, G_est):
    """Plot the true and estimated frequency responses."""
    plt.figure(figsize=(12, 6))

    # Magnitude plot
    plt.subplot(121)
    plt.semilogx(frequencies, 20 * np.log10(np.abs(G_true)), 'b-', label='True')
    plt.semilogx(frequencies, 20 * np.log10(np.abs(G_est)), 'r--', label='Estimated')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Magnitude (dB)')

    # Phase plot
    plt.subplot(122)
    plt.semilogx(frequencies, np.angle(G_true) * 180 / np.pi, 'b-', label='True')
    plt.semilogx(frequencies, np.angle(G_est) * 180 / np.pi, 'r--', label='Estimated')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Phase (degrees)')

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate test system
    A, B, C, D = generate_test_system()

    # Generate frequency response data
    frequencies = np.logspace(-2, 2, 200)
    G_true = generate_frequency_data(A, B, C, D, frequencies)
    G_noisy = generate_frequency_data(A, B, C, D, frequencies, noise_level=0.05)

    # Perform subspace identification
    A_est, C_est, S = subspace_id_simple(G_noisy, frequencies, order=2)

    # Generate frequency response of estimated model
    sys_est = signal.StateSpace(A_est, np.ones((2, 1)), C_est, [[0]], dt=1.0)
    _, mag_est, phase_est = signal.dbode(sys_est, w=frequencies)
    G_est = mag_est * np.exp(1j * phase_est)

    # Plot results
    plot_results(frequencies, G_true, G_est)

    # Plot singular values
    plt.figure()
    plt.semilogy(S, 'bo-')
    plt.grid(True)
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title('Hankel Matrix Singular Values')
    plt.show()