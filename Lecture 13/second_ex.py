import numpy as np
from scipy import signal, linalg
import matplotlib.pyplot as plt
from scipy.linalg import block_diag


class ModalSystem:
    """Class representing a system with multiple resonant modes with better numerical conditioning"""

    def __init__(self, modes, damping_ratios, gains):
        """
        Initialize system with multiple modes using better numerical scaling
        """
        self.modes = np.array(modes)
        self.damping_ratios = np.array(damping_ratios)
        self.gains = np.array(gains)

        # Create state space matrices with better scaling
        n_modes = len(modes)
        A_blocks = []
        B_blocks = []
        C_blocks = []

        # Build block diagonal matrices with better scaling
        for i in range(n_modes):
            wn = modes[i]
            zeta = damping_ratios[i]
            k = gains[i]

            # Scale the state-space realization
            scale = np.sqrt(wn)

            A_block = np.array([
                [-2 * zeta * wn / scale, -wn / scale],
                [wn / scale, 0]
            ])
            B_block = np.array([[k / scale], [0]])
            C_block = np.array([[scale, 0]])

            A_blocks.append(A_block)
            B_blocks.append(B_block)
            C_blocks.append(C_block)

        # Construct full matrices
        self.A = block_diag(*A_blocks)
        self.B = np.vstack(B_blocks)
        self.C = np.hstack(C_blocks)
        self.D = np.zeros((1, 1))


def generate_frequency_data(system, frequencies, noise_std=0.0):
    """Generate frequency response data using direct computation"""
    n_freqs = len(frequencies)
    G = np.zeros(n_freqs, dtype=complex)

    for i, w in enumerate(frequencies):
        # Compute frequency response directly
        resp = system.C @ np.linalg.solve(1j * w * np.eye(system.A.shape[0]) - system.A, system.B)
        G[i] = resp.flatten()[0]

    # Add noise if specified
    if noise_std > 0:
        noise = noise_std * (np.random.randn(n_freqs) + 1j * np.random.randn(n_freqs))
        G = G + noise

    return G


def identify_system(G, frequencies, order, num_block_rows=20):
    """Identify system using numerically stable subspace method"""
    # Scale frequencies to improve numerical conditioning
    max_freq = np.max(frequencies)
    frequencies_scaled = frequencies / max_freq

    # Construct block Hankel matrix using scaled frequency response
    N = len(frequencies)
    dt = 2 * np.pi / (N * np.max(frequencies_scaled))
    h = np.real(np.fft.ifft(G) * N)

    # Build Hankel matrix with proper scaling
    H = np.zeros((num_block_rows, N - num_block_rows))
    for i in range(num_block_rows):
        for j in range(N - num_block_rows):
            H[i, j] = h[(i + j) % N]

    # Scale Hankel matrix
    scale_factor = np.max(np.abs(H))
    H = H / scale_factor

    # SVD with full matrices
    U, S, Vh = np.linalg.svd(H, full_matrices=False)

    # Truncate to desired order
    U1 = U[:, :order]
    S1 = S[:order]
    V1 = Vh[:order, :]

    # Estimate A and C with regularization
    U1_up = U1[:-1, :]
    U1_down = U1[1:, :]

    # Improved regularization
    reg = 1e-8 * np.max(np.abs(U1_up.T @ U1_up))
    A_est = linalg.solve(U1_up.T @ U1_up + reg * np.eye(order),
                         U1_up.T @ U1_down)

    # Scale back the state-space realization
    A_est = A_est * max_freq
    C_est = U1[0:1, :] * scale_factor
    B_est = np.ones((order, 1))  # Simplified B estimation
    D_est = np.zeros((1, 1))

    return A_est, B_est, C_est, D_est, S


def plot_identification_results(freqs, G_true, G_est, S):
    """Plot the identification results"""
    plt.figure(figsize=(15, 5))

    # Magnitude plot with better scaling
    plt.subplot(131)
    plt.semilogx(freqs, 20 * np.log10(np.abs(G_true)), 'b-', label='True')
    plt.semilogx(freqs, 20 * np.log10(np.abs(G_est)), 'r--', label='Identified')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Magnitude (dB)')

    # Phase plot
    plt.subplot(132)
    true_phase = np.unwrap(np.angle(G_true))
    est_phase = np.unwrap(np.angle(G_est))
    plt.semilogx(freqs, true_phase * 180 / np.pi, 'b-', label='True')
    plt.semilogx(freqs, est_phase * 180 / np.pi, 'r--', label='Identified')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Phase (deg)')

    # Singular values
    plt.subplot(133)
    plt.semilogy(S / S[0], 'bo-')
    plt.grid(True)
    plt.xlabel('Index')
    plt.ylabel('Normalized Singular Value')
    plt.title('Hankel Matrix Singular Values')

    plt.tight_layout()
    plt.show()


# Example usage with better numerical conditioning
if __name__ == "__main__":
    # Create a system with 3 modes - using better frequency spacing
    modes = [10, 25, 40]  # rad/s - better separated modes
    damping = [0.05, 0.04, 0.03]  # increased damping for better conditioning
    gains = [1.0, 0.7, 0.5]  # scaled gains

    # Create true system
    true_system = ModalSystem(modes, damping, gains)

    # Generate frequency response data with better frequency spacing
    frequencies = np.logspace(0, 2, 200)  # reduced number of points
    G_true = generate_frequency_data(true_system, frequencies)
    G_noisy = generate_frequency_data(true_system, frequencies, noise_std=0.005)

    # Identify system
    order = 6  # 2 * number of modes
    A_id, B_id, C_id, D_id, S = identify_system(G_noisy, frequencies, order)

    # Generate frequency response of identified system using direct computation
    G_id = np.zeros(len(frequencies), dtype=complex)
    for i, w in enumerate(frequencies):
        resp = C_id @ np.linalg.solve(1j * w * np.eye(A_id.shape[0]) - A_id, B_id)
        G_id[i] = resp.flatten()[0]

    # Plot results
    plot_identification_results(frequencies, G_true, G_id, S)