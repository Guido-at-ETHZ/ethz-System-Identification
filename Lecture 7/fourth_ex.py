import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


class CorruptedDataEstimator:
    def __init__(self, system_order=2):
        """
        Initialize estimator for handling corrupted measurements

        Args:
            system_order (int): Known system order
        """
        self.system_order = system_order

    def mask_corrupted_data(self, y, bad_samples):
        """
        Create a mask to identify valid measurements

        Args:
            y (numpy.ndarray): Output measurements
            bad_samples (numpy.ndarray): Indices of corrupted samples

        Returns:
            numpy.ndarray: Boolean mask of valid samples
        """
        mask = np.ones(len(y), dtype=bool)
        mask[bad_samples] = False
        return mask

    def interpolate_corrupted_data(self, y, bad_samples):
        """
        Interpolate corrupted samples using linear interpolation

        Args:
            y (numpy.ndarray): Output measurements
            bad_samples (numpy.ndarray): Indices of corrupted samples

        Returns:
            numpy.ndarray: Measurements with interpolated values
        """
        y_interpolated = y.copy()

        for idx in bad_samples:
            # Find nearest valid neighbors
            lower_valid = max([i for i in range(idx) if i not in bad_samples], default=0)
            upper_valid = min([i for i in range(idx + 1, len(y)) if i not in bad_samples], default=len(y) - 1)

            # Linear interpolation
            y_interpolated[idx] = np.interp(
                idx,
                [lower_valid, upper_valid],
                [y[lower_valid], y[upper_valid]]
            )

        return y_interpolated

    def weighted_least_squares_estimation(self, u, y, bad_samples):
        """
        Perform weighted least squares estimation with corrupted data

        Args:
            u (numpy.ndarray): Input measurements
            y (numpy.ndarray): Output measurements
            bad_samples (numpy.ndarray): Indices of corrupted samples

        Returns:
            tuple: Estimated parameters and their covariance
        """
        # Interpolate corrupted measurements
        y_interpolated = self.interpolate_corrupted_data(y, bad_samples)

        # Create regression matrix
        N = len(u)
        Phi = np.zeros((N - self.system_order, 2 * self.system_order))

        # Construct regression matrix
        for i in range(N - self.system_order):
            # Past outputs
            for j in range(self.system_order):
                Phi[i, j] = -y_interpolated[i + self.system_order - j - 1]

            # Past inputs
            for j in range(self.system_order):
                Phi[i, self.system_order + j] = u[i + self.system_order - j - 1]

        # Target vector
        Y = y_interpolated[self.system_order:]

        # Compute weights (reduce weight for interpolated samples)
        weights = np.ones(len(Y))
        for idx in bad_samples[bad_samples >= self.system_order]:
            weights[idx - self.system_order] = 0.5  # Reduce weight for interpolated samples

        # Weighted least squares
        Phi_weighted = Phi * np.sqrt(weights)[:, np.newaxis]
        Y_weighted = Y * np.sqrt(weights)

        # Solve weighted least squares
        theta_hat, residuals, _, _ = np.linalg.lstsq(Phi_weighted, Y_weighted, rcond=None)

        # Compute covariance
        noise_variance = np.sum(residuals) / (len(Y) - 2 * self.system_order)
        cov_matrix = noise_variance * np.linalg.inv(Phi.T @ Phi)

        return theta_hat, cov_matrix

    def full_data_least_squares(self, u, y):
        """
        Standard least squares estimation without corrupted data handling

        Args:
            u (numpy.ndarray): Input measurements
            y (numpy.ndarray): Output measurements

        Returns:
            tuple: Estimated parameters and their covariance
        """
        # Create regression matrix
        N = len(u)
        Phi = np.zeros((N - self.system_order, 2 * self.system_order))

        # Construct regression matrix
        for i in range(N - self.system_order):
            # Past outputs
            for j in range(self.system_order):
                Phi[i, j] = -y[i + self.system_order - j - 1]

            # Past inputs
            for j in range(self.system_order):
                Phi[i, self.system_order + j] = u[i + self.system_order - j - 1]

        # Target vector
        Y = y[self.system_order:]

        # Standard least squares
        theta_hat, residuals, _, _ = np.linalg.lstsq(Phi, Y, rcond=None)

        # Compute covariance
        noise_variance = np.sum(residuals) / (len(Y) - 2 * self.system_order)
        cov_matrix = noise_variance * np.linalg.inv(Phi.T @ Phi)

        return theta_hat, cov_matrix


def main():
    # Simulate data with corrupted measurements
    np.random.seed(42)
    N = 500

    # Generate true system parameters
    true_a = [-0.6, 0.2]  # AR coefficients
    true_b = [0.3, 0.1]  # Input coefficients

    # Generate input signal
    u = np.random.randn(N)

    # Generate output signal
    y = np.zeros(N)
    for k in range(2, N):
        y[k] = -true_a[0] * y[k - 1] - true_a[1] * y[k - 2] + true_b[0] * u[k - 1] + true_b[1] * u[
            k - 2] + np.random.normal(0, 0.1)

    # Introduce corrupted samples
    bad_samples = np.random.choice(range(N), size=20, replace=False)
    y_corrupted = y.copy()
    y_corrupted[bad_samples] = np.nan

    # Create estimator
    estimator = CorruptedDataEstimator()

    # (a) Estimate with corrupted data handling
    print("(a) Corrupted Data Handling:")
    theta_hat_interpolated, cov_interpolated = estimator.weighted_least_squares_estimation(
        u, y_corrupted, bad_samples
    )
    print("Interpolated Estimate:")
    print("AR Coefficients:", theta_hat_interpolated[:2])
    print("Input Coefficients:", theta_hat_interpolated[2:])

    # (b) Compare with full data estimation
    print("\n(b) Variance Comparison:")
    theta_hat_full, cov_full = estimator.full_data_least_squares(u, y)

    # Print variance information
    print("Full Data Covariance Diagonal:")
    print(np.diag(cov_full))
    print("\nInterpolated Data Covariance Diagonal:")
    print(np.diag(cov_interpolated))

    # (c) Visualization
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.title('Parameter Estimates Comparison')
    plt.bar(
        np.arange(4) - 0.2,
        np.concatenate([true_a, true_b]),
        width=0.4,
        label='True Parameters',
        alpha=0.7
    )
    plt.bar(
        np.arange(4) + 0.2,
        theta_hat_interpolated,
        width=0.4,
        label='Interpolated Estimate',
        alpha=0.7
    )
    plt.xlabel('Parameter Index')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Covariance Comparison')
    plt.bar(
        np.arange(4) - 0.2,
        np.diag(cov_full),
        width=0.4,
        label='Full Data',
        alpha=0.7
    )
    plt.bar(
        np.arange(4) + 0.2,
        np.diag(cov_interpolated),
        width=0.4,
        label='Interpolated',
        alpha=0.7
    )
    plt.xlabel('Parameter Index')
    plt.ylabel('Variance')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()