import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class PulseResponseEstimator:
    def __init__(self, max_pulse_length=30, confidence_level=0.95):
        """
        Initialize pulse response estimation parameters

        Args:
            max_pulse_length (int): Maximum length of pulse response
            confidence_level (float): Confidence interval for estimate
        """
        self.max_pulse_length = max_pulse_length
        self.confidence_level = confidence_level

    def estimate_pulse_response(self, u, y):
        """
        Estimate pulse response using least squares method

        Args:
            u (numpy.ndarray): Input signal
            y (numpy.ndarray): Output signal

        Returns:
            numpy.ndarray: Estimated pulse response coefficients
        """
        # Construct regression matrix (Toeplitz matrix)
        N = len(u)
        Phi = np.zeros((N - self.max_pulse_length, self.max_pulse_length))

        for i in range(N - self.max_pulse_length):
            for j in range(self.max_pulse_length):
                Phi[i, j] = u[i + j]

        # Target vector
        Y = y[self.max_pulse_length:]

        # Least squares solution
        g_hat = np.linalg.lstsq(Phi, Y, rcond=None)[0]

        return g_hat

    def estimate_noise_variance(self, u, y, g_hat):
        """
        Estimate noise variance using residuals

        Args:
            u (numpy.ndarray): Input signal
            y (numpy.ndarray): Output signal
            g_hat (numpy.ndarray): Estimated pulse response

        Returns:
            float: Estimated noise variance
        """
        # Construct regression matrix
        N = len(u)
        Phi = np.zeros((N - self.max_pulse_length, self.max_pulse_length))

        for i in range(N - self.max_pulse_length):
            for j in range(self.max_pulse_length):
                Phi[i, j] = u[i + j]

        # Target vector
        Y = y[self.max_pulse_length:]

        # Compute residuals
        y_hat = Phi @ g_hat
        residuals = Y - y_hat

        # Estimate variance (unbiased estimate)
        sigma2 = np.sum(residuals ** 2) / (len(Y) - self.max_pulse_length)

        return sigma2

    def compute_covariance_matrix(self, u, sigma2):
        """
        Compute covariance matrix for pulse response estimate

        Args:
            u (numpy.ndarray): Input signal
            sigma2 (float): Estimated noise variance

        Returns:
            numpy.ndarray: Covariance matrix of pulse response estimate
        """
        # Construct information matrix
        N = len(u)
        Phi = np.zeros((N - self.max_pulse_length, self.max_pulse_length))

        for i in range(N - self.max_pulse_length):
            for j in range(self.max_pulse_length):
                Phi[i, j] = u[i + j]

        # Compute covariance matrix
        cov_matrix = sigma2 * np.linalg.inv(Phi.T @ Phi)

        return cov_matrix

    def compute_confidence_intervals(self, g_hat, cov_matrix):
        """
        Compute confidence intervals for pulse response estimate

        Args:
            g_hat (numpy.ndarray): Estimated pulse response
            cov_matrix (numpy.ndarray): Covariance matrix

        Returns:
            tuple: Lower and upper confidence bounds
        """
        # Compute standard errors
        std_errors = np.sqrt(np.diag(cov_matrix))

        # Compute t-value for given confidence level and degrees of freedom
        df = len(g_hat) - 1
        t_value = stats.t.ppf((1 + self.confidence_level) / 2, df)

        # Compute confidence intervals
        lower_bound = g_hat - t_value * std_errors
        upper_bound = g_hat + t_value * std_errors

        return lower_bound, upper_bound

    def handle_corrupted_measurement(self, u, y, corrupt_index):
        """
        Remove corrupted measurement from data

        Args:
            u (numpy.ndarray): Input signal
            y (numpy.ndarray): Output signal
            corrupt_index (int): Index of corrupted measurement

        Returns:
            tuple: Cleaned input and output signals
        """
        # Create masks to remove corrupted measurement
        mask = np.ones(len(y), dtype=bool)
        mask[corrupt_index] = False

        u_cleaned = u[mask]
        y_cleaned = y[mask]

        return u_cleaned, y_cleaned

    def plot_pulse_response(self, g_hat, lower_bound, upper_bound):
        """
        Plot estimated pulse response with confidence intervals

        Args:
            g_hat (numpy.ndarray): Estimated pulse response
            lower_bound (numpy.ndarray): Lower confidence bound
            upper_bound (numpy.ndarray): Upper confidence bound
        """
        plt.figure(figsize=(10, 6))

        # Pulse response plot
        plt.plot(g_hat, label='Estimated Pulse Response', color='blue')
        plt.fill_between(
            range(len(g_hat)),
            lower_bound,
            upper_bound,
            alpha=0.3,
            color='blue',
            label=f'{self.confidence_level * 100}% Confidence Interval'
        )

        plt.title('Pulse Response Estimate with Confidence Intervals')
        plt.xlabel('Lag')
        plt.ylabel('Pulse Response Coefficient')
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    # Simulate data for demonstration
    np.random.seed(42)
    N = 500

    # Simulate a simple stable system
    true_pulse_response = np.exp(-np.linspace(0, 10, 30) / 5)

    # Generate input and output signals
    u = np.random.randn(N)
    noise = 0.1 * np.random.randn(N)
    y = np.convolve(u, true_pulse_response, mode='same') + noise

    # Create estimator
    estimator = PulseResponseEstimator()

    # (a) Estimate pulse response
    g_hat = estimator.estimate_pulse_response(u, y)
    print("Pulse Response Estimation Method:")
    print("1. Constructed Toeplitz regression matrix")
    print("2. Used least squares to estimate coefficients")
    print("3. Constrained to maximum length of 30")

    # (b) Compute covariance matrix
    sigma2 = estimator.estimate_noise_variance(u, y, g_hat)
    cov_matrix = estimator.compute_covariance_matrix(u, sigma2)
    print(f"\nNoise Variance Estimate: {sigma2:.4f}")

    # (c) Plot with confidence intervals
    lower_bound, upper_bound = estimator.compute_confidence_intervals(g_hat, cov_matrix)
    estimator.plot_pulse_response(g_hat, lower_bound, upper_bound)

    # (d) Handling corrupted measurement
    print("\nHandling Corrupted Measurement:")
    u_cleaned, y_cleaned = estimator.handle_corrupted_measurement(u, y, corrupt_index=100)
    print(f"Removed measurement at index 100")
    print(f"Original data length: {len(y)}")
    print(f"Cleaned data length: {len(y_cleaned)}")


if __name__ == "__main__":
    main()