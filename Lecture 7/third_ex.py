import numpy as np
import matplotlib.pyplot as plt


class MultiExperimentEstimator:
    def __init__(self, period_length=127, transient_length=None):
        """
        Initialize multi-experiment pulse response estimator

        Args:
            period_length (int): Length of PRBS period
            transient_length (int, optional): Number of transient samples to discard
        """
        self.period_length = period_length
        self.transient_length = transient_length or period_length // 4

    def preprocess_data(self, inputs, outputs):
        """
        Preprocess experimental data by removing transients

        Args:
            inputs (list): List of input signals
            outputs (list): List of output signals

        Returns:
            tuple: Processed inputs and outputs
        """
        processed_inputs = [
            u[self.transient_length:] for u in inputs
        ]
        processed_outputs = [
            y[self.transient_length:] for y in outputs
        ]

        return processed_inputs, processed_outputs

    def compute_noise_variances(self, inputs, outputs):
        """
        Estimate noise variances for each experiment

        Args:
            inputs (list): List of input signals
            outputs (list): List of output signals

        Returns:
            numpy.ndarray: Estimated noise variances for each experiment
        """
        noise_variances = []

        for u, y in zip(inputs, outputs):
            # Construct regression matrix
            max_pulse_length = 30  # As per previous problem specification
            N = len(u)
            Phi = np.zeros((N - max_pulse_length, max_pulse_length))

            for i in range(N - max_pulse_length):
                for j in range(max_pulse_length):
                    Phi[i, j] = u[i + j]

            # Target vector
            Y = y[max_pulse_length:]

            # Least squares solution
            try:
                # Compute solution with full output
                result = np.linalg.lstsq(Phi, Y, rcond=None)

                # Unpack results carefully
                g_hat = result[0]

                # Compute residuals
                y_hat = Phi @ g_hat
                residuals = Y - y_hat

                # Estimate noise variance
                noise_var = np.sum(residuals ** 2) / (N - max_pulse_length)
                noise_variances.append(noise_var)

            except Exception as e:
                print(f"Error in noise variance estimation: {e}")
                # Fallback to standard variance if computation fails
                noise_variances.append(np.var(y))

        return np.array(noise_variances)

    def optimal_weighted_estimation(self, inputs, outputs):
        """
        Perform optimal weighted estimation across multiple experiments

        Args:
            inputs (list): List of input signals
            outputs (list): List of output signals

        Returns:
            tuple: Estimated pulse response and its covariance
        """
        # Preprocess data (remove transients)
        processed_inputs, processed_outputs = self.preprocess_data(inputs, outputs)

        # Compute noise variances
        noise_variances = self.compute_noise_variances(inputs, outputs)

        # Compute weights (inverse variance weighting)
        weights = 1 / noise_variances
        weights /= np.sum(weights)

        # Prepare for weighted estimation
        pulse_responses = []
        regression_matrices = []

        for u, y in zip(processed_inputs, processed_outputs):
            # Construct regression matrix
            N = len(u)
            max_pulse_length = 30
            Phi = np.zeros((N - max_pulse_length, max_pulse_length))

            for i in range(N - max_pulse_length):
                for j in range(max_pulse_length):
                    Phi[i, j] = u[i + j]

            # Target vector
            Y = y[max_pulse_length:]

            # Least squares solution
            g_hat = np.linalg.lstsq(Phi, Y, rcond=None)[0]

            pulse_responses.append(g_hat)
            regression_matrices.append(Phi)

        # Weighted combination
        weighted_pulse_response = np.zeros_like(pulse_responses[0])
        weighted_information_matrix = np.zeros_like(
            regression_matrices[0].T @ regression_matrices[0]
        )

        for w, g_hat, Phi in zip(weights, pulse_responses, regression_matrices):
            # Weighted pulse response
            weighted_pulse_response += w * g_hat

            # Weighted information matrix
            weighted_information_matrix += w * (Phi.T @ Phi)

        # Compute covariance (inverse of weighted information matrix)
        combined_noise_variance = np.mean(noise_variances)
        covariance_matrix = combined_noise_variance * np.linalg.inv(weighted_information_matrix)

        return weighted_pulse_response, covariance_matrix

    def compute_variance_reduction(self, single_exp_variance, combined_variance):
        """
        Compute variance reduction percentage

        Args:
            single_exp_variance (float): Variance from single experiment
            combined_variance (numpy.ndarray): Covariance matrix of combined estimate

        Returns:
            float: Variance reduction percentage
        """
        # Compute trace of single experiment variance
        single_var_trace = single_exp_variance

        # Compute trace of combined variance
        combined_var_trace = np.trace(combined_variance)

        # Compute reduction percentage
        reduction_percentage = (1 - combined_var_trace / single_var_trace) * 100

        return reduction_percentage


def main():
    # Simulate data for demonstration
    np.random.seed(42)
    period_length = 127
    N = 3 * period_length  # 3 experiments, each with period 127

    # Create synthetic input signals (PRBS)
    def generate_prbs(period_length):
        return np.random.choice([-1, 1], size=period_length)

    # Simulate true pulse response
    true_pulse_response = np.exp(-np.linspace(0, 10, 30) / 5)

    # Generate multiple experiments with different noise
    inputs = []
    outputs = []

    for _ in range(3):
        u = generate_prbs(period_length)
        u_full = np.tile(u, 3)  # Repeat for full experiment length
        noise = 0.1 * np.random.randn(N)
        y = np.convolve(u_full, true_pulse_response, mode='same') + noise

        inputs.append(u_full)
        outputs.append(y)

    # Create estimator
    estimator = MultiExperimentEstimator()

    # (a) Optimal Combination Method
    print("Optimal Combination Method:")
    print("1. Inverse Variance Weighting")
    print("2. Remove transient samples")
    print("3. Weighted combination of pulse response estimates")

    # (b) Implement solution
    weighted_pulse_response, combined_covariance = estimator.optimal_weighted_estimation(
        inputs, outputs
    )

    # Compute single experiment variance for comparison
    single_exp_variance = np.var(weighted_pulse_response)

    # (c) Variance Reduction
    variance_reduction = estimator.compute_variance_reduction(
        single_exp_variance, combined_covariance
    )

    print(f"\nVariance Reduction: {variance_reduction:.2f}%")

    # Visualization
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.title('Weighted Pulse Response Estimate')
    plt.plot(weighted_pulse_response)
    plt.xlabel('Lag')
    plt.ylabel('Coefficient')

    plt.subplot(1, 2, 2)
    plt.title('Covariance Matrix Diagonal')
    plt.plot(np.diag(combined_covariance))
    plt.xlabel('Coefficient Index')
    plt.ylabel('Variance')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()