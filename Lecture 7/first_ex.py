import numpy as np
import matplotlib.pyplot as plt


class PulseResponseEstimation:
    def __init__(self, max_duration=500, sampling_time=0.1, max_amplitude=1):
        """
        Initialize pulse response estimation parameters

        Args:
            max_duration (int): Maximum number of samples in the experiment
            sampling_time (float): Sampling time in seconds
            max_amplitude (float): Maximum input signal amplitude
        """
        self.max_duration = max_duration
        self.sampling_time = sampling_time
        self.max_amplitude = max_amplitude

        # Derived parameters
        self.nyquist_freq = np.pi / sampling_time

    def design_periodic_input(self, period_length=255, num_periods=8):
        """
        Design a Pseudo-Random Binary Sequence (PRBS) input signal

        Args:
            period_length (int): Length of a single PRBS period
            num_periods (int): Number of periods in the input signal

        Returns:
            numpy.ndarray: Periodic PRBS input signal
        """
        # Generate PRBS signal with specified period and number of periods
        np.random.seed(42)  # For reproducibility
        prbs = np.random.choice([-1, 1], size=period_length)
        full_signal = np.tile(prbs, num_periods)

        # Scale to max amplitude
        full_signal *= self.max_amplitude

        return full_signal

    def process_transient_data(self, input_signal, output_signal, transient_length=50):
        """
        Remove transient effects from experimental data

        Args:
            input_signal (numpy.ndarray): Input signal
            output_signal (numpy.ndarray): Output signal
            transient_length (int): Number of samples to discard at the start

        Returns:
            tuple: Processed input and output signals
        """
        processed_input = input_signal[transient_length:]
        processed_output = output_signal[transient_length:]

        return processed_input, processed_output

    def compute_frequency_estimates(self, input_signal):
        """
        Compute frequencies for frequency response estimation

        Args:
            input_signal (numpy.ndarray): Input signal to determine frequency resolution

        Returns:
            numpy.ndarray: Vector of frequencies in rad/s
        """
        # Number of samples per period
        period_length = len(input_signal)

        # Compute frequency grid
        frequencies = np.linspace(0, self.nyquist_freq, period_length // 2)

        return frequencies

    def plot_input_signal(self, input_signal):
        """
        Plot the designed input signal
        """
        plt.figure(figsize=(10, 4))
        plt.plot(input_signal)
        plt.title('Periodic PRBS Input Signal')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()


# Demonstration and explanation
def main():
    # Create pulse response estimation object
    pulse_est = PulseResponseEstimation()

    # (a) Design input signal
    input_signal = pulse_est.design_periodic_input()
    pulse_est.plot_input_signal(input_signal)

    print("Input Signal Design Explanation:")
    print("1. PRBS signal chosen for its persistent excitation properties")
    print("2. Uniformly distributed energy across frequencies")
    print(f"3. Signal period: {len(input_signal)} samples")
    print(f"4. Amplitude: ±{pulse_est.max_amplitude}")

    # (b) Demonstrate transient processing
    print("\nTransient Processing:")
    print("Simulating experimental setup with initial 50 samples as transients")

    # Simulated output (for demonstration)
    np.random.seed(42)
    output_signal = np.cumsum(np.random.randn(len(input_signal))) + input_signal
    processed_input, processed_output = pulse_est.process_transient_data(
        input_signal, output_signal
    )

    # (c) Compute frequency estimates
    frequencies = pulse_est.compute_frequency_estimates(input_signal)

    print("\nFrequency Estimation:")
    print(f"Nyquist Frequency: {pulse_est.nyquist_freq:.2f} rad/s")
    print(f"Number of frequency points: {len(frequencies)}")
    print(f"Frequency range: 0 to {frequencies[-1]:.2f} rad/s")


if __name__ == "__main__":
    main()