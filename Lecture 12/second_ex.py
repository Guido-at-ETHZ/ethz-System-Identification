import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class IndirectIdentification:
    def __init__(self, Ts=0.1):
        """
        Initialize the identification system
        Ts: sampling time
        """
        self.Ts = Ts

    def generate_reference(self, T, amplitude=1.0):
        """
        Generate a multi-sine reference signal
        """
        t = np.arange(0, T, self.Ts)
        r = np.zeros_like(t)

        # Sum of sines at different frequencies
        frequencies = [0.1, 0.5, 1.0, 2.0]
        for freq in frequencies:
            r += amplitude / len(frequencies) * np.sin(2 * np.pi * freq * t)

        return t, r

    def compute_closed_loop_tf(self, G_num, G_den, C_num, C_den):
        """
        Compute closed-loop transfer function manually
        """
        # Multiply G and C numerators and denominators
        num = np.convolve(G_num, C_num)
        den = np.convolve(G_den, C_den)

        # Compute 1 + GC
        aug_num = np.zeros(max(len(num), len(den)))
        aug_num[:len(num)] = num
        aug_den = np.zeros(max(len(num), len(den)))
        aug_den[:len(den)] = den

        cl_den = aug_den + aug_num
        cl_num = num

        return cl_num, cl_den

    def true_system(self):
        """
        Define the true system G(s) and controller C(s)
        G(s) = 1/(s+1)
        C(s) = 2 + 1/s (PI controller)
        """
        # Discretize the system using ZOH
        G_num = [1]
        G_den = [1, 1]
        C_num = [2, 1]
        C_den = [1, 0]

        G_d = signal.cont2discrete((G_num, G_den), self.Ts, method='zoh')
        C_d = signal.cont2discrete((C_num, C_den), self.Ts, method='tustin')

        return G_d[0].flatten(), G_d[1].flatten(), C_d[0].flatten(), C_d[1].flatten()

    def simulate_closed_loop(self, t, r, noise_std=0.1):
        """
        Simulate the closed-loop system
        """
        G_num, G_den, C_num, C_den = self.true_system()

        # Compute closed-loop transfer function
        T_num, T_den = self.compute_closed_loop_tf(G_num, G_den, C_num, C_den)

        # Normalize coefficients
        T_num = T_num / np.max(np.abs(T_num))
        T_den = T_den / np.max(np.abs(T_den))

        # Simulate
        y = signal.lfilter(T_num, T_den, r)
        y = y + noise_std * np.random.randn(len(t))

        return y

    def identify_closed_loop(self, t, r, y):
        """
        Identify closed-loop transfer function using prediction error method
        """

        def tf_predict(params, u):
            """Helper function to simulate transfer function"""
            b = np.array([params[0], params[1]])
            a = np.array([1.0, params[2], params[3]])

            # Normalize coefficients
            b = b / np.max(np.abs(b))
            a = a / np.max(np.abs(a))

            return signal.lfilter(b, a, u)

        def cost_function(params):
            """Prediction error cost function"""
            y_pred = tf_predict(params, r)
            return np.mean((y - y_pred) ** 2)

        # Initial parameter guess
        initial_params = [0.1, 0.1, 0.5, 0.5]

        # Optimize
        result = minimize(cost_function, initial_params, method='Nelder-Mead')

        return result.x

    def plot_results(self, t, r, y, y_true):
        """
        Plot time domain results
        """
        plt.figure(figsize=(12, 6))
        plt.plot(t, r, 'g-', label='Reference')
        plt.plot(t, y, 'b-', label='Measured Output')
        plt.plot(t, y_true, 'r--', label='True Output')
        plt.grid(True)
        plt.legend()
        plt.title('Time Domain Response')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.show()


def main():
    # Initialize identification object
    id_sys = IndirectIdentification(Ts=0.1)

    # Generate reference signal
    T = 100  # simulation time
    t, r = id_sys.generate_reference(T)

    # Simulate system
    y = id_sys.simulate_closed_loop(t, r)

    # Simulate system without noise for comparison
    y_true = id_sys.simulate_closed_loop(t, r, noise_std=0)

    # Identify closed-loop model
    cl_params = id_sys.identify_closed_loop(t, r, y)
    print("Identified closed-loop parameters:", cl_params)

    # Plot results
    id_sys.plot_results(t, r, y, y_true)


if __name__ == "__main__":
    main()