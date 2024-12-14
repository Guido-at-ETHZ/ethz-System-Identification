import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from typing import Dict, Tuple, NamedTuple


class ARARMAXIdentification:
    @staticmethod
    def generate_system_data(N: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate synthetic system data with ARARMAX characteristics

        Args:
            N (int): Number of data points

        Returns:
            Tuple of input signal, output signal, and true system parameters
        """
        # True system parameters
        true_system = {
            'A': [1, -1.5, 0.7],  # AR parameters
            'B': [0, 0.5, 0.3],  # Input parameters
            'C': [1, -0.6],  # MA noise parameters
            'D': [1, -0.4]  # AR noise parameters
        }

        # Generate input: pseudo-random binary signal
        np.random.seed(42)
        u = 2 * (np.random.rand(N) > 0.5) - 1

        # Preallocate output and noise
        y = np.zeros(N)
        e = np.random.normal(0, 0.1, N)  # White noise

        # Simulate system with ARARMAX dynamics
        for k in range(2, N):
            # Input contribution
            input_term = (true_system['B'][1] * u[k - 1] +
                          true_system['B'][2] * u[k - 2])

            # Autoregressive output terms
            ar_terms = (-true_system['A'][1] * y[k - 1] -
                        true_system['A'][2] * y[k - 2])

            # Noise model
            noise_ma = true_system['C'][1] * e[k - 1]
            noise_ar = -true_system['D'][1] * e[k - 2]

            # Combine terms
            y[k] = input_term + ar_terms + e[k] + noise_ma + noise_ar

        return u, y, true_system

    @staticmethod
    def predict_ararmax(params: np.ndarray, u: np.ndarray,
                        na: int, nb: int, nc: int, nd: int) -> np.ndarray:
        """
        Prediction function for ARARMAX model using optimization

        Args:
            params (np.ndarray): Flattened parameter vector
            u (np.ndarray): Input signal
            na, nb, nc, nd (int): Model order parameters

        Returns:
            np.ndarray: Predicted output
        """
        # Reconstruct parameters
        A = np.concatenate(([1], params[:na]))
        B = np.concatenate(([0], params[na:na + nb]))
        C = np.concatenate(([1], params[na + nb:na + nb + nc]))
        D = np.concatenate(([1], params[na + nb + nc:]))

        N = len(u)
        y_pred = np.zeros(N)

        for k in range(2, N):
            # Input contribution
            input_term = B[1] * u[k - 1] + B[2] * u[k - 2] if nb > 1 else 0

            # Autoregressive output terms
            ar_terms = -A[1] * y_pred[k - 1] - A[2] * y_pred[k - 2] if na > 1 else 0

            y_pred[k] = input_term + ar_terms

        return y_pred

    @classmethod
    def estimate_ararmax(cls, u: np.ndarray, y: np.ndarray,
                         max_orders: Dict[str, int]) -> Dict:
        """
        Estimate ARARMAX model parameters using grid search and optimization

        Args:
            u (np.ndarray): Input signal
            y (np.ndarray): Output signal
            max_orders (Dict): Maximum orders for model parameters

        Returns:
            Dict: Best estimated parameters and model information
        """
        best_aic = np.inf
        best_params = None
        best_model = None

        # Grid search over model orders
        for na in range(1, max_orders['na'] + 1):
            for nb in range(1, max_orders['nb'] + 1):
                for nc in range(1, max_orders['nc'] + 1):
                    for nd in range(1, max_orders['nd'] + 1):
                        try:
                            # Initial parameter guess
                            initial_guess = np.zeros(na + nb + nc + nd)

                            # Cost function for optimization
                            def cost_function(params):
                                y_pred = cls.predict_ararmax(params, u, na, nb, nc, nd)
                                return np.mean((y - y_pred) ** 2)

                            # Optimize parameters
                            result = optimize.minimize(
                                cost_function,
                                initial_guess,
                                method='Nelder-Mead'
                            )

                            # Compute AIC
                            y_pred = cls.predict_ararmax(result.x, u, na, nb, nc, nd)
                            residual_norm = np.mean((y - y_pred) ** 2)
                            num_params = na + nb + nc + nd
                            aic = 2 * num_params + len(y) * np.log(residual_norm)

                            # Track best model
                            if aic < best_aic:
                                best_aic = aic
                                best_params = result.x
                                best_model = {
                                    'orders': {'na': na, 'nb': nb, 'nc': nc, 'nd': nd},
                                    'aic': aic,
                                    'residual_norm': residual_norm
                                }
                        except Exception as e:
                            print(f"Model estimation failed: {e}")
                            continue

        return {
            'params': best_params,
            'model_info': best_model
        }

    @staticmethod
    def run_demonstration():
        """
        Main demonstration function for ARARMAX identification
        """
        # Generate synthetic data
        N = 1000
        u, y, true_system = ARARMAXIdentification.generate_system_data(N)

        # Define maximum orders for grid search
        max_orders = {
            'na': 3, 'nb': 3, 'nc': 3, 'nd': 3
        }

        # Estimate ARARMAX model
        result = ARARMAXIdentification.estimate_ararmax(u, y, max_orders)

        # Simulate prediction
        y_pred = ARARMAXIdentification.predict_ararmax(
            result['params'],
            u,
            result['model_info']['orders']['na'],
            result['model_info']['orders']['nb'],
            result['model_info']['orders']['nc'],
            result['model_info']['orders']['nd']
        )

        # Performance metrics
        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))

        # Plotting
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(y, label='True Output', color='blue')
        plt.plot(y_pred, label='Predicted Output', color='red', linestyle='--')
        plt.title('ARARMAX Model: True vs Predicted Output')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(y - y_pred, label='Prediction Error', color='green')
        plt.title('Prediction Error')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Print results
        print("ARARMAX Model Identification Results:")
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print("\nModel Orders:")
        print(result['model_info']['orders'])
        print("\nAIC:", result['model_info']['aic'])


# Run the demonstration
if __name__ == "__main__":
    ARARMAXIdentification.run_demonstration()