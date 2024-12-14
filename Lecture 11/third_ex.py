import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def generate_system_data(N):
    """Generate data from the given system."""
    u = np.random.randn(N)  # Random input
    e = 0.1 * np.random.randn(N)  # Noise
    y = np.zeros(N)

    # Implementing the system:
    # y(k) = 0.8*y(k-1) - 0.2*y(k-2) + u(k-1) + 0.5*u(k-2) + e(k) + 0.3*e(k-1)
    for k in range(2, N):  # Start from 2 because we need two previous values
        y[k] = 0.8*y[k-1] - 0.2*y[k-2] + \
               u[k-1] + 0.5*u[k-2] + \
               e[k] + 0.3*e[k-1]

    return y, u, e

# Generate data
N = 1000
y, u, e = generate_system_data(N)

# Split data for training and validation
train_idx = int(0.7 * N)
y_train = y[:train_idx]
u_train = u[:train_idx]
y_val = y[train_idx:]
u_val = u[train_idx:]

class ARXModel:
    def __init__(self, na, nb):
        self.na = na  # Order of A polynomial
        self.nb = nb  # Order of B polynomial
        self.theta = None

    def create_regressor_matrix(self, y, u):
        N = len(y)
        max_lag = max(self.na, self.nb)
        phi = np.zeros((N-max_lag, self.na + self.nb))

        for i in range(self.na):
            phi[:, i] = -y[max_lag-i-1:-i-1] if i < N-max_lag else -y[max_lag-i-1:]

        for i in range(self.nb):
            phi[:, self.na+i] = u[max_lag-i-1:-i-1] if i < N-max_lag else u[max_lag-i-1:]

        return phi, y[max_lag:]

    def fit(self, y, u):
        phi, y_est = self.create_regressor_matrix(y, u)
        self.theta = np.linalg.lstsq(phi, y_est, rcond=None)[0]
        return self

    def predict(self, y, u):
        phi, _ = self.create_regressor_matrix(y, u)
        return phi @ self.theta

class ARMAXModel:
    def __init__(self, na, nb, nc):
        self.na = na  # Order of A polynomial
        self.nb = nb  # Order of B polynomial
        self.nc = nc  # Order of C polynomial
        self.theta = None

    def simulate(self, params, y, u):
        na, nb, nc = self.na, self.nb, self.nc
        N = len(y)
        y_pred = np.zeros(N)
        e = np.zeros(N)

        # Extract parameters
        a_params = params[:na]
        b_params = params[na:na + nb]
        c_params = params[na + nb:]

        # Initial conditions
        y_pred[:max(na, nb)] = y[:max(na, nb)]

        # Simulate system
        for k in range(max(na, nb, nc), N):
            # Fix the array slicing to ensure correct indexing
            y_hist = y_pred[k - na:k][::-1]
            u_hist = u[k - nb:k][::-1]
            e_hist = e[k - nc:k][::-1]

            y_pred[k] = -np.sum(a_params * y_hist) + \
                        np.sum(b_params * u_hist) + \
                        np.sum(c_params * e_hist)
            e[k] = y[k] - y_pred[k]

        return y_pred, e

    def cost_function(self, params, y, u):
        _, e = self.simulate(params, y, u)
        return np.sum(e**2)

    def fit(self, y, u):
        # Initial parameter guess
        initial_params = np.zeros(self.na + self.nb + self.nc)

        # Optimize parameters
        result = minimize(
            self.cost_function,
            initial_params,
            args=(y, u),
            method='BFGS'
        )

        self.theta = result.x
        return self

    def predict(self, y, u):
        if self.theta is None:
            raise ValueError("Model must be fitted before making predictions")
        y_pred, _ = self.simulate(self.theta, y, u)
        return y_pred

# Fit both models
arx_model = ARXModel(na=2, nb=2)
arx_model.fit(y_train, u_train)

armax_model = ARMAXModel(na=2, nb=2, nc=1)
armax_model.fit(y_train, u_train)

# Make predictions
y_pred_arx = arx_model.predict(y_val, u_val)
y_pred_armax = armax_model.predict(y_val, u_val)

# Calculate performance metrics
def calculate_fit(y_true, y_pred):
    """Calculate fit percentage."""
    return (1 - np.linalg.norm(y_true - y_pred) /
            np.linalg.norm(y_true - np.mean(y_true))) * 100


def calculate_metrics(y_true, y_pred):
    """Calculate multiple performance metrics."""
    # Ensure equal lengths by using the shorter length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    mse = np.mean((y_true - y_pred) ** 2)
    fit = calculate_fit(y_true, y_pred)
    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'Fit': fit
    }

# Calculate and print metrics
arx_metrics = calculate_metrics(y_val, y_pred_arx)
armax_metrics = calculate_metrics(y_val, y_pred_armax)

print("\nARX Model Performance:")
for metric, value in arx_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nARMAX Model Performance:")
for metric, value in armax_metrics.items():
    print(f"{metric}: {value:.4f}")

# Ensure all arrays have same length for plotting
min_len = min(len(y_val), len(y_pred_arx), len(y_pred_armax))
y_val_plot = y_val[:min_len]
y_pred_arx_plot = y_pred_arx[:min_len]
y_pred_armax_plot = y_pred_armax[:min_len]

# Plotting
plt.figure(figsize=(15, 10))

# Plot 1: Validation Data and Predictions
plt.subplot(3, 1, 1)
plt.plot(y_val_plot, 'k-', label='True', alpha=0.7)
plt.plot(y_pred_arx_plot, 'r--', label='ARX', alpha=0.7)
plt.plot(y_pred_armax_plot, 'b--', label='ARMAX', alpha=0.7)
plt.legend()
plt.title('Model Predictions vs True Output')
plt.grid(True)

# Plot 2: Prediction Errors
plt.subplot(3, 1, 2)
plt.plot(y_val_plot - y_pred_arx_plot, 'r-', label='ARX', alpha=0.7)
plt.plot(y_val_plot - y_pred_armax_plot, 'b-', label='ARMAX', alpha=0.7)
plt.legend()
plt.title('Prediction Errors')
plt.grid(True)

# Plot 3: Error Distributions
plt.subplot(3, 1, 3)
plt.hist(y_val_plot - y_pred_arx_plot, bins=50, color='r', alpha=0.5, label='ARX', density=True)
plt.hist(y_val_plot - y_pred_armax_plot, bins=50, color='b', alpha=0.5, label='ARMAX', density=True)
plt.legend()
plt.title('Error Distributions')
plt.grid(True)

plt.tight_layout()
plt.show()