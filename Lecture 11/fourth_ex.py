import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def generate_system_data(N):
    """Generate data from the given system."""
    np.random.seed(42)  # For reproducibility
    u = np.random.randn(N)
    e = 0.1 * np.random.randn(N)
    y = np.zeros(N)

    # System: y(k) = 0.8*y(k-1) - 0.2*y(k-2) + u(k-1) + 0.5*u(k-2) + e(k) + 0.3*e(k-1)
    for k in range(2, N):
        y[k] = 0.8 * y[k - 1] - 0.2 * y[k - 2] + u[k - 1] + 0.5 * u[k - 2] + e[k] + 0.3 * e[k - 1]

    return y, u


class ARXModel:
    """ARX Model Implementation"""

    def __init__(self, na, nb):
        self.na = na
        self.nb = nb
        self.theta = None

    def create_regressor_matrix(self, y, u):
        N = len(y)
        max_lag = max(self.na, self.nb)
        phi = np.zeros((N - max_lag, self.na + self.nb))

        for i in range(self.na):
            phi[:, i] = -y[max_lag - 1 - i:-i - 1 if i < N - max_lag else None]
        for i in range(self.nb):
            phi[:, self.na + i] = u[max_lag - 1 - i:-i - 1 if i < N - max_lag else None]

        return phi, y[max_lag:]

    def fit(self, y, u):
        phi, y_train = self.create_regressor_matrix(y, u)
        self.theta = np.linalg.lstsq(phi, y_train, rcond=None)[0]
        return self

    def predict(self, y, u):
        phi, _ = self.create_regressor_matrix(y, u)
        return phi @ self.theta


class ARMAXModel:
    """ARMAX Model Implementation"""

    def __init__(self, na, nb, nc):
        self.na = na
        self.nb = nb
        self.nc = nc
        self.theta = None

    def predict_one_step(self, y, u, e, k, params):
        """Predict one step ahead"""
        a_params = params[:self.na]
        b_params = params[self.na:self.na + self.nb]
        c_params = params[self.na + self.nb:]

        # Get the required past values (with proper indexing)
        y_past = y[k - self.na:k][::-1] if k >= self.na else np.pad(y[max(0, k - self.na):k][::-1],
                                                                    (self.na - min(k, self.na), 0))
        u_past = u[k - self.nb:k][::-1] if k >= self.nb else np.pad(u[max(0, k - self.nb):k][::-1],
                                                                    (self.nb - min(k, self.nb), 0))
        e_past = e[k - self.nc:k][::-1] if k >= self.nc else np.pad(e[max(0, k - self.nc):k][::-1],
                                                                    (self.nc - min(k, self.nc), 0))

        # Compute prediction
        y_pred = -np.sum(a_params * y_past) + \
                 np.sum(b_params * u_past) + \
                 np.sum(c_params * e_past)

        return y_pred

    def compute_prediction_error(self, params, y, u):
        """Compute prediction errors"""
        N = len(y)
        e = np.zeros(N)
        y_pred = np.zeros(N)

        # Compute predictions and errors
        for k in range(max(self.na, self.nb, self.nc), N):
            y_pred[k] = self.predict_one_step(y, u, e, k, params)
            e[k] = y[k] - y_pred[k]

        return e[max(self.na, self.nb, self.nc):]

    def fit(self, y, u):
        def objective(params):
            e = self.compute_prediction_error(params, y, u)
            return np.sum(e ** 2)

        # Initial parameter guess
        initial_params = np.zeros(self.na + self.nb + self.nc)

        # Optimize
        result = minimize(objective, initial_params, method='BFGS')
        self.theta = result.x
        return self

    def predict(self, y, u):
        N = len(y)
        y_pred = np.zeros(N)
        e = np.zeros(N)

        for k in range(max(self.na, self.nb, self.nc), N):
            y_pred[k] = self.predict_one_step(y, u, e, k, self.theta)
            e[k] = y[k] - y_pred[k]

        return y_pred


# Generate data
N = 1000
y, u = generate_system_data(N)

# Split data
train_size = int(0.7 * N)
y_train, y_test = y[:train_size], y[train_size:]
u_train, u_test = u[:train_size], u[train_size:]

# Fit models
arx_model = ARXModel(na=2, nb=2)
arx_model.fit(y_train, u_train)

armax_model = ARMAXModel(na=2, nb=2, nc=1)
armax_model.fit(y_train, u_train)

# Make predictions
y_pred_arx = arx_model.predict(y_test, u_test)
y_pred_armax = armax_model.predict(y_test, u_test)


# Compute metrics
# Modify the way we handle predictions and compute metrics:

def compute_metrics(y_true, y_pred):
    # Make sure we use the same length for both arrays
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    var_acc = 1 - np.var(y_true - y_pred) / np.var(y_true)
    return {'MSE': mse, 'RMSE': rmse, 'VAF': var_acc}


# When computing metrics and plotting, use:
max_lag = max(arx_model.na, arx_model.nb, armax_model.na, armax_model.nb, armax_model.nc)

# Print metrics
print("\nARX Model Performance:")
arx_metrics = compute_metrics(y_test[max_lag:], y_pred_arx[max_lag:])
for metric, value in arx_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nARMAX Model Performance:")
armax_metrics = compute_metrics(y_test[max_lag:], y_pred_armax[max_lag:])
for metric, value in armax_metrics.items():
    print(f"{metric}: {value:.4f}")

# For plotting, also use the same ranges:
plt.figure(figsize=(15, 12))

# Plot 1: True vs Predicted
plt.subplot(311)
plt.plot(y_test[max_lag:], 'k-', label='True', alpha=0.7)
plt.plot(y_pred_arx[max_lag:], 'r--', label='ARX', alpha=0.7)
plt.plot(y_pred_armax[max_lag:], 'b--', label='ARMAX', alpha=0.7)
plt.legend()
plt.title('Model Predictions vs True Output')
plt.grid(True)

# Plot 2: Prediction Errors
plt.subplot(312)
plt.plot(y_test[max_lag:] - y_pred_arx[max_lag:], 'r-', label='ARX Error', alpha=0.7)
plt.plot(y_test[max_lag:] - y_pred_armax[max_lag:], 'b-', label='ARMAX Error', alpha=0.7)
plt.legend()
plt.title('Prediction Errors')
plt.grid(True)

# Plot 3: Error Distributions
plt.subplot(313)
plt.hist(y_test[max_lag:-2] - y_pred_arx[max_lag:], bins=50, color='r', alpha=0.5, label='ARX', density=True)
plt.hist(y_test[max_lag:-2] - y_pred_armax[max_lag:], bins=50, color='b', alpha=0.5, label='ARMAX', density=True)
plt.legend()
plt.title('Error Distributions')
plt.grid(True)

plt.tight_layout()
plt.show()