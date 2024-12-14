import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

# Given data
u = np.array([1, -1, 1, -1, 0, 0, 1, -1, 1, -1])
y = np.array([0.5, -0.8, 1.2, -1.5, -0.3, 0.2, 0.9, -1.1, 1.4, -1.7])

# ARX Model parameters
na = 2  # Order of A polynomial
nb = 2  # Order of B polynomial


def create_regressor_matrix(y, u, na, nb):
    """Create regressor matrix for ARX model."""
    N = len(y)
    max_lag = max(na, nb)
    phi = np.zeros((N - max_lag, na + nb))

    # Fill in y regressors
    for i in range(na):
        phi[:, i] = -y[max_lag - 1 - i:N - 1 - i]

    # Fill in u regressors
    for i in range(nb):
        phi[:, na + i] = u[max_lag - 1 - i:N - 1 - i]

    return phi


# Create regressor matrix
phi = create_regressor_matrix(y, u, na, nb)

# Create output vector (removing initial samples due to lags)
y_est = y[max(na, nb):]

# Least squares estimation
theta_hat = np.linalg.inv(phi.T @ phi) @ phi.T @ y_est

# Print estimated parameters
print("Estimated parameters:")
print(f"a1 = {theta_hat[0]:.4f}")
print(f"a2 = {theta_hat[1]:.4f}")
print(f"b1 = {theta_hat[2]:.4f}")
print(f"b2 = {theta_hat[3]:.4f}")


# Model validation - generate predicted output
def predict_arx(u, y, theta, na, nb):
    N = len(u)
    y_pred = np.zeros(N)
    y_pred[:max(na, nb)] = y[:max(na, nb)]  # Initialize with actual values

    for k in range(max(na, nb), N):
        y_pred[k] = -theta[0] * y_pred[k - 1] - theta[1] * y_pred[k - 2] + \
                    theta[2] * u[k - 1] + theta[3] * u[k - 2]

    return y_pred


# Generate predictions
y_pred = predict_arx(u, y, theta_hat, na, nb)

# Calculate residuals
residuals = y - y_pred

# Plotting
plt.figure(figsize=(12, 8))

# Plot 1: Measured vs Predicted Output
plt.subplot(2, 1, 1)
plt.plot(y, 'b-', label='Measured')
plt.plot(y_pred, 'r--', label='Predicted')
plt.grid(True)
plt.legend()
plt.title('ARX Model - Measured vs Predicted Output')
plt.xlabel('Sample')
plt.ylabel('Output')

# Plot 2: Residuals
plt.subplot(2, 1, 2)
plt.plot(residuals, 'k.')
plt.grid(True)
plt.title('Prediction Residuals')
plt.xlabel('Sample')
plt.ylabel('Residual')

plt.tight_layout()
plt.show()

# Calculate fit metrics
mse = np.mean(residuals ** 2)
print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.6f}")
print(f"Root Mean Squared Error: {np.sqrt(mse):.6f}")

# Correlation analysis of residuals
acf_residuals = np.correlate(residuals, residuals, mode='full') / len(residuals)
acf_residuals = acf_residuals[len(acf_residuals) // 2:]  # Take only positive lags

print("\nResidual Analysis:")
print(f"Residual autocorrelation at lag 1: {acf_residuals[1]:.4f}")