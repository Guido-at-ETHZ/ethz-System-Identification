import numpy as np
import matplotlib.pyplot as plt

# Given data
u = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0])
y = np.array([0, 0.5, 0.25, 0.1, 0.6, 0.3, 0.15, 0.05, 0.55, 0.25])


# a) Form regression matrix Φ and output vector Y
def create_regression_matrix(y, u, na=1, nb=1):
    """
    Create regression matrix for ARX model
    na: order of A polynomial
    nb: order of B polynomial
    """
    N = len(y)
    # We lose na initial conditions
    N_eff = N - max(na, nb)

    # Initialize regression matrix
    phi = np.zeros((N_eff, na + nb))
    Y = np.zeros(N_eff)

    # Fill regression matrix
    for i in range(N_eff):
        k = i + max(na, nb)  # Current time index
        # Fill A polynomial terms (past outputs)
        for j in range(na):
            phi[i, j] = -y[k - j - 1]
        # Fill B polynomial terms (past inputs)
        for j in range(nb):
            phi[i, j + na] = u[k - j - 1]
        Y[i] = y[k]

    return phi, Y


# Create regression matrix and output vector
Phi, Y = create_regression_matrix(y, u)

# b) Calculate parameter estimates using least squares
theta_hat = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ Y
a1_hat, b1_hat = theta_hat


# c) Compute one-step ahead predictions
def compute_predictions(y, u, a1, b1):
    """Compute one-step ahead predictions"""
    N = len(y)
    y_pred = np.zeros(N)
    # First prediction needs initial condition
    y_pred[0] = 0  # Since we don't have previous data

    # Compute predictions
    for k in range(1, N):
        y_pred[k] = -a1 * y[k - 1] + b1 * u[k - 1]

    return y_pred


y_pred = compute_predictions(y, u, a1_hat, b1_hat)

# d) Calculate prediction errors
e = y - y_pred

# Plot results
plt.figure(figsize=(15, 10))

# Plot 1: Input and Output Data
plt.subplot(3, 1, 1)
plt.plot(u, 'b.-', label='Input u(k)')
plt.plot(y, 'r.-', label='Output y(k)')
plt.grid(True)
plt.legend()
plt.title('Input and Output Data')

# Plot 2: Actual vs Predicted Output
plt.subplot(3, 1, 2)
plt.plot(y, 'r.-', label='Actual y(k)')
plt.plot(y_pred, 'g.-', label='Predicted ŷ(k|k-1)')
plt.grid(True)
plt.legend()
plt.title('Actual vs Predicted Output')

# Plot 3: Prediction Errors
plt.subplot(3, 1, 3)
plt.plot(e, 'k.-', label='Prediction Error e(k)')
plt.grid(True)
plt.legend()
plt.title('Prediction Errors')
plt.xlabel('Sample k')

plt.tight_layout()

# Print results
print(f"Estimated parameters:")
print(f"a1_hat = {a1_hat:.4f}")
print(f"b1_hat = {b1_hat:.4f}")
print(f"\nPrediction error statistics:")
print(f"Mean error: {np.mean(e):.4f}")
print(f"Variance of error: {np.var(e):.4f}")
print(f"RMS error: {np.sqrt(np.mean(e ** 2)):.4f}")