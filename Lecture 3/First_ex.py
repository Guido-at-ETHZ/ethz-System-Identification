import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([1, 2, 3])
y = np.array([2.1, 3.8, 6.2])
sigma_sq = 0.1

# Form matrices
Phi = x.reshape(-1, 1)  # Make it a column vector
Y = y.reshape(-1, 1)

# Compute least squares estimate
theta_hat = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ Y
print(f"Least squares estimate θ̂ = {float(theta_hat):.3f}")  # Convert to float before formatting

# c) Variance of the estimate
var_theta = sigma_sq * np.linalg.inv(Phi.T @ Phi)
print(f"Variance of estimate = {float(var_theta):.3f}")

# d) Verify it minimizes squared error
# Compare with nearby values
theta_test = np.linspace(float(theta_hat)-0.5, float(theta_hat)+0.5, 100)
errors = np.zeros_like(theta_test)
for i, t in enumerate(theta_test):
    pred = Phi * t
    errors[i] = np.sum((Y - pred)**2)

# Plot the squared error
plt.figure(figsize=(10, 6))
plt.plot(theta_test, errors)
plt.axvline(theta_hat, color='r', linestyle='--', label=f'θ̂={float(theta_hat):.3f}')
plt.xlabel('θ')
plt.ylabel('Squared Error')
plt.title('Squared Error vs θ')
plt.grid(True)
plt.legend()
plt.show()

# Plot the data and fitted line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data')
x_line = np.linspace(0, 4, 100)
plt.plot(x_line, float(theta_hat) * x_line, 'r--',
         label=f'Fitted line (θ={float(theta_hat):.3f})')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data and Least Squares Fit')
plt.grid(True)
plt.legend()
plt.show()