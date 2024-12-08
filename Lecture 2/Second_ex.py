import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Given measurements and variances
measurements = np.array([22.5, 23.1, 22.8, 22.2])
variances = np.array([0.01, 0.04, 0.04, 0.09])

# Prior distribution parameters
prior_mean = 23.0
prior_var = 0.25

# 1. BLUE Estimation
inv_variances = 1/variances
weights_blue = inv_variances / np.sum(inv_variances)
estimate_blue = np.sum(weights_blue * measurements)
variance_blue = 1/np.sum(inv_variances)

# 2. Maximum Likelihood Estimation
# For normal distributions with known variances, MLE is weighted average
estimate_mle = estimate_blue  # In this case, same as BLUE

# 3. MAP Estimation
# Combine prior with measurements
posterior_var = 1/(1/prior_var + np.sum(inv_variances))
posterior_mean = posterior_var * (prior_mean/prior_var +
                                np.sum(measurements * inv_variances))

print("BLUE Estimation:")
print(f"Weights: {weights_blue}")
print(f"Estimate: {estimate_blue:.3f}°C")
print(f"Variance: {variance_blue:.5f}°C²")
print(f"\nMAP Estimation:")
print(f"Estimate: {posterior_mean:.3f}°C")
print(f"Variance: {posterior_var:.5f}°C²")

# Plot the distributions
x = np.linspace(21.5, 23.5, 1000)
prior = norm.pdf(x, prior_mean, np.sqrt(prior_var))
likelihood = norm.pdf(x, estimate_blue, np.sqrt(variance_blue))
posterior = norm.pdf(x, posterior_mean, np.sqrt(posterior_var))

plt.figure(figsize=(10, 6))
plt.plot(x, prior, 'g--', label='Prior')
plt.plot(x, likelihood, 'b-', label='Likelihood')
plt.plot(x, posterior, 'r-', label='Posterior')
plt.legend()
plt.xlabel('Temperature (°C)')
plt.ylabel('Density')
plt.title('Temperature Estimation Distributions')
plt.grid(True)
plt.show()