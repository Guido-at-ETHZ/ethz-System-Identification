import numpy as np

# Given measurements and variances
measurements = np.array([5.2, 5.0, 5.3])
variances = np.array([0.04, 0.09, 0.01])

# 1. Calculate BLUE weights
# Using formula: wi = (1/σi²) / Σ(1/σj²)
inv_variances = 1/variances
weights_blue = inv_variances / np.sum(inv_variances)

# 2. Calculate simple average weights
weights_simple = np.ones_like(measurements) / len(measurements)

# 3. Calculate estimates
estimate_blue = np.sum(weights_blue * measurements)
estimate_simple = np.sum(weights_simple * measurements)

# 4. Calculate variance of BLUE estimate
# Var(BLUE) = 1/Σ(1/σi²)
variance_blue = 1/np.sum(inv_variances)

# 5. Calculate variance of simple average
# Var(simple) = Σσi²/n²
variance_simple = np.sum(variances)/(len(measurements)**2)

print(f"BLUE Weights: {weights_blue}")
print(f"BLUE Estimate: {estimate_blue:.3f}V")
print(f"BLUE Estimate Variance: {variance_blue:.5f}")
print("\nSimple Average Weights: {weights_simple}")
print(f"Simple Average Estimate: {estimate_simple:.3f}V")
print(f"Simple Average Variance: {variance_simple:.5f}")

# Compute MSE for both methods (assuming true value is unknown)
# Note: This is the variance for unbiased estimators
print(f"\nBLUE MSE: {variance_blue:.5f}")
print(f"Simple Average MSE: {variance_simple:.5f}")