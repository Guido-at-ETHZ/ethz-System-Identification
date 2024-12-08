import numpy as np
import matplotlib.pyplot as plt

# Sensor characteristics
sensors = {
    'A': {'reading': 25.2, 'bias': 0.5, 'variance': 0.04},
    'B': {'reading': 24.8, 'bias': -0.3, 'variance': 0.09},
    'C': {'reading': 25.0, 'bias': 0.2, 'variance': 0.01}
}

# True temperature (for validation)
true_temp = 24.7


def calculate_mse_components(weights):
    """Calculate bias, variance and MSE for given weights"""
    readings = np.array([s['reading'] for s in sensors.values()])
    biases = np.array([s['bias'] for s in sensors.values()])
    variances = np.array([s['variance'] for s in sensors.values()])

    # Calculate estimate
    estimate = np.sum(weights * readings)

    # Calculate bias
    bias = np.sum(weights * biases)  # Linear combination of biases

    # Calculate variance
    variance = np.sum(weights ** 2 * variances)  # Uncorrelated sensors

    # Calculate MSE
    mse = bias ** 2 + variance

    return estimate, bias, variance, mse


# Try different weight combinations
weight_sets = {
    'Equal Weights': np.array([1 / 3, 1 / 3, 1 / 3]),
    'BLUE Weights': None,  # Will calculate below
    'MSE Optimal': None  # Will calculate below
}

# Calculate BLUE weights (inverse variance weighting)
variances = np.array([s['variance'] for s in sensors.values()])
weight_sets['BLUE Weights'] = (1 / variances) / np.sum(1 / variances)

# Find MSE optimal weights using optimization
from scipy.optimize import minimize


def mse_objective(weights):
    _, _, _, mse = calculate_mse_components(weights)
    return mse


# Constraint: weights sum to 1
constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, 1) for _ in range(3)]
result = minimize(mse_objective, weight_sets['Equal Weights'],
                  constraints=constraint, bounds=bounds)
weight_sets['MSE Optimal'] = result.x

# Calculate and display results for each weighting scheme
print("Comparison of Different Weighting Schemes:\n")
for name, weights in weight_sets.items():
    estimate, bias, variance, mse = calculate_mse_components(weights)
    print(f"{name}:")
    print(f"Weights: [{', '.join([f'{w:.3f}' for w in weights])}]")
    print(f"Estimate: {estimate:.3f}°C")
    print(f"Bias: {bias:.3f}°C")
    print(f"Variance: {variance:.5f}°C²")
    print(f"MSE: {mse:.5f}°C²")
    print(f"Error from true: {abs(estimate - true_temp):.3f}°C\n")

# Plot comparison
results = []
for name, weights in weight_sets.items():
    _, bias, variance, mse = calculate_mse_components(weights)
    results.append({
        'Method': name,
        'Bias²': bias ** 2,
        'Variance': variance,
        'MSE': mse
    })

fig, ax = plt.subplots(figsize=(10, 6))
bottom = np.zeros(3)
width = 0.5

for component in ['Bias²', 'Variance']:
    values = [r[component] for r in results]
    ax.bar([r['Method'] for r in results], values, width,
           label=component, bottom=bottom)
    bottom += values

ax.set_ylabel('Error Components')
ax.set_title('Comparison of MSE Components')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()