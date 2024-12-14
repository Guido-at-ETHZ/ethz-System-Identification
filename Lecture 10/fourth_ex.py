import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def exercise4_james_stein():
    """
    Implementation and analysis of James-Stein estimator compared to LS
    """
    # Setup parameters
    N = 10  # dimension
    n_trials = 1000  # number of Monte Carlo trials
    sigma = 1.0  # noise standard deviation

    # Different true parameter norms to test
    theta_norms = np.logspace(-1, 2, 20)

    # Storage for results
    mse_ls = np.zeros(len(theta_norms))
    mse_js = np.zeros(len(theta_norms))
    bias_js = np.zeros(len(theta_norms))
    var_js = np.zeros(len(theta_norms))

    # Monte Carlo simulation
    for i, norm in enumerate(theta_norms):
        # Create true parameter with specified norm
        theta_true = np.ones(N) * norm / np.sqrt(N)

        # Storage for this set of trials
        ls_errors = np.zeros(n_trials)
        js_errors = np.zeros(n_trials)
        js_estimates = np.zeros((n_trials, N))

        for j in range(n_trials):
            # Generate noisy observation
            Y = theta_true + sigma * np.random.randn(N)

            # LS estimate (just Y in this case)
            theta_ls = Y

            # James-Stein estimate
            theta_js = james_stein_estimator(Y, sigma)

            # Store results
            ls_errors[j] = np.sum((theta_ls - theta_true) ** 2)
            js_errors[j] = np.sum((theta_js - theta_true) ** 2)
            js_estimates[j] = theta_js

        # Compute MSE
        mse_ls[i] = np.mean(ls_errors)
        mse_js[i] = np.mean(js_errors)

        # Compute bias and variance for JS
        bias_js[i] = np.sum((np.mean(js_estimates, axis=0) - theta_true) ** 2)
        var_js[i] = np.mean([np.sum((est - np.mean(js_estimates, axis=0)) ** 2)
                             for est in js_estimates])

    # Plot results
    plot_results(theta_norms, mse_ls, mse_js, bias_js, var_js)

    # Additional analysis
    analyze_shrinkage_effect(N, sigma)


def james_stein_estimator(Y, sigma):
    """
    Compute James-Stein estimate
    """
    N = len(Y)
    Y_norm_sq = np.sum(Y ** 2)

    # Compute shrinkage factor
    shrinkage = max(0, 1 - (N - 2) * sigma ** 2 / Y_norm_sq)

    return shrinkage * Y


def plot_results(theta_norms, mse_ls, mse_js, bias_js, var_js):
    """
    Plot comprehensive results of the analysis
    """
    plt.figure(figsize=(15, 10))

    # Plot MSE comparison
    plt.subplot(221)
    plt.loglog(theta_norms, mse_ls, 'b-', label='LS')
    plt.loglog(theta_norms, mse_js, 'r-', label='James-Stein')
    plt.grid(True)
    plt.legend()
    plt.xlabel('||θ||')
    plt.ylabel('MSE')
    plt.title('MSE Comparison')

    # Plot relative MSE improvement
    plt.subplot(222)
    plt.semilogx(theta_norms, (mse_ls - mse_js) / mse_ls * 100, 'g-')
    plt.grid(True)
    plt.xlabel('||θ||')
    plt.ylabel('Relative MSE Improvement (%)')
    plt.title('James-Stein Improvement over LS')

    # Plot bias-variance decomposition
    plt.subplot(223)
    plt.loglog(theta_norms, bias_js, 'b-', label='Bias²')
    plt.loglog(theta_norms, var_js, 'r-', label='Variance')
    plt.loglog(theta_norms, bias_js + var_js, 'k--', label='Total MSE')
    plt.grid(True)
    plt.legend()
    plt.xlabel('||θ||')
    plt.ylabel('Error Components')
    plt.title('Bias-Variance Decomposition of JS')

    # Plot MSE ratio
    plt.subplot(224)
    plt.semilogx(theta_norms, mse_js / mse_ls, 'b-')
    plt.grid(True)
    plt.xlabel('||θ||')
    plt.ylabel('MSE Ratio (JS/LS)')
    plt.title('Relative Efficiency')

    plt.tight_layout()
    plt.show()


def analyze_shrinkage_effect(N, sigma):
    """
    Analyze how shrinkage factor changes with observation norm
    """
    Y_norms = np.logspace(-1, 2, 1000)
    shrinkage_factors = np.maximum(0, 1 - (N - 2) * sigma ** 2 / Y_norms ** 2)

    plt.figure(figsize=(10, 5))

    plt.semilogx(Y_norms, shrinkage_factors)
    plt.grid(True)
    plt.xlabel('||Y||')
    plt.ylabel('Shrinkage Factor')
    plt.title('James-Stein Shrinkage Factor vs Observation Norm')

    # Add threshold line
    threshold = np.sqrt((N - 2) * sigma ** 2)
    plt.axvline(threshold, color='r', linestyle='--',
                label=f'Threshold = {threshold:.2f}')
    plt.legend()

    plt.show()


def theoretical_analysis():
    """
    Print theoretical insights about the James-Stein estimator
    """
    print("\nTheoretical Analysis of James-Stein Estimator:")
    print("1. The JS estimator is a shrinkage estimator that pulls the LS estimate")
    print("   toward zero by a data-dependent factor.")
    print("\n2. The shrinkage factor depends on:")
    print("   - The dimension N")
    print("   - The noise variance σ²")
    print("   - The norm of the observation ||Y||²")
    print("\n3. Key properties:")
    print("   - Always improves on LS for N ≥ 3")
    print("   - Maximum improvement when ||θ|| is small")
    print("   - Asymptotically equivalent to LS as ||θ|| → ∞")
    print("   - Inadmissible but better than LS in total MSE")


if __name__ == "__main__":
    np.random.seed(0)  # for reproducibility

    # Run main analysis
    exercise4_james_stein()

    # Print theoretical insights
    theoretical_analysis()
