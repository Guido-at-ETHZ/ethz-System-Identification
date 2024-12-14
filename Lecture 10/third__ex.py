import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def exercise3_ill_conditioned():
    # Generate ill-conditioned problem
    N = 100
    t = np.linspace(0, 10, N)
    Phi = np.column_stack([
        np.sin(0.1 * t),
        np.sin(0.101 * t)
    ])
    theta_true = np.array([1, 2])
    y = Phi @ theta_true + 0.01 * np.random.randn(N)

    # 1. Standard LS Analysis
    # Compute and display condition number
    cond_num = np.linalg.cond(Phi.T @ Phi)
    print(f'Condition number of Phi.T @ Phi: {cond_num:.2e}')

    # Try standard LS with different noise realizations
    num_trials = 100
    ls_estimates = np.zeros((2, num_trials))
    for i in range(num_trials):
        y_noisy = Phi @ theta_true + 0.01 * np.random.randn(N)
        ls_estimates[:, i] = np.linalg.solve(Phi.T @ Phi, Phi.T @ y_noisy)

    # 2. Ridge Regression Analysis
    gamma_vals = np.logspace(-6, 2, 50)
    ridge_estimates = np.zeros((2, len(gamma_vals)))

    for i, gamma in enumerate(gamma_vals):
        ridge_estimates[:, i] = np.linalg.solve(
            Phi.T @ Phi + gamma * np.eye(2),
            Phi.T @ y
        )

    # 3. Cross-validation for optimal gamma
    K = 5  # number of folds
    cv_errors = np.zeros(len(gamma_vals))
    fold_size = N // K

    for i, gamma in enumerate(gamma_vals):
        cv_error = 0
        for k in range(K):
            # Split data
            val_idx = np.arange(k * fold_size, (k + 1) * fold_size)
            train_idx = np.setdiff1d(np.arange(N), val_idx)

            # Train model
            Phi_train = Phi[train_idx]
            y_train = y[train_idx]
            theta = np.linalg.solve(
                Phi_train.T @ Phi_train + gamma * np.eye(2),
                Phi_train.T @ y_train
            )

            # Validate
            Phi_val = Phi[val_idx]
            y_val = y[val_idx]
            cv_error += np.mean((y_val - Phi_val @ theta) ** 2)

        cv_errors[i] = cv_error / K

    # Find best gamma
    best_idx = np.argmin(cv_errors)
    best_gamma = gamma_vals[best_idx]
    best_ridge = np.linalg.solve(
        Phi.T @ Phi + best_gamma * np.eye(2),
        Phi.T @ y
    )

    # Plot results
    plot_results(theta_true, ls_estimates, ridge_estimates, cv_errors,
                 gamma_vals, best_gamma, best_ridge, Phi, y, best_idx)

    # Additional analysis
    analyze_stability(Phi, theta_true, best_gamma)


def plot_results(theta_true, ls_estimates, ridge_estimates, cv_errors,
                 gamma_vals, best_gamma, best_ridge, Phi, y, best_idx):
    """Plot comprehensive results"""
    plt.figure(figsize=(15, 10))

    # Plot LS estimates scatter
    plt.subplot(221)
    plt.scatter(ls_estimates[0], ls_estimates[1], alpha=0.3, label='LS estimates')
    plt.plot(theta_true[0], theta_true[1], 'r*', markersize=15, label='True')
    plt.title('LS Estimates with Different Noise Realizations')
    plt.xlabel('θ₁')
    plt.ylabel('θ₂')
    plt.legend()
    plt.grid(True)

    # Plot ridge path
    plt.subplot(222)
    plt.plot(ridge_estimates[0], ridge_estimates[1], 'b-', label='Ridge path')
    plt.plot(theta_true[0], theta_true[1], 'r*', markersize=15, label='True')
    plt.plot(best_ridge[0], best_ridge[1], 'go', markersize=10, label='Best Ridge')
    plt.title('Ridge Regression Path')
    plt.xlabel('θ₁')
    plt.ylabel('θ₂')
    plt.legend()
    plt.grid(True)

    # Plot CV error
    plt.subplot(223)
    plt.semilogx(gamma_vals, cv_errors)
    plt.plot(best_gamma, cv_errors[best_idx], 'ro')
    plt.title('Cross-validation Error vs γ')
    plt.xlabel('γ')
    plt.ylabel('CV Error')
    plt.grid(True)

    # Plot fit
    plt.subplot(224)
    t = np.linspace(0, 10, len(y))
    plt.plot(t, y, 'k.', label='Data')
    plt.plot(t, Phi @ theta_true, 'r-', label='True')
    plt.plot(t, Phi @ best_ridge, 'b--', label='Best Ridge')
    plt.title('Data Fit')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def analyze_stability(Phi, theta_true, best_gamma):
    """Analyze numerical stability"""
    # Compare condition numbers
    cond_original = np.linalg.cond(Phi.T @ Phi)
    cond_ridge = np.linalg.cond(Phi.T @ Phi + best_gamma * np.eye(2))

    print("\nStability Analysis:")
    print(f"Original condition number: {cond_original:.2e}")
    print(f"Ridge condition number: {cond_ridge:.2e}")
    print(f"Condition number reduction: {cond_original / cond_ridge:.2f}x")

    # Analyze sensitivity to perturbations
    delta = 1e-6
    perturbed_y = Phi @ theta_true + delta * np.random.randn(len(Phi))

    # LS solution
    ls_perturbed = np.linalg.solve(Phi.T @ Phi, Phi.T @ perturbed_y)

    # Ridge solution
    ridge_perturbed = np.linalg.solve(
        Phi.T @ Phi + best_gamma * np.eye(2),
        Phi.T @ perturbed_y
    )

    print("\nSensitivity Analysis:")
    print(f"LS perturbation norm: {np.linalg.norm(ls_perturbed - theta_true):.2e}")
    print(f"Ridge perturbation norm: {np.linalg.norm(ridge_perturbed - theta_true):.2e}")


if __name__ == "__main__":
    np.random.seed(0)  # for reproducibility
    exercise3_ill_conditioned()
